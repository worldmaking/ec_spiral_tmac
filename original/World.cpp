#include "World.h"

#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"
#include "cinder/Camera.h"
#include "cinder/Font.h"
#include "cinder/ImageIo.h"
#include "cinder/Log.h"
#include "cinder/Perlin.h"
#include "cinder/Rand.h"
#include "cinder/Thread.h"
#include "cinder/Utilities.h"
#include "cinder/Vector.h"

#include <deque>

// there is only one world:
static World world;

double now = 0;
double fluid_amount = 0.05;
int fluid_passes = 14;
int plane = 0;
std::deque<int32_t> organism_pool;

SineData OrganismSound::__sinedata;
Noise OrganismSound::noise;
double OrganismSound::grainwindow[AN_AUDIO_BUFFER_SIZE];
double OrganismSound::sinewindow[AN_AUDIO_BUFFER_SIZE];
Reverb reverb;
AmbiDomain * ambi_bus = 0;
AmbiDomain headphoneL, headphoneR;

// backwards compat:
World * global = &world;

World& World::get() { return world; }


// TODO: not sure if this math makes sense
template<typename T, glm::precision P>
glm::tquat<T, P> quat_toward_point(glm::tquat<T, P> const & self, glm::tvec3<T, P> const & v, glm::tquat<T, P> const & q, glm::tvec3<T, P> const & tp)
{
	// local z
	glm::tvec3<T, P> vz = quat_uz(q);

	// local to other
	glm::tvec3<T, P> axis = glm::normalize(tp - v);

	// half-way z to axis
	axis = glm::mix(vz, axis, 0.5);

	T len_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];

	if (len_sq < (10e-6f)) {
		//rotate 90 degrees (take the -x axis of the current orientation)
		//potentially need more info if this is time based because
		//there is a unique 90 degree rotation depending on previous
		//position of target point.  Here it's arbitrarily decided.
		axis = -quat_ux(q);

		glm::tquat<T, P> q_inv = glm::inverse(q);
		//q.inverse(q_inv);

		// TODO: left or right multiply?
		// right multiply: orientation = orientation * offset;
		// When you right-multiply, the offset orientation is in model space. When you left-multiply, the offset is in world space. Both of these can be useful for different purposes.

		return q_inv * self; //not_multiply(q_inv);
							 // or  self * q_inv?
	}
	else {
		if (len_sq < 0.001) {
			axis = glm::normalize(axis);
		}

		glm::tquat<T, P> q1;

		glm::tquat<T, P> rot = glm::angleAxis(M_PI, axis);

		q1 = rot * q;

		//flip_about_z();

		return glm::tquat<T, P>(-q1.z, q1.y, -q1.x, q1.w);
	}

	//printf("axis: %f %f %f  %f\n", axis[0], axis[1], axis[2], len_sq);
}

/// AUDIO ///

inline void ambiEncodeFuMa(AmbiDomain& ws, double x, double y, double z) {
	static const double c1_sqrt2 = 0.707106781186548;
	// 0th order
	ws.w = c1_sqrt2;				// W channel (shouldn't it be already defined?)
									// 1st order components are simply the direction vector
	ws.x = x;						// X = cos(A)cos(E)
	ws.y = y;						// Y = sin(A)cos(E)
	ws.z = z;						// Z = sin(E)
									/*
									// 2nd order
									double x2 = x * x;
									double y2 = y * y;
									double z2 = z * z;
									ws.u = x2 - y2;					// U = cos(2A)cos2(E) = xx-yy
									ws.v = 2. * x * y;				// V = sin(2A)cos2(E) = 2xy
									ws.s = 2. * z * x;				// S = cos(A)sin(2E) = 2zx
									ws.t = 2. * z * y;				// T = sin(A)sin(2E) = 2yz
									ws.r = 1.5 * z2 - 0.5;			// R = 1.5sin2(E)-0.5 = 1.5zz-0.5
									// 3rd order
									static const double c8_11		= 8./11.;
									static const double c40_11		= 40./11.;
									double pre = c40_11 * z2 - c8_11;
									ws.p = x * (x2 - 3. * y2);		// P = cos(3A)cos3(E) = X(X2-3Y2)
									ws.q = y * (y2 - 3. * x2);		// Q = sin(3A)cos3(E) = Y(3X2-Y2)
									ws.n = z * (x2 - y2) * 0.5;		// N = cos(2A)sin(E)cos2(E) = Z(X2-Y2)/2
									ws.o = x * y * z;				// O = sin(2A)sin(E)cos2(E) = XYZ
									ws.l = pre * x;					// L = 8cos(A)cos(E)(5sin2(E) - 1)/11 = 8X(5Z2-1)/11
									ws.m = pre * y;					// M = 8sin(A)cos(E)(5sin2(E) - 1)/11 = 8Y(5Z2-1)/11
									ws.k = z * (2.5 * z2 - 1.5);	// K = sin(E)(5sin2(E) - 3)/2 = Z(5Z2-3)/2
									*/
}

inline void organism_sound(int type, organism& o) {

	double mindist = world.sound_mindist;
	double distscale = world.sound_distscale;

	static const double ms2samps = 44100. / 1000.;

	Sound& s = o.sound;
	if (s.type && type != SOUND_DEATH) return;	// already playing (but death sound interrupts)

												// calculate az, el, dist.
												// bail if distance is too great.
	glm::dvec3 rel0;
	rel0.x = o.thing.pos.x - world.pos.x;
	rel0.y = o.thing.pos.y - world.pos.y;
	rel0.z = o.thing.pos.z - world.pos.z;

	// (the alternative would be to apply rotation to the speakers!)
	// rotate rel into the current view:
	rel0 = quat_unrotate(world.orient, rel0); // or quat_unrotate?

	glm::dvec3 rel = rel0;
	double dist2 = rel.x*rel.x + rel.y*rel.y + rel.z*rel.z;

	// normalize to world:
	double dist2n = (mindist + sqrt(sqrt(dist2 * distscale)));
	if (dist2n > 1.) return;	// too far away

								//printf("organism_sound %d %d %f %d %f\n", o.thing.id, type, dist2, DIM*DIM, dist2n);


	s.start = world.sound_latency + rand() % world.sound_distribution;
	s.elapsed = 0;
	s.dur = 0;
	s.param_age = o.age;

	// normalize rel:
	double dist = sqrt(dist2);
	double scale = 1. / dist;
	rel.x *= scale;
	rel.y *= scale;
	rel.z *= scale;


	// coordinate change for ambi:
	ambiEncodeFuMa(s.ambi, rel.x, rel.y, rel.z);

	s.attenuation = world.sound_volume * sqrt(1. - dist2n);

	// to get the pan angle, use dot product of rel with the world.ux?





	switch (type) {
	case SOUND_BIRTH:

		s.dur = int(ms2samps * (10. + 0.2 * s.param_age));
		s.param_rand = World::urandom() * s.param_age / 40.;
		s.phasei_4 = 0;
		s.pincr_4 = 0;
		s.phasei_5 = 0;
		s.pincr_5 = 0;
		o.flash = 1;

		break;
	case SOUND_EAT:

		s.dur = int(ms2samps * 5.);
		s.m_cutoff_1 = 100 * ((rand() % 10) + 10);
		s.m_x_2 = 0;
		s.m_x_3 = 0;
		s.m_y_4 = 0;
		s.m_y_5 = 0;
		o.flash = 1;

		break;
		//case SOUND_MEET: break;
	case SOUND_DEATH:

		s.dur = int(ms2samps * (12. + s.param_age * 0.5));
		s.param_rand = (rand() % 5) * -2;
		s.m_history_1 = 0;
		s.m_phase = 0;
		s.x1 = 0;
		s.y1 = 0;

		break;
	default:
		break;
	}

	// mark as ready:
	s.type = type;

	// also send:
	SoundEvent& sound = world.topSound();
	sound.state = 1;
	sound.type = type;
	sound.pos.x = (float)rel0.x;
	sound.pos.y = (float)rel0.y;
	sound.pos.z = (float)rel0.z;
	sound.p0 = (float)s.param_age;
	world.commitTopSound();
}

/// ORGANISM POOL ///


void organism_pool_push(organism& o) {
	//printf("push beetle %d %d\n", b, world.organism_pool.size());
	organism_pool.push_back(o.thing.id);
	//o.recycle = 1;
	o.thing.alive = 0;
}

int32_t organism_pool_pop() {
	int b = -1;
	if (!organism_pool.empty()) {
		b = organism_pool.front();
		organism_pool.pop_front();
	}
	return b;
}

void organism_pool_clear() {
	while (!organism_pool.empty()) organism_pool_pop();
}

/// VOXEL HASHSPACE ///


inline uint32_t voxel_hash(glm::dvec3 pos) {
	static const int32_t DIMX = DIM;
	static const int32_t DIMY = DIM;
	static const int32_t DIMZ = DIM;

	int32_t x = int32_t(pos.x), y = int32_t(pos.y), z = int32_t(pos.z);
	if (x < 0 || x >= DIMX || y < 0 || y >= DIMY || z < 0 || z >= DIMZ) {
		return INVALID_VOXEL_HASH;
	}
	return x + DIMX*(y + DIMY*z);
}

// assumes pos is > 0 in all axes:
inline uint32_t voxel_hashf(glm::vec3 pos) {
	static const int32_t DIMX = DIM;
	static const int32_t DIMY = DIM;

	int32_t x = int32_t(pos.x), y = int32_t(pos.y), z = int32_t(pos.z);
	x &= DIMX - 1;
	y &= DIMX - 1;
	z &= DIMX - 1;
	return x + DIMX*(y + DIMY*z);
}

inline uint32_t voxel_hashf_safe(glm::vec3 pos) {
	static const int32_t DIMX = DIM;
	static const int32_t DIMY = DIM;
	static const float OFFSET = float(2 ^ 30);

	int32_t x = int32_t(OFFSET + pos.x), y = int32_t(OFFSET + pos.y), z = int32_t(OFFSET + pos.z);
	x &= DIMX - 1;
	y &= DIMX - 1;
	z &= DIMX - 1;
	return x + DIMX*(y + DIMY*z);
}

// assumes pos is > 0 in all axes:
inline uint32_t voxel_hashf(float xf, float yf, float zf) {
	static const int32_t DIMX = DIM;
	static const int32_t DIMY = DIM;

	int32_t x = int32_t(xf), y = int32_t(yf), z = int32_t(zf);
	x &= DIMX - 1;
	y &= DIMX - 1;
	z &= DIMX - 1;
	return x + DIMX*(y + DIMY*z);
}

inline uint32_t voxel_hash_safe(double xf, double yf, double zf) {
	static const int32_t DIMX = DIM;
	static const int32_t DIMY = DIM;
	static const double OFFSET = double(2 ^ 30);

	int32_t x = int32_t(OFFSET + xf), y = int32_t(OFFSET + yf), z = int32_t(OFFSET + zf);
	x &= DIMX - 1;
	y &= DIMX - 1;
	z &= DIMX - 1;
	return x + DIMX*(y + DIMY*z);
}

inline uint32_t voxel_hashf_safe(float xf, float yf, float zf) {
	static const int32_t DIMX = DIM;
	static const int32_t DIMY = DIM;
	static const float OFFSET = float(2 ^ 30);

	int32_t x = int32_t(OFFSET + xf), y = int32_t(OFFSET + yf), z = int32_t(OFFSET + zf);
	x &= DIMX - 1;
	y &= DIMX - 1;
	z &= DIMX - 1;
	return x + DIMX*(y + DIMY*z);
}



inline void voxel_push_particle(Voxel& v, particle& p) {
	p.neighbor = v.particles;
	v.particles = &p;
}

inline particle * voxel_top_particle(Voxel& v) {
	return v.particles;
}

inline particle * voxel_pop_particle(Voxel& v) {
	particle * p = voxel_top_particle(v);
	if (p) {
		v.particles = p->neighbor;
		p->neighbor = 0;
	}
	return p;
}

inline void voxel_clear_particles(Voxel& v) {
	v.particles = 0;
}

void World::setup() {

	fluid.initialize(DIM, DIM, DIM);
	noisefield.initialize(DIM, DIM, DIM, 4);
	for (int i = 0; i<NUM_VOXELS; i++) {
		glm::vec4 * n = (glm::vec4 *)noisefield[i];
		*n = glm::linearRand(glm::vec4(0.), glm::vec4(1.));
		boundary[i] = 1;
	}

	for (int i = 0; i<AN_AUDIO_BUFFER_SIZE; i++) {
		double norm = i / (double)AN_AUDIO_BUFFER_SIZE;
		double a = pow(1. - norm, 2.5);
		double b = sin(sin(M_PI*a));
		double c = a*b*1.8;
		OrganismSound::grainwindow[i] = c;
		OrganismSound::sinewindow[i] = sin(norm * M_PI * 2.);
	}

	dsp_initialize(44100., 1024);

	// configure headphones (not quite the same spatial position as view plane)
	glm::dvec3 l;
	l.x = 1.;
	l.y = -0.05;
	l.z = 0.1;
	l = glm::normalize(l);
	ambiEncodeFuMa(headphoneL, l.x, l.y, l.z);
	ambiEncodeFuMa(headphoneR, -l.x, l.y, l.z);

	// build stalkMesh:
	double radius = 1.;
	int columns = 6;
	int rows = 6;
	int i = 0;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < columns; c++) {

			double angle0 = M_PI * 2.0 * c / columns;
			double angle1 = M_PI * 2.0 * (c + 1) / columns;
			double x0 = radius * cos(angle0);
			double y0 = radius * sin(angle0);
			double x1 = radius * cos(angle1);
			double y1 = radius * sin(angle1);

			// QUAD:
			stalkMesh[i].normal = glm::vec3(-sin(angle0), -cos(angle0), 0);
			stalkMesh[i].pos = glm::vec3(x0, y0, r / (rows));
			i = i + 1;
			stalkMesh[i].normal = glm::vec3(-sin(angle0), -cos(angle0), 0);
			stalkMesh[i].pos = glm::vec3(x0, y0, (r + 1) / (rows));
			i = i + 1;
			stalkMesh[i].normal = glm::vec3(-sin(angle1), -cos(angle1), 0);
			stalkMesh[i].pos = glm::vec3(x1, y1, (r + 1) / (rows));
			i = i + 1;
			stalkMesh[i].normal = glm::vec3(-sin(angle1), -cos(angle1), 0);
			stalkMesh[i].pos = glm::vec3(x1, y1, r / (rows));
			i = i + 1;
		}
	}

	// build organismMesh
	for (int i = 0; i < NUM_ORGANISM_VERTICES; i++) {

		double segment = i / ORGANISM_SEGMENTS_PER_RIB;
		double p0 = segment / (ORGANISM_SEGMENTS_PER_RIB);

		double step = i % ORGANISM_SEGMENTS_PER_RIB;
		double p1 = step / ORGANISM_SEGMENTS_PER_RIB;

		double fin = M_PI * p1;
		double sweep = 2 * M_PI * p0;

		double r = 0.8 * sin(sin(fin * -0.5)); // the basic rib radius

											   // over time, wings may appear on certain fins.
											   // the 'nose' is at step == 0
											   // wings could be at segment % 3 == 0
		double change = abs(sin(i * M_PI / 6.));

		double cosfin = cos(fin);
		double sinfin = sin(fin);

		organismMesh[i].pos = glm::vec3(r, cosfin, sinfin);
		organismMesh[i].normal = glm::vec3(i, sweep, change);

	}

	pos = glm::dvec3();
	dpos = glm::dvec3();

	near_clip = 0.1;
	far_clip = DIM;
	focal = 10;
	fovy = 90;
	eyesep = 0.03;

	for (int i = 0; i < NUM_COLLECTORS; i++) {
		particleCollector& pc = collectors[i];
		pc.intensity = 0.f;
		pc.pos = glm::vec3();
		pc.orient = glm::quat();
		pc.type = !(i % 2);
		pc.count = 0;
		pc.vibrate = 0.f;
	}

	orient = glm::dquat();

	//mat3 m = glm::toMat3(orient);

	ux = quat_ux(orient); //m[0]; //orient:ux();
	uy = quat_uy(orient); //m[1]; //orient:uy();
	uz = quat_uz(orient); //m[2]; //orient:uz();

						  // no. samples that sound events are randomized over
	sound_distribution = 44100 / 10;
	// minimum delay between invoking a sound and actually hearing it (in samples)
	sound_latency = 44100 / 100;

	fog_offset = 14;
	fog_density = .1;

	// fluid configuration:
	fluid_viscosity = 0.001;
	fluid_boundary_damping = .2;
	fluid_noise_count = 32;
	fluid_noise = 8.;
	fluid_hand_push = 1.;
	organism_fluid_push = -5;
	stalk_fluid_push = 0.1;// -2.5;


	particle_noise = 0.0000; //0.0001;
	particle_move_scale = 2.;

	particlesize = 0.04;
	particle_entropy = 0.01; // maximum energy lost per frame;
	particle_push = 0.5;
	particle_move_xfade = 0.6;
	// interp factor by which dust influences fluid (think inertia); 0 = no influence
	particle_move_feedback = 0.2;
	// amount by which dust move slows down per frame (0..1)
	particle_move_decay = 2.5;
	// interp factor per frame by which dust accumulates energy
	particle_nrg_recover = 0.0001;
	// global multiplier to dust velocity

	// basic birth size:
	organism_size = 0.8;
	organism_decay_rate = 0.99;
	organism_min = int(NUM_ORGANISMS * 0.15);
	organism_move_xfade = 0.5;
	organism_move_purity = 0.2;
	organism_reproduction_threshold = 0.6;
	// % of dust energy transferred to organism per frame
	organism_digest_rate = 0.2; // 0.05;
								// at this age, digestion rate is 50% efficient (an exponential decay)
	organism_digest_age = 100;
	organism_decay_threshold = 0.1;
	// how quickly we move tward particles to eat:
	organism_eat_lunge = 0.25;

	stalk_length = 1;
	stalk_width = 0.04;

	// stalk dynamics:
	stalk_flow = 5;// 20; // strength of fluid force
	stalk_spring = 0.9; // 1; // strength of force that tries to maintain normal stalk length 
	stalk_move_xfade = 0.3; // immediacy at which forces are applied (0..1)
	stalk_damping = 0.07; // 0.09; // scalar applied to all stalk velocities

	sound_volume = 0.5;
	sound_mindist = 0.0;
	sound_distscale = 0.008;

	updating = 1;

	reset();

	printf("World created\n");
}


void World::reset() {


	//pos = glm::dvec3(DIM/2, DIM/2, DIM/2);
	pos = glm::dvec3(0, 0, 0);
	base = glm::dvec3(0, 0, 0);

	//	// fill with noise:
	//	rng.fill(particlePositions, cv::RNG::UNIFORM, 0., double(DIM));
	//	rng.fill(particleColors, cv::RNG::UNIFORM, 0., 1.);
	//	rng.fill(particleVels, cv::RNG::UNIFORM, -0.01, 0.01);

	landscape.initialize(DIM, DIM, DIM, 4);

	//	spheres.clear();
	//	spheres.push_back(glm::vec4(0.2, 0.2, 0.2, 0.2));
	//	landscape_bake();

	// basic distance field is a union of randomized spheres:
	float minradius = 0.05f;
	float stretchradius = 0.5f; // max radius = minradius + stretchradius
	int spherecount = 10;
	int cluster_size = 3;

	{
		spheres.clear();
		for (int i = 0; i<spherecount; i++) {
			float phase = i / (float)spherecount;
			float r = minradius + stretchradius*phase*phase;
			glm::vec3 cluster_center(urandom(), urandom(), urandom());
			for (int j = 0; j<cluster_size; j++) {
				spheres.push_back(glm::vec4(cluster_center + glm::sphericalRand(1.f) * r * 0.5f, r));
			}
		}

		landscape_bake();
	}

	for (int i = 0; i < NUM_ORGANISMS; i++) {
		organism& o = organisms[i];
		o.thing.id = i;
		o.thing.type = ORGANISM;
		o.thing.alive = 1;
		o.thing.nrg = ci::Rand::randFloat(); // ci::Rand::randFloat(0., 1.);


		o.thing.pos = glm::dvec3(0.5 + ci::Rand::randFloat(0., double(DIM - 1)),
			0.5 + ci::Rand::randFloat(0., double(DIM - 1)),
			0.5 + ci::Rand::randFloat(0., double(DIM - 1)));
		o.thing.dpos = glm::dvec3(0.);

		//printf("organism pos %f %f %f\n", o.thing.pos.x, o.thing.pos.y, o.thing.pos.z);

		// build quat from euler:
		glm::dvec3 euler(ci::Rand::randFloat(0.f, (float)M_PI), ci::Rand::randFloat(0.f, (float)M_PI), ci::Rand::randFloat(0.f, (float)M_PI));
		o.orient = glm::dquat(euler);
		o.dorient = glm::dquat(glm::dvec3(ci::Rand::randFloat(-0.1f, 0.1f), ci::Rand::randFloat(-0.1f, 0.1f), ci::Rand::randFloat(-0.1f, 0.1f)));
		o.turn = glm::dvec3(0.);
		o.vary = ci::Rand::randFloat(0.f, 0.4f);// //?
		o.flash = ci::Rand::randFloat(0.f, 1.f);// //?
		o.speed = ci::Rand::randFloat(0.f, 1.f);// //?
		o.age = 0;
		o.sound.type = SOUND_NONE;
	}
	activeorganisms = NUM_ORGANISMS;

	for (unsigned int i = 0; i < NUM_STALKS; i++) {
		stalk& o = stalks[i];

		o.thing.id = i;
		o.thing.type = STALK;
		o.thing.alive = 1;
		o.thing.nrg = ci::Rand::randFloat(0., 1.);

		o.color = glm::dvec3(ci::Rand::randFloat(0., 1.), 0.1, ci::Rand::randFloat(0., 1.));
		o.length = 1.;
		o.numchildren = 0;
		o.parent = 0;

		
		if (i > (NUM_STALKS/10)) {

			o.parent = &stalks[ci::Rand::randInt(i)];
			while (o.parent->numchildren >= NUM_STALK_CHILDREN) {
				o.parent = &stalks[ci::Rand::randInt(i)];
			}
			o.parent->children[o.parent->numchildren] = &stalks[i];
			o.parent->numchildren++;
			
			glm::dvec3 euler(ci::Rand::randFloat(0.f, (float)M_PI), ci::Rand::randFloat(0.f, (float)M_PI), ci::Rand::randFloat(0.f, (float)M_PI));
			o.orient = glm::dquat(euler);

			o.thing.pos = o.parent->thing.pos - quat_uz(o.orient);
			o.thickness = o.parent->thickness * 0.9;
		}
		else {
			o.thing.pos = glm::dvec3(ci::Rand::randFloat(float(DIM)), ci::Rand::randFloat(float(DIM)), ci::Rand::randFloat(float(DIM)));
			glm::dvec3 euler(ci::Rand::randFloat((float)M_PI), ci::Rand::randFloat((float)M_PI), ci::Rand::randFloat((float)M_PI));
			o.orient = glm::dquat(euler);
			o.thickness = stalk_width;
		}
 
	}
	for (int i = 0; i < NUM_STALKS; i++) {
		stalk& o = stalks[i];
		if (o.parent == 0) {
			o.type = STALK_ROOT; // root
		}
		else if (o.numchildren == 0) {
			o.type = STALK_LEAF; // leaf
		}
		else {
			o.type = STALK_BRANCH;
		}
	}

	for (int i = 0; i < NUM_GHOST_PARTICLES; i++) {
		ghostpoint& p = ghostpoints[i];
		p.pos = glm::linearRand(glm::vec4(0.f), glm::vec4((float)DIM));
		//		p.color = glm::vec4(1.f);
	}

	ghostbegin = 0;
	ghostcount = 0;

	for (int i = 0; i < NUM_PARTICLES; i++) {
		particle& o = particles[i];
		particle_base& base = particle_bases[i];
		o.id = i;
		o.collector = -1;

		base.pos = glm::vec3(ci::Rand::randFloat(0.f, float(DIM)), ci::Rand::randFloat(0.f, float(DIM)), ci::Rand::randFloat(0.f, float(DIM)));
		o.neighbor = NULL;

		bool isfood = urandom() < 0.5;
		o.isfood = isfood;

		double speed = urandom();
		double nrg = urandom();
		o.energy = nrg;

		if (isfood) {
			base.color = glm::vec4(
				1,   //1 - (1 -d.nrg) + d.speed * 30.;
				1.21 - nrg*.8 + speed * 10.,
				speed * 2. - 0.5,
				1
				);
		}
		else {

			// excreted... tend towards blue
			double r = 0.5 - (nrg * 0.5) + speed * 0.25 + 0.2;
			base.color = glm::vec4(
				r,
				r * 0.5 + speed * 0.25 + 0.4,
				1,
				1
				);
		}
	}

}


void apply_fluid_boundary2(glm::vec3 * velocities, const glm::vec4 * landscape, const size_t dim0, const size_t dim1, const size_t dim2) {

	const float influence_offset = -world.fluid_boundary_damping;
	const float influence_scale = 1.f / world.fluid_boundary_damping;

	// probably don't need the triple loop here -- could do it cell by cell.
	int i = 0;
	for (size_t z = 0; z<dim2; z++) {
		for (size_t y = 0; y<dim1; y++) {
			for (size_t x = 0; x<dim0; x++, i++) {

				const glm::vec4& land = landscape[i];
				const float distance = fabsf(land.w);
				//const float inside = sign(land.w);	// do we care?
				const float influence = clamp((distance + influence_offset) * influence_scale, 0., 1.);


				glm::vec3& vel = velocities[i];
				glm::vec3 veln = vec_safenormalize(vel);
				float speed = glm::length(vel);

				const glm::vec3 normal = glm::vec3(land);	// already normalized.

															// get the projection of vel onto normal axis
															// i.e. the component of vel that points in either normal direction:
				glm::vec3 normal_component = normal * (dot(vel, normal));

				// remove this component from the original velocity:
				glm::vec3 without_normal_component = vel - normal_component;

				// and re-scale to original magnitude:
				glm::vec3 rescaled = vec_safenormalize(without_normal_component) * speed;

				// update:
				vel = mix(rescaled, vel, influence);

			}
		}
	}
}

void World::fluid_update(double dt) {

	now = now + dt;
	// update fluid
	Field3D<>& velocities = fluid.velocities;

	const size_t stride0 = velocities.stride(0);
	const size_t stride1 = velocities.stride(1);
	const size_t stride2 = velocities.stride(2);
	const size_t dim0 = velocities.dimx();
	const size_t dim1 = velocities.dimy();
	const size_t dim2 = velocities.dimz();
	const size_t dimwrap0 = dim0 - 1;
	const size_t dimwrap1 = dim1 - 1;
	const size_t dimwrap2 = dim2 - 1;
	glm::vec3 * data = (glm::vec3 *)velocities.front().ptr();
	float * boundary = world.boundary;

	// and some turbulence:

	//	for (int i=0; i<rng.uniform(0, fluid_noise_count); i++) {
	//		// pick a cell at random:
	//		glm::vec3 * cell = data + rng.uniform(0, dim0*dim1*dim2);
	//		// add a random vector:
	//		*cell = glm::sphericalRand(rng.uniform(0.1, fluid_noise));
	//	}

	apply_fluid_boundary2(data, (glm::vec4 *)landscape.ptr(), dim0, dim1, dim2);

	velocities.diffuse(global->fluid_viscosity, fluid_passes);
	// apply boundaries:
	//apply_fluid_boundary(data, boundary, dim0, dim1, dim2);
	apply_fluid_boundary2(data, (glm::vec4 *)landscape.ptr(), dim0, dim1, dim2);
	// stabilize:
	fluid.project(fluid_passes / 2);
	// advect:
	velocities.advect(velocities.back(), 1.);
	// apply boundaries:
	//apply_fluid_boundary(data, boundary, dim0, dim1, dim2);
	apply_fluid_boundary2(data, (glm::vec4 *)landscape.ptr(), dim0, dim1, dim2);

	// clear gradients:
	fluid.gradient.front().zero();
	fluid.gradient.back().zero();
}

void World::landscape_update(double dt) {

	// compare to all spheres:
	for (auto & sphere : spheres) {
		//glm::dvec3 move;
		//fluid.velocities.front().read_interp<double>(sphere.x*DIM, sphere.x*DIM, sphere.x*DIM, &move.x);


		glm::vec3 move = glm::sphericalRand(dt * urandom() * 0.05);

		sphere.x = euc_fmod(sphere.x + move.x, 1.);
		sphere.y = euc_fmod(sphere.y + move.y, 1.);
		sphere.z = euc_fmod(sphere.z + move.z, 1.);
	}

	landscape_bake();

	// use to update fluid?
}

void World::landscape_bake() {



	// bake distance field into a voxel field
	// (ported from jit.gen):
	float relativewrap_offset = 10.5f; // why?
	const size_t dim0 = landscape.dimx();
	const size_t dim1 = landscape.dimy();
	const size_t dim2 = landscape.dimz();

	//assert(dim0 == DIM);

	const float dim0r = 1.f / (float)dim0;
	const float dim1r = 1.f / (float)dim1;
	const float dim2r = 1.f / (float)dim2;
	glm::vec4 * cells = (glm::vec4 *)landscape.ptr();
	float * bptr = boundary;
	for (size_t z = 0; z<dim2; z++) {
		for (size_t y = 0; y<dim1; y++) {
			for (size_t x = 0; x<dim0; x++) {

				// the voxel cell point, converted to texture dim coordinates:
				glm::vec3 texcoord(x*dim0r, y*dim1r, z*dim2r);

				// result (normal in .xyz and distance in .w):
				// randomized initial normal:
				// maxed out initial distance:
				glm::vec4 result(0., 0., 0., 2.f);

				// compare to all spheres:
				for (auto sphere : spheres) {

					//if (i==0) printf("sphere %f %f %f %f\n", sphere.x, sphere.y, sphere.z, sphere.w);

					// get texcoord relative to sphere center:
					// and wrap into toroidal relative space:
					glm::vec3 reltexcoord = mod((texcoord - glm::vec3(sphere)) + relativewrap_offset, 1.f) - 0.5f;
					// reltexcoord is now the shortest vector from sphere center to p


					// find intersection with unit sphere
					// (this is the normal):
					glm::vec3 surfacenormal = glm::normalize(reltexcoord);

					//if (i==0) printf("sphere %f %f %f\n", reltexcoord.x, reltexcoord.y, reltexcoord.z);

					// now scale this to the radius of the object
					// to find the surface intersection with the actual sphere
					// note: still relative to spherecenter
					// (this only works for spheres)
					glm::vec3 surfacepoint = surfacenormal * sphere.w;

					//if (i==0) printf("sphere %f %f %f\n", surfacepoint.x, surfacepoint.y, surfacepoint.z);

					// get the vector from this intersection to our voxel:
					// again, put in toroidal space
					glm::vec3 rel = mod((reltexcoord - surfacepoint) + relativewrap_offset, 1.f) - 0.5f;
					// ...in order to derive the distance:
					float distance = glm::length(rel);

					// inside?
					bool inside = glm::length(reltexcoord) < sphere.w;
					if (inside) {
						distance = -distance;
						//surfacenormal = -surfacenormal; // for a two-sided surface
					}

					// the solid union is to take the lesser (innermost, not shorter!) distance:
					if (distance < result.w) {

						// pack the result into a vec4:
						result = glm::vec4(surfacenormal, distance);
					}
				}

				//printf("result %f %f %f %f\n", result.x, result.y, result.z, result.w);

				// boundary is a scaling factor, should be 1 in empty space where abs(dist) >> 0,
				// and 0 at the surface (dist == 0)

				//*bptr++ = MIN(1., fluid_boundary_damping*fabs(result.w));
				*cells++ = result;
			}
		}
	}
}



void World::dsp_initialize(double sr, long blocksize) {
	samplerate = sr;

	printf("set sr %f blocksize %ld\n", sr, blocksize);

	for (int i = 0; i<NUM_ORGANISMS; i++) {
		Sound& s = organisms[i].sound;
		s.type = 0;
		((OrganismSound&)(s)).reset(samplerate);
	}
	reverb.reset(samplerate);

	if (ambi_bus) free(ambi_bus);
	ambi_bus = (AmbiDomain *)malloc(sizeof(glm::dvec4) * blocksize);
}


void World::perform(long frames) {
	// clear bus:
	memset(ambi_bus, 0, sizeof(AmbiDomain) * frames);

	if (!updating) return;

	for (int i = 0; i<NUM_ORGANISMS; i++) {
		organism& o = organisms[i];
		Sound& s = o.sound;

		// active?
		if (s.type) {
			// due?
			if (s.start < frames) {
				// calculate the time window:
				int start = 0;
				if (s.start > 0) {
					//printf("start %d %d %d\n", s.start, s.type, frames);
					start = s.start;
					s.start = 0;
				}
				// does it also end this block?
				int end = start + s.dur - s.elapsed;
				if (end >= frames) end = frames;

				// play the sound
				switch (o.sound.type) {
				case SOUND_BIRTH: {
					((OrganismSound &)s).perform_birth(start, end, ambi_bus);
				} break;
				case SOUND_EAT: {
					((OrganismSound &)s).perform_eat(start, end, ambi_bus);
				} break;
					//case SOUND_MEET: break;
				case SOUND_DEATH: {
					((OrganismSound &)s).perform_death(start, end, ambi_bus);
				} break;
				default:
					break;
				}

				// mark sound as done (ready for new sound)
				if (s.elapsed >= s.dur) {
					s.type = 0;
				}
			}
			else {
				// getting nearer to being due:
				s.start -= frames;
			}
		}
	}

	// apply reverb.
	reverb.perform(frames, ambi_bus);
}

void World::performStereo(float * L, float * R, long frames) {
	perform(frames);

	for (int i = 0; i<frames; i++) {
		L[i] = headphoneL.w * ambi_bus[i].w
			+ headphoneL.x * ambi_bus[i].x
			+ headphoneL.y * ambi_bus[i].y
			+ headphoneL.z * ambi_bus[i].z;

		R[i] = headphoneR.w * ambi_bus[i].w
			+ headphoneR.x * ambi_bus[i].x
			+ headphoneR.y * ambi_bus[i].y
			+ headphoneR.z * ambi_bus[i].z;
	}
}


void World::update_move(double dt) {
	if (updating) {
		// must be in main thread:
		stalks_move(1);
		organisms_move(1);
		particles_move(1);
	}
	userconfidence *= 0.995;
	userconfidence_filtered = linear_interp(0.1, userconfidence, userconfidence_filtered);
}


void World::stalks_move(double dt) {
	for (int i = 0; i<NUM_STALKS; i++) {
		stalk& self = stalks[i];
		if (self.thing.alive) {

			double x = self.thing.pos.x;
			double y = self.thing.pos.y;
			double z = self.thing.pos.z;

			// is there food here?
			// TODO: only leaf nodes should eat:
			if (self.type == STALK_LEAF) {
				uint32_t hash = voxel_hashf_safe(x, y, z);
				Voxel& voxel = world.voxels_waste[hash];
				particle * p = voxel_top_particle(voxel);
				if (p && !p->isfood) {
					if (p->energy >= 0.5) {
						particle_base& base = particle_bases[p->id];

						// eat it
						voxel_pop_particle(voxel);
						p->owner = (Thing *)&self;

						if (p->collector >= 0) {
							particleCollector& pc = collectors[p->collector];
							pc.vibrate += 1.f;
						}

						//post("stalk eat %d %p", i, p);

						// change state:
						p->digested = 0.;

						//a->dpos += (d->pos - a->pos) * 0.1;
						//fluid->mixFlow(a->pos, a->dpos, stalk_fluid_push);

						// move toward what we just ate:
						self.thing.dpos.x += 0.25 * wrapdiff((double)base.pos.x, self.thing.pos.x, (double)DIM);
						self.thing.dpos.y += 0.25 * wrapdiff((double)base.pos.y, self.thing.pos.y, (double)DIM);
						self.thing.dpos.z += 0.25 * wrapdiff((double)base.pos.z, self.thing.pos.z, (double)DIM);
					}
				}
			}

			x += self.thing.dpos.x;
			y += self.thing.dpos.y;
			z += self.thing.dpos.z;


			glm::vec3 push(self.thing.dpos.x, self.thing.dpos.y, self.thing.dpos.z);
			push *= stalk_fluid_push;
			fluid.velocities.front().add(self.thing.pos, &push.x);

			// wrap around the observer:
			self.thing.pos.x = euc_fmod(x - base.x, (double)DIM) + base.x;
			self.thing.pos.y = euc_fmod(y - base.y, (double)DIM) + base.y;
			self.thing.pos.z = euc_fmod(z - base.z, (double)DIM) + base.z;

			if (self.parent) {
				stalk& parent = *self.parent;

				// vector between them:
				glm::dvec3 diff;
				diff.x = wrapdiff(self.thing.pos.x, parent.thing.pos.x, (double)DIM);
				diff.y = wrapdiff(self.thing.pos.y, parent.thing.pos.y, (double)DIM);
				diff.z = wrapdiff(self.thing.pos.z, parent.thing.pos.z, (double)DIM);
				self.length = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
			}
		}
	}
}


void World::organisms_move(double dt) {
	const double local_organism_eat_lunge = organism_eat_lunge;

	for (int i = 0; i<NUM_ORGANISMS; i++) {
		organism& self = organisms[i];
		if (self.thing.alive) {
			double x = self.thing.pos.x;
			double y = self.thing.pos.y;
			double z = self.thing.pos.z;

			// NOT THREAD SAFE.
			// all voxel list manipulations should be in a single thread.
			uint32_t hash = voxel_hash_safe(x, y, z);
			// is there food here?
			Voxel& voxel = voxels_food[hash];
			particle * p = voxel_top_particle(voxel);
			if (p) {
				if (p->isfood && p->energy >= 0.5) {
					// eat it

					particle_base& base = particle_bases[p->id];

					voxel_pop_particle(voxel);
					p->owner = (Thing *)&self;

					if (p->collector >= 0) {
						particleCollector& pc = collectors[p->collector];
						pc.vibrate += 1.f;
					}

					// change state:
					p->isfood = 0;

					//post("organism eat %d %p", i, p);

					// move toward what we just ate:
					self.thing.dpos.x += local_organism_eat_lunge * wrapdiff((double)base.pos.x, self.thing.pos.x, (double)DIM);
					self.thing.dpos.y += local_organism_eat_lunge * wrapdiff((double)base.pos.y, self.thing.pos.y, (double)DIM);
					self.thing.dpos.z += local_organism_eat_lunge * wrapdiff((double)base.pos.z, self.thing.pos.z, (double)DIM);

					organism_sound(SOUND_EAT, self);
				}
			}

			x += self.thing.dpos.x;
			y += self.thing.dpos.y;
			z += self.thing.dpos.z;

			glm::vec3 push(self.thing.dpos.x, self.thing.dpos.y, self.thing.dpos.z);
			push *= organism_fluid_push;
			fluid.velocities.front().add(self.thing.pos, &push.x);

			// wrap around the observer:
			self.thing.pos.x = euc_fmod(x - base.x, (double)DIM) + base.x;
			self.thing.pos.y = euc_fmod(y - base.y, (double)DIM) + base.y;
			self.thing.pos.z = euc_fmod(z - base.z, (double)DIM) + base.z;

			// increment orientation by angular velocity
			//self.orient.not_multiply(self.dorient);
			// dorient in model space, so put on right-hand side:
			self.orient = self.orient * self.dorient;
			self.orient = glm::normalize(self.orient);

		}
	}
}

// must happen at frame-rate:
void World::particles_move(double dt) {
	const float dimf = (float)DIM;
	const float rdimf = 1.f / dimf;
	const glm::vec3 worldbase = glm::vec3(base);

	// clear the voxels here, since this routine will repopulate them with particles
	clear_voxels_particles();

	// loop over all particles:
	for (int i = 0; i<NUM_PARTICLES; i++) {
		particle& self = particles[i];
		particle_base& base = particle_bases[i];

		Thing const * owner = self.owner;
		if (owner) {

			// move with owner:
			base.pos = owner->pos;

			// special case for stalks:
			if (owner->type == STALK) {
				stalk& s = *(stalk *)owner;

				// let stalk show that it is digesting:
				s.hasdust = 1;
				s.color.x = 2.5;

				// move along stalk:
				// get stalk direction:
				glm::dvec3 uz = quat_uz(s.orient);
				base.pos += uz * (self.digested);

				// move to next segment:
				self.digested += 0.005 * (1. + urandom());
				if (self.digested > 1.) {
					self.digested -= 1.;
					if (s.parent) {
						self.owner = (Thing *)s.parent;

					}
					else {

						// release:
						self.owner = 0;
						self.isfood = 1;
					}
				} // stalk case
			}
			else {
				base.pos += glm::ballRand(0.02f);
			}

		}
		else {


			// free-roaming:
			const float x = euc_fmodf1(base.pos.x + self.dpos.x, dimf, rdimf);
			const float y = euc_fmodf1(base.pos.y + self.dpos.y, dimf, rdimf);
			const float z = euc_fmodf1(base.pos.z + self.dpos.z, dimf, rdimf);

			// position in hashspace:
			uint32_t hash = voxel_hashf_safe(x, y, z);
			if (self.energy > 0.5) {
				if (self.isfood) {
					voxel_push_particle(world.voxels_food[hash], self);
				}
				else {
					voxel_push_particle(world.voxels_waste[hash], self);
				}
			}


			// wrap around the observer:
			base.pos.x = euc_fmodf1(x - worldbase.x, dimf, rdimf) + worldbase.x;
			base.pos.y = euc_fmodf1(y - worldbase.y, dimf, rdimf) + worldbase.y;
			base.pos.z = euc_fmodf1(z - worldbase.z, dimf, rdimf) + worldbase.z;
		}
	}
}


void World::organisms_update(double dt) {

	// if no organisms, capacity is 1
	// if full of organisms, capacity is 0
	// typically want it steadily > 0.5
	double capacity = 1. - (activeorganisms / double(NUM_ORGANISMS));
	organism_decay_rate = 1. - pow(1. + capacity, -15.);

	const double local_organism_decay_rate = organism_decay_rate;
	const double local_organism_reproduction_threshold = organism_reproduction_threshold;
	const double local_organism_decay_threshold = organism_decay_threshold;
	const double local_organism_move_xfade = organism_move_xfade;

	const float local_organism_reproduce_age = 500;

	// repopulate if too thin:
	if (activeorganisms < organism_min) {
		for (int i = 0; i<4; i++) {
			int32_t id = organism_pool_pop();
			if (id >= 0) {

				organism& self = organisms[id];
				activeorganisms++;

				self.age = 0;
				self.thing.nrg = 0.5;
				self.flash = 0;
				self.sound.type = SOUND_NONE;

				// make seeding happen far away from us:

				self.thing.pos = pos + glm::sphericalRand(double(DIM / 2));

				//self.thing.pos.x = DIM * urandom();
				//self.thing.pos.y = DIM * urandom();
				//self.thing.pos.z = DIM * urandom();

				self.thing.dpos.x = 0;
				self.thing.dpos.y = 0;
				self.thing.dpos.z = 0;

				glm::dvec3 axis = glm::sphericalRand(1.);
				self.orient = glm::angleAxis(M_PI*srandom(), axis);

				axis = glm::sphericalRand(1.);
				self.dorient = glm::angleAxis(0.17*srandom(), axis);

				self.thing.alive = 1;

				//post("seed organism %d", id);
			}
		}
	}

	for (int i = 0; i<NUM_ORGANISMS; i++) {
		organism& self = organisms[i];
		if (self.thing.alive) {
			self.age++;

			// show organism growth (this is now in the shader)
			self.vary = self.age * 0.001;
			self.flash *= 0.9;

			// metabolism:
			self.thing.nrg *= local_organism_decay_rate;

			//if (i==0) post("energy %f", self.thing.nrg);

			if (self.thing.nrg < local_organism_decay_threshold) {
				// it dies:
				//post("death %d", self.thing.id);
				organism_pool_push(self);
				activeorganisms--;
				self.age = 0;
				self.thing.alive = 0;

				organism_sound(SOUND_DEATH, self);

			}
			else {

				double x = self.thing.pos.x;
				double y = self.thing.pos.y;
				double z = self.thing.pos.z;

				// test status for death/reproduction:
				if (self.age > local_organism_reproduce_age
					&& self.thing.nrg > local_organism_reproduction_threshold
					&& !organism_pool.empty()) {
					// it reproduces:
					organism_sound(SOUND_BIRTH, self);
					self.thing.nrg *= 0.45;

					int32_t id = organism_pool_pop();
					if (id >= 0) {
						//post("reproduce %d: %d", self.thing.id, id);

						organism& child = organisms[id];

						// "eject it!"
						//child.thing.dpos = vec_safenormalize(self.thing.dpos);
						//child.thing.dpos *= 10.;

						child.thing.pos = self.thing.pos;
						//child.dpos.set(srandom()*0.1, srandom()*0.1, srandom()*0.1);
						//child.turn.set(0, 0, 0);
						//child.speed = 0.01*(urandom() + 0.0001);
						//child.orient.fromAxisAngle(180*srandom(), srandom(), srandom(), srandom());
						//child.dorient.fromAxisAngle(10*srandom(), srandom(), srandom(), srandom());

						child.thing.nrg = self.thing.nrg * 0.4;
						child.age = 0;
						child.vary = 0;
						child.thing.alive = 1;

						activeorganisms++;
					}
				}

				// get flow at current position
				glm::dvec3 flow;
				fluid.velocities.front().read_interp<double>(x, y, z, &flow.x);

				// adjust velocity
				self.speed = 0.01 * (self.age + 1.) / (self.age + 130.); //1; //vec::mag(flow);
																		 // (negative since we assume forward is -uz)
				glm::dvec3 vel = quat_uz(self.orient) * -self.speed;

				// dpos = mix(dpos, vel+flow*0.2, xfade):
				double f = 0.2;
				self.thing.dpos.x += local_organism_move_xfade * (vel.x + (flow.x * f) - self.thing.dpos.x);
				self.thing.dpos.y += local_organism_move_xfade * (vel.y + (flow.y * f) - self.thing.dpos.y);
				self.thing.dpos.z += local_organism_move_xfade * (vel.z + (flow.z * f) - self.thing.dpos.z);

				//if (i==0) post("pos %f %f %f vel %f %f %f speed %f", x, y, z, vel.x, vel.y, vel.z, self.speed);

				// get divergence between own movement & current:
				glm::dvec3 diff = flow - vel;
				// move to local coordinate frame:
				diff = glm::rotate(self.orient, diff);

				// TODO: none of this makes sense
				// even diff should be normalized here.
				glm::dquat reorient = glm::angleAxis(0.5, (diff));
				self.dorient = glm::slerp(self.dorient, reorient, 0.5);
				self.dorient = glm::normalize(self.dorient);
				//if (i==1) post("dorient %f %f %f %f", self.dorient.x, self.dorient.y, self.dorient.z, self.dorient.w);

			}
		}
	}

	//printf("population size: %i\n", activeorganisms);
}

void World::stalks_update(double dt) {
	double local_stalk_decay_rate = 0.96;
	double local_stalk_flow = stalk_flow;
	double local_stalk_move_xfade = stalk_move_xfade;
	double local_stalk_damping = stalk_damping;
	double local_stalk_spring = stalk_spring;
	double local_stalk_length = stalk_length;
	//stalk_length, stalk_spring, stalk_friction, stalk_flow, stalk_fluid_push, stalk_move_xfade, stalk_damping;

	for (int i = 0; i<NUM_STALKS; i++) {
		stalk& self = stalks[i];
		if (self.thing.alive) {

			double x = self.thing.pos.x;
			double y = self.thing.pos.y;
			double z = self.thing.pos.z;

			// metabolism:
			self.thing.nrg *= local_stalk_decay_rate;

			self.hasdust = 0;	// we'll update this as we go

								// for a smoother movement
								// dpos is only gradually affected by the forces
								// in this case, the forces are the local flow plus the branch tensions

								// get flow at current position
			glm::dvec3 flow;
			fluid.velocities.front().read_interp<double>(x, y, z, &flow.x);

			// assume force was calculated on previous frame
			// add field flow to force
			self.force += flow * local_stalk_flow;

			// gradually apply to object
			self.thing.dpos = glm::mix(self.thing.dpos, self.force, local_stalk_move_xfade) * local_stalk_damping;

			// now clear force:
			self.force = glm::dvec3(0);

			// smooth color change:
			self.color.x += 0.1 * (0.1 - self.color.x);
			self.color.y = self.color.x;
			self.color.z = 1 - self.color.x*0.7;
		}
	}


	for (int i = 0; i<NUM_STALKS; i++) {
		stalk& self = world.stalks[i];
		if (self.thing.alive) {

			// apply branch force
			if (self.parent) {
				stalk& parent = *self.parent;

				// TODO: may need to move some of this to move() to keep stalks connected

				// vector between them:
				glm::dvec3 diff;
				diff.x = wrapdiff(self.thing.pos.x, parent.thing.pos.x, (double)DIM);
				diff.y = wrapdiff(self.thing.pos.y, parent.thing.pos.y, (double)DIM);
				diff.z = wrapdiff(self.thing.pos.z, parent.thing.pos.z, (double)DIM);

				// this also becomes our new direction:
				//self.orient.toward_point(diff.x, diff.y, diff.z, self.orient, 0, 0, 0);

				// WTF is going on here
				self.orient = quat_toward_point(self.orient, diff, self.orient, glm::dvec3(0., 0., 0.));
				self.orient = glm::normalize(self.orient);

				double len = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
				double intensity = local_stalk_spring * (len - local_stalk_length);

				// straigtening factor:
				glm::dvec3 norm = vec_safenormalize(diff);

				glm::dvec3 change;
				change.x = (norm.x * intensity); // - (vel.x * stalk_friction);
				change.y = (norm.y * intensity); // - (vel.x * stalk_friction);
				change.z = (norm.z * intensity); // - (vel.x * stalk_friction);

				self.force.x -= change.x;
				self.force.y -= change.y;
				self.force.z -= change.z;

				parent.force.x += change.x;
				parent.force.y += change.y;
				parent.force.z += change.z;

				//b->force -= (change + twist * 0.1);
			}
		}
	}
}

// could happen independently of frame-rate:
void World::particles_update(double dt) {

	const float local_particle_noise = (float)particle_noise;
	const float local_particle_move_scale = (float)particle_move_scale;

	const double local_particle_move_decay = particle_move_decay;
	const double local_particle_move_xfade = particle_move_xfade;

	const double local_particle_nrg_recover = particle_nrg_recover;
	const float local_organism_digest_rate = (float)organism_digest_rate;

	const float local_particle_hand_decay = (1.f / 90.f) / 3.f;
	const float local_particle_hand_noise = 0.001f;
	const float local_particle_hand_sporadic_loss = 0.9f;
	const int local_collector_capacity = 10;

	for (int i = 0; i < NUM_COLLECTORS; i++) {
		collectors[i].count = 0;
	}

	// loop over all particles:
	for (int i = 0; i<NUM_PARTICLES; i++) {
		particle& self = world.particles[i];
		particle_base& base = world.particle_bases[i];

		Thing * owner = self.owner;
		if (owner) {
			// being digested:
			self.speed = 0;

			if (owner->alive) {
				// transfer energy to owner:
				double transfer = self.energy * local_organism_digest_rate;
				// TODO: what if self.energy becomes negative??
				// shouldn't we eject in that case?
				//transfer = MAX(transfer, 0.);
				if (transfer <= 0.) {
					owner = 0;
					self.energy = 0.f;
				}
				else {
					self.energy -= transfer;
					owner->nrg += transfer;
					// * organism_digest_age/(r->age+organism_digest_age);
				}
			}
			else {
				// owner died; liberation!
				self.owner = 0;
			}
		}
		else {
			// FREE ROAMING PARTICLE
			self.collector = -1;

			// compute new velocity:
			float x = base.pos.x;
			float y = base.pos.y;
			float z = base.pos.z;

			glm::vec3 flow;
			fluid.velocities.front().read_interp<float>(x, y, z, &flow.x);

			// add some deviation (brownian)
			glm::vec3 disturb = glm::ballRand(local_particle_noise);
			self.dpos = disturb + local_particle_move_scale * flow;

			// near a collector?
			if (self.energy > 0.5) {
				float hand_min = 0.0003f; // minimum radius of hand attraction -- below this the force becomes null (omnidirectional)
				float hand_range = 0.5f; // maximum radius beyond hand_min of hand attraction -- above this there is no effect
				float hand_min2 = hand_min * hand_min;
				float hand_range2 = hand_range * hand_range;
				glm::vec3 nav_vel = world.dpos * dt;// world.lefthand_vel;

				for (int i = 0; i < NUM_COLLECTORS; i++) {
					particleCollector& pc = collectors[i];
					if (pc.count < local_collector_capacity
						&& self.isfood == pc.type
						&& pc.intensity > 0.1f) {
						const glm::vec3& pos = pc.pos;
						const glm::vec3& uz = quat_uz(pc.orient);

						glm::vec3 hrel = pos - base.pos;
						// distance-squared:
						float d2 = dot(hrel, hrel);
						// normalized distance:
						float dn = ((d2 - hand_min2) / (hand_range2));

						if (dn < 1.f) {
							float idn = 1.f - dn; // high when close to controller

												  // move toward center, but also tangentially to it:
							glm::vec3 f = hrel * (0.2f);
							glm::vec3 g = glm::cross(f, uz);

							glm::vec3 dpos = f // centrpetal force
								+ g //glm::vec3(f.y, f.z, f.x) // perpendicular centrifugal force for orbit
								- f * (0.2f*idn*idn*idn*idn)// repulsive force 
															//+ glm::ballRand(local_particle_hand_noise * idn*idn) // noise near center
								+ nav_vel;

							float mixture = 2. * (self.energy - 0.5);
							mixture = powf(mixture, 0.5);
							self.dpos = mix(self.dpos, dpos, mixture);

							if (urandom() < local_particle_hand_decay) self.energy *= local_particle_hand_sporadic_loss;

							pc.count++;
							self.collector = i;

						}
					}
				}
			}

			// free roaming particles gradually absorb redness:
			self.energy += local_particle_nrg_recover * (1. - self.energy);

		}

		// pass in state via color:
		base.color.x = self.energy;
		base.color.y = self.speed;
		base.color.z = self.isfood ? 0 : self.owner ? 1 : 2;
	}
}