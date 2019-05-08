#ifndef __AN__World__
#define __AN__World__

#include "al_glm.h"
#include "an_math.h"
#include "an_field3d.h"
#include "genlib.h"
//#include "sounds.h"

#include <vector>

static const int DIM = 32;
static const int DIMWRAP = 31;
static const int NUM_VOXELS = 32768; // DIM^3
static const uint32_t INVALID_VOXEL_HASH = 0xffffffff;

static const int ORGANISM_SEGMENTS_PER_RIB = 16;
static const int NUM_ORGANISM_VERTICES = 256;	// 16*16
static const int NUM_STALK_VERTICES = 144; // 6*6*4
static const int NUM_STALK_CHILDREN = 3;

static const int NUM_ORGANISMS = 500; //0;
static const int NUM_STALKS = 400;
static const int NUM_PARTICLES = (int)48e3;
static const int NUM_GHOST_PARTICLES = 217088;
static const int NUM_COLLECTORS = 2;


static const int AN_AUDIO_BUFFER_SIZE = (1 << 14);
static const int AN_AUDIO_BUFFER_WRAP = (AN_AUDIO_BUFFER_SIZE - 1);
static const int NUM_SOUNDS = 512;

enum Sounds {
	SOUND_NONE = 0,
	SOUND_MEET = 1,
	SOUND_EAT = 2,
	SOUND_BIRTH = 3,
	SOUND_DEATH = 4
};

enum Things {
	NONE = 0,
	ORGANISM = 1,
	STALK = 2
};

enum StalkTypes {
	STALK_ROOT = 0,
	STALK_BRANCH = 1,
	STALK_LEAF = 2
};

typedef struct Thing {
	glm::dvec3 pos, dpos;
	double nrg, pad;
	int id, type, alive;
} Thing;

typedef struct Sound {
	int type, pad;	// e.g. SOUND_EAT
	int start, end, dur, elapsed; // in samples

	AmbiDomain ambi;
	double attenuation;	// 1 - distance squared, normalized in 0..1 over audible range

	double samplerate;
	double samples_to_seconds;
	double f2i;
	double param_age;
	double param_rand;

	union {

		struct {
			// Birth:
			//SineCycle m_cycle_4; // uint32_t x2, double
			//SineCycle m_cycle_5; // uint32_t x2, double
			uint32_t phasei_4, pincr_4;
			uint32_t phasei_5, pincr_5;
		};

		struct {
			// Eat:
			double m_y_5;
			double m_x_2;
			double m_x_3;
			double m_y_4;
			double m_cutoff_1;
		};

		struct {
			// Death:
			double x1, y1; //DCBlock m_dcblock_5;	//
			double m_history_1;
			double m_phase;
		};
	};

} Sound;

typedef struct OrganismSound : public Sound {

	static SineData __sinedata;
	static Noise noise;

	static double grainwindow[AN_AUDIO_BUFFER_SIZE];
	static double sinewindow[AN_AUDIO_BUFFER_SIZE];

	static inline double window_lerp(double * window, double a) {
		double widx = a * AN_AUDIO_BUFFER_SIZE;
		int widx0 = int(widx);
		int widx1 = widx0 + 1;
		double wa = widx - widx0;
		widx1 &= (AN_AUDIO_BUFFER_SIZE - 1);	// wrap after getting wa!
		widx0 &= (AN_AUDIO_BUFFER_SIZE - 1);
		double w0 = window[widx0];
		double w1 = window[widx1];
		return w0 + wa*(w1 - w0);
	}

	inline void reset(double __sr) {
		samplerate = __sr;
		samples_to_seconds = (1 / samplerate);
		param_age = 0;
		param_rand = 0;
		f2i = 4294967296.0 / samplerate;
	}

	inline void perform_birth(int n0, int n1, AmbiDomain * ambi_bus) {
		const double * sinedata = __sinedata.mData;

		// play it over buffer indices [start, end)
		for (int n = n0; n<n1; n++, elapsed++) {
			double in1 = elapsed / (double)dur;	// the ramp

												//m_cycle_4.phase((param_rand + in1*0.2));
												//phasei_4 = (param_rand + in1*0.2) * 4294967296.0;
			phasei_4 = uint32_t((param_rand + in1*0.2) * 4294967296.0);

			//double cycle_18057 = m_cycle_4(__sinedata);
			// divide uint32_t range down to buffer size (32-bit to 14-bit)
			uint32_t idx_4 = phasei_4 >> 18;
			// compute fractional portion and divide by 18-bit range
			const double frac_4 = (phasei_4 & 262143) * 3.81471181759574e-6;
			// index safely in 14-bit range:
			const double y0_4 = sinedata[idx_4];
			const double y1_4 = sinedata[(idx_4 + 1) & 16383];
			const double cycle_18057 = linear_interp(frac_4, y0_4, y1_4);
			phasei_4 += pincr_4;
			//double cycleindex_18058 = m_cycle_4.phase();

			double mul_18052 = (cycle_18057 * -20);
			double add_18051 = (mul_18052 + 70);
			double mul_18050 = (add_18051 * 15);

			//m_cycle_5.freq(mul_18050);
			pincr_5 = uint32_t(mul_18050 * f2i);

			//double cycle_18048 = m_cycle_5(__sinedata);
			// divide uint32_t range down to buffer size (32-bit to 14-bit)
			uint32_t idx_5 = phasei_5 >> 18;

			// compute fractional portion and divide by 18-bit range
			const double frac_5 = (phasei_5 & 262143) * 3.81471181759574e-6;
			// index safely in 14-bit range:
			const double y0_5 = sinedata[idx_5];
			const double y1_5 = sinedata[(idx_5 + 1) & 16383];
			const double cycle_18048 = linear_interp(frac_5, y0_5, y1_5);
			phasei_5 += pincr_5;
			//double cycleindex_18049 = m_cycle_5.phase();

			// assign results to output buffer;
			double src = (cycle_18048 * 0.3);

			// enveloping:
			double env = window_lerp(grainwindow, in1);
			src *= env;
			src *= attenuation;

			// spatializing:
			ambi_bus[n].w += src * ambi.w;
			ambi_bus[n].x += src * ambi.x;
			ambi_bus[n].y += src * ambi.y;
			ambi_bus[n].z += src * ambi.z;
		};
	};

	inline void perform_eat(int n0, int n1, AmbiDomain * ambi_bus) {
		double cf = m_cutoff_1;
		// the main sample loop;
		// play it over buffer indices [start, end)
		// play it over buffer indices [start, end)

		for (int n = n0; n<n1; n++, elapsed++) {
			double in1 = elapsed / (double)dur;	// the ramp

			double noise_21482 = noise();
			double mul_21481 = (noise_21482 * 10);
			double expr_21485 = ((in1 * 9) + 90);
			double x = mul_21481;
			double q = expr_21485;
			double bw = safediv(cf, q);
			double r = exp(((-bw) * 0.00014247585730566));
			double c1 = ((2 * r) * cos((cf * 0.00014247585730566)));
			double c2 = ((-r) * r);
			double y = ((((1 - r) * (x - (r * m_x_3))) + (c1 * m_y_4)) + (c2 * m_y_5));
			m_y_5 = m_y_4;
			m_y_4 = y;
			m_x_3 = m_x_2;
			m_x_2 = x;
			double src = y;

			// enveloping:
			double env = window_lerp(grainwindow, in1);
			src *= env;



			src *= attenuation;

			// spatializing:
			ambi_bus[n].w += src * ambi.w;
			ambi_bus[n].x += src * ambi.x;
			ambi_bus[n].y += src * ambi.y;
			ambi_bus[n].z += src * ambi.z;

		};

	};

	inline void perform_death(int n0, int n1, AmbiDomain * ambi_bus) {
		double expr_21831 = (75 - (param_age * 0.02));

		// play it over buffer indices [start, end)
		for (int n = n0; n<n1; n++, elapsed++) {
			double in1 = elapsed / (double)dur;	// the ramp

			double freq = (((in1 * param_rand) + 13) * expr_21831);

			// begin rect
			double rate = freq * samples_to_seconds;

			// self-oscillation factor
			double expr_19966 = (54. * safepow((0.5 - rate), 6.));
			double mul_19139 = (expr_19966 * 0.5);
			double mul_19135 = (m_history_1 * m_history_1);
			double mul_19134 = (mul_19135 * -1.);
			double mul_19152 = (mul_19139 * mul_19134);

			// phasor
			const double pincr = freq * samples_to_seconds;
			m_phase = wrap(m_phase + pincr, 0., 1.);
			double phasor_19151 = m_phase;
			double expr_19967 = ((phasor_19151 * 2) - 1);
			// to radians:
			double mul_19154 = ((expr_19967 + mul_19152) * 3.1415926535898);
			double sin_19156 = sin(mul_19154);
			// averaging filter:
			double mix_21832 = (sin_19156 + (0.55 * (m_history_1 - sin_19156)));
			double mix_19147 = mix_21832;
			// HF boost:
			double expr_19964 = ((mix_19147 * 1.9) - (m_history_1 * 0.9));

			// apply compensation
			double rsub_19142 = (1 - (rate * 2));
			double rect_with_dc = (expr_19964 * rsub_19142);

			//double rect_out = m_dcblock_5(rect_with_dc);
			double y = rect_with_dc - x1 + y1*0.9997;
			x1 = rect_with_dc;
			y1 = y;
			double rect_out = y;

			double history_19146_next_19968 = mix_19147;
			// end rect

			double mul_19971 = (rect_out * -0.125);
			double src = mul_19971;
			m_history_1 = history_19146_next_19968;

			// enveloping:
			double env = window_lerp(grainwindow, in1);
			src *= env;


			src *= attenuation;

			// spatializing:
			ambi_bus[n].w += src * ambi.w;
			ambi_bus[n].x += src * ambi.x;
			ambi_bus[n].y += src * ambi.y;
			ambi_bus[n].z += src * ambi.z;

		};
	};

} OrganismSound;

struct SoundEvent {
	int state, type; // unused = 0;
	glm::vec3 pos;
	float p0;
};

typedef struct organism {
	Thing thing;
	glm::dquat orient, dorient;
	glm::dvec3 turn;
	double vary, flash, speed;
	int age;

	Sound sound;
} organism;

typedef struct particle_base {
	glm::vec3 pos; float pad;
	glm::vec4 color;
} particle_base;

typedef struct particle {
	glm::vec3 dpos; float speed;
	double energy, digested;
	int id, isfood;
	int collector, unused;
	struct particle * neighbor;
	Thing * owner;
} particle;

typedef struct ghostpoint {
	glm::vec4 pos;
	//	glm::vec4 color;
} ghostpoint;

typedef struct Voxel {
	particle * particles;
} Voxel;

typedef struct stalk {
	Thing thing;
	glm::dquat orient;
	glm::dvec3 pos;
	glm::dvec3 force;
	glm::dvec3 color;
	int hasdust, numchildren;
	double thickness, length;
	struct stalk * parent;
	struct stalk * children[NUM_STALK_CHILDREN];
	int type;
} stalk;

typedef struct vertex {
	glm::vec3 pos;
	glm::vec3 normal;
} vertex;

typedef struct particleCollector {
	glm::quat orient;
	glm::vec3 pos;
	float intensity, vibrate;
	int type; // food or waste
	int count;
} particleCollector;

struct World {

	organism organisms[NUM_ORGANISMS];
	stalk stalks[NUM_STALKS];
	particle_base particle_bases[NUM_PARTICLES];
	particle particles[NUM_PARTICLES];

	ghostpoint ghostpoints[NUM_GHOST_PARTICLES];

	Voxel voxels_food[NUM_VOXELS];
	Voxel voxels_waste[NUM_VOXELS];
	float density[NUM_VOXELS];
	float boundary[NUM_VOXELS];

	vertex organismMesh[NUM_ORGANISM_VERTICES];
	vertex stalkMesh[NUM_STALK_VERTICES];

	glm::dquat orient;
	glm::dvec3 pos, ux, uy, uz;
	glm::dvec3 dpos;
	glm::dvec3 base;	// the origin of the current world grid
	glm::dvec3 ghostpos;	// the position of the ghost in the world
	glm::dvec3 lightpos;

	particleCollector collectors[NUM_COLLECTORS];

	//glm::vec3 lefthand, righthand;
	//glm::vec3 lefthand_vel, righthand_vel;
	//glm::quat lefthand_orient, righthand_orient;
	//float lefthand_confidence, righthand_confidence;

	double fovy;
	double near_clip;
	double far_clip;
	double focal;
	double eyesep;
	double fog_offset, fog_density;

	double fluid_viscosity, fluid_diffusion, fluid_decay, fluid_boundary_damping, fluid_noise;
	double fluid_hand_push;

	double particlesize;
	double particle_noise, particle_entropy, particle_push;
	double particle_move_xfade, particle_move_feedback, particle_move_decay, particle_nrg_recover, particle_move_scale;

	double organism_size, organism_decay_rate, organism_move_xfade, organism_move_purity, organism_fluid_push, organism_reproduction_threshold, organism_digest_rate, organism_digest_age, organism_decay_threshold, organism_eat_lunge;

	double stalk_length, stalk_width, stalk_spring, stalk_friction, stalk_flow, stalk_fluid_push, stalk_move_xfade, stalk_damping;

	int fluid_noise_count;
	int organism_min;
	int activeparticles, activeorganisms;
	int sound_latency, sound_distribution;
	int frame, updating;

	int ghostbegin, ghostcount;

	double userconfidence, userconfidence_filtered;

	double sound_volume, sound_mindist, sound_distscale;
	double samplerate;

	bool fluidrunning = 0;
	Fluid3D<> fluid;
	Array<> noisefield;
	Array<> landscape;

	std::vector<glm::vec4> spheres;

	//cv::Mat particlePositions, particleColors, particleVels;
	//cv::RNG rng;

	SoundEvent sounds[NUM_SOUNDS];
	int nextSound;

	static World& get();

	void setup();
	void reset();

	void update_move(double dt);

	void animate(double dt)
	{
		frame++;
		lightpos = pos + ux * 4.;

		// this was for nudging the view, but that doesn't make sense for Oculus
		//	glm::dvec3 flow;
		//	fluid.velocities.front().read_interp<double>(pos.x, pos.y, pos.z, &flow.x);
		//	flow = vec_safenormalize(flow);

		if (updating) {

			// add some random turbulence:
			for (int i = 0; i < 10; i++) {
				glm::dvec3 pos = glm::dvec3(urandom() * DIM, urandom() * DIM, urandom() * DIM);
				glm::dvec3 vel = glm::sphericalRand(0.01);
				an_fluid_insert_velocity(pos, vel);
			}

			stalks_update(dt);
			organisms_update(dt);
			particles_update(dt);
		}
	};

	void setPos(glm::dvec3 const & v) {
		static glm::dvec3 world_center = glm::dvec3(DIM / 2., DIM / 2., DIM / 2.);
		pos = vec_fixnan(v);
		base = pos - world_center;
		//post("pos %f %f %f base %f %f %f", pos.x, pos.y, pos.z, base.x, base.y, base.z);
	}

	glm::dvec3 getPos() const { return pos; }

	void setOrient(glm::dquat const & q) {
		orient = q;
		ux = quat_ux(q);
		uy = quat_uy(q);
		uz = quat_uz(q);
		//post("ux %f %f %f", ux.x, ux.y, ux.z);
		//post("uy %f %f %f", uy.x, uy.y, uy.z);
		//post("uz %f %f %f", uz.x, uz.y, uz.z);
	}

	void landscape_update(double dt);
	void landscape_bake();

	void particles_update(double dt);
	void particles_move(double dt);

	void organisms_update(double dt);
	void organisms_move(double dt);

	void stalks_update(double dt);
	void stalks_move(double dt);

	void fluid_update(double dt);

	void dsp_initialize(double samplerate, long blocksize);
	void perform(long frames);
	void performStereo(float * L, float * R, long frames);
	void perform64(double **ins, long numins, double **outs, long numouts, long frames);

	void clear_voxels_particles() {
		memset(voxels_food, 0, sizeof(voxels_food));
		memset(voxels_waste, 0, sizeof(voxels_waste));
	}

	static double urandom() {
		return (double)rand() / (double)RAND_MAX;
		//return world.rng.uniform(0., 1.);
	}

	static double srandom() {
		return urandom() * 2. - 1.;
		//return world.rng.uniform(-1., 1.);
	}

	glm::vec3 an_fluid_get_velocity(double x, double y, double z) {
		glm::vec3 res;
		fluid.velocities.front().read_interp(x, y, z, &res.x);
		return res;
	}

	void an_fluid_insert_velocity(const glm::dvec3 pos, const glm::dvec3 vel) {
		// interpolated add:
		glm::vec3 v;
		v.x = (float)vel.x;
		v.y = (float)vel.y;
		v.z = (float)vel.z;
		fluid.velocities.front().add(pos, &v.x);
	}

	SoundEvent& topSound() {
		return sounds[nextSound];
	}

	void commitTopSound() {
		nextSound = (nextSound + 1) % NUM_SOUNDS;
	}
};

#endif