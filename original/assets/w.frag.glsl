#include "lib.glsl"
#version 150

uniform mat4 ciModelView, ciProjectionMatrix;
uniform sampler3D landtex1, landtex2, noisetex;
uniform float now, landtexmix;

in vec2 T;
in vec3 origin, direction;
in float world_dim;
out vec4 outColor;

float dim = 32.;
float near = 0.1 / dim;	// in texture-space
float far = 1.;			// in texture-space
float surface_noise = 3.5;
float surface_scale = 2.;

// basic step size is half a texel:
float stepsize = 0.5/dim;
// surface sharpness (how close a ray needs to get before it aborts)
float eps =  0.125 * stepsize;

#define MAX_STEPS 64
float inv_max_steps = 1./float(MAX_STEPS);

// lighting:
vec3 color1 = vec3(0.47, 0.5, 0.5); //vec3(0.39, 0.13, 0.22); vec3(0.9, 0.3, 0.2);
vec3 color2 = vec3(0.3, 0.3, 0.4); //vec3(0.2, 0.2, 0.2);
vec3 ambient = vec3(0.1);//0.1

// lighting position
vec3 light1 = vec3(3.2, 6.4, 9.6);
vec3 light2 = vec3(6.4, -9.6, 3.2);

// what color to paint pixels that took too many steps to render:
vec4 overstep_color = vec4(0., 0., 0., 1.);

float fog_density = 0.3;
vec3 fog_color = vec3(0.);
float fog_offset = 14.;

// pos should be world position (modelview * vertex)
// that is (p * dim)
vec3 fog(vec3 color, vec3 pos) {
	// fog parameters
	float distance0 = length(pos);
	float distance = max(distance0-fog_offset, 0.);
	float fogExponent = distance*fog_density;
	float fogFactor = exp2(-abs(fogExponent));
	float z = clamp(-2.*(pos.z)-0.5, 0., 1.);	// 0.5 is the nearness
	return vec3(mix(fog_color, z*color.rgb, fogFactor));

}

#define PI 3.14159265359
#define EPS 0.004

float rand(vec2 coordinate) {
	return fract(sin(dot(coordinate.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// Floating-point modulo
// The result (the remainder) has same sign as the divisor.
// Similar to matlab's mod(); Not similar to fmod() -   Mod(-3,4)= 1   fmod(-3,4)= -3
// y must be > 0
float euc_fmod(float x, float modulo) {
	float m = x - modulo * floor(x/modulo);
	// handle boundary cases resulted from floating-point cut off:
	if (m >= modulo) return 0.;   // Mod(-1e-16             , 360.    ): m= 360.
	if (m < 0.) {
		if (modulo+m == modulo) return 0.; // just in case...
		else return modulo+m; 	// Mod(106.81415022205296 , _TWO_PI ): m= -1.421e-14
	}
	return m;
}

// returns the shortest distance from x to y on a toroidal line of length m:
float wrapdist(float x, float y, float m) {
	float halfm = m/2.0;
	return abs(euc_fmod((x - y) + halfm, m) - halfm);
}

vec3 euc_fmod3(vec3 p, vec3 modulo) {
	return vec3(euc_fmod(p.x, modulo.x),
				euc_fmod(p.y, modulo.y),
				euc_fmod(p.z, modulo.z));
}

/*
 // returns shortest signed vector from x to y in a toroidal space of dim m:
vec3 wrapdiff(vec3 x, vec3 y, float m) {
	float half = m/2.0;
	return (euc_fmod3((x - y) + half, m) - half);
}
 */

float erf_guts(in float x) {
	const float a=8.0*(PI-3.0)/(3.0*PI*(4.0-PI));
	float x2=x*x;
	return exp(-x2 * (4.0/PI + a*x2) / (1.0+a*x2));
}

// "error function": integral of exp(-x*x)
float erf(in float x) {
	float sign=1.0;
	if (x<0.0) sign=-1.0;
	return sign*sqrt(1.0-erf_guts(x));
}

// erfc = 1.0-erf, but with less roundoff
float erfc(float x) {
	if (x>3.0) { //<- hits zero sig. digits around x==3.9
				 // x is big -> erf(x) is very close to +1.0
				 // erfc(x)=1-erf(x)=1-sqrt(1-guts)=approx +guts/2
		return 0.5*erf_guts(x);
	} else {
		return 1.0-erf(x);
	}
}

void rX(inout vec3 p, float a) {
	float c,s;vec3 q=p;
	c = cos(a); s = sin(a);
	p.y = c * q.y - s * q.z;
	p.z = s * q.y + c * q.z;
}

void rY(inout vec3 p, float a) {
	float c,s;vec3 q=p;
	c = cos(a); s = sin(a);
	p.x = c * q.x + s * q.z;
	p.z = -s * q.x + c * q.z;
}

void rZ(inout vec3 p, float a) {
	float c,s;vec3 q=p;
	c = cos(a); s = sin(a);
	p.x = c * q.x - s * q.y;
	p.y = s * q.x + c * q.y;
}

// absmin
float absmin(float a, float b) {
	return abs(a) < abs(b) ? a : b;
}

// polynomial smooth min (k = 0.1);
float smin( float a, float b, float k )
{
	float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
	return mix( b, a, h ) - k*h*(1.0-h);
}

float smax( float a, float b, float k )
{
	float k1 = k*k;
	float k2 = 1./k1;
	return log( exp(k2*a) + exp(k2*b) )*k1;
}

float ssub(in float A, in float B, float k) {
	return smax(A, -B, k);
}

float opSub(in float A, in float B) {
	return max(A, -B);
}

float opShell(in float A, in float B, in float t) {
	return max(A-t*0.5, -B-t*0.5);
}

vec3 opRep(in vec3 p, in vec3 c) {
	return mod(p,c)-0.5*c;
}

vec3 opRepEuc(in vec3 p, in vec3 c) {
	return euc_fmod3(p,c)-0.5*c;
}

// plane defined by a normal and a point on the plane
// note: it is not efficient to render planes in sphere tracing
// but it might still be useful as a component within CSG for example.
float sdPlane(in vec3 p, in vec3 normal, in vec3 pt) {
	return length(dot(p, normal) - pt);
}

// note: scaling the sphere into an ellipsoid is not well-formed
float sdSphere(in vec3 p, in float r) {
	return length(p)-r;
}

float sdBox(in vec3 p, in vec3 b ) {
	vec3 d = abs(p) - b; 				// utilize symmetry of cuboid
	float ri = max(d.x, max(d.y, d.z)); // distance inside box
	float ro = length(max(d, 0.)); 		// distance outside box
										// return ri > 0. ? ro : ri;
	return min(ri, 0.) + max(ro, 0.);	// cheaper than a conditional
}

// aligned to Z axis, radius r, length l
float sdCylinder(in vec3 p, in float r, in float l) {
	// intersection of:
	// infinite cylinder radius r, and
	// box length l (z axis only)
	return max(length(p.xy)-r, abs(p.z) - l);
}

// like cylinder, but with r dependent on p.z
float sdCone(in vec3 p, in float r, in float l) {
	// is there a way to do this without the division?
	float a = (p.z*0.5+0.5)/l;
	return max(length(p.xy)-r*a, abs(p.z) - l);
}

float sdTorus(in vec3 p, in float R, in float r) {
	// length(p.xy)-R is a disc radius R on z-axis
	// i.e. torus is like two 2D circles combined
	vec2 xy = vec2(length(p.xy)-R, p.z);
	return length(xy)-r;
}

// NOTE scale := f(p/s)*s

// Euclidean distance metric is length(p)
// other distance metrics:
// Manhattan distance metric is (p.x+p.y+p.z) -- where diamond is the 'circle'
// Chessboard distance metric is max(p.x, max(p.y, p.z)) -- where square is the 'circle'
// qnorm is the generalized case; manhat when q == 1, euc when q == 2, chess when q == infinity, ...
float manhattan(in vec3 p) {
	return abs(p.x)+abs(p.y)+abs(p.z);
}
float chessboard(in vec3 p) {
	vec3 a = abs(p);
	return max(a.x, max(a.y, a.z));
}
float qnorm(vec3 p, float q) {
	return pow(pow(p.x, q) + pow(p.y, q) + pow(p.z, q), 1./q);
}
// convert q metrics to euc metrics:
// (assumes q < 2)
float q2euc(in float d, in float q) {
	float a = sqrt(3.)/3.;	// const 0.57735026918963
							// vec3 b = qnorm(vec3(a), q)
	float b = pow(pow(a, q)*3., 1./q);
	return d / b;
}
float manhattan2euc(in float d) {
	return d * 0.57735026918962;		// 1./sqrt(3)
}

float sdDiamond(in vec3 p, float r) {
	return manhattan2euc(manhattan(p) - r);
}

// rounded cube, by using a higher-order distance metric
float sdRcube(in vec3 p, float r) {
	float n = 4.;	// should be an even number!
	return q2euc(qnorm(p, n) - r, n);
}

vec3 closest_point_on_line_segment(vec3 P, vec3 A, vec3 B) {
	vec3 AB = B-A;
	float l2 = dot(AB, AB);	// length squared
	
	if (l2 < EPS) {
		// line is too short, just use an endpoint
		return A;
	}
	
	// Consider the line extending the segment,
	// parameterized as A + t (AB).
	// We find projection of point p onto the line.
	// It falls where t = [(AP) . (AB)] / |AB|^2
	
	vec3 AP = P-A;
	float t = dot(AP, AB) / l2;
	
	if (t < 0.0) {
		return A; 	// off A end
	} else if (t > 1.0) {
		return B; 	// off B end
	} else {
		return A + t * AB; // on segment
	}
}

// i.e. distance to line segment, with smoothness r
float sdCapsule1(vec3 p, vec3 a, vec3 b, float r) {
	vec3 p1 = closest_point_on_line_segment(p, a, b);
	return distance(p, p1) - r;
}

// this seems to be equivalent to above -- but so much simpler!
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
	vec3 pa = p - a, ba = b - a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return length( pa - ba*h ) - r;
}

// another shape is a 'skeleton' curve
// find the nearest point on curve to p
// then apply offset radius to turnw it into a curved cylinder



float scene_demo(vec3 p) {
	
	/*
	 float minradius = 0.05;
	 float stretchradius = 0.4; // max radius = minradius + stretchradius
	 int spherecount = 10;
	 int cluster_size = 1;
	 
	 {
		spheres.clear();
		for (int i=0; i<spherecount; i++) {
	 float phase = i / (float)spherecount;
	 float r = minradius + stretchradius*phase*phase;
	 glm::vec3 cluster_center(urandom(), urandom(), urandom());
	 for (int j=0; j<cluster_size; j++) {
	 spheres.push_back(glm::vec4(cluster_center + glm::sphericalRand(1.f) * r * 0.5f, r));
	 }
		}
	 */
	
	
	vec3 pr = opRepEuc(p, vec3(1));
	
	float a = min(
			   sdSphere(pr + vec3(0, 0.4, 0.2), 0.5),
			   sdBox(pr + vec3(0.25, -0.3, -0.5), vec3(0.6))
	);
	float b = min(
			   sdSphere(pr + vec3(0.1, 0.4, -0.12), 0.7),
			   sdBox(pr + vec3(-0.25, 0.7, 0.45), vec3(0.4))
				  );
	float c = min(
			   sdSphere(pr + vec3(0.4, -0.1, 0.2), 0.3),
			   sdBox(pr + vec3(0.5, -0.23, 0.5), vec3(0.4))
				  );
	float d = min(
			   sdSphere(pr + vec3(0.4, 0, -0.2), 0.2f),
			   sdBox(pr + vec3(-0.3, 0.5, -0.25), vec3(0.5))
				  );
	//return min(min(a, b), min(c, d));
	
	vec3 p1 = p;
	p1.x = sin(1.*PI*p.x + 2.*PI*p.y + 4.*PI*p.z);
	p1.y = sin(3.*PI*p.x + 1.*PI*p.y + 5.*PI*p.z);
	p1.z = sin(3.*PI*p.x + 2.*PI*p.y + 1.*PI*p.z);
	
	vec3 sz1 = vec3(0.01);
	float b1 = sdSphere(sz1*1.5*p1, sz1.x);
	
	
	vec3 p3 = p;
	p3.x = sin(7.*PI*p.x + 4.*PI*p.y + 1.*PI*p.z);
	p3.y = sin(6.*PI*p.x + 3.*PI*p.y + 2.*PI*p.z);
	p3.z = sin(5.*PI*p.x + 2.*PI*p.y + 3.*PI*p.z);
	
	vec3 sz3 = vec3(0.03);
	float b3 = sdSphere(sz3*2.5*p3, sz1.x);
	
	
	vec3 sz2 = vec3(0.4);
	float b2 = sdSphere(opRep(p, sz2*2.0+0.1), sz2.x);
	
	return smin(b1, b2, 0.); //smin(b2, b1, 0.2);
}

// for a given point in texture space, return the (signed) distance to the nearest surface
// distance is also in texture space
// stored in 4th component of field texture:
float scene(vec3 p) {
	//return scene_demo(p);
	float a = texture(landtex1, p).w;
	float b = texture(landtex2, p).w;
	return mix(a, b, landtexmix);
}

#define NORMAL_EPS 0.02

// returns both normal (gradient)
// gets normal via tetrahedron rather than cube, 4 taps rather than 6
vec3 normal4(in vec3 p)
{
	vec2 e = vec2(-NORMAL_EPS, NORMAL_EPS);
	float n = 1. / (4.*NORMAL_EPS*NORMAL_EPS);	// precomputed normalization factor
									// tetrahedral points:
	float t1 = scene(p + e.yxx), t2 = scene(p + e.xxy);
	float t3 = scene(p + e.xyx), t4 = scene(p + e.yyy);
	return (e.yxx*t1 + e.xxy*t2 + e.xyx*t3 + e.yyy*t4) * n;
}

// returns both normal (gradient) and curvature
// gets normal via tetrahedron rather than cube, 4 taps rather than 6
// for curvature, pass in curv = distance (i.e. DE(p))
vec4 norcurv(in vec3 p)
{
	vec2 e = vec2(-NORMAL_EPS, NORMAL_EPS) * 6.;
	// precomputed normalization factor
	float n = 1. / (4.*e.x*e.x);	
	
	// tetrahedral points:
	float t1 = scene(p + e.yxx), t2 = scene(p + e.xxy);
	float t3 = scene(p + e.xyx), t4 = scene(p + e.yyy);
	vec3 gradient = (e.yxx*t1 + e.xxy*t2 + e.xyx*t3 + e.yyy*t4) * n;
	
	
	/*
		t1-t is the rate of change of t  from p to p+e.xxx 
			etc. is t1+t2+t3+t4 - 4*t
			average by * 0.25 
		distance of p to p+e.xxx is:
			sqrt(e.x*e.x * 3.) = e.x * sqrt(3.)
		so normalize by 
			1./(e.x * 1.732050807568877)
		
		
	*/
	float t = scene(p);
	float cn = 0.25 / (e.x * sqrt(3.));
	float curvature = cn * (t1 + t2 + t3 + t4 - 4.*t);
	
	return vec4(gradient, curvature);
}

// NOT USED!
// for a given point in texture space, return the direction to the nearest surface:
vec3 scene_normal(vec3 p) {
	//vec3 gpu = norcurv(p);
	//vec3 gpu = normal4(p);
	vec3 cpu = texture(landtex1, p).xyz;
	return cpu;
}

// cheap fake ao, but not really working...
float ambient_occlusion(vec3 p, vec3 rd, float ao_scale) {
#define AO_SAMPLES 4
	float ao = 0.;			// initial occlusion
	float ao_falloff = 0.5;	// exponential decay factor
	float ao_step = stepsize*0.25;	// step size
	
	// take four samples stepping back along the normal
	for (int i=1; i<= AO_SAMPLES; i++) {
		float t1 = ao_step * float(i);
		vec3 p1 = p + rd*t1;
		// compare distance field at t1 distance away from p:
		float d = abs(scene(p1));
		// if t1 > d, concave area
		float diff = (t1 - d) * ao_falloff;
		// exponential falloff:
		ao_falloff = ao_falloff*ao_falloff;
		ao += diff;
	}
	
	return 1. - ao_scale * ao;
}

// for gl_FragDepth:
float computeDepth(vec3 p) {
	float dfar = gl_DepthRange.far;
	float dnear = gl_DepthRange.near;
	
	vec4 eye_space_pos = ciModelView * vec4(p, 1.);
	vec4 clip_space_pos = ciProjectionMatrix * eye_space_pos;
	float ndc_depth = clip_space_pos.z / clip_space_pos.w;
	
	// standard perspective:
	return (((dfar-dnear) * ndc_depth) + dnear + dfar) / 2.0;
}


void main() {
	
	// the ray origin at this pixel (in texture space):
	vec3 ro = origin;
	// the ray direction at this pixel (normalized):
	vec3 rd = normalize(direction);
	// stereo shift:
	//ro += cross(normalize(rd), up * parallax;
	
	// distance along ray (in texture space):
	float t = near;
	// ray limit (in texture space):
	float tmax = 1.;
	// p is current ray point, also in texture space:
	vec3 p = ro + rd * t;
	
	// initial color:
	vec3 color = vec3(0.);
	
	// the distance from the current point to the nearest surface:
	float d = scene(p);
	float side = sign(d);	// did we start inside or outside? TODO: use this to determine increment

	float intensity = 0.;
	float halo = 0.;
	
	float halo_accum = inv_max_steps;
	
	int i = 0;
	for (; i < MAX_STEPS; i++) {

		if (abs(d) < eps) {
			// hit surface

			
			
			//To hit eps exactly we want to 'backtrack' a little.
			//How much to backtrack is difficult to know, but we can assume interpolation between current and previous point
			// old point 't' is 'd' distance "outside"
			// new point 't2' is 'd2' distance "inside"
			float t2 = t + (d);		// t at new point
			float d2 = eps - (d);	// distance t is inside
			// weighted average of d & d2 is applied to t and t2:
			t = (t*d2 + t2*d)/(d2 + d);
			p = ro + rd * t;

			d = eps;
			break;
			
		} else if (t > tmax) {
			t = tmax;
			p = ro + rd*tmax;
			// overshot world
			break;
		}
		
		float advance = abs(d) * 0.9;
		t += advance;
		//p += rd*advance;
		p = ro + rd * t;
		
		// get distance at current location:
		//d = scene_demo(p);
		d = scene(p);
		
		halo += halo_accum;
	}
	
	if (d==eps) {
		// ray has hit a surface
		// calculate lighting for the ray position
		
		// compute the intersection in absolute world space:
		vec3 worldposition = p * world_dim;
		// want to return this to view space:
		vec3 viewposition = (ciModelView * vec4(worldposition, 1.)).xyz;

		
		
		// get normal (at point just before we landed)
		vec4 nc = norcurv(p);
		vec3 normal = nc.xyz;
		float curv = nc.w;
		
		vec3 variance = normal + normal4(p * 8.);//texture(landtex, p * 8.).xyz;
		float surface_noise = variance.x * 3.;
		float surface_scale = variance.y * 2.;
		
		// temporal disturbance:
		vec3 jitter = texture(noisetex, texture(noisetex, variance + vec3(now)).xyz).xyz;
		
		// surface variation:
		// TODO: consider using something dynamic & meaningful here
		// e.g. fluid flow, chemical density, etc.
		//vec4 disturb = texture(noisetex, normal * 0.2 * surface_scale + p * 0.5);
		// TODO:
		vec4 disturb = texture(noisetex, p * surface_scale);
		
		//disturb = texture(noisy, disturb.xyz);
		//float amt = 0.5 * sin(now + disturb.a * 3.14);
//		normal = normalize(normal);
		normal = normalize(normal + disturb.xyz * 0.1 + jitter * 0.25);
		
		// do lighting:
		
		
		// TODO:
		//	 consider adding some specular effect (especially from grain)
		//	 consider fresnel effect (could nicely mask the ray overshoots)
		
		
		float amt = surface_noise * disturb.w;
		vec3 ldir1 = normalize(light1 + disturb.xyz * amt);
		vec3 ldir2 = normalize(light2 - disturb.xyz * amt);
		vec3 onesided = color1 * max(0.,dot(ldir1, normal)) + color2 * max(0.,dot(ldir2, normal));
		vec3 twosided = color1 * abs(dot(ldir1, normal))    + color2 * abs(dot(ldir2, normal));
		vec3 lighting = ambient + mix(onesided, twosided, 0.8);
		
		color = lighting;
		//color = mix(color, lighting, 0.8);
		
		// fake ambient occlusion from the number of steps!
		// (kind of anti-halo)
		// for light halo, use += and change overstep color to white
		// trouble is, it introduces banding because of the integer steps.
		// using halo*halo reduces the banding a lot.
		//color += vec3(halo * halo);
		color -= vec3(halo * halo);
		
		//color *= vec3(0.3-(curv*0.5));
		color += 0.1-0.5*abs(curv);
		
		//float ao_scale = 10.; 	// how deep the shadow effect
		//color *= ambient_occlusion(p, normal, ao_scale);
		
		// fog effect:
		color = fog(color, viewposition);
		
		//color = viewposition;
		
		//color = vec3(normal);
		//color = vec3(curv);
	}
	
	//color = vec3(abs(texture(landtex, vec3(T, origin.z)).w)); // does landtex work?
	//color = p;
	//color = vec3(d == eps);
	
	//color = vec3(halo * 4.);	// indicates number of steps taken
	//color = vec3(t*0.5);		// ray length
	
	//color = rd+0.5;
	
	outColor = vec4(color, 1.);
	gl_FragDepth = computeDepth(p * dim);
	
	
}
