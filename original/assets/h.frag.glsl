#include "lib.glsl"
#version 150


#define STEPS 64
#define EPS 0.003
#define GLOW 0.2
#define GLOWDISTANCE 0.1
#define FAR 3.0
#define PI 3.14159265359

uniform float time;
uniform mat4 ciModelViewProjectionMatrix;
uniform sampler3D noisetex;

in vec4	Color;
in vec3 WorldPosition;
in vec3 VertexPosition;
in vec3 ray;
in vec3 lightp;
in vec3 param;
in mat4 mvp;


in vec2 trackpadValue;
in float trackpadButton;
in float triggerValue;

out vec4 outColor;

vec3 us = vec3(0.667, 1., 0.4); // scalar factor to get back to uniform


vec4 La = vec4( 0.35, 0.1, 0.4, 1); // -- light ambient
vec4 Li = vec4( 0.75, 0.55, 0.1, 1); // -- light incident (diffuse)
vec4 Ks = vec4( 0.98, 0.13, 0.1, 1);
vec4 Ka = vec4( 0.2, 0.2, 0.25, 1);
vec4 Kd = vec4( 0.15, 0.5, 0.5, 1);
float Ns = 15.;						// shininess

float fog_density = 1.;
vec3 fog_color = vec3(0.);
float fog_offset = 14.;

float gamma = 2.2;
vec3 gamma3 = vec3(1.0 / gamma);

vec4 foga(vec4 color, vec3 pos) {
	// fog parameters
	float distance0 = length(pos);
	float distance = max(distance0-fog_offset, 0.);
	float fogExponent = distance*fog_density;
	float fogFactor = exp2(-abs(fogExponent));
	float z = clamp(-8.*(pos.z+0.02), 0., 1.);	// 0.5 is the nearness
	return vec4(mix(fog_color, color.rgb, fogFactor), color.a * z);
}

vec3 xDir = vec3(EPS,0,0);
vec3 yDir = vec3(0,EPS,0);
vec3 zDir = vec3(0,0,EPS);

float rand(vec2 coordinate) {
	return fract(sin(dot(coordinate.xy, vec2(12.9898, 78.233))) * 43758.5453);
}



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

vec3 rX(in vec3 p, float a) {
	float c,s;vec3 q=p;
	c = cos(a); s = sin(a);
	p.y = c * q.y - s * q.z;
	p.z = s * q.y + c * q.z;
	return p;
}

vec3 rY(in vec3 p, float a) {
	float c,s;vec3 q=p;
	c = cos(a); s = sin(a);
	p.x = c * q.x + s * q.z;
	p.z = -s * q.x + c * q.z;
	return p;
}

vec3 rZ(in vec3 p, float a) {
	float c,s;vec3 q=p;
	c = cos(a); s = sin(a);
	p.x = c * q.x - s * q.y;
	p.y = s * q.x + c * q.y;
	return p;
}

float displace( in vec3 p ) {
	vec3 p1 = p * us;  // uniform scale
	float f = 9.; //4.*sin(time * 8.);
	return sin(f*p1.x + time*1.)*cos(f*p1.y + time*2.)*sin(f*p1.z + time*3.);
}

// polynomial smooth min (k = 0.1);
float smin( float a, float b, float k ) {
	float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
	return mix( b, a, h ) - k*h*(1.0-h);
}

// an alternative form, but more expensive
float smin0( float a, float b, float k) {
	float res = exp( -k*a ) + exp( -k*b );
	return -log( res )/k;
}

float smax( float a, float b, float k )
{
	float k1 = k*k;
	float k2 = 1./k1;
	return log( exp(k2*a) + exp(k2*b) )*k1;
}

vec2 smax2( vec2 a, vec2 b, float k )
{
	float k1 = k*k;
	float k2 = 1./k1;
	return vec2(
		log( exp(k2*a.x) + exp(k2*b.x) )*k1,
		log( exp(k2*a.y) + exp(k2*b.y) )*k1
		);
}

vec2 smin2( vec2 a, vec2 b, float k )
{
	vec2 res = exp( -k*a ) + exp( -k*b );
	return -log( res )/k;
}

float sub(in float A, in float B) {
	return max(A, -B);
}

float ssub(in float A, in float B, float k) {
	return smax(A, -B, k);
}


// Euclidean distance metric is length(p) -- this is the only one that preserves rotational symmetry (its primitive is a circle)
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
float qnorm2(vec2 p, float q) {
	return pow(pow(p.x, q) + pow(p.y, q), 1./q);
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

// rounded cube, by using a higher-order distance metric
float sdRcubeN(in vec3 p, float r, float n) {
	return q2euc(qnorm(p, n) - r, n);
}

// plane defined by a normal and a point on the plane
// note: it is not efficient to render planes in sphere tracing
// but it might still be useful as a component within CSG for example.
float sdPlane(in vec3 p, in vec3 normal, in vec3 pt) {
	return length(dot(p, normal) - pt);
}

// simpler version for  a regular ground plane:
float sdPlaneY(in vec3 p) {
	return p.y;
}

float sdBox(in vec3 p, in vec3 b ) {
	vec3 d = abs(p) - b; 				// utilize symmetry of cuboid
	float ri = max(d.x, max(d.y, d.z)); // distance inside box
	float ro = length(max(d, 0.)); 		// distance if outside box
										// return ri > 0. ? ro : ri;
	return min(ri, 0.) + ro;	// cheaper than a conditional
}

float sdSphere(in vec3 p, in float r) {
	return length(p)-r;
}


// this one seems slightly better quality:
float sdEllipsoid(in vec3 p, in vec3 s) {
	// apply scalar:
	vec3 rs = 1./s;
	float f = length(p*rs);
	float d = f-1.;
	// (reciprocal of) analytic gradient:
	vec3 s2 = pow(rs, vec3(2));  // vec3(s.x*s.x, s.y*s.y, s.z*s.z);
	float gr = f/(length(p*s2));
	return d*gr;
}

// iq has this version, which seems a lot simpler?
float sdEllipsoid1( in vec3 p, in vec3 r ) {
	return (length( p/r ) - 1.0) * min(min(r.x,r.y),r.z);
}

// aligned to Z axis, radius r, length l
float sdCylinder(in vec3 p, in float r, in float l) {
	// intersection of:
	// infinite cylinder radius r, and
	// box length l (z axis only)
	return max(length(p.xy)-r, abs(p.z) - l);
}

float sdSegmentedCylinder(in vec3 p, in float r, in float l, in float segment_angle) {
	// intersection of:
	// infinite cylinder radius r, and
	// box length l (z axis only)
	
	float a = atan(p.y, p.x);
	float d = length(p.xy);
	// quantize it
	//a = segment_angle * floor(a / segment_angle);
	a = mod(a, 2.*segment_angle) - segment_angle;
	vec2 p2 = vec2(d*cos(a), d*sin(a));
	return max(qnorm2(p2.xy,40.)-r, abs(p.z) - l);
}

// like cylinder, but with r dependent on p.z
float sdCone(in vec3 p, in float r, in float l) {
	// is there a way to do this without the division?
	float a = (p.z*0.5+0.5)/l;
	return max(length(p.xy)-r*a, abs(p.z) - l);
}

// according to comments in https://www.shadertoy.com/view/Xds3zN this is more accurate:
float sdCone1( in vec3 p, in vec3 c ) {
	vec2 q = vec2( length(p.xz), p.y );
	vec2 v = vec2( c.z*c.y/c.x, -c.z );
	vec2 w = v - q;
	vec2 vv = vec2( dot(v,v), v.x*v.x );
	vec2 qv = vec2( dot(v,w), v.x*w.x );
	vec2 d = max(qv,0.0)*qv/vv;
	return sqrt( dot(w,w) - max(d.x,d.y) )* sign(max(q.y*v.x-q.x*v.y,w.y));
}

float sdConeSection( in vec3 p, in float h, in float r1, in float r2 )
{
	float d1 = -p.y - h;
	float q = p.y - h;
	float si = 0.5*(r1-r2)/h;
	float d2 = max( sqrt( dot(p.xz,p.xz)*(1.0-si*si)) + q*si - r2, q );
	return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
}

float sdTorus(in vec3 p, in float R, in float r) {
	// length(p.xy)-R is a disc radius R on z-axis
	// i.e. torus is like two 2D circles combined
	vec2 xy = vec2(length(p.xy)-R, p.z);
	return length(xy)-r;
}


////////////////////////////////////////

float opSub(in float A, in float B) {
	return max(A, -B);
}

float opShell(in float A, in float B, in float t) {
	return max(A-t*0.5, -B-t*0.5);
}

vec3 opRep(in vec3 p, in vec3 c) {
	return mod(p,c)-0.5*c;
}

// NOTE scale := f(p/s)*s



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
float sdCapsule(vec3 p, vec3 a, vec3 b, float ra, float rb) {
	vec3 pa = p - a, ba = b - a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return length( pa - ba*h ) - mix(ra, rb, h);
}

// very similar, but what is ll?
vec2 sdSegment2( vec3 a, vec3 b, vec3 p, float ll ) {
	vec3 pa = p - a;
	vec3 ba = b - a;
	float h = clamp( dot(pa,ba)*ll, 0.0, 1.0 );
	return vec2( length( pa - ba*h ), h );
}

float sdCapsule2(vec3 p, vec3 a, vec3 b, float ra, float rb) {
	vec3 pa = p - a, ba = b - a;
	float t = dot(pa,ba)/dot(ba,ba);	// phase on line from a to b
	float h = clamp( t, 0.0, 1.0 );
	
	// add some ripple:
	float h1 = h + 0.2*sin(PI * 4. * (t*t + time* 0.3));
	
	// basic distance:
	vec3 rel = pa - ba*h;
	float d = length(rel);
	
	d = d - mix(ra, rb, h1);
	
	return d;
}

vec2 de_head(in vec3 p) {
	return vec2(
		sdTorus(rX(p - vec3(0, 0, -0.28), -0.3), 0.3, 0.1),
		0.);
}

vec2 de_shaft(in vec3 p) {
	return vec2(
		sdBox(rX(p - vec3(0, 0.24, 0.14), -0.1), vec3(0.12, 0.1, 0.32)),
		1.);
}


vec2 de_button(in vec3 p) {
	return vec2(
		sdSphere(p - vec3(0, 0.4, 0.15), 0.3),
		2.);
}

vec2 de_trigger(in vec3 p) {
	float trigger = mod(time * 2., 1.);
	vec3 trig_center = vec3(0, 0.05, -0.04);
	vec3 trig_radius = vec3(0.00066, 0.001, 0.0004);
	return vec2(
		sdEllipsoid1(rX(p - trig_center*2., 0.1+trigger) + trig_center, trig_radius),
		3.);
}

vec2 DE1(in vec3 p) {
	vec2 a = de_head(p);
	vec2 b = de_shaft(p);
	
	vec2 bs = de_button(p);
	vec2 t = de_trigger(p);
	
	b = smax2(b, bs, 0.15);
	b = min(b,t);
	vec2 c = smin2(a, b, 0.3);
	
	// a little reshaping:
	c.s = sub(c.s, sdEllipsoid(p - vec3(0., 0.6, -0.02), vec3(0.2, 0.3, 0.12)));
	c.s += displace(p * 3.) * (0.1 * p.z);
	
	return c;
}

vec4 DE(in vec3 p) {
	float trigger = triggerValue; //mod(time * 2., 1.);
	vec3 bmat = vec3(0.5);
	vec3 tmat = Color.rgb; //vec3(0., 0.5, 1.);
	vec3 mat = bmat;
	
	float limit = sdSphere(p, 0.1);
	
	// handle:
	float handle = sdBox(rX(p - vec3(0, 0.24, 0.14), -0.1), vec3(0.12, 0.1, 0.32));
	float handle_curve = sdSphere(p - vec3(0, 0.4, 0.15), 0.3);
	handle = smax(handle, handle_curve, 0.15);
	
	// trigger:
	vec3 trig_center = vec3(0, 0.05, -0.04);
	vec3 trig_radius = vec3(0.00066, 0.001, 0.0004);
	float trig = sdEllipsoid1(rX(p - trig_center*2., 0.1+trigger) + trig_center, trig_radius);
	
	// trackpad:
	float trackpad = sdEllipsoid(p - vec3(0., 0.6, -0.02), vec3(0.2, 0.3, 0.12));
	
	// head:
	float head = sdTorus(rX(p - vec3(0, 0, -0.28), -0.3), 0.3, 0.1);
	
	float d = handle;
	
	mat = d < trig ? mat : tmat;
	d = min(d,trig);
	
	mat = head < d ? bmat : mat;
	d = smin(head, d, 0.3);
	
	mat = d > -trackpad ? mat : tmat;
	d = sub(d, trackpad);
	
	d += displace(p * 3.) * (0.1 * p.z);
	
	return vec4(mat, d);
}


// returns normal (gradient)
// gets normal via tetrahedron rather than cube, 4 taps rather than 6
vec3 normal4(in vec3 p)
{
	vec2 e = vec2(-EPS, EPS);
	float n = 1. / (4.*EPS*EPS);	// precomputed normalization factor
									// tetrahedral points:
	float t1 = DE(p + e.yxx).s, t2 = DE(p + e.xxy).s;
	float t3 = DE(p + e.xyx).s, t4 = DE(p + e.yyy).s;
	return (e.yxx*t1 + e.xxy*t2 + e.xyx*t3 + e.yyy*t4) * n;
}

// returns both normal (gradient) and curvature
// gets normal via tetrahedron rather than cube, 4 taps rather than 6
// for curvature, pass in curv = distance (i.e. DE(p))
vec3 norcurv(in vec3 p, inout float curv)
{
	vec2 e = vec2(-1., 1.) * EPS;
	float n = 1. / (4.*EPS*EPS);	// precomputed normalization factor
									// tetrahedral points:
	float t1 = DE(p + e.yxx).s, t2 = DE(p + e.xxy).s;
	float t3 = DE(p + e.xyx).s, t4 = DE(p + e.yyy).s;
	vec3 grad = (e.yxx*t1 + e.xxy*t2 + e.xyx*t3 + e.yyy*t4) * n;
	//curv = .125/(e.x*e.x) * (t1 + t2 + t3 + t4 - 4. * DE(p).s);
	curv = .125/(e.x*e.x) * (t1 + t2 + t3 + t4 - 4. * curv);
	return grad;
}


float ao(vec3 p, vec3 n, float d, float i) {
	float o;
	for (o=1.;i>0.;i--) {
		o-=(i*d-abs(DE(p+n*i*d).s))/pow(2.,i);
	}
	return o;
}

// http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k ) {
	float res = 1.0;
	for( float t=mint; t < maxt; )
	{
		float h = DE(ro + rd*t).s;
		if( h<0.001 )
			return 0.0;
		res = min( res, k*h/t );
		t += h;
	}
	return res;
}

vec4 shade(in vec3 p, in vec3 rd, in float distance, in float steps) {
	vec4 noise = 0.3*texture(noisetex, p*1.5+0.5);
	
	vec3 color = Color.rgb;
	color.g = color.b;
	
	//color *= noise.w;
	
	// fog -- trickier, as p is in object coordinates; we need camera coordinates. but for the size of the objects we are dealing with, perhaps just a vertex-based fog is sufficient?
	
	float curv = (distance);
	vec3 n = norcurv(p, curv);
	
	// surface texture:
	//n = normalize(mix(n, noise.xyz, 0.4 * sin(time)));
	n = normalize(n + noise.xyz);
	
	// TODO: vertex shader should put lightposition into objectspace
	vec3 lightDir = normalize(lightp - p);
	
	
	vec3 ambient = color * 0.2;
	
	vec3 diffuseColor = vec3(1.);
	
	float ndotl = dot(n, lightDir);
	
	vec3 diffuse = diffuseColor * color * abs(ndotl);
	//vec3 diffuse = diffuseColor * color * max(0.,ndotl);
	
	// cheat:
	//diffuse = mix(color, diffuse, ndotl*0.5+0.5);
	
	vec3 specularColor = vec3(0.9);
	float shininess = 2.;
	vec3 specular = vec3(0.);
	if (ndotl > 0.) specular = specularColor * pow(max(0.0, dot(reflect(lightDir, n), rd)), shininess);
	// cheat:
	//specular = specularColor * pow(abs(dot(reflect(lightDir, n), rd)), shininess);
	
	color.rgb = ambient + diffuse + specular;
	
	//color.rgb += 0.5*lightColor * dot(n, lightDir);
	
	//color.rgb *= ao(p, n, 0.25, 5.);
	//color.rgb *= 0.5+softshadow(p, lightDir, 0.1, 2., 10.);
	
	//color = vec3(0.5)+(n);
	//color += 0.1*vec3(curv);
	
	return vec4(color, Color.a);
}

// p is in clip space
// for gl_FragDepth:
float computeDepthE(vec4 p) {
	float dfar = gl_DepthRange.far;
	float dnear = gl_DepthRange.near;
	// standard perspective:
	return (((dfar-dnear) * p.z / p.w) + dnear + dfar) * 0.5;
}

// p is in object space
// for gl_FragDepth:
float computeDepth(vec3 p) {
	//return computeDepthE(ciModelViewProjectionMatrix * vec4(p, 1.));
	return computeDepthE(mvp * vec4(p, 1.));
}

void main( void )
{
	vec3 rd = normalize(ray);
	vec3 ro = VertexPosition.xyz;
	float offset = 0.;
	
	// TRACE:
	vec3 p = ro;
	vec3 result = ro;
	float d = 0.;
	float glow = 0.25;
	float t = 0.0;	// EPS?
	int steps = 0;
	float stepsize = 1.;
	outColor = vec4(0.);
	int contact = 0;
	
	for (; steps < STEPS; steps++) {
		vec4 result = DE(p);
		d = abs(result.w);
		if (d < EPS) {
			contact++;
			
			//vec4 s = shade(p, rd, d, 0.);
			vec3 mat = result.rgb;
			vec4 s = vec4(mat, 0.8);
			
			outColor += s * 0.03; //vec4(0.1+float(steps)/float(STEPS));
			
			// push on:
			d = EPS*1.;
		}
		
		t += d * stepsize;
		
		p = ro + t*rd;
		if (t >= FAR) break;
		
		glow += 1./float(STEPS);
		
	}
	
	if (contact > 0) {
		
		// TODO: refine point with 3-5 iterations of "Discontinuity reduction" as proposed in Enhanced Sphere Tracing; Benjamin Keinert1 Henry Schafer1 Johann Korndorfer Urs Ganse2 Marc Stamminger1
		
		//outColor = vec4(0.1+float(steps)/float(STEPS));
		//outColor = shade(p, rd, d, 0.) * 0.5;
		gl_FragDepth = computeDepth(p);
		
		//outColor = foga(outColor, WorldPosition);
		// tint & premultiply
		outColor.rgb *= outColor.a;
		// gamma
		outColor.rgb = pow(outColor.rgb, gamma3);
	} else {
		outColor = vec4(0.+float(steps)/float(STEPS));
		//outColor += vec4(0.1);
		//gl_FragDepth = 0.9999;
		discard;
	}
	
	//outColor = vec4(ray,1.);
	//outColor = vec4(param,1.);
}