#include "lib.glsl"
#version 150


#define STEPS 24
#define EPS 0.01
#define FAR 2.0
#define PI 3.14159265359

uniform float time;
uniform mat4 ciModelViewProjectionMatrix;
uniform sampler3D noisetex;

in vec4	Color;
in vec3 WorldPosition;
in vec3 ray_origin;
in vec3 ray;
in vec3 lightp;
//in vec3 param;
in mat4 mvp;
in vec4 orient;

out vec4 outColor;

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
	float z = clamp(-8.*(pos.z+0.12), 0., 1.);	// 0.5 is the nearness
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


/*
 
 float smin(float a, float b, float k) {
 float h = clamp(.5+.5*(b-a)/k, 0.0, 1.0 );
 return mix(b,a,h)-k*h*(1.-h);
 }
 
 float smin(float a, float b, float blendRadius) {
 float c = saturate(0.5 + (b - a) * (0.5 / blendRadius));
 return lerp(b, a, c) - blendRadius * c * (1.0 - c);
 }
 
 vec2 rep(vec2 p) {
 float a = atan(p.y, p.x);
 a = mod(a, 2.0*PI/18.) - PI/18.;
 return length(p)*vec2(cos(a), sin(a));
 }
 
 
 float spikedBall(vec3 p) {
 p = mod(p, 8.0) - 4.0;
 float d = length(p) - 1.2;
 p.xz = rep(p.xz); p.xy = rep(p.xy);
 return smin(d, length(p.yz)-.1+abs(.15*(p.x-1.0)), 0.1);
 }
 
 float capsules(vec3 p) {
 vec3 q = floor(p/4.0);
 p = mod(p, 4.0) - 2.0;
 p.yz = p.yz*cos(iGlobalTime + q.z) + vec2(-p.z, p.y)*sin(iGlobalTime + q.z);
 p.xy = p.xy*cos(iGlobalTime + q.x) + vec2(-p.y, p.x)*sin(iGlobalTime + q.x);
 p.zx = p.zx*cos(iGlobalTime + q.y) + vec2(-p.x, p.z)*sin(iGlobalTime + q.y);
 
 float angle = .3*cos(iGlobalTime)*p.x;
 p.xy = cos(angle)*p.xy + sin(angle)*vec2(-p.y, p.x); p.x += 1.0;
 float k = clamp(2.0*p.x/4.0, 0.0, 1.0); p.x -= 2.*k;
 return length(p) - .5;
 }
 
 float cubeMap(vec3 p, vec3 n) {
 float a = texture2D(iChannel0, p.yz).r;
 float b = texture2D(iChannel0, p.xz).r;
 float c = texture2D(iChannel0, p.xy).r;
 n = abs(n);
 return (a*n.x + b*n.y + c*n.z)/(n.x+n.y+n.z);
 }
 
 vec3 bumpMap(vec3 p, vec3 n, float c) {
 vec2 q = vec2(0.0, .5);
	vec3 grad = -1.0*(vec3(cubeMap(p+q.yxx, n), cubeMap(p+q.xyx, n), cubeMap(p+q.xxy, n))-c)/q.y;
 vec3 t = grad - n*dot(grad, n);
 return normalize(n - t);
 }
 
 vec3 shade(vec3 ro, vec3 rd, float t) {
 vec3 p = ro + t*rd, n = normal(p);
 
 vec3 green = pow(vec3(93,202,49)/255., vec3(2.2));
 vec3 yellow = pow(vec3(255,204,0)/255., vec3(2.2));
 
 float k = cubeMap(.5*p, n);
 n = bumpMap(.5*p, n, k);
 
 vec3 col = mix(green, yellow, k)*(1.0-dot(-rd,n));
 if (spikedBall(p) < capsules(p)) {
 p = mod(p, 8.0) - 4.0;
 col *= 1.0/(1.0 + .5*dot(p, p));
 }
 
 return col*exp(-.008*t*t);
 }
 
 
 vec3 image(in vec2 fragCoord ) {
	vec2 uv = (-iResolution.xy + 2.0*fragCoord.xy) / iResolution.y;
 uv *= 1.0 + .1*dot(uv,uv);
 
 vec3 ro = vec3(iGlobalTime, iGlobalTime, cos(iGlobalTime));
 vec3 rd = normalize(lookat(ro, ro+vec3(cos(.1*iGlobalTime), sin(.1*iGlobalTime), 1.0))*vec3(uv, -1.0)); // direo do raio.
 
 // based on eiffie's antialiasing method (https://www.shadertoy.com/view/XsSXDt)
 vec3 col = vec3(0.0);
 vec4 stack = vec4(-1.0); bool grab = true;
 float t = 0.0, d = EPS, od = d, pix = 4.0/iResolution.x, w = 1.8, s = 0.0;
 for (int i = 0; i < STEPS; ++i) {
 d = map(ro + t*rd);
 if (w > 1.0 && (od + d < s)) {
 s -= w*s; w = 1.0;
 } else {
 s = d * w;
 if (d <= od) grab = true;
 else if (grab && stack.w < 0. && od < pix*(t-od)) {
 stack.w = t-od; stack = stack.wxyz;
 grab = false;
 }
 if (d < EPS || t > FAR) break;
 }
 od = d; t += s;
 }
 col = d < EPS ? shade(ro, rd, t) : col;
 
 for (int i = 0; i < 4; ++i) {
 if (stack[i] < 0.0) break;
 d = map(ro + stack[i]*rd);
 col = mix(shade(ro, rd, stack[i]), col, clamp(d/(pix*stack[i]), 0.0, 1.0));
 }
 
 col = smoothstep(0., .7, col);
 col = pow(col, vec3(1.0/2.2));
 
	return col;
 }
 
 */

float displace( in vec3 p ) {
	float f = 10.*sin(time*0.01);
	return sin(f*p.x)*sin(f*p.y)*sin(f*p.z);
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


float naive(in vec3 p) {
	
	p = p.yxz;
	
	// basic symmetry:
	p.y = abs(p.y);
	
	// blobbies
	vec3 A = vec3(0., 0., -0.5);
	vec3 B = vec3(0., 0., 0.5);
	float w = 0.125*abs(2.+0.5*sin(14.*p.z - 8.8*time));
	//float w = 0.4;
	float z = 0.25;
	float y = 0.5;
	float a = sdCapsule1(p, vec3(0., 0., -0.25), vec3(0., y, z), w*w);
	float b = sdCapsule2(p, vec3(0., -0., -0.25), vec3(z, w, y), 0.125, 0.1);
	//float a = 0.7;
	//float b = 0.7;
	
	float d = smin(a, b, 0.5);
	
	return ssub(d, sdEllipsoid1(p.yzx, vec3(0.25, 0.5, 0.05)), 0.125);
}

// r is radius
// s is scalar squish, vec3(1.) gives a circle
float naive1(in vec3 p) {
	vec3 s = vec3(8,1,1);
	float r = 0.5 + 0.000005*sin(time);
	
	return length(p*s) - r;
}

float numerical(in vec3 p) {
	// the naive call:
	float f = naive(p);
	
	// get the gradient:
	vec2 e = vec2(-EPS, EPS);
	float n = 1. / (4.*EPS*EPS);	// precomputed normalization factor
									// tetrahedral points:
	float t1 = naive(p + e.yxx);
	float t2 = naive(p + e.xxy);
	float t3 = naive(p + e.xyx);
	float t4 = naive(p + e.yyy);
	
	// derive normal (whose magnitude is the gradient):
	// (BTW we might want to keep a hold of this normal...?)
	vec3 normal = (e.yxx*t1 + e.xxy*t2 + e.xyx*t3 + e.yyy*t4) * n;
	float g = length(normal);
	
	// scale accordingly:
	return f / g;
}

float DE(in vec3 p) {
	vec3 c = p + vec3(0, 0, 0.5);
	
	///float a = sdEllipsoid(p,vec3(1./16.,0.5,0.5));
	//float b = sdEllipsoid(p,vec3(0.2,1./16.,0.5 * sin(time)));
	
	float a = sdEllipsoid(p,vec3(1./16.,0.5,0.5));
	
	//return numerical(p);
	//return naive(p);
	//return sdSphere(p, 0.5);
	//return sdBox(c, vec3(0.1, 0.1, 0.5));
	//return sdRcube(c, 0.25);
	float b = sdCylinder(c, 0.2, 0.5);
	
	return smin(a, b, 0.2);

}


float DE2(in vec3 p) {
	float e = 0.01; // should be resolution-dependent
	
	float f = naive(p);
	float g = length( vec2(dFdx(f), dFdy(f))/e );
	return f/g;
}

float DE3(in vec3 p) {
	float e = 0.001; // should be resolution-dependent
	float f = naive(p);
	float g = length( vec3(naive(p+vec3(e,0.0,0.))-naive(p-vec3(e,0.0,0.)),
						   naive(p+vec3(0.0,e,0.))-naive(p-vec3(0.0,e,0.)),
						   naive(p+vec3(0.0,0.,e))-naive(p-vec3(0.0,0.,e))) )/(2.0*e);
	return f / g;
	
}



// another shape is a 'skeleton' curve
// find the nearest point on curve to p
// then apply offset radius to turnw it into a curved cylinder

float DE1(in vec3 p) {
	
	/*
	 // warp:
	 //p.x += sin(time + p.z * 3.); // not distance-preserving
	 
	 //wp = rY(p, 2.5*p.y);
	 
	 float s = 1. + 0.02*sin(time * 4.);
	 
	 float a = sdBox(p, vec3(param.y));
	 
	 
	 float b = sdSphere(p/s, 0.9)*s;
	 
	 //b += 0.1*sin(time + p.z * 3.); // not distance-preserving
	 //return smin(a, b, 0.1);
	 float c = smin(a, b, 0.1);
	 //return a;
	 
	 //c += displace(p)*0.04;
	 
	 
	 //vec3 pr = opRep(p, vec3(0.9 + 0.05*sin(time))) ;
	 
	 float r1 = (param.x + 0.5) + 0.05*sin(time);
	 float r2 = (param.z + 0.5) - 0.05*sin(time);
	 
	 vec3 pr = opRep(p, vec3(r1, r1, r2)) ;
	 float d = sdSphere(pr, min(r1,r2)*0.5);
	 
	 
	 //return opSub(c, d);
	 
	 //return sdDiamond(p, 1.);
	 //return sdRcube(p, 1.);
	 //return sdTorus(p, 0.7, 0.3);
	 //return sdCone(p, normalize(vec2(1)));
	 
	 */
	
	/*
	 // blobbies
	 vec3 A = vec3(0., 0., -0.5);
	 vec3 B = vec3(0., 0., 0.5);
	 float w = 0.125*abs(3.+sin(4.*p.z + 8.*time));
	 //float w = 0.4;
	 float z = 0.5;
	 float y = 0.5;
	 float a = sdCapsule(p, vec3(0., 0., -0.5), vec3(0., y, z), w);
	 float b = sdCapsule(p, vec3(0., -0., -0.5), vec3(0., -y, z), w);
	 
	 return smin(a, b, w);
	 */
	
	float a = sdBox(p, vec3(0.5));
	float b = sdSphere(p, 0.5);
	return opShell(a, b, 0.2*sin(time * 3. + PI*2.*Color.r));
}

// returns normal (gradient)
// gets normal via tetrahedron rather than cube, 4 taps rather than 6
vec3 normal4(in vec3 p)
{
	vec2 e = vec2(-EPS, EPS);
	float n = 1. / (4.*EPS*EPS);	// precomputed normalization factor
									// tetrahedral points:
	float t1 = DE(p + e.yxx), t2 = DE(p + e.xxy);
	float t3 = DE(p + e.xyx), t4 = DE(p + e.yyy);
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
	float t1 = DE(p + e.yxx), t2 = DE(p + e.xxy);
	float t3 = DE(p + e.xyx), t4 = DE(p + e.yyy);
	vec3 grad = (e.yxx*t1 + e.xxy*t2 + e.xyx*t3 + e.yyy*t4) * n;
	//curv = .125/(e.x*e.x) * (t1 + t2 + t3 + t4 - 4. * DE(p));
	curv = .125/(e.x*e.x) * (t1 + t2 + t3 + t4 - 4. * curv);
	return grad;
}

float ao(vec3 p, vec3 n, float d, float i) {
	float o;
	for (o=1.;i>0.;i--) {
		o-=(i*d-abs(DE(p+n*i*d)))/pow(2.,i);
	}
	return o;
}

// http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k ) {
	float res = 1.0;
	for( float t=mint; t < maxt; )
	{
		float h = DE(ro + rd*t);
		if( h<0.001 )
			return 0.0;
		res = min( res, k*h/t );
		t += h;
	}
	return res;
}

vec4 shade(in vec3 p, in vec3 rd, in float distance, in float steps) {
	vec4 noise = texture(noisetex, p*(p.y+0.5)+0.5);
	
	
	vec3 n = normal4(p);
	
	// surface texture:
	n = normalize(mix(n, noise.xyz, 0.4 * sin(2.*time - 2.*p.z)));
	
	// TODO: vertex shader should put lightposition into objectspace
	vec3 lightDir = normalize(-lightp);
	
	
	vec3 color = vec3(0.5);
	
	//color *= noise.w;
	
	// fog -- trickier, as p is in object coordinates; we need camera coordinates. but for the size of the objects we are dealing with, perhaps just a vertex-based fog is sufficient?
	
	vec3 pt = p+0.5;
	pt.x = abs(pt.x);
	vec3 ambient = texture(noisetex, pt*0.2).xyz*0.25;
	
	vec3 diffuseColor = vec3(0.8);
	
	float ndotl = dot(n, lightDir);
	
	vec3 diffuse = diffuseColor * max(0.,ndotl);
	
	vec3 specularColor = ambient; //vec3(0.2, 0.5, 0.2);
	float shininess = 200.;
	vec3 specular = vec3(0.);
	if (ndotl > 0.) specular = specularColor * pow(max(0.0, dot(reflect(lightDir, n), rd)), shininess);
	
	color.rgb = ambient + diffuse + specular;
	
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
	vec3 ro = ray_origin - rd;
	float offset = 0.;
	
	// TRACE:
	vec3 p = ro;
	vec3 result = ro;
	float d = 0.;
	float glow = -0.15;
	float glow_rate = 1./float(STEPS);
	float t = 0.0;	// EPS?
	int steps = 0;
	outColor = vec4(0.);
	int contact = 0;
	
	for (; steps < STEPS; steps++) {
		d = DE(p);
		
		t += d;
		p = ro + t*rd;
		
		if (t >= FAR) break;
		if (abs(d) < EPS) {
			contact++;
			break;
		}
		
		glow += glow_rate;
	}
	
	if (contact > 0) {
		
		// TODO: refine point with 3-5 iterations of "Discontinuity reduction" as proposed in Enhanced Sphere Tracing; Benjamin Keinert1 Henry Schafer1 Johann Korndorfer Urs Ganse2 Marc Stamminger1
		
		//outColor = vec4(0.1+float(steps)/float(STEPS));
		//outColor = shade(p, rd, d, 0.) * 0.5;
		gl_FragDepth = computeDepth(p);
		
		//vec4 c = shade(p, rd, d, 0.);
		
		//outColor.rgb *= c.rgb;
		
		//outColor = foga(outColor, WorldPosition);
		// tint & premultiply
		//outColor.rgb *= outColor.a;
		// gamma
		//outColor.rgb = pow(outColor.rgb, gamma3);
		
		
		//outColor = vec4(glow);
		
		vec3 n = normalize(quat_rotate(orient, normal4(p)));
		float diffuse = max(0.,dot(lightp, n))
			+ 0.2*max(0.,dot(lightp.yzx, n));
		float amb = 0.1;
		outColor = vec4(vec3(amb + diffuse), 1.);
		
	} else if (glow > 0.) {
		//outColor = vec4(0.2+0.*float(steps)/float(STEPS));
		//outColor += vec4(0.1);
		//gl_FragDepth = 0.9999;
		discard;
		//outColor = vec4(glow);
		//gl_FragDepth = computeDepth(p);
	} else {
		//outColor = vec4(0., 0., 1., 0.5);
		discard;
	}
	
	//outColor = vec4(ro,1.);
	//outColor = vec4(rd, 1.);
	//outColor = vec4(param,1.);
}

