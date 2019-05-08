#include "lib.glsl"
#version 150

uniform sampler2D uTex0;
//uniform sampler3D uLandTex;
uniform float uNow;
uniform mat4 ciModelView;

in vec4	Color;
in vec3 ViewPosition;
in vec4 position;
in float energy;
in float noisiness;

out vec4 outColor;

vec3 ldir = normalize(vec3(-0.4, -0.1, 1));

float fog_density = 1.;
vec3 fog_color = vec3(0.);
float fog_offset = 14.;

float gamma = 2.2;
vec3 gamma3 = vec3(1.0 / gamma);

vec3 fog(vec3 color, vec3 pos) {
	// fog parameters
	float distance = max(length(pos)-fog_offset, 0.);
	float fogExponent = distance*fog_density; // simpler because density is 1.
	float fogFactor = exp2(-abs(fogExponent));
	return mix(fog_color, color, fogFactor);
}

vec4 foga(vec4 color, vec3 pos) {
	// fog parameters
	float distance0 = length(pos);
	float distance = max(distance0-fog_offset, 0.);
	float fogExponent = distance*fog_density;
	float fogFactor = exp2(-abs(fogExponent));
	float z = clamp(-8.*(pos.z+0.12), 0., 1.);	// 0.5 is the nearness
	return vec4(mix(fog_color, color.rgb, fogFactor), color.a * z);
}

float pi = 3.141592653589793;

float radius = 0.3;
vec2 scene(in vec3 p) {
	// radius depends on location:
	float r = radius * (0.75 + snoise(vec4(p + position.xyz, uNow) * noisiness) * 0.25);
	return vec2(length(p) - r, r);
}

#define NORMAL_EPS 0.02
// returns both normal (gradient)
// gets normal via tetrahedron rather than cube, 4 taps rather than 6
vec3 normal4(in vec3 p)
{
	vec2 e = vec2(-NORMAL_EPS, NORMAL_EPS);
	float n = 1. / (4.*NORMAL_EPS*NORMAL_EPS);	// precomputed normalization factor
												// tetrahedral points:
	float t1 = scene(p + e.yxx).s, t2 = scene(p + e.xxy).s;
	float t3 = scene(p + e.xyx).s, t4 = scene(p + e.yyy).s;
	return (e.yxx*t1 + e.xxy*t2 + e.xyx*t3 + e.yyy*t4) * n;
}

vec3 lighting(vec3 pos, vec3 nor, vec3 ro, vec3 rd) {
	vec3 dir1 = normalize(vec3(0, 1, 0));
	vec3 col1 = vec3(3.0, 0.7, 0.4);
	vec3 dif1 = col1 * orenNayarDiffuse(dir1, -rd, nor, 0.15, 1.0);
	vec3 spc1 = col1 * gaussianSpecular(dir1, -rd, nor, 0.15);
	
	//vec3 dir2 = normalize(vec3(0.4, -1, 0.4));
	//vec3 col2 = vec3(0.4, 0.8, 0.9);
	//vec3 dif2 = col2 * orenNayarDiffuse(dir2, -rd, nor, 0.15, 1.0);
	//vec3 spc2 = col2 * gaussianSpecular(dir2, -rd, nor, 0.15);
	
	return dif1 + spc1; // + dif2 + spc2;
}

void main( void )
{
	// scene is a unit cube, center is 0,0,0, bounds are +- 0.5, view-aligned
	// start on the front face (z == 0.5)
	vec3 ro = vec3(gl_PointCoord.xy - 0.5, 0.5);
	ro.y = -ro.y;
	// head forward (-z)
	vec3 rd = vec3(0, 0, -1);
	// make it view-dependent:
	//vec3 vp = (ciModelView * position).xyz;
	vec3 vp = ViewPosition.xyz;
	
	rd = normalize(vp);
	vec2 dr = scene(ro);
	vec3 p = ro + dr.s*rd;
	vec3 n = normal4(p);
	
	// mask on a disk:
	float mask = 1. - 4.*clamp(length(ro.xy) - 0.4, 0., 1.);
	
	// distance of surface from center of particle
	float l = length(p);
	// whether we are pushed in or out of the ideal sphere:
	float inside = l - dr.t;
	float a; // brightness
	if (inside < 0.05) {
		float diffuse = max(0., dot(ldir, n));
		a = diffuse * 0.8;
		// add ambient:
		a += 0.1 + 0.2*energy;
	} else {
		a = 0.02/inside;
	}
	
	outColor = vec4(a * mask);
	outColor = foga(outColor, ViewPosition);
	// tint & premultiply
	outColor.rgb *= Color.rgb * outColor.a;
	// gamma
	outColor.rgb = pow(outColor.rgb, gamma3);
}