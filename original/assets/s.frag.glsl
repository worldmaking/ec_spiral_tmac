#include "lib.glsl"
#version 150

uniform vec3 lightPosition;

in vec4	Color;
in vec3	Normal;
in vec3 ViewPosition;

out vec4 outColor;

/*
 previously:
 vec4 La= vec4( 0.26, 0.4, 0.1, 1); --// -- light ambient
 vec4 Li= vec4( 0.35, 0.4, 0.4, 1); --// -- light incident (diffuse)
 vec4 Ks= vec4( 0.8, 0.8, 0.3, 1);
 vec4 Ka= vec4( 0.1, 0., 0.2, 1);
 vec4 Kd= vec4( 0.5, 1, 1, 1);
 */

vec4 La = vec4( 0.8, 0.8, 0.8, 1); // -- light ambient
vec4 Li = vec4( 0.85, 0.8, 0.4, 1); // -- light incident (diffuse)
vec4 Ks = vec4( 0.9, 0.7, 0.7, 1);
vec4 Ka = vec4( 0.4, 0.4, 0.5, 1);
vec4 Kd = vec4( 0.63, 0.3, 0.47, 1);
float Ns = 10.;						// shininess


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

void main( void )
{
	vec3 Nn = normalize(Normal);
	
	//ambient contribution
	vec4 ambient = La*Ka;
	
	//diffuse contribution
	vec3 L = normalize(lightPosition - ViewPosition);
	vec4 diffuse = Kd*Li*abs(dot(Nn, L));
	
	//calculate specular contribution
	vec3 V = normalize(-ViewPosition);
	//average of lighting and view vector)  not true reflection vector
	vec3 H = normalize(L + V);
	vec4 specular = Ks*Li * pow(abs(dot(Nn,H)), Ns);
	
	// fog calculate
	outColor = (ambient * Color + diffuse + specular);
	//outColor = Color;
	//gl_FragColor = 0.2 + gl_Color + 0.01 * gl_FragColor;	// for debug
	outColor = foga(outColor, ViewPosition);
	outColor.a *= 0.3;
	//gl_FragColor.rgb = vec3(0.5);
	
	//outColor.rgb *= outColor.a; // pre-multiply
	
	// gamma correct
	//outColor.rgb = pow(outColor.rgb, gamma3);
	
	//outColor = vec4(1.);
	
}