#include "lib.glsl"
#version 150

uniform vec3 lightPosition;

in vec4	Color;
in vec3	Normal;
in vec3 WorldPosition;

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

vec4 foga(vec4 color, vec3 pos) {
	// fog parameters
	float distance0 = length(pos);
	float distance = max(distance0-fog_offset, 0.);
	float fogExponent = distance*fog_density;
	float fogFactor = exp2(-abs(fogExponent));
	float z = clamp(-8.*(pos.z+0.12), 0., 1.);	// 0.5 is the nearness
	return vec4(mix(fog_color, color.rgb, fogFactor), color.a * z);
}

float gamma = 2.2;
vec3 gamma3 = vec3(1.0 / gamma);

void main( void )
{
	vec3 normal = normalize( -Normal );

	//	float diffuse = max( dot( normal, vec3( 0, 0, -1 ) ), 0 );
//	//oColor = texture( uTex0, TexCoord.st ) * Color * diffuse;
//	
//	oColor = Color * diffuse;
	
	
	//ambient contribution
	vec4 ambient = La*Ka;
	
	//diffuse contribution
	vec3 L = normalize(lightPosition - WorldPosition);
	vec4 diffuse = Kd*Li*abs(dot(normal, L));
	
	//calculate specular contribution
	vec3 V = normalize(-WorldPosition);
	//average of lighting and view vector)  not true reflection vector
	vec3 H = normalize(L + V);
	vec4 specular = Ks*Li * pow(abs(dot(normal,H)), Ns);
	
	outColor = (ambient * Color + diffuse + specular);
	outColor.a = Color.a;
	
	outColor.rgba = foga(outColor.grba, WorldPosition);
	
	outColor.rgb *= outColor.a; // pre-multiply by alpha
	
	
	// gamma correct
	//outColor.rgb = pow(outColor.rgb, gamma3);
	
}