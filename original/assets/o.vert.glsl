#include "lib.glsl"
#version 150

uniform mat4	ciModelView, ciProjectionMatrix;
uniform mat4	ciModelViewProjection;
uniform mat3	ciNormalMatrix;

in vec4		ciPosition;
in vec2		ciTexCoord0;
in vec3		ciNormal;
in vec4		ciColor;
in vec3		vInstancePosition; // per-instance position variable
in vec4		vInstanceOrientation;
in vec4		vInstanceParams;

out lowp vec4	Color;
out highp vec3	Normal;
out highp vec3  WorldPosition;

void main( void )
{
	float scale = 0.3;
	
	float vary = vInstanceParams.x * 0.5;
	float flash = vInstanceParams.y * 2.;
	float nrg = vInstanceParams.z;
	float twisty = vInstanceParams.w;
	
	Color 		= vec4(0.2, 1., flash*2.+0.2, clamp(6.*nrg, 0., 1.));
	
	// organism:
	// BAKED
	float r = ciPosition.x;				// rib radius
	float cosfin = ciPosition.y;
	float sinfin = ciPosition.z;
	float i = ciNormal.x;
	float sweep = ciNormal.y;			// rib phase 0..1
	float change = ciNormal.z;			// whether this rib vertex can grow
	
	// DYNAMIC
	r += 2. * change * vary; // the growth factor
	float twist = 0.1;
	sweep += twist * sin((i * vary) * 2. + i * 0.01);	// the twist factor
	float cossweep = cos(sweep);
	float sinsweep = sin(sweep);
	
	vec3 V = r * vec3(cosfin, sinfin*sinsweep, sinfin*cossweep);
	vec3 N = vec3(sinfin, cosfin*cossweep, cosfin*sinsweep);
	
	V = vInstancePosition + quat_rotate(vInstanceOrientation, V * scale);
	vec4 V4 = ciModelView * vec4(V, 1.);
	
	WorldPosition = V4.xyz;
	gl_Position	= ciProjectionMatrix * V4;
	Normal		= ciNormalMatrix * quat_rotate(vInstanceOrientation, normalize(N));
}
