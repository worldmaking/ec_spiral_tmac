#include "lib.glsl"
#version 150

uniform mat4	ciModelView, ciProjectionMatrix;
uniform mat4	ciModelViewProjection;
uniform mat3	ciNormalMatrix;

in vec4		ciPosition;
in vec2		ciTexCoord0;
in vec3		ciNormal;
in vec4		ciColor;

// per-instance variables
in vec3		vInstancePosition;
in vec4		vInstanceOrientation, vInstanceOrientation2;
in vec2		vInstanceSize, vInstanceSize2;
in vec3		vInstanceColor, vInstanceColor2;
in float 	vInstanceType;

out lowp vec4	Color;
out highp vec3	Normal;
out highp vec3  ViewPosition;

void main( void )
{
	
	vec3 V = ciPosition.xyz;
	
	
	float a = V.z;
	 
	// apply thickness:
	V.xy = V.xy * mix(vInstanceSize.x, vInstanceSize2.x, a);
	// rescale to length:
	V.z = vInstanceSize.y * a;
	
	
	V = vInstancePosition + quat_rotate(vInstanceOrientation, V);
	
	vec3 n = normalize(ciNormal);
	vec3 n1 = quat_rotate(vInstanceOrientation, n);
	vec3 n2 = quat_rotate(vInstanceOrientation2, n);
	Normal = ciNormalMatrix * mix(n1, n2, a);
	
	vec3 startcolor = vInstanceColor;
	vec3 endcolor = vInstanceColor2;
	if (vInstanceType < 0.5) {
		endcolor = vec3(0, 1, 0);
	}
	if (vInstanceType > 1.5) {
		//startcolor = vec3(1, 0, 0);
	}
	
	Color.rgb = mix(startcolor, endcolor, a);
	Color.a = 0.5;
	
	float stretch = abs(1. - vInstanceSize.y);
	Color.rgb += vec3((stretch), 0., 0.);
	
	// world to eye space transform
	vec4 vertex = ciModelView * vec4(V, 1);
	ViewPosition = vertex.xyz;
	
	gl_Position = ciProjectionMatrix * vertex;
}
