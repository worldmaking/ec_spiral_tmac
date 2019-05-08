#include "lib.glsl"
#version 150

uniform mat4	ciModelView;
uniform mat4	ciModelViewProjection;
uniform mat4 ciProjectionMatrixInverse, ciViewMatrixInverse;
uniform float pointSize;

in vec4		ciPosition;
in vec4		ciColor;

out lowp vec4	Color;
out highp vec3  ViewPosition;
out highp vec4  position;
out lowp float energy;
out lowp float noisiness;

void main( void )
{
	// vertex in object space
	position = ciPosition;
	// vertex in camera space:
	ViewPosition = (ciModelView * ciPosition).xyz;
	// vertex in screen space:
	gl_Position	= ciModelViewProjection * ciPosition;
	
	energy = ciColor.x;
	float speed = ciColor.y;
	float state = ciColor.z;
	
	noisiness = 0.;
	
	if (state <= 0.) {
		// it is food:
		vec3 no_energy = vec3(0.2, 0.2, 0.);
		vec3 hi_energy = vec3(0.8, 0.1, 0.);
		Color.rgb = mix(no_energy, hi_energy, energy);
		
		noisiness = (1.-energy) * 5.;
	} else if (state <= 1.) {
		// being digested... 
		Color.r = 0.25;
		Color.g = 0.25;
		Color.b = (1.-energy);
		
		Color.rgb = vec3(0.5);
		
		noisiness = (1.-energy) * 5.;
	} else {
		// excreted... tend toward blue
		//energy = energy > 0.5 ? 1. : 0.;
		
		vec3 no_energy = vec3(0.2);
		vec3 hi_energy = vec3(0.03, 0.1, 0.3);
		Color.rgb = mix(no_energy, hi_energy, energy);
		
		
		noisiness = (1.-energy) * 5.;
	}
	Color.a = ciColor.a;
	
	//Color.rgb = vec3(0.5);
	//Color.r = energy* 50.;
	
	gl_PointSize = pointSize / gl_Position.w; // // * vertex.w / gl_Position.z;
}
