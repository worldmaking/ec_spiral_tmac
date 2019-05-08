#include "lib.glsl"
#version 150

uniform mat4	ciModelView, ciProjectionMatrix, ciViewMatrix, ciModelMatrix;
uniform mat4	ciProjectionMatrixInverse, ciModelViewInverse, ciModelMatrixInverse, ciViewMatrixInverse;
uniform mat4	ciModelViewProjection;
uniform mat3	ciNormalMatrix;
uniform vec4 triggers;
uniform float hand;

vec3 lightPosition = vec3(10, 1, 1);

in vec4		ciPosition;
in vec4		ciColor;

out lowp vec4	Color;
out highp vec3  VertexPosition;
out highp vec3  WorldPosition;
out highp vec3 ray;
out highp vec3 lightp;
out highp vec3 param;
out highp mat4 mvp;

out lowp vec2 trackpadValue;
out lowp float trackpadButton;
out lowp float triggerValue;

vec4 leftColor = vec4(0.8, 0.1, 0., 1.);
vec4 rightColor = vec4(0.03, 0.1, 0.3, 1.);

void main(void) {

	vec4 vertex = ciPosition; 						// object space
	
	vec4 mvvertex = ciViewMatrix * ciModelMatrix * vertex; 	// camera space
	vec4 mvpvertex = ciProjectionMatrix * mvvertex;	// clip space
	gl_Position = mvpvertex;
	
	mvp = ciProjectionMatrix * ciViewMatrix * ciModelMatrix;
	
	// pick a point behind the the front frace
	// (in clip space, all rays are parallel to Z)
	vec4 back = gl_Position + vec4(0, 0, 1, 0 );
	// bring this back into object space:
	back = ciProjectionMatrixInverse * back;
	back = ciViewMatrixInverse * back;
	back = ciModelMatrixInverse * back;
	
	// do this in frag shader instead
	ray = (vertex.xyz/vertex.w - back.xyz/back.w);
		
	// TODO: this should actually be just the modelmatrix inverse,
	// since light position is view-independent
	lightp = (ciViewMatrixInverse * vec4(lightPosition, 1.)).xyz;
	
	// perform standard transform on vertex
	
	Color = hand > 0.5 ? rightColor : leftColor;
	VertexPosition = vertex.xyz;
	WorldPosition = mvvertex.xyz;
	param = vec3(0.5);
	
	trackpadValue = triggers.xy;
	trackpadButton = triggers.z;
	triggerValue = triggers.w;
}

