#include "lib.glsl"
#version 150

uniform mat4	ciModelView, ciProjectionMatrix, ciViewMatrix, ciModelMatrix;
uniform mat4	ciProjectionMatrixInverse, ciModelViewInverse, ciModelMatrixInverse, ciViewMatrixInverse;
uniform mat4	ciModelViewProjection;
uniform mat3	ciNormalMatrix;
//uniform vec3 lightPosition;
vec3 lightPosition = vec3(-1, 10, -10);

in vec4		ciPosition;
in vec4		ciColor;
in mat4		vInstanceMatrix;
in mat4		vInstanceMatrixInverse;
in vec3		vInstanceParams;

out lowp vec4	Color;
out highp vec3  VertexPosition;
out highp vec3  WorldPosition;
out highp vec3 ray;
out highp vec3 lightp;
out highp vec3 param;
out highp mat4 mvp;

float scale = 0.4; // 0.8

void main(void) {

	vec4 vertex = ciPosition; 						// object space
	vertex.xyz *= scale;
	
	vec4 mvertex = vInstanceMatrix * vertex;
	vec4 mvvertex = ciViewMatrix * mvertex; 	// camera space
	vec4 mvpvertex = ciProjectionMatrix * mvvertex;	// clip space
	gl_Position = mvpvertex;
	
	mvp = ciProjectionMatrix * ciViewMatrix * vInstanceMatrix;
	
	// pick a point behind the the front frace
	// (in clip space, all rays are parallel to Z)
	vec4 back = gl_Position + vec4(0, 0, 1, 0 );
	// bring this back into object space:
	back = ciProjectionMatrixInverse * back;
	back = ciViewMatrixInverse * back;
	back = vInstanceMatrixInverse * back;
	
	// do this in frag shader instead
	ray = (vertex.xyz/vertex.w - back.xyz/back.w);
		
	// TODO: this should actually be just the modelmatrix inverse,
	// since light position is view-independent
	lightp = (ciViewMatrixInverse * vec4(lightPosition, 1.)).xyz;
	
	// perform standard transform on vertex
	
	Color = vec4(0.3, 0.6, 0.5, 1.);
	VertexPosition = vertex.xyz / scale;
	WorldPosition = mvvertex.xyz;
	param = vec3(0.5);
}
