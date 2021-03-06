#include "lib.glsl"
#version 150

uniform mat4 ciProjectionMatrixInverse, ciViewMatrixInverse;

in vec4	ciPosition;
in vec2	ciTexCoord0;

//out highp vec3 origin, direction;
out highp vec2 T, P;

void main()
{
	T = ciTexCoord0;
	P = ciPosition.xy;
	gl_Position = ciPosition;
	
	/*T = ciTexCoord0;
	vec4 coord = vec4(T*2.-1., 0., 1.);
	gl_Position = coord;
	gl_Position.z = -gl_Position.z;

	vec4 v = ciProjectionMatrixInverse * coord;
	mat4 mvi = ciViewMatrixInverse;
	vec3 viewpos = vec3(mvi[3]);
	mvi[3][0] = 0.;
	mvi[3][1] = 0.;
	mvi[3][2] = 0.;
	vec4 v1 = mvi * v;
	
	// ray direction:
	// (to blend with GL scene, need to apply inverse projection matrix)
	vec3 rd = v.xyz;
	//vec3 rd = (unfrustum(focal_plane.x, focal_plane.y, focal_plane.z, focal_plane.w, near, far) * in_vertex).xyz;
	
	// rotate by current orientation:
	// (ok to do in vertex shader because of planar screen assumption)
	//direction = quat_rotate(orient, rd);
	
	direction = v1.xyz;
	
	// the ray origin at this vertex (converted to texture space):
	//origin = (pos + quat_rotate(orient, eyepos)) / dim;
	origin = viewpos/dim;
	 */
}