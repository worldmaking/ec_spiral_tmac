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
in mat4		vInstanceMatrix1;
in mat4		vInstanceMatrixInverse1;
in vec3		vInstanceParams;
in vec4		vInstanceOrient;

out lowp vec4	Color;
out highp vec3  ray_origin;
out highp vec3  WorldPosition;
out highp vec3 ray;
out highp vec3 lightp;
out highp vec3 param;
out highp mat4 mvp;
out highp vec4 orient;

/*
 Can this be done simply via a screen-aligned quad (or circle)?
 ignore the ciPosition data initially, just transform vec3(0) by instance matrix (rot doesn't matter), view matrix, then extend by vertex .xy before applying projection
 -> That gets us a screen-aligned quad.
 
 Now how to reverse engineer front & back faces?
 */

void main(void) {
	mvp = ciProjectionMatrix * ciViewMatrix * vInstanceMatrix;
	
	vec4 vertex = ciPosition;
	// this doesn't really matter, it's just the screen-space bounding box
	// maybe there's a way to align it to the creature better?
	vec2 uv = vec2(vertex.x, vertex.y)*3.;
	
	vec4 viewpos4 = ciViewMatrixInverse[3];
	vec3 viewpos = viewpos4.xyz/viewpos4.w; // maybe not necessary?
	
	
	/*
	// arrange onto screen:
	vec4 wvertex = vertex - vec4(0., 0., 0.5, 0.);
	vec4 mvertex = vInstanceMatrix * wvertex;
	// camera space
	vec4 mvvertex = ciViewMatrix * mvertex;
	// clip space
	vec4 mvpvertex = ciProjectionMatrix * mvvertex;
	gl_Position = mvpvertex;
	 
	 // pick a point behind the the front frace
	 // (in clip space, all rays are parallel to Z)
	 vec4 back = gl_Position + vec4(0, 0, 1, 0 );
	 // bring this back into object space:
	 back = ciProjectionMatrixInverse * back;
	 back = ciViewMatrixInverse1 * back;
	 back = vInstanceMatrixInverse * back;
	 */
	
	
	/* 
	 take object-space center (origin) and transform into view space
	 then displace in XY plane according to ciPosition, simply to create a screen-aligned quad
	 then project to generate gl_Position
	 */
	vec4 wvertex = vec4(0., 0., 0., 1.);
	vec4 mvertex = vInstanceMatrix1 * wvertex;
	// camera space
	vec4 mvvertex = ciViewMatrix * mvertex;
	mvvertex.xy += uv;
	//vec4 mvvertex = mvertex;
	// clip space
	vec4 mvpvertex = ciProjectionMatrix * mvvertex;
	gl_Position = mvpvertex;
	
	/*
	 we can now extrude this point forward & back along the world-space ray
	 simply by shifting in clip-space Z, then unprojecting
	 
	 
	 
	 should be possible to create a ray origin and direction
	 by extruding the quad along Z in clip space
	 
	 this defines a front and back location in object space,
	 */
	
	/*
	 we can now extrude this point forward & back along the world-space ray
	 simply by shifting in clip-space Z, then undoing the projection & view
	 
	 in effect, we are creating a screen-aligned frustum out of the quad
	 and using its front & rear faces for the ray entry & exit points
	 */
	vec4 front = gl_Position;
	vec4 back = gl_Position + vec4(0, 0, 1., 0 );
	front = ciProjectionMatrixInverse * front;
	back = ciProjectionMatrixInverse * back;
	// front & back are now in view space
	// idea: since we know the view matrix has no scale, and we already screen-aligned our vertex
	// the only thing to remove here is the translation
	
	
	front = ciViewMatrixInverse * front;
	back = ciViewMatrixInverse * back;
	// front & back are now in world space
	// we can bring this into object space too:
	front = vInstanceMatrixInverse * front;
	back = vInstanceMatrixInverse * back;
	// front & back are now in object space, relative to 0,0,0
	// do this in frag shader instead?
	ray = (front.xyz/front.w - back.xyz/back.w);
	// origin is ostensibly front.xyz, but the distance can be wonky
	// instead try taking the billboard plane and subtracting the ray?
	ray_origin = front.xyz/front.w;
	
	//ray = quat_unrotate(vInstanceOrient, ray);
	
	// TODO: this should actually be just the modelmatrix inverse,
	// since light position is view-independent
	lightp = normalize(lightPosition);
	
	// perform standard transform on vertex
	Color = vec4(0.3, 0.6, 0.5, 1.);
	WorldPosition = mvvertex.xyz; // used for depth
	param = vec3(0.5);
	orient = vInstanceOrient;
}
