#include "lib.glsl"
#version 150 core

uniform mat4		ciModelViewProjection, ciViewMatrix, ciProjectionMatrix;
uniform mat4		uTrackingMatrix;
uniform usampler2D	uTextureDepth;
uniform sampler2D	uTextureDepthToCameraTable;
uniform vec4 uOrient;

in vec2				ciPosition;

out float			vDepth;
out vec2			vTexCoord0;

void main( void )
{
	vTexCoord0	= ciPosition;

	vec2 uv		= vTexCoord0;
	uv.t		= 1.0 - uv.t; // why? some Cinder bug?

	vDepth		= texture( uTextureDepth, uv ).r; // in mm, apparently
	vec3 pos	= vec3( texture( uTextureDepthToCameraTable, vTexCoord0 ).xy * vDepth, vDepth );
	// to meters:
	vec4 p = vec4( pos * 0.001, 1.0 );
	//p.y = -p.y;
	p.xyz = quat_rotate(uOrient, p.xyz);
	
	p.z = p.z-1.;  // because oculus tracker has origin 1m in front
	

	
	//gl_Position = ciModelViewProjection * p;
	//gl_Position = ciProjectionMatrix * p;
	gl_Position = ciProjectionMatrix * ciViewMatrix * p;

};
 