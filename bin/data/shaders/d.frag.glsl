#include "lib.glsl"
#version 150

uniform samplerCube uTextureCube;
uniform mat4 ciModelView, ciProjectionMatrix;

float pi = 3.141592653589793;
float twopi = 6.28318530717959;

in vec2 T, P;
//in vec3 origin, direction;
out vec4 outColor;

void main() {
	
	// get ray direction from pixel coordinate:
	float x = sin(P.x * pi);
	float y = P.y;
	float z = cos(P.x * pi);
	vec3 TV = vec3(x, y, -z);
	
	//TV = normalize(TV);	// seems unneccesary
	vec3 rgb = texture(uTextureCube, TV).rgb;
	
	outColor = vec4(T, 0., 1.);
	outColor = vec4(TV, 1.);
	outColor = vec4(rgb, 1.);
	
}
