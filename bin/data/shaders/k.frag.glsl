#version 150 core

uniform sampler2D	uTextureColor;
uniform sampler2D	uTextureDepthToColorTable;
uniform sampler2D	uTextureBody;

in float			vDepth;	// apparently in mm
in vec2				vTexCoord0;

out vec4			gl_FragColor;

void main( void )
{
	if (vDepth <= 0.0 || vDepth >= 4096.0)  discard; // not a relevant point
	//if (texture(uTextureBody, vTexCoord0).r > 0.5) discard;	// not a detected body
	
	//vec2 uv			= texture( uTextureDepthToColorTable, vTexCoord0 ).rg;
	//gl_FragColor	= texture( uTextureColor, uv ) + 0.25;

	gl_FragColor = vec4(0.25);

}
 