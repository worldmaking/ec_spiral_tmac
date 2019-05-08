

mat4 unfrustum(float l, float r, float b, float t, float near, float far) {
	float W = r-l;
	float W2 = r+l;
	float H = t-b;
	float H2 = t+b;
	float D = far-near;
	float D2 = far+near;
	float n2 = near*2.;
	float fn2 = far*n2;
	
	return mat4(
				W/n2, 0., 0., 0.,
				0., H/n2, 0., 0.,
				0., 0., 0., -D/fn2,
				W2/n2, H2/n2, -1., D2/fn2
				);
}

vec4 quat_fromeuler(float az, float el, float ba) {
	float c1 = cos(az * 0.5);
	float c2 = cos(el * 0.5);
	float c3 = cos(ba * 0.5);
	float s1 = sin(az * 0.5);
	float s2 = sin(el * 0.5);
	float s3 = sin(ba * 0.5);
	// equiv Q1 = Qy * Qx; -- since many terms are zero
	float tw = c1*c2;
	float tx = c1*s2;
	float ty = s1*c2;
	float tz =-s1*s2;
	// equiv Q2 = Q1 * Qz; -- since many terms are zero
	return vec4(
				tx*c3 + ty*s3,
				ty*c3 - tx*s3,
				tw*s3 + tz*c3,
				tw*c3 - tz*s3
				);
}

// equiv. quat_rotate(quat_conj(q), v):
// q must be a normalized quaternion
vec3 quat_unrotate(in vec4 q, in vec3 v) {
	// return quat_mul(quat_mul(quat_conj(q), vec4(v, 0)), q).xyz;
	// reduced:
	vec4 p = vec4(
				  q.w*v.x - q.y*v.z + q.z*v.y,  // x
				  q.w*v.y - q.z*v.x + q.x*v.z,  // y
				  q.w*v.z - q.x*v.y + q.y*v.x,  // z
				  q.x*v.x + q.y*v.y + q.z*v.z   // w
				  );
	return vec3(
				p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y,  // x
				p.w*q.y + p.y*q.w + p.z*q.x - p.x*q.z,  // y
				p.w*q.z + p.z*q.w + p.x*q.y - p.y*q.x   // z
				);
}

//	q must be a normalized quaternion
vec3 quat_rotate(vec4 q, vec3 v) {
	vec4 p = vec4(
				  q.w*v.x + q.y*v.z - q.z*v.y,	// x
				  q.w*v.y + q.z*v.x - q.x*v.z,	// y
				  q.w*v.z + q.x*v.y - q.y*v.x,	// z
				  -q.x*v.x - q.y*v.y - q.z*v.z	// w
				  );
	return vec3(
				p.x*q.w - p.w*q.x + p.z*q.y - p.y*q.z,	// x
				p.y*q.w - p.w*q.y + p.x*q.z - p.z*q.x,	// y
				p.z*q.w - p.w*q.z + p.y*q.x - p.x*q.y	// z
				);
}

float gaussianSpecular(
					   vec3 lightDirection,
					   vec3 viewDirection,
					   vec3 surfaceNormal,
					   float shininess) {
	vec3 H = normalize(lightDirection + viewDirection);
	float theta = acos(dot(H, surfaceNormal));
	float w = theta / shininess;
	return exp(-w*w);
}

float orenNayarDiffuse(
					   vec3 lightDirection,
					   vec3 viewDirection,
					   vec3 surfaceNormal,
					   float roughness,
					   float albedo) {
	float PI = 3.141592653589793;
	
	float LdotV = dot(lightDirection, viewDirection);
	float NdotL = dot(lightDirection, surfaceNormal);
	float NdotV = dot(surfaceNormal, viewDirection);
	
	float s = LdotV - NdotL * NdotV;
	float t = mix(1.0, max(NdotL, NdotV), step(0.0, s));
	
	float sigma2 = roughness * roughness;
	float A = 1.0 + sigma2 * (albedo / (sigma2 + 0.13) + 0.5 / (sigma2 + 0.33));
	float B = 0.45 * sigma2 / (sigma2 + 0.09);
	
	return albedo * max(0.0, NdotL) * (A + B * s / t) / PI;
}

float beckmannDistribution(float x, float roughness) {
	float NdotH = max(x, 0.0001);
	float cos2Alpha = NdotH * NdotH;
	float tan2Alpha = (cos2Alpha - 1.0) / cos2Alpha;
	float roughness2 = roughness * roughness;
	float denom = 3.141592653589793 * roughness2 * cos2Alpha * cos2Alpha;
	return exp(tan2Alpha / roughness2) / denom;
}

float beckmannSpecular(
					   vec3 lightDirection,
					   vec3 viewDirection,
					   vec3 surfaceNormal,
					   float roughness) {
	return beckmannDistribution(dot(surfaceNormal, normalize(lightDirection + viewDirection)), roughness);
}

float cookTorranceSpecular(
						   vec3 lightDirection,
						   vec3 viewDirection,
						   vec3 surfaceNormal,
						   float roughness,
						   float fresnel) {
	
	float VdotN = max(dot(viewDirection, surfaceNormal), 0.0);
	float LdotN = max(dot(lightDirection, surfaceNormal), 0.0);
	
	//Half angle vector
	vec3 H = normalize(lightDirection + viewDirection);
	
	//Geometric term
	float NdotH = max(dot(surfaceNormal, H), 0.0);
	float VdotH = max(dot(viewDirection, H), 0.000001);
	float LdotH = max(dot(lightDirection, H), 0.000001);
	float G1 = (2.0 * NdotH * VdotN) / VdotH;
	float G2 = (2.0 * NdotH * LdotN) / LdotH;
	float G = min(1.0, min(G1, G2));
	
	//Distribution term
	float D = beckmannDistribution(NdotH, roughness);
	
	//Fresnel term
	float F = pow(1.0 - VdotN, fresnel);
	
	//Multiply terms and done
	return  G * F * D / max(3.14159265 * VdotN, 0.000001);
}

vec4 mod289(vec4 x) {
	return x - floor(x * (1.0 / 289.0)) * 289.0; }

float mod289(float x) {
	return x - floor(x * (1.0 / 289.0)) * 289.0; }

vec4 permute(vec4 x) {
	return mod289(((x*34.0)+1.0)*x);
}

float permute(float x) {
	return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
	return 1.79284291400159 - 0.85373472095314 * r;
}

float taylorInvSqrt(float r)
{
	return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 grad4(float j, vec4 ip)
{
	const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
	vec4 p,s;
	
	p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
	p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
	s = vec4(lessThan(p, vec4(0.0)));
	p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;
	
	return p;
}

// (sqrt(5) - 1)/4 = F4, used once below
#define F4 0.309016994374947451

float snoise(vec4 v)
{
	const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
						 0.276393202250021,  // 2 * G4
						 0.414589803375032,  // 3 * G4
						 -0.447213595499958); // -1 + 4 * G4
	
	// First corner
	vec4 i  = floor(v + dot(v, vec4(F4)) );
	vec4 x0 = v -   i + dot(i, C.xxxx);
	
	// Other corners
	
	// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
	vec4 i0;
	vec3 isX = step( x0.yzw, x0.xxx );
	vec3 isYZ = step( x0.zww, x0.yyz );
	//  i0.x = dot( isX, vec3( 1.0 ) );
	i0.x = isX.x + isX.y + isX.z;
	i0.yzw = 1.0 - isX;
	//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
	i0.y += isYZ.x + isYZ.y;
	i0.zw += 1.0 - isYZ.xy;
	i0.z += isYZ.z;
	i0.w += 1.0 - isYZ.z;
	
	// i0 now contains the unique values 0,1,2,3 in each channel
	vec4 i3 = clamp( i0, 0.0, 1.0 );
	vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
	vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );
	
	//  x0 = x0 - 0.0 + 0.0 * C.xxxx
	//  x1 = x0 - i1  + 1.0 * C.xxxx
	//  x2 = x0 - i2  + 2.0 * C.xxxx
	//  x3 = x0 - i3  + 3.0 * C.xxxx
	//  x4 = x0 - 1.0 + 4.0 * C.xxxx
	vec4 x1 = x0 - i1 + C.xxxx;
	vec4 x2 = x0 - i2 + C.yyyy;
	vec4 x3 = x0 - i3 + C.zzzz;
	vec4 x4 = x0 + C.wwww;
	
	// Permutations
	i = mod289(i);
	float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
	vec4 j1 = permute( permute( permute( permute (
												  i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
										+ i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
							   + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
					  + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));
	
	// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
	// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
	vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;
	
	vec4 p0 = grad4(j0,   ip);
	vec4 p1 = grad4(j1.x, ip);
	vec4 p2 = grad4(j1.y, ip);
	vec4 p3 = grad4(j1.z, ip);
	vec4 p4 = grad4(j1.w, ip);
	
	// Normalise gradients
	vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;
	p4 *= taylorInvSqrt(dot(p4,p4));
	
	// Mix contributions from the five corners
	vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
	vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
	m0 = m0 * m0;
	m1 = m1 * m1;
	return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
				   + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;
	
}
