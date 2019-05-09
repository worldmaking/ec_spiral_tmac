#ifndef AN_an_math_h
#define AN_an_math_h

#include "al_glm.h"

#include <math.h>
#include <float.h>

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#ifndef __INT32_TYPE__
#define __INT32_TYPE__ int
#endif
//#ifdef __GNUC__
#include <stdint.h>
//#else
//#include "msc_stdint.h"
//#endif
#define inf (__DBL_MAX__)
#define AN_UINT_MAX                (4294967295)
#define TWO_TO_32             (4294967296.0)

// system constants
#define AN_DBL_EPSILON (DBL_EPSILON)

#define AN_PI (3.14159265358979323846264338327950288)
#define AN_PI_OVER_2 (1.57079632679489661923132169163975144)
#define AN_PI_OVER_4 (0.785398163397448309615660845819875721)
#define AN_1_OVER_LOG_2 (1.442695040888963)
// assumes v is a 64-bit double:
#define AN_IS_NAN_DOUBLE(v)			(((((uint32_t *)&(v))[1])&0x7fe00000)==0x7fe00000)
#define AN_FIX_NAN_DOUBLE(v)		(AN_IS_NAN_DOUBLE(v)?0.:(v))
#define AN_IS_DENORM_DOUBLE(v)	((((((uint32_t *)&(v))[1])&0x7fe00000)==0)&&((v)!=0.))
#define AN_FIX_DENORM_DOUBLE(v)	((v)=AN_IS_DENORM_DOUBLE(v)?0.f:(v))
#define AN_QUANT(f1,f2)			(floor((f1)*(f2)+0.5)/(f2))

#define DOUBLE_FLOOR(v) ( (long)(v) - ((v)<0.0 && (v)!=(long)(v)) )
#define DOUBLE_CEIL(v) ( (((v)>0.0)&&((v)!=(long)(v))) ? 1+(v) : (v) )

#define FLOAT_FLOOR(v) ( (int)(v) - ((v)<0.f && (v)!=(int)(v)) )
#define FLOAT_CEIL(v) ( (((v)>0.f)&&((v)!=(int)(v))) ? 1+(v) : (v) )

// this won't work for negative numbers:
#define DOUBLE_FRAC(v) ( ((v)>=0.0) ? (v)-(long)(v) : (-v)-(long)(v) )

// euclidean modulo:
inline int euc(int i, int n) {
	int r = i % n;
	return r + n & (-(r < 0));
}

// Floating-point modulo
// The result (the remainder) has same sign as the divisor.
// Similar to matlab's mod(); Not similar to fmod() -   Mod(-3,4)= 1   fmod(-3,4)= -3
// y must be > 0
template<typename T, typename T1>
T euc_fmod(T x, T1 y) {
	//static_assert(!std::numeric_limits<T>::is_exact , "Mod: floating-point type expected");
	//if (0 == y) return x;
	double m = x - y * floor(x/y);
	// handle boundary cases resulted from floating-point cut off:
	//if (y > 0)   {           // modulo range: [0..y)
	if (m >= y) return T(0);   // Mod(-1e-16             , 360.    ): m= 360.
	if (m < 0.) {
		if (y+m == y) return T(0); // just in case...
		else return T(y+m); 	// Mod(106.81415022205296 , _TWO_PI ): m= -1.421e-14
	}
	return T(m);
}

// here's an idea
// separate integer & fractional components
// bitwrap the integer part
// add the fractional part
// special handling for negative inputs
inline float euc_fmod_bitwrap(float x, float dim, int wrap) {
	long ix = (long)(x < 0.f ? x - 1.f : x);
	float fx = (x - (float)ix);
	float wx = float(ix & wrap);
	float result = fx + wx;
	if (result >= dim || result < 0.f) result = 0.f;
	return result;
}

// ry is 1.f/y
// faster, because the division is lifted.
inline float euc_fmodf1(float x, float y, float ry) {
	//static_assert(!std::numeric_limits<T>::is_exact , "Mod: floating-point type expected");
	//if (0 == y) return x;
	float div = x * ry;
	float m = x - y * FLOAT_FLOOR(div);	// yes this is faster than floorf.
	// handle boundary cases resulted from floating-point cut off:
	//if (y > 0)   {           // modulo range: [0..y)
	if (m >= y) return 0.f;   // Mod(-1e-16             , 360.    ): m= 360.
	if (m < 0.) {
		if (y + m == y) return 0.f; // just in case...
		else return (y + m); 	// Mod(106.81415022205296 , _TWO_PI ): m= -1.421e-14
	}
	return (m);
}



// returns the shortest distance from x to y on a toroidal line of length m:
template<typename T>
T wrapdist(T x, T y, T m) {
	T half = m/T(2.0);
	return fabs(euc_fmod((x - y) + half, m) - half);
}

// returns shortest signed vector from x to y in a toroidal space of dim m:
template<typename T>
T wrapdiff(T x, T y, T m) {
	T half = m/T(2.0);
	return (euc_fmod((x - y) + half, m) - half);
}

/*
	Quick tut on glm:
	@see http://glm.g-truc.net/0.9.4/api/a00141.html
	@see http://glm.g-truc.net/0.9.5/api/modules.html
	
	# Vectors
	
	v3 = vec3(0.);
	v4 = vec4(v3, 1.);
	
	v.x; // access v.x, v.y, v.z etc.
	
	i = v.length();					// no. elements in type
	s = glm::length(v);				// length of vector
	s = glm::length2(v);				// length squared
	
	v = glm::normalize(v);		// will create NaNs if vector is zero length
	v = glm::cross(v1, v2);
	s = glm::dot(v1, v2);
	s = glm::distance(v1, v2);
	v = glm::faceforward(vn, vi, vref);	//If dot(Nref, I) < 0.0, return N, otherwise, return -N.
	v = glm::reflect(vi, vn);	// reflect vi around vn
	v = glm::refract(vi, vn, eta);
	
	v = glm::cos(v);			// sin, tan, acos, atanh, etc.
	v = glm::atan(v1, v2);		// aka atan2
	v = glm::degrees(v);		// radians()
	
	v = glm::abs(v);			// ceil, floor, fract, trunc
	v = glm::pow(v1, v2);		// exp, exp2, log, log2, sqrt, inversesqrt
	v = glm::mod(v, v_or_s);	// x - y * floor(x / y)
	v = glm::modf(v, &iv);		// returns fract, stores integer part in iv
	v = glm::round(v);			// direction of 0.5 implementation defined
	v = glm::roundEven(v);		// 0.5 rounds to nearest even integer
	v = glm::sign(v);
	v = glm::clamp(v, vmin, vmax);	// min, max
	v = glm::fma(va, vb, vc);	// return a*b+c
	v = glm::mix(v1, v2, a);
	
	// Returns 0.0 if x <= edge0 and 1.0 if x >= edge1 and performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
	v = glm::smoothstep(v0, v1, vx);
	v = glm::step(e, v);		// v<e ? 0 : 1
	
	
	v<bool> = glm::isnan(v);	// isinf
	v<bool> = glm::equal(v1, v2);	// notEqual, lessThanEqual, etc.
	bool = glm::any(v<bool);	// also glm::all()
	
	# Matrices:
	
	m = mat4(1.); // or mat4();	// identity matrix
	m[3] = vec4(1, 1, 0, 1);	// set 4th column (translation)
	
	v = m * v;					// vertex transformation
	
	// (matrix types store their values in column-major order)
	glm::value_ptr():
	
	// e.g.
	glVertex3fv(glm::value_ptr(v));		// or glVertex3fv(&v[0]);
	glLoadMatrixfv(glm::value_ptr(m));	// or glLoadMatrixfv(&m[0][0]);
 
	m = glm::make_mat4(ptr);	// also make_mat3, make_mat3x2, etc.
	
	v = glm::column(m, idx);
	v = glm::row(m, idx);
	m = glm::transpose(m);
	m = glm::inverse(m);
	s = glm::determinant(m);
	m = glm::matrixCompMult(m1, m2);	// component-wise multiplication
	m = glm::outerProduct(vc, vr);		// generate mat by vc * vr
	
	// Map the specified vertex into window coordinates.
	v = glm::project(vec3 v, mat4 model, mat4 proj, vec4 viewport);
	v = glm::unProject(vec3 win, mat4 model, mat4 proj, vec4 viewport);
	
	m = glm::frustum(l, r, b, t, near, far);
	m = glm::ortho(l, r, b, t);					// for 2D
	m = glm::ortho(l, r, b, t, near, far);
	m = glm::infinitePerspective(fovy, aspect, near, far);
	m = glm::perspective(fovy, aspect, near, far);
	
	m = glm::lookat(eye, at, up);
	
	// define a picking region
	m = glm::pickMatrix(vec2_center, vec2_delta, vec4_viewport);
	
	m = glm::rotate(m, angle, axis);
	m = glm::scale(m, v);
	m = glm::translate(m, v);
	
	m = glm::affineInverse(m);	// Fast inverse for affine matrix.
	m = glm::inverseTranspose(m);
	
	# Quaternions
	// uses wxyz order:
	q = quat(w, x, y, z);
	
	q = q * rot;				// rot is in model space (local)
	q = rot * q;				// rot is in world space (global)
	Remember to normalize quaternions periodically!
	
	s = glm::length(q);
	s = glm::pitch(q);			// also roll(q), yaw(q)
	
	q = glm::normalize(q);
	q = glm::conjugate(q);
	q = glm::inverse(q);
	q = glm::dot(q1, q2);
	v = glm::cross(q, v);
	
	q = glm::lerp(q1, q2, s);
	q = glm::mix(q1, q2, s);
	q = glm::slerp(q1, q2, s);
	
	// also greaterThan, greaterThanEqual, lessThan, notEqual, etc.
	vec4_bool = glm::equal(q1, q2);
	
	## conversions:
	
	q = glm::angleAxis(angle, x, y, z);
	q = glm::angleAxis(angle, axis);
	
	a = glm::angle(q);
	axis = glm::axis(q);
	
	m = glm::mat3_cast(q);
	m = glm::mat4_cast(q);
	q = glm::quat_cast(m);		// from mat3 or mat4
	
	pitch_yaw_roll = glm::eulerAngles(q);
	
	# Random / Noise
	
	s = glm::noise1(v);
	vec2 = glm::noise2(v);		// etc. noise3, noise4
	
	s = glm::perlin(v);			// classic perlin noise
	s = glm::perlin(v, v_rep);	// periodic perlin noise
	s = glm::simplex(v);		// simplex noise
	
	## generate vec<n> or scalar:
	gaussRand(mean, deviation);
	linearRand(min, max);
	
	## generate vec3:
	ballRand(radius);			// coordinates are regulary distributed within the volume of a ball
	circularRand(radius);		// coordinates are regulary distributed on a circle
	diskRand(radius);			// coordinates are regulary distributed within the area of a disk
	sphericalRand(radius);		// coordinates are regulary distributed on a sphere
 */

template<typename T, typename T1, glm::precision P>
inline T sample(T * data, glm::tvec2<T1, P> const & coord, int stridey) {
	// warning: no bounds checking!
	T c00 = data[(int)(coord.x) + (int)(coord.y)*stridey];
	T c01 = data[(int)(coord.x) + (int)(coord.y+1.f)*stridey];
	T c10 = data[(int)(coord.x+1.f) + (int)(coord.y)*stridey];
	T c11 = data[(int)(coord.x+1.f) + (int)(coord.y+1.f)*stridey];
	float bx = coord.x - (int)coord.x;
	float by = coord.y - (int)coord.y;
	float ax = 1.f - bx;
	float ay = 1.f - by;
	return c00 * ax * ay
	+ c01 * ax * by
	+ c10 * bx * ay
	+ c11 * bx * by;
}


template<typename T>
T computeNormal(T const & a, T const & b, T const & c) {
	return glm::normalize(glm::cross(c - a, b - a));
}


template<typename T, glm::precision P>
inline glm::tvec3<T, P> vec_fixnan(glm::tvec3<T, P> const & v) {
	return glm::tvec3<T, P>(
		isnan(v.x) ? T(0) : v.x,
		isnan(v.y) ? T(0) : v.y,
		isnan(v.z) ? T(0) : v.z
		);
}

/*
	wrap a point in the region [base, base+dim)
 */
template<typename T, glm::precision P>
inline glm::tvec3<T, P> vec_relativewrap(glm::tvec3<T, P> const & v, glm::tvec3<T, P> const & base, const T dim) {
	return glm::tvec3<T, P>(
							euc_fmod(v.x - base.x, dim) + base.x,
							euc_fmod(v.y - base.y, dim) + base.y,
							euc_fmod(v.z - base.z, dim) + base.z
							);
}

/*
	if the length is zero, then return a random unit vector
 */
template<typename T, glm::precision P>
inline glm::tvec3<T, P> vec_safenormalize(glm::tvec3<T, P> const & v) {
	static double min2 = std::numeric_limits< T >::min() * std::numeric_limits< T >::min();
	T len2 = glm::length2(v);
	if (len2 > min2) {
		return v / T(sqrt(len2));
	} else {
		return glm::sphericalRand(T(1));
	}
}

/*
	vec3 ux = quat_ux(q);
	
	Basically:
 mat3 m = glm::toMat3(q);
 ux = m[0];
 uy = m[1];
 uz = m[2];
 */
/*
template<typename T, glm::precision P>
inline glm::tvec3<T, P> quat_ux(glm::tquat<T, P> const & q) {
	return glm::tvec3<T, P>(
							T(1) - T(2) * ((q.y * q.y) + (q.z * q.z)),
							T(2) * ((q.x * q.y) + (q.w * q.z)),
							T(2) * ((q.x * q.z) - (q.w * q.y))
							);
}

template<typename T, glm::precision P>
inline glm::tvec3<T, P> quat_uy(glm::tquat<T, P> const & q) {
	return glm::tvec3<T, P>(
							T(2) * ((q.x * q.y) - (q.w * q.z)),
							T(1) - 2 * ((q.x * q.x) + (q.z * q.z)),
							T(2) * ((q.y * q.z) + (q.w * q.x))
							);
}

template<typename T, glm::precision P>
inline glm::tvec3<T, P> quat_uz(glm::tquat<T, P> const & q) {
	return glm::tvec3<T, P>(
							T(2) * ((q.x * q.z) + (q.w * q.y)),
							T(2) * ((q.y * q.z) - (q.w * q.x)),
							T(1) - T(2) * ((q.x * q.x) + (q.y * q.y))
							);
}

//	q must be a normalized quaternion
template<typename T, glm::precision P>
glm::tvec3<T, P> quat_unrotate(glm::tquat<T, P> const & q, glm::tvec3<T, P> const & v) {
	// return quat_mul(quat_mul(quat_conj(q), vec4(v, 0)), q).xyz;
	// reduced:
	glm::tvec4<T, P> p(
				  q.w*v.x - q.y*v.z + q.z*v.y,  // x
				  q.w*v.y - q.z*v.x + q.x*v.z,  // y
				  q.w*v.z - q.x*v.y + q.y*v.x,  // z
				  q.x*v.x + q.y*v.y + q.z*v.z   // w
				  );
	return glm::tvec3<T, P>(
				p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y,  // x
				p.w*q.y + p.y*q.w + p.z*q.x - p.x*q.z,  // y
				p.w*q.z + p.z*q.w + p.x*q.y - p.y*q.x   // z
				);
}

//	q must be a normalized quaternion
template<typename T, glm::precision P>
glm::tvec3<T, P> quat_rotate(glm::tquat<T, P> const & q, glm::tvec3<T, P> const & v) {
	glm::tvec4<T, P> p(
				  q.w*v.x + q.y*v.z - q.z*v.y,	// x
				  q.w*v.y + q.z*v.x - q.x*v.z,	// y
				  q.w*v.z + q.x*v.y - q.y*v.x,	// z
				  -q.x*v.x - q.y*v.y - q.z*v.z	// w
				  );
	return glm::tvec3<T, P>(
				p.x*q.w - p.w*q.x + p.z*q.y - p.y*q.z,	// x
				p.y*q.w - p.w*q.y + p.x*q.z - p.z*q.x,	// y
				p.z*q.w - p.w*q.z + p.y*q.x - p.x*q.y	// z
				);
}
*/
///////////////////////////////////////////////////////////////////////////////////////

inline double an_isnan(double v) { return AN_IS_NAN_DOUBLE(v); }
inline double fixnan(double v) { return AN_FIX_NAN_DOUBLE(v); }
inline double fixdenorm(double v) { return AN_FIX_DENORM_DOUBLE(v); }
inline double isdenorm(double v) { return AN_IS_DENORM_DOUBLE(v); }

inline double safemod(double f, double m) {
	if (m > AN_DBL_EPSILON || m < -AN_DBL_EPSILON) {
		if (m<0)
			m = -m; // modulus needs to be absolute value
		if (f >= m) {
			if (f >= (m*2.)) {
				double d = f / m;
				d = d - (long)d;
				f = d * m;
			}
			else {
				f -= m;
			}
		}
		else if (f <= (-m)) {
			if (f <= (-m*2.)) {
				double d = f / m;
				d = d - (long)d;
				f = d * m;
			}
			else {
				f += m;
			}
		}
	}
	else {
		f = 0.0; //don't divide by zero
	}
	return f;
}


inline double safediv(double num, double denom) {
	return denom == 0. ? 0. : num / denom;
}

// fixnan for case of negative base and non-integer exponent:
inline double safepow(double base, double exponent) {
	return fixnan(pow(base, exponent));
}

inline double absdiff(double a, double b) { return fabs(a - b); }

inline double exp2(double v) { return pow(2., v); }

inline double trunc(double v) {
	double epsilon = (v<0.0) * -2 * 1E-9 + 1E-9;
	// copy to long so it gets truncated (probably cheaper than floor())
	long val = long(v + epsilon);
	return val;
}

inline double sign(double v) { return v > 0. ? 1. : v < 0. ? -1. : 0.; }

inline long is_poweroftwo(long x) {
	return (x & (x - 1)) == 0;
}

inline uint64_t next_power_of_two(uint64_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;
	return v;
}

inline double fold(double v, double lo1, double hi1) {
	double lo;
	double hi;
	if (lo1 == hi1) { return lo1; }
	if (lo1 > hi1) {
		hi = lo1; lo = hi1;
	}
	else {
		lo = lo1; hi = hi1;
	}
	const double range = hi - lo;
	long numWraps = 0;
	if (v >= hi) {
		v -= range;
		if (v >= hi) {
			numWraps = (long)((v - lo) / range);
			v -= range * (double)numWraps;
		}
		numWraps++;
	}
	else if (v < lo) {
		v += range;
		if (v < lo) {
			numWraps = (long)((v - lo) / range) - 1;
			v -= range * (double)numWraps;
		}
		numWraps--;
	}
	if (numWraps & 1) v = hi + lo - v;	// flip sign for odd folds
	return v;
}

inline double wrap(double v, double lo1, double hi1) {
	double lo;
	double hi;
	if (lo1 == hi1) return lo1;
	if (lo1 > hi1) {
		hi = lo1; lo = hi1;
	}
	else {
		lo = lo1; hi = hi1;
	}
	const double range = hi - lo;
	if (v >= lo && v < hi) return v;
	if (range <= 0.000000001) return lo;	// no point...
	const long numWraps = long((v - lo) / range) - (v < lo);
	return v - range * double(numWraps);
}

// this version gives far better performance when wrapping is relatively rare
// and typically double of wraps is very low (>1%)
// but catastrophic if wraps is high (1000%+)
inline double wrapfew(double v, double lo, double hi) {
	const double range = hi - lo;
	while (v >= hi) v -= range;
	while (v < lo) v += range;
	return v;
}

inline double phasewrap_few(double val) {
	return wrapfew(val, -AN_PI, AN_PI);
}

inline double phasewrap(double val) {
	const double twopi = AN_PI*2.;
	const double oneovertwopi = 1. / twopi;
	if (val >= twopi || val <= twopi) {
		double d = val * oneovertwopi;	//multiply faster
		d = d - (long)d;
		val = d * twopi;
	}
	if (val > AN_PI) val -= twopi;
	if (val < -AN_PI) val += twopi;
	return val;
}

/// 8th order Taylor series approximation to a cosine.
/// r must be in [-pi, pi].
inline double cosT8(double r) {
	const double t84 = 56.;
	const double t83 = 1680.;
	const double t82 = 20160.;
	const double t81 = 2.4801587302e-05;
	const double t73 = 42.;
	const double t72 = 840.;
	const double t71 = 1.9841269841e-04;
	if (r < AN_PI_OVER_4 && r > -AN_PI_OVER_4) {
		double rr = r*r;
		return 1. - rr * t81 * (t82 - rr * (t83 - rr * (t84 - rr)));
	}
	else if (r > 0.) {
		r -= AN_PI_OVER_2;
		double rr = r*r;
		return -r * (1. - t71 * rr * (t72 - rr * (t73 - rr)));
	}
	else {
		r += AN_PI_OVER_2;
		double rr = r*r;
		return r * (1. - t71 * rr * (t72 - rr * (t73 - rr)));
	}
}

inline double sin_fast(const double r) {
	const double y = (4. / AN_PI) * r + (-4. / (AN_PI*AN_PI)) * r * fabs(r);
	return 0.225 * (y * fabs(y) - y) + y;   // Q * y + P * y * abs(y)
}

inline double sinP7(double n) {
	double nn = n*n;
	return n * (3.138982 + nn * (-5.133625 + nn * (2.428288 - nn * 0.433645)));
}

inline double sinP9(double n) {
	double nn = n*n;
	return n * (AN_PI + nn * (-5.1662729 + nn * (2.5422065 + nn * (-0.5811243 + nn * 0.0636716))));
}

inline double sinT7(double r) {
	const double t84 = 56.;
	const double t83 = 1680.;
	const double t82 = 20160.;
	const double t81 = 2.4801587302e-05;
	const double t73 = 42.;
	const double t72 = 840.;
	const double t71 = 1.9841269841e-04;
	if (r < AN_PI_OVER_4 && r > -AN_PI_OVER_4) {
		double rr = r*r;
		return r * (1. - t71 * rr * (t72 - rr * (t73 - rr)));
	}
	else if (r > 0.) {
		r -= AN_PI_OVER_2;
		double rr = r*r;
		return 1. - rr * t81 * (t82 - rr * (t83 - rr * (t84 - rr)));
	}
	else {
		r += AN_PI_OVER_2;
		double rr = r*r;
		return -1. + rr * t81 * (t82 - rr * (t83 - rr * (t84 - rr)));
	}
}

// use these if r is not known to be in [-pi, pi]:
inline double cosT8_safe(double r) { return cosT8(phasewrap(r)); }
inline double sin_fast_safe(double r) { return sin_fast(phasewrap(r)); }
inline double sinP7_safe(double r) { return sinP7(phasewrap(r)); }
inline double sinP9_safe(double r) { return sinP9(phasewrap(r)); }
inline double sinT7_safe(double r) { return sinT7(phasewrap(r)); }

inline double minimum(double x, double y) { return (y<x ? y : x); }
inline double maximum(double x, double y) { return (x<y ? y : x); }

inline double clamp(double x, double minVal, double maxVal) {
	return minimum(maximum(x, minVal), maxVal);
}

template<typename T>
inline T smoothstep(double e0, double e1, T x) {
	T t = clamp(safediv(x - T(e0), T(e1 - e0)), 0., 1.);
	return t*t*(T(3) - T(2)*t);
}

inline double mix(double x, double y, double a) {
	return x + a*(y - x);
}

inline double scale(double in, double inlow, double inhigh, double outlow, double outhigh, double power)
{
	double value;
	double inscale = safediv(1., inhigh - inlow);
	double outdiff = outhigh - outlow;

	value = (in - inlow) * inscale;
	if (value > 0.0)
		value = pow(value, power);
	else if (value < 0.0)
		value = -pow(-value, power);
	value = (value * outdiff) + outlow;

	return value;
}

inline double linear_interp(double a, double x, double y) {
	return x + a*(y - x);
}

inline double cosine_interp(double a, double x, double y) {
	const double a2 = (1. - cosT8_safe(a*AN_PI)) / 2.;
	return(x*(1. - a2) + y*a2);
}

inline double cubic_interp(double a, double w, double x, double y, double z) {
	const double a2 = a*a;
	const double f0 = z - y - w + x;
	const double f1 = w - x - f0;
	const double f2 = y - w;
	const double f3 = x;
	return(f0*a*a2 + f1*a2 + f2*a + f3);
}

// Breeuwsma catmull-rom spline interpolation
inline double spline_interp(double a, double w, double x, double y, double z) {
	const double a2 = a*a;
	const double f0 = -0.5*w + 1.5*x - 1.5*y + 0.5*z;
	const double f1 = w - 2.5*x + 2 * y - 0.5*z;
	const double f2 = -0.5*w + 0.5*y;
	return(f0*a*a2 + f1*a2 + f2*a + x);
}

template<typename T1, typename T2>
inline T1 neqp(T1 x, T2 y) {
	return ((((x) != T1(y))) ? (x) : T1(0));
}

template<typename T1, typename T2>
inline T1 gtp(T1 x, T2 y) { return ((((x) > T1(y))) ? (x) : T1(0)); }
template<typename T1, typename T2>
inline T1 gtep(T1 x, T2 y) { return ((((x) >= T1(y))) ? (x) : T1(0)); }
template<typename T1, typename T2>
inline T1 ltp(T1 x, T2 y) { return ((((x) < T1(y))) ? (x) : T1(0)); }
template<typename T1, typename T2>
inline T1 ltep(T1 x, T2 y) { return ((((x) <= T1(y))) ? (x) : T1(0)); }

inline double fract(double x) { double unused; return modf(x, &unused); }

// log2(x) = log(x)/log(2)
template<typename T>
inline T log2(T x) {
	return log(x)*AN_1_OVER_LOG_2;
}

inline double atodb(double in) {
	return (in <= 0.) ? -999. : (20. * log10(in));
}

inline double dbtoa(double in) {
	return pow(10., in * 0.05);
}

inline double ftom(double in, double tuning = 440.) {
	return 69. + 17.31234050465299 * log(safediv(in, tuning));
}

inline double mtof(double in, double tuning = 440.) {
	return tuning * exp(.057762265 * (in - 69.0));
}

inline double mstosamps(double ms, double samplerate = 44100.) {
	return samplerate * ms * 0.001;
}

inline double sampstoms(double s, double samplerate = 44100.) {
	return 1000. * s / samplerate;
}

inline double triangle(double phase, double p1) {
	phase = wrap(phase, 0., 1.);
	p1 = clamp(p1, 0., 1.);
	if (phase < p1)
		return (p1) ? phase / p1 : 0.;
	else
		return (p1 == 1.) ? phase : 1. - ((phase - p1) / (1. - p1));
}



#endif
