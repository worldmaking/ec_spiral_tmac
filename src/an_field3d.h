#ifndef FIELD3D_H
#define FIELD3D_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "an_math.h"


template<typename T=float>
class Array {
public:

	// temporary hack because the one in al_Function gave a bad result
	// for e.g. wrap<double>(-64.0, -32.0);
	template<typename T>
	inline static T wrap(T v, const T hi=T(1.), const T lo=T(0.)){
		if(lo == hi) return lo;
		//if(v >= hi){
		if(!(v < hi)){
			T diff = hi - lo;
			v -= diff;
			if(!(v < hi)) v -= diff * (T)(uint32_t)((v - lo)/diff);
		}
		else if(v < lo){
			T diff = hi - lo;
			v += diff;
			if(v < lo) v += diff * (T)(uint32_t)(((lo - v)/diff) + 1);
			if(v==diff) return lo;
		}
		return v;
	}

	// dims should be powers of 2.
	void initialize(int dimx, int dimy, int dimz, int components = 1) {
		mDimX = dimx;
		mDimY = dimy;
		mDimZ = dimz;
		mDimWrapX = mDimX-1;
		mDimWrapY = mDimY-1;
		mDimWrapZ = mDimZ-1;
		mDim3 = mDimX*mDimY*mDimZ;
		mComponents = components;
		mLength = mDim3 * mComponents;
		mSize = sizeof(T) * mLength;
		mStride[0] = mComponents;
		mStride[1] = mStride[0] * mDimX;
		mStride[2] = mStride[1] * mDimY;
		
		//printf("created Field3D %d %d %d [%d] len %d size %d stride %d %d %d\n", mDimX, mDimY, mDimZ, mComponents, mLength, mSize,  mStride[0], mStride[1], mStride[2]);
		
		// allocate:
		data = (T *)malloc(mSize);
		memset(data, 0, mSize);
	}
	void initialize(glm::dvec3 dim, int components = 1) {
		initialize(dim.x, dim.y, dim.z, components);
	}
	
	unsigned length() const { return mLength; }
	unsigned size() const { return mSize; }
	unsigned components() const { return mComponents; }
	unsigned dimx() const { return (unsigned)mDimX; }
	unsigned dimy() const { return (unsigned)mDimY; }
	unsigned dimz() const { return (unsigned)mDimZ; }
	unsigned stride(int dim) const { return (unsigned)mStride[dim]; }
	
	void print() {
		printf("Array %dx%dx%d[%d]\n", mDimX, mDimY, mDimZ, mComponents);
	}
	
	template<typename T1>
	void read_interp(const glm::dvec3 pos, T1 * data) const;
	
	template<typename T1>
	void read_interp(double x, double y, double z, T1 * val) const;
	
	template<typename T1>
	inline void read(int x, int y, int z, T1 * val) const;
	
	void add(Array<T>& src) {
		//this->print(); src.print();
		// assumes dim/comonents etc. match
		for (int i=0; i<mLength; i++) {
			data[i] += src.data[i];
		}
	}
	// interpolated add:
	inline void add(const glm::dvec3 pos, const T * elems) {
		add(pos.x, pos.y, pos.z, elems);
	}
	
	
	
	
	// interpolated add:
	inline void add(const glm::vec3 pos, const T * elems) {
		add(pos.x, pos.y, pos.z, elems);
	}
	
	void add(double x, double y, double z, const T * elems);
	
	void zero() {
		for (unsigned int i=0; i<mLength; i++) { data[i] = 0; }
	}
	
	void scale(T v) {
		for (int i=0; i<mLength; i++) { data[i] *= v; }
	}
	
	// get the ith cell:
	T * operator[](int i) { return data + i*mStride[0]; }
	
	T * ptr() { return data; }
	// pointer index of a particular cell:
	size_t index(int x, int y, int z) const;
	// access a particular element:
	T * cell(int x, int y, int z, int k=0) const;
	
	// returns true if the vector v is outside of the field boundaries
	inline bool oob(glm::dvec3 v);
	
	T * data;
	size_t mDimX, mDimY, mDimZ, mDim3, mDimWrapX, mDimWrapY, mDimWrapZ, mComponents, mLength, mSize;
	size_t mStride[3];
};

/*!
	Field processing often requires double-buffering
 */

template<typename T=float>
class Field3D {
public:
	
	// dims should be powers of 2.
	void initialize(int dimx, int dimy, int dimz, int components = 1) {
		mDimX = dimx;
		mDimY = dimy;
		mDimZ = dimz;
		mDim3 = mDimX*mDimY*mDimZ;
		mDimWrapX = mDimX-1;
		mDimWrapY = mDimY-1;
		mDimWrapZ = mDimZ-1;
		mFront = 1;
		
		mArray0.initialize(mDimX, mDimY, mDimZ, components);
		mArray1.initialize(mDimX, mDimY, mDimZ, components);
	}
	void initialize(glm::dvec3 dim, int components = 1) {
		initialize(dim.x, dim.y, dim.z, components);
	}
	
	unsigned length() const { return mArray0.length(); }
	unsigned size() const { return mArray0.mSize; }
	unsigned components() const { return mArray0.mComponents; }
	unsigned dimx() const { return mArray0.mDimX; }
	unsigned dimy() const { return mArray0.mDimY; }
	unsigned dimz() const { return mArray0.mDimZ; }
	unsigned stride(int dim) const { return (unsigned)mArray0.mStride[dim]; }
	
	// front is what is currently interacted with
	// back is used for intermediate processing
	Array<T>& front() { return mFront ? mArray0 : mArray1; }
	Array<T>& back() { return mFront ? mArray1 : mArray0; }
	const Array<T>& front() const { return mFront ? mArray0 : mArray1; }
	const Array<T>& back() const { return mFront ? mArray1 : mArray0; }
	
	// swap buffers:
	void swap() { mFront = !mFront; }
	
	void diffuse(double diffusion, unsigned passes=14);
	/*
	 /// read the intensity at a particular location:
	 template<typename T1>
	 void read(const Vec<3,T1> pos, T * elems) const;
	 
	 
	 /// multiply the front array:
	 void scale(T v);
	 /// src must have matching layout
	 void scale(const Array& src);
	 /// src must have matching layout, except for components (which may be 1)
	 void scale1(const Array& src);
	 /// multiply by 1./(src+1.), as a 'damping' factor:
	 void damp(const Array& src);	// src must have matching layout
	 /// add to front array. src must have matching layout
	 void add(Array& src);
	 /// add uniform vector to front array. vec must have as many components as the field
	 void add(T * vec);
	 /// add intensity at a particular location:
	 template<typename T1>
	 void add(const Vec<3,T1> pos, const T * elems);
	 void add(int x, int y, int z, const T * elems);
	 /// single component case:
	 template<typename T1>
	 void add(const Vec<3,T1> pos, T elem);
	 // fill with noise:
	 void adduniform(rnd::Random<>& rng, T scalar = T(1));
	 void adduniformS(rnd::Random<>& rng, T scalar = T(1));
	 // fill with sines:
	 void setHarmonic(T px=T(1), T py=T(1), T pz=T(1));
	 // 3-component fields only: scale velocities at boundaries
	 void boundary();
	 
	 // advect a field.
	 // velocity field should have 3 components
	 void advect(const Array& velocities, T rate = T(1.));
	 static void advect(Array& dst, const Array& src, const Array& velocities, T rate = T(1.));
	 
	 //
	 //	Clever part of Jos Stam's work.
	 //		A velocity field can become divergent (have regions that are purely emanating or aggregating)
	 //			violating the definition of an incompressible fluid
	 //		But, since a velocity field can be seen as an incompressible velocity field + a gradient field,
	 //			we can subtract a gradient field from our bad velocity field to get an incompressible one
	 //		To calculate this gradient field and then subtract it, we use this function:
	 //
	 // grabs the previous instantaneous magnitude of velocity gradient
	 void calculateGradientMagnitude(Array& gradient);
	 void subtractGradientMagnitude(const Array& gradient);
	 
	 void relax(double a, int iterations);
	 */
	
	void advect(const Array<T>& velocities, T rate) {
		swap();
		advect(front(), back(), velocities, rate);
	}
	void advect(Array<T>& dst, const Array<T>& src, const Array<T>& velocities, T rate);
	
	void calculateGradientMagnitude(Array<T>& gradient);
	void subtractGradientMagnitude(const Array<T>& gradient);
	
	size_t mDimX, mDimY, mDimZ, mDim3, mDimWrapX, mDimWrapY, mDimWrapZ;
	volatile int mFront;	// which one is the front buffer?
	Array<T> mArray0, mArray1; //mArrays[2];	// double-buffering
};

template<typename T=float>
class Fluid3D {
public:
	enum BoundaryMode {
		NONE = 0,
		CLAMP = 1,
		FIELD = 2
	};
	
	void initialize(int dimx, int dimy, int dimz) {
		velocities.initialize(dimx, dimy, dimz, 3);
		gradient.initialize(dimx, dimy, dimz, 1);
		printf("initialized\n");
		//boundaries.initialize(dimx, dimy, dimz, 1);
	}
	void initialize(glm::dvec3 dim) {
		initialize(dim.x, dim.y, dim.z);
	}
	
	void readVelocity(const glm::dvec3 pos, glm::dvec3& vel) const {
		velocities.front().read_interp(pos, &vel.x);
	}
	
	/*
	 Fluid3D(int dimx=32, int dimy=32, int dimz=32)
	 :	velocities(3, dimx, dimy, dimz),
		gradient(1, dimx, dimy, dimz),
		boundaries(1, Array::type<T>(), dimx, dimy, dimz),
		passes(14),
		viscocity(0.00001),
		selfadvection(0.9),
		selfdecay(0.99),
		selfbackgroundnoise(0.001),
		mBoundaryMode(CLAMP)
	 {
		// set all values to T(1):
		T one = 1;
		boundaries.set3d(&one);
	 }
	 ~Fluid3D() {}
	 
	 template<typename T1>
	 void addVelocity(const Vec<3,T1> pos, const Vec<3,T> vel) {
		velocities.add(pos, vel.elems());
	 }
	 void addVelocity(int x, int y, int z, const Vec<3,T> vel) {
		velocities.add(x, y, z, vel.elems());
	 }
	 
	 template<typename T1>
	 void addGradient(const Vec<3,T1> pos, float& g) {
		gradient.add(pos, &g);
	 }
	 
	 template<typename T1>
	 void readGradient(const Vec<3,T1> pos, float& g) const {
		gradient.read(pos, &g);
	 }
	 
	 /// fluid simulation step
	 // TODO: add dt param
	 void update() {
		// VELOCITIES:
		// add a bit of random noise:
		velocities.adduniformS(rng, selfbackgroundnoise);
		// assume new data is in front();
		// smoothen the new data:
		velocities.diffuse(viscocity, passes);
		// zero velocities at boundaries:
		boundary();
		// (diffused data now in velocities.front())
		// stabilize:
		project();
		// (projected data now in velocities.front())
		// advect velocities:
		velocities.advect(velocities.back(), selfadvection);
		// zero velocities at boundaries:
		boundary();
		// (advected data now in velocities.front())
		// stabilize again:
		project();
		// (projected data now in velocities.front())
		velocities.scale(selfdecay);
		// zero velocities at boundaries:
		boundary();
		
		gradient.front().zero();
		gradient.back().zero();
	 }
	 
	 void boundary() {
		switch (mBoundaryMode) {
	 case CLAMP:
	 velocities.boundary();
	 break;
	 case FIELD:
	 velocities.scale1(boundaries);
	 break;
	 default:
	 break;
		}
	 }
	 
	 
	 
	 void boundary(BoundaryMode b) { mBoundaryMode = b; }
	 */
	
	void project(int passes) {
		gradient.back().zero();
		// prepare new gradient data:
		velocities.calculateGradientMagnitude(gradient.front());
		// diffuse it:
		gradient.diffuse(0.5, passes);
		// subtract from current velocities:
		velocities.subtractGradientMagnitude(gradient.front());
	}
	
	Field3D<T> velocities, gradient;
	//Array<T> boundaries;
	//unsigned passes;
	//T viscocity, selfadvection, selfdecay, selfbackgroundnoise;
	//BoundaryMode mBoundaryMode;
};

// returns true if the vector v is outside of the field boundaries
template <typename T>
inline bool Array<T>::oob(glm::dvec3 v) {
	return (v.x < 0 || v.x >= dimx()
		 || v.y < 0 || v.y >= dimy()
		 || v.z < 0 || v.z >= dimz());
}

template<typename T>
inline size_t Array<T>::index(int x, int y, int z) const {
	return	((x&mDimWrapX) * stride(0)) +
	((y&mDimWrapY) * stride(1)) +
	((z&mDimWrapZ) * stride(2));
}

template<typename T>
inline T * Array<T>::cell(int x, int y, int z, int k) const {
	return data + (index(x, y, z) + k);
}

template<typename T>
template<typename T1>
inline void Array<T>::read_interp(const glm::dvec3 pos, T1 * val) const {
	read_interp(pos.x, pos.y, pos.z, val);
}

template<typename T>
template<typename T1>
inline void Array<T>::read(int x, int y, int z, T1 * val) const {
	x = wrap<int>(x, (int)mDimX, 0.);
	y = wrap<int>(y, (int)mDimY, 0.);
	z = wrap<int>(z, (int)mDimZ, 0.);
	T * paaa = cell(x, y, z);
	for (size_t p=0; p<mComponents; p++) {
		val[p] = (T1)(paaa[p]);
	}
}


template<typename T>
template<typename T1>
inline void Array<T>::read_interp(double x, double y, double z, T1 * val) const {
	x = wrap<double>(x, (double)mDimX, 0.);
	y = wrap<double>(y, (double)mDimY, 0.);
	z = wrap<double>(z, (double)mDimZ, 0.);
	// convert 0..1 field indices to 0..(d-1) cell indices
	const unsigned xa = (const unsigned)DOUBLE_FLOOR(x);
	const unsigned ya = (const unsigned)DOUBLE_FLOOR(y);
	const unsigned za = (const unsigned)DOUBLE_FLOOR(z);
	unsigned xb = xa+1;	if (xb == mDimX) xb = 0;
	unsigned yb = ya+1;	if (yb == mDimY) yb = 0;
	unsigned zb = za+1;	if (zb == mDimZ) zb = 0;
	// get the normalized 0..1 interp factors, of x,y,z:
	double xbf = DOUBLE_FRAC(x);
	double xaf = 1.f - xbf;
	double ybf = DOUBLE_FRAC(y);
	double yaf = 1.f - ybf;
	double zbf = DOUBLE_FRAC(z);
	double zaf = 1.f - zbf;
	// get the interpolation corner weights:
	double faaa = xaf * yaf * zaf;
	double faab = xaf * yaf * zbf;
	double faba = xaf * ybf * zaf;
	double fabb = xaf * ybf * zbf;
	double fbaa = xbf * yaf * zaf;
	double fbab = xbf * yaf * zbf;
	double fbba = xbf * ybf * zaf;
	double fbbb = xbf * ybf * zbf;
	// get the cell addresses for each neighbor:
	T * paaa = cell(xa, ya, za);
	T * paab = cell(xa, ya, zb);
	T * paba = cell(xa, yb, za);
	T * pabb = cell(xa, yb, zb);
	T * pbaa = cell(xb, ya, za);
	T * pbab = cell(xb, ya, zb);
	T * pbba = cell(xb, yb, za);
	T * pbbb = cell(xb, yb, zb);
	// for each plane of the field, do the 3D interp:
	for (size_t p=0; p<mComponents; p++) {
		val[p] =(T)((double(paaa[p]) * faaa) +
					(double(pbaa[p]) * fbaa) +
					(double(paba[p]) * faba) +
					(double(paab[p]) * faab) +
					(double(pbab[p]) * fbab) +
					(double(pabb[p]) * fabb) +
					(double(pbba[p]) * fbba) +
					(double(pbbb[p]) * fbbb) );
	}
}

template<typename T>
inline void Array<T>::add(double x, double y, double z, const T * val) {
	x = wrap<double>(x, (double)mDimX, 0.);
	y = wrap<double>(y, (double)mDimY, 0.);
	z = wrap<double>(z, (double)mDimZ, 0.);
	const unsigned xa = (const unsigned)DOUBLE_FLOOR(x);
	const unsigned ya = (const unsigned)DOUBLE_FLOOR(y);
	const unsigned za = (const unsigned)DOUBLE_FLOOR(z);
	unsigned xb = xa+1;	if (xb == mDimX) xb = 0;
	unsigned yb = ya+1;	if (yb == mDimY) yb = 0;
	unsigned zb = za+1;	if (zb == mDimZ) zb = 0;
	// get the normalized 0..1 interp factors, of x,y,z:
	double xbf = DOUBLE_FRAC(x);
	double xaf = 1.f - xbf;
	double ybf = DOUBLE_FRAC(y);
	double yaf = 1.f - ybf;
	double zbf = DOUBLE_FRAC(z);
	double zaf = 1.f - zbf;
	// get the interpolation corner weights:
	double faaa = xaf * yaf * zaf;
	double faab = xaf * yaf * zbf;
	double faba = xaf * ybf * zaf;
	double fabb = xaf * ybf * zbf;
	double fbaa = xbf * yaf * zaf;
	double fbab = xbf * yaf * zbf;
	double fbba = xbf * ybf * zaf;
	double fbbb = xbf * ybf * zbf;
	T * paaa = cell(xa, ya, za);
	T * paab = cell(xa, ya, zb);
	T * paba = cell(xa, yb, za);
	T * pabb = cell(xa, yb, zb);
	T * pbaa = cell(xb, ya, za);
	T * pbab = cell(xb, ya, zb);
	T * pbba = cell(xb, yb, za);
	T * pbbb = cell(xb, yb, zb);
	// for each plane of the field, do the 3D interp:
	for (uint8_t p=0; p<mComponents; p++) {
		T tmp = val[p];
		paaa[p] += T(tmp * faaa);
		paab[p] += T(tmp * faab);
		paba[p] += T(tmp * faba);
		pabb[p] += T(tmp * fabb);
		pbaa[p] += T(tmp * fbaa);
		pbab[p] += T(tmp * fbab);
		pbba[p] += T(tmp * fbba);
		pbbb[p] += T(tmp * fbbb);
	}
}

template<typename T>
inline void Field3D<T> :: advect(Array<T>& dst, const Array<T>& src, const Array<T>& velocities, T rate) {
	const size_t stride0 = src.stride(0);
	const size_t stride1 = src.stride(1);
	const size_t stride2 = src.stride(2);
	const size_t dim0 = src.mDimX;
	const size_t dim1 = src.mDimY;
	const size_t dim2 = src.mDimZ;
	const size_t dimwrap0 = dim0-1;
	const size_t dimwrap1 = dim1-1;
	const size_t dimwrap2 = dim2-1;
	
	float * outptr = dst.data;
	float * velptr = velocities.data;
	
	const size_t vstride0 = velocities.stride(0);
	const size_t vstride1 = velocities.stride(1);
	const size_t vstride2 = velocities.stride(2);
	
#define CELL(p, x, y, z, k) (((T *)((p) + (((x)&dimwrap0)*stride0) +  (((y)&dimwrap1)*stride1) +  (((z)&dimwrap2)*stride2)))[(k)])
#define VCELL(p, x, y, z, k) (((T *)((p) + (((x)&dimwrap0)*vstride0) +  (((y)&dimwrap1)*vstride1) +  (((z)&dimwrap2)*vstride2)))[(k)])
	
	for (size_t z=0;z<dim2;z++) {
		for (size_t y=0;y<dim1;y++) {
			for (size_t x=0;x<dim0;x++) {
				// back trace: (current cell offset by vector at cell)
				T * bp  = &(CELL(outptr, x, y, z, 0));
				T * vp	= &(VCELL(velptr, x, y, z, 0));
				glm::dvec3 pos;
				pos.x = x - rate * vp[0];
				pos.y = y - rate * vp[1];
				pos.z = z - rate * vp[2];
				
				// read interpolated input field value into back-traced location:
				src.read_interp(pos, bp);
			}
		}
	}
#undef CELL
#undef VCELL
}

// Gauss-Seidel relaxation scheme:
template<typename T>
inline void Field3D<T> :: diffuse(double diffusion, unsigned passes) {
	swap();
	Array<T>& out = front();
	const Array<T>& in = back();
	const size_t stride0 = out.mStride[0];
	const size_t stride1 = out.mStride[1];
	const size_t stride2 = out.mStride[2];
	const size_t components = out.mComponents;
	const T * iptr = in.data;
	T * optr = out.data;
	double div = 1.0/((1.+6.*diffusion));
#define INDEX(p, x, y, z) (p + ((((x)&mDimWrapX)*stride0) + (((y)&mDimWrapY)*stride1) + (((z)&mDimWrapZ)*stride2)))
	
	for (unsigned n=0 ; n<passes ; n++) {
		for (size_t z=0;z<mDimZ;z++) {
			for (size_t y=0;y<mDimY;y++) {
				for (size_t x=0;x<mDimX;x++) {
					const T * prev =	INDEX(iptr, x,	y,	z);
					T *		  next =	INDEX(optr, x,	y,	z);
					const T * va00 =	INDEX(optr, x-1,y,	z);
					const T * vb00 =	INDEX(optr, x+1,y,	z);
					const T * v0a0 =	INDEX(optr, x,	y-1,z);
					const T * v0b0 =	INDEX(optr, x,	y+1,z);
					const T * v00a =	INDEX(optr, x,	y,	z-1);
					const T * v00b =	INDEX(optr, x,	y,	z+1);
					for (size_t k=0;k<components;k++) {
						next[k] = T(div*(
										 prev[k] +
										 diffusion * (
													  va00[k] + vb00[k] +
													  v0a0[k] + v0b0[k] +
													  v00a[k] + v00b[k]
													  )
										 ));
					}
				}
			}
		}
	}
#undef INDEX
}


template<typename T>
inline void Field3D<T> :: calculateGradientMagnitude(Array<T>& gradient) {
	const size_t stride0 = stride(0);
	const size_t stride1 = stride(1);
	const size_t stride2 = stride(2);
	const size_t gstride0 = gradient.stride(0);
	const size_t gstride1 = gradient.stride(1);
	const size_t gstride2 = gradient.stride(2);
	
#define CELL(p, x, y, z, k) (((T *)((p) + (((x)&mDimWrapX)*stride0) +  (((y)&mDimWrapY)*stride1) +  (((z)&mDimWrapZ)*stride2)))[(k)])
#define CELLG(p, x, y, z) (((T *)((p) + (((x)&mDimWrapX)*gstride0) +  (((y)&mDimWrapY)*gstride1) +  (((z)&mDimWrapZ)*gstride2)))[0])
	
	// calculate gradient.
	// previous instantaneous magnitude of velocity gradient
	//		= average of velocity gradients per axis:
	const double hx = -0.5/mDimX; //1./3.; //0.5/mDim;
	const double hy = -0.5/mDimY; //1./3.; //0.5/mDim;
	const double hz = -0.5/mDimZ; //1./3.; //0.5/mDim;
	float * iptr = front().data;
	float * gptr = gradient.data;
	
	for (size_t z=0;z<mDimZ;z++) {
		for (size_t y=0;y<mDimY;y++) {
			for (size_t x=0;x<mDimX;x++) {
				// gradients per axis:
				const T xgrad = CELL(iptr, x+1,y,	z,	0) - CELL(iptr, x-1,y,	z,	0);
				const T ygrad = CELL(iptr, x,	y+1,z,	1) - CELL(iptr, x,	y-1,z,	1);
				const T zgrad = CELL(iptr, x,	y,	z+1,2) - CELL(iptr, x,	y,	z-1,2);
				// gradient at current cell:
				const T grad = T(hx*xgrad + hy*ygrad + hz*zgrad);
				// add to 1-plane field
				CELLG(gptr, x, y, z) += grad;
			}
		}
	}
#undef CELL
#undef CELLG
}

template<typename T>
inline void Field3D<T> :: subtractGradientMagnitude(const Array<T>& gradient) {
	const size_t stride0 = stride(0);
	const size_t stride1 = stride(1);
	const size_t stride2 = stride(2);
	const size_t gstride0 = gradient.stride(0);
	const size_t gstride1 = gradient.stride(1);
	const size_t gstride2 = gradient.stride(2);
	
#define INDEX(p, x, y, z) ((T *)(p + ((((x)&mDimWrapX)*stride0) + (((y)&mDimWrapY)*stride1) + (((z)&mDimWrapZ)*stride2))))
#define CELLG(p, x, y, z) (((T *)((p) + (((x)&mDimWrapX)*gstride0) +  (((y)&mDimWrapY)*gstride1) +  (((z)&mDimWrapZ)*gstride2)))[0])
	
	// now subtract gradient from current field:
	float * gptr = gradient.data;
	float * optr = front().data;
	//const double h = 1.; ///3.;
	const float hx = mDimX * 0.5f;
	const float hy = mDimY * 0.5f;
	const float hz = mDimZ * 0.5f;
	for (size_t z=0;z<mDimZ;z++) {
		for (size_t y=0;y<mDimY;y++) {
			for (size_t x=0;x<mDimX;x++) {
				// cell to update:
				T * vel = INDEX(optr, x, y, z);
				// gradients per axis:
				vel[0] -= hx * ( CELLG(gptr, x+1,y,	z  ) - CELLG(gptr, x-1,y,	z  ) );
				vel[1] -= hy * ( CELLG(gptr, x,	y+1,z  ) - CELLG(gptr, x,	y-1,z  ) );
				vel[2] -= hz * ( CELLG(gptr, x,	y,	z+1) - CELLG(gptr, x,	y,	z-1) );
			}
		}
	}
	
#undef INDEX
#undef CELLG
}

#endif
