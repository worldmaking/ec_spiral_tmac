#ifndef GENLIB_H
#define GENLIB_H 1

#include "an_math.h"

#include <stdio.h>




#define C74_CONST const

#define DSP_GEN_MAX_SIGNALS 16

// DATA_MAXIMUM_ELEMENTS * 8 bytes = 256 mb limit
#define DATA_MAXIMUM_ELEMENTS	(33554432)

typedef double t_sample;
typedef char *t_ptr;
typedef long t_genlib_err;
typedef enum {
	GENLIB_ERR_NONE =			0,	///< No error
	GENLIB_ERR_GENERIC =		-1,	///< Generic error
	GENLIB_ERR_INVALID_PTR =	-2,	///< Invalid Pointer
	GENLIB_ERR_DUPLICATE =		-3,	///< Duplicate
	GENLIB_ERR_OUT_OF_MEM =		-4,	///< Out of memory
	
	GENLIB_ERR_LOOP_OVERFLOW =  100,	// too many iterations of loops in perform()
	GENLIB_ERR_NULL_BUFFER =	101	// missing signal data in perform()
	
} e_genlib_errorcodes;

// opaque interface to float32 buffer:
typedef struct _genlib_buffer t_genlib_buffer;
typedef struct {
	char b_name[256];	///< name of the buffer
	float *b_samples;	///< stored with interleaved channels if multi-channel
	long b_frames;		///< number of sample frames (each one is sizeof(float) * b_nchans bytes)
	long b_nchans;		///< number of channels
	long b_size;		///< size of buffer in floats
	float b_sr;			///< sampling rate of the buffer
	long b_modtime;		///< last modified time ("dirty" method)
	long b_rfu[57];		///< reserved for future use
} t_genlib_buffer_info;

// opaque interface to float64 buffer:
typedef struct _genlib_data t_genlib_data;
typedef struct {
	int					dim, channels;
	double *			data;
} t_genlib_data_info;

inline unsigned long genlib_ticks(void) { return 0; }

inline void genlib_report_error(const char *s) {
	fprintf(stderr, "%s\n", s);
}

inline void genlib_report_message(const char *s) {
	fprintf(stdout, "%s\n", s);
}

inline void genlib_set_zero64(double *memory, long size) {
	long i;
	for (i = 0; i < size; i++, memory++) {
		*memory = 0.;
	}
}

inline void * genlib_sysmem_newptr(size_t size) { return (void *)malloc(size); }
inline void genlib_sysmem_freeptr(void *ptr) { free(ptr); }
inline void * genlib_sysmem_resizeptr(void *ptr, size_t newsize) { return (void *)realloc(ptr, newsize); }


inline void genlib_data_setbuffer(t_genlib_data *b, void *ref) {
	genlib_report_error("not supported for export targets\n");
}

typedef struct {
	t_genlib_data_info	info;
	double				cursor;	// used by genDelay
								//t_symbol *			name;
} t_dsp_gen_data;

inline t_genlib_data * genlib_obtain_data_from_reference(void *ref)
{
	t_dsp_gen_data * self = (t_dsp_gen_data *)malloc(sizeof(t_dsp_gen_data));
	self->info.dim = 0;
	self->info.channels = 0;
	self->info.data = 0;
	self->cursor = 0;
	return (t_genlib_data *)self;
}

inline t_genlib_err genlib_data_getinfo(t_genlib_data *b, t_genlib_data_info *info) {
	t_dsp_gen_data * self = (t_dsp_gen_data *)b;
	info->dim = self->info.dim;
	info->channels = self->info.channels;
	info->data = self->info.data;
	return GENLIB_ERR_NONE;
}

inline void genlib_data_release(t_genlib_data *b) {
	t_dsp_gen_data * self = (t_dsp_gen_data *)b;
	
	if (self->info.data) {
		genlib_sysmem_freeptr(self->info.data);
		self->info.data = 0;
	}
}

inline long genlib_data_getcursor(t_genlib_data *b) {
	t_dsp_gen_data * self = (t_dsp_gen_data *)b;
	return (long)self->cursor;
}

inline void genlib_data_setcursor(t_genlib_data *b, long cursor) {
	t_dsp_gen_data * self = (t_dsp_gen_data *)b;
	self->cursor = cursor;
}

inline void genlib_data_resize(t_genlib_data *b, long s, long c) {
	t_dsp_gen_data * self = (t_dsp_gen_data *)b;
	
	size_t sz, oldsz, copysz;
	double * old = 0;
	double * replaced = 0;
	int i, j, copydim, copychannels, olddim, oldchannels;
	
	//printf("data resize %d %d\n", s, c);
	
	// cache old for copying:
	old = self->info.data;
	olddim = self->info.dim;
	oldchannels = self->info.channels;
	
	// limit [data] size:
	if (s * c > DATA_MAXIMUM_ELEMENTS) {
		s = DATA_MAXIMUM_ELEMENTS/c;
		genlib_report_message("warning: constraining [data] to < 256MB");
	}
	// bytes required:
	sz = sizeof(double) * s * c;
	oldsz = sizeof(double) * olddim * oldchannels;
	
	if (old && sz == oldsz) {
		// no need to re-allocate, just resize
		// careful, audio thread may still be using it:
		if (s > olddim) {
			self->info.channels = c;
			self->info.dim = s;
		} else {
			self->info.dim = s;
			self->info.channels = c;
		}
		
		genlib_set_zero64(self->info.data, s * c);
		return;
		
	} else {
		
		// allocate new:
		replaced = (double *)genlib_sysmem_newptr(sz);
		
		// check allocation:
		if (replaced == 0) {
			genlib_report_error("allocating [data]: out of memory");
			// try to reallocate with a default/minimal size instead:
			if (s > 512 || c > 1) {
				genlib_data_resize((t_genlib_data *)self, 512, 1);
			} else {
				// if this fails, then Max is kaput anyway...
				genlib_data_resize((t_genlib_data *)self, 4, 1);
			}
			return;
		}
		
		// fill with zeroes:
		genlib_set_zero64(replaced, s * c);
		
		// copy in old data:
		if (old) {
			// frames to copy:
			// clamped:
			copydim = olddim > s ? s : olddim;
			// use memcpy if channels haven't changed:
			if (c == oldchannels) {
				copysz = sizeof(double) * copydim * c;
				//post("reset resize (same channels) %p %p, %d", self->info.data, old, copysz);
				memcpy(replaced, old, copysz);
			} else {
				// memcpy won't work if channels have changed,
				// because data is interleaved.
				// clamp channels copied:
				copychannels = oldchannels > c ? c : oldchannels;
				//post("reset resize (different channels) %p %p, %d %d", self->info.data, old, copydim, copychannels);
				for (i = 0; i<copydim; i++) {
					for (j = 0; j<copychannels; j++) {
						replaced[j + i*c] = old[j + i*oldchannels];
					}
				}
			}
		}
		
		// now update info:
		if (old == 0) {
			self->info.data = replaced;
			self->info.dim = s;
			self->info.channels = c;
		} else {
			// need to be careful; the audio thread may still be using it
			// since dsp_gen_data is preserved through edits
			// the order of resizing has to be carefully done
			// to prevent indexing out of bounds
			// (or maybe I'm being too paranoid here...)
			if (oldsz > sz) {
				// shrink size first
				if (s > olddim) {
					self->info.channels = c;
					self->info.dim = s;
				} else {
					self->info.dim = s;
					self->info.channels = c;
				}
				self->info.data = replaced;
			} else {
				// shrink size after
				self->info.data = replaced;
				if (s > olddim) {
					self->info.channels = c;
					self->info.dim = s;
				} else {
					self->info.dim = s;
					self->info.channels = c;
				}
			}
			
			// done with old:
			genlib_sysmem_freeptr(old);
			
		}
		
	}
}

// other notification:
inline void genlib_reset_complete(void *data);


struct Delta {
	double history;
	Delta() { reset(); }
	inline void reset(double init = 0) { history = init; }

	inline double operator()(double in1) {
		double ret = in1 - history;
		history = in1;
		return ret;
	}
};
struct Change {
	double history;
	Change() { reset(); }
	inline void reset(double init = 0) { history = init; }

	inline double operator()(double in1) {
		double ret = in1 - history;
		history = in1;
		return sign(ret);
	}
};

struct Rate {
	double phase, diff, mult, invmult, prev;
	int wantlock, quant;

	Rate() { reset(); }

	inline void reset() {
		phase = diff = prev = 0;
		mult = invmult = 1;
		wantlock = 1;
		quant = 1;
	}

	inline double perform_lock(double in1, double in2) {
		// did multiplier change?
		if (in2 != mult && !an_isnan(in2)) {
			mult = in2;
			invmult = safediv(1., mult);
			wantlock = 1;
		}
		double diff = in1 - prev;

		if (diff < -0.5) {
			diff += 1;
		}
		else if (diff > 0.5) {
			diff -= 1;
		}

		if (wantlock) {
			// recalculate phase
			phase = (in1 - AN_QUANT(in1, quant)) * invmult
				+ AN_QUANT(in1, quant * mult);
			diff = 0;
			wantlock = 0;
		}
		else {
			// diff is always between -0.5 and 0.5
			phase += diff * invmult;
		}

		if (phase > 1. || phase < -0.) {
			phase = phase - (long)(phase);
		}

		prev = in1;

		return phase;
	}

	inline double perform_cycle(double in1, double in2) {
		// did multiplier change?
		if (in2 != mult && !an_isnan(in2)) {
			mult = in2;
			invmult = safediv(1., mult);
			wantlock = 1;
		}
		double diff = in1 - prev;

		if (diff < -0.5) {
			if (wantlock) {
				wantlock = 0;
				phase = in1 * invmult;
				diff = 0;
			}
			else {
				diff += 1;
			}
		}
		else if (diff > 0.5) {
			if (wantlock) {
				wantlock = 0;
				phase = in1 * invmult;
				diff = 0;
			}
			else {
				diff -= 1;
			}
		}

		// diff is always between -0.5 and 0.5
		phase += diff * invmult;

		if (phase > 1. || phase < -0.) {
			phase = phase - (long)(phase);
		}

		prev = in1;

		return phase;
	}

	inline double perform_off(double in1, double in2) {
		// did multiplier change?
		if (in2 != mult && !an_isnan(in2)) {
			mult = in2;
			invmult = safediv(1., mult);
			wantlock = 1;
		}
		double diff = in1 - prev;

		if (diff < -0.5) {
			diff += 1;
		}
		else if (diff > 0.5) {
			diff -= 1;
		}

		phase += diff * invmult;

		if (phase > 1. || phase < -0.) {
			phase = phase - (long)(phase);
		}

		prev = in1;

		return phase;
	}
};

struct DCBlock {
	double x1, y1;
	inline void reset() { x1 = 0; y1 = 0; }

	inline double operator()(double in1) {
		double y = in1 - x1 + y1*0.9997;
		x1 = in1;
		y1 = y;
		return y;
	}
};


struct Phasor {
	double phase;
	Phasor() { reset(); }
	void reset(double v = 0.) { phase = v; }
	inline double operator()(double freq, double invsamplerate) {
		const double pincr = freq * invsamplerate;
		//phase = wrapfew(phase + pincr, 0., 1.); // faster for low frequencies, but explodes with high frequencies
		phase = wrap(phase + pincr, 0., 1.);
		return phase;
	}
};

struct PlusEquals {
	double count;
	PlusEquals() { reset(); }
	void reset(double v = 0.) { count = v; }
	inline double operator()(double incr, double reset, double min, double max) {
		count = reset ? min : wrap(count + incr, min, max);
		return count;
	}
	inline double operator()(double incr = 1., double reset = 0., double min = 0.) {
		count = reset ? min : count + incr;
		return count;
	}
};

struct MulEquals {
	double count;
	MulEquals() { reset(); }
	void reset(double v = 0.) { count = v; }
	inline double operator()(double incr, double reset, double min, double max) {
		count = reset ? min : wrap(fixdenorm(count*incr), min, max);
		return count;
	}
	inline double operator()(double incr = 1., double reset = 0., double min = 0.) {
		count = reset ? min : fixdenorm(count*incr);
		return count;
	}
};

struct Sah {
	double prev, output;
	Sah() { reset(); }
	void reset(double o = 0.) {
		output = prev = o;
	}

	inline double operator()(double in, double trig, double thresh) {
		if (prev <= thresh && trig > thresh) {
			output = in;
		}
		prev = trig;
		return output;
	}
};

struct Train {
	double phase;
	double state;
	Train() { reset(); }
	void reset(double p = 0) { phase = p; state = 0.; }

	inline double operator()(double pulseinterval, double width, double pulsephase) {
		if (width <= 0.) {
			state = 0.;	// no pulse!
		}
		else if (width >= 1.) {
			state = 1.; // constant pulse!
		}
		else {
			const double interval = maximum(pulseinterval, 1.);	// >= 1.
			const double p1 = clamp(pulsephase, 0., 1.);	// [0..1]
			const double p2 = p1 + width;						// (p1..p1+1)
			const double pincr = 1. / interval;				// (0..1]
			phase += pincr;									// +ve
			if (state) {	// on:
				if (phase > p2) {
					state = 0.;				// turn off
					phase -= (int)(1. + phase - p2);	// wrap phase back down
				}
			}
			else {		// off:
				if (phase > p1) {
					state = 1.;				// turn on.
				}
			}
		}
		return state;
	}
};

struct Noise {
	unsigned long last;
	static long uniqueTickCount(void) {
		static long lasttime = 0;
		long time = genlib_ticks();
		return (time <= lasttime) ? (++lasttime) : (lasttime = time);
	}

	Noise() { reset(); }
	Noise(double seed) { reset(seed); }
	void reset() { last = uniqueTickCount() * uniqueTickCount(); }
	void reset(double seed) { last = (unsigned long)seed; }

	inline double operator()() {
		last = 1664525L * last + 1013904223L;
		unsigned long itemp = 0x3f800000 | (0x007fffff & last);
		return ((*(float *)&itemp) * 2.0) - 3.0;
	}
};

struct genDelay {
	double * memory;
	uint64_t size, wrap, maxdelay;
	uint64_t reader, writer;
	
	genDelay() : memory(0) {
		size = wrap = maxdelay = 0;
		reader = writer = 0;
	}
	~genDelay() {
		if (memory) {
			genlib_sysmem_freeptr(memory);
		}
	}
	
	inline void reset(const char * name, long d) {
		// if needed, acquire the Data's global reference:
		if (memory == 0) {
			
			// scale maxdelay to next highest power of 2:
			maxdelay = d;
			size = uint64_t(maximum(double(maxdelay),2.));
			size = next_power_of_two(size);
			wrap = size-1;
			
			memory = (double *)genlib_sysmem_newptr(sizeof(double) * (size_t)size);
			writer = 0;
		}
		
		// subsequent reset should zero the memory & heads:
		genlib_set_zero64(memory, (long)size);
		writer = 0;
		reader = writer;
	}
	
	// called at bufferloop end, updates read pointer time
	inline void step() {
		reader++;
		if (reader >= size) reader = 0;
	}
	
	inline void write(double x) {
		writer = reader;	// update write ptr
		memory[writer] = x;
	}
	
	inline double read_step(double d) {
		// extra half for nice rounding:
		// min 1 sample delay for read before write (r != w)
		const double r = double(size + reader) - clamp(d-0.5, double(reader != writer), double(maxdelay));
		long r1 = long(r);
		return memory[r1 & wrap];
	}
	
	inline double read_linear(double d) {
		// min 1 sample delay for read before write (r != w)
		double c = clamp(d, double(reader != writer), double(maxdelay));
		const double r = double(size + reader) - c;
		long r1 = long(r);
		long r2 = r1+1;
		double a = r - (double)r1;
		double x = memory[r1 & wrap];
		double y = memory[r2 & wrap];
		return linear_interp(a, x, y);
	}
	
	inline double read_cosine(double d) {
		// min 1 sample delay for read before write (r != w)
		const double r = double(size + reader) - clamp(d, double(reader != writer), double(maxdelay));
		long r1 = long(r);
		long r2 = r1+1;
		double a = r - (double)r1;
		double x = memory[r1 & wrap];
		double y = memory[r2 & wrap];
		return cosine_interp(a, x, y);
	}
	
	// cubic requires extra sample of compensation:
	inline double read_cubic(double d) {
		// min 1 sample delay for read before write (r != w)
		// plus extra 1 sample compensation for 4-point interpolation
		const double r = double(size + reader) - clamp(d, 1.+double(reader != writer), double(maxdelay));
		long r1 = long(r);
		long r2 = r1+1;
		long r3 = r1+2;
		long r4 = r1+3;
		double a = r - (double)r1;
		double w = memory[r1 & wrap];
		double x = memory[r2 & wrap];
		double y = memory[r3 & wrap];
		double z = memory[r4 & wrap];
		return cubic_interp(a, w, x, y, z);
	}
	
	// spline requires extra sample of compensation:
	inline double read_spline(double d) {
		// min 1 sample delay for read before write (r != w)
		// plus extra 1 sample compensation for 4-point interpolation
		const double r = double(size + reader) - clamp(d, 1.+double(reader != writer), double(maxdelay));
		long r1 = long(r);
		long r2 = r1+1;
		long r3 = r1+2;
		long r4 = r1+3;
		double a = r - (double)r1;
		double w = memory[r1 & wrap];
		double x = memory[r2 & wrap];
		double y = memory[r3 & wrap];
		double z = memory[r4 & wrap];
		return spline_interp(a, w, x, y, z);
	}
};

template<typename T>
struct DataInterface {
	long dim, channels;
	T * mData;
	int modified;
	
	DataInterface() : dim(0), channels(1), mData(0), modified(0) {}
	
	// raw reading/writing/overdubbing (internal use only, no bounds checking)
	inline double read(long index, long channel=0) const {
		return mData[channel+index*channels];
	}
	inline void write(double value, long index, long channel=0) {
		mData[channel+index*channels] = value;
		modified = 1;
	}
	inline void overdub(double value, long index, long channel=0) {
		mData[channel+index*channels] += value;
		modified = 1;
	}
	
	// averaging overdub (used by splat)
	inline void blend(double value, long index, long channel, double alpha) {
		long offset = channel+index*channels;
		const double old = mData[offset];
		mData[offset] = old + alpha * (value - old);
		modified = 1;
	}
	
	inline void read_ok(long index, long channel=0, bool ok=1) const {
		return ok ? mData[channel+index*channels] : T(0);
	}
	inline void write_ok(double value, long index, long channel=0, bool ok=1) {
		if (ok) mData[channel+index*channels] = value;
	}
	inline void overdub_ok(double value, long index, long channel=0, bool ok=1) {
		if (ok) mData[channel+index*channels] += value;
	}
	
	// Bounds strategies:
	inline long index_clamp(long index) const { return clamp(index, 0, dim-1); }
	inline long index_wrap(long index) const { return wrap(index, 0, dim); }
	inline long index_fold(long index) const { return fold(index, 0, dim); }
	inline bool index_oob(long index) const { return (index < 0 || index >= dim); }
	inline bool index_inbounds(long index) const { return (index >=0 && index < dim); }
	
	// channel bounds:
	inline long channel_clamp(long c) const { return clamp(c, 0, channels-1); }
	inline long channel_wrap(long c) const { return wrap(c, 0, channels); }
	inline long channel_fold(long c) const { return fold(c, 0, channels); }
	inline bool channel_oob(long c) const { return (c < 0 || c >= channels); }
	inline bool channel_inbounds(long c) const { return !channel_oob(c); }
	
	// Indexing strategies:
	// [0..1] -> [0..(dim-1)]
	inline double phase2index(double phase) const { return phase * (dim-1); }
	// [0..1] -> [min..max]
	inline double subphase2index(double phase, long min, long max) const {
		min = index_clamp(min);
		max = index_clamp(max);
		return min + phase * (max-min);
	}
	// [-1..1] -> [0..(dim-1)]
	inline double signal2index(double signal) const { return phase2index((signal+1.) * 0.5); }
	
	inline double peek(double index, long channel=0) const {
		const long i = (long)index;
		if (index_oob(i) || channel_oob(channel)) {
			return 0.;
		} else {
			return read(i, channel);
		}
	}
	
	inline double index(double index, long channel=0) const {
		channel = channel_clamp(channel);
		// no-interp:
		long i = (long)index;
		// bound:
		i = index_clamp(i);
		return read(i, channel);
	}
	
	inline double cell(double index, long channel=0) const {
		channel = channel_clamp(channel);
		// no-interp:
		long i = (long)index;
		// bound:
		i = index_wrap(i);
		return read(i, channel);
	}
	
	inline double cycle(double phase, long channel=0) const {
		channel = channel_clamp(channel);
		double index = phase2index(phase);
		// interp:
		long i1 = (long)index;
		long i2 = i1+1;
		const double alpha = index - (double)i1;
		// bound:
		i1 = index_wrap(i1);
		i2 = index_wrap(i2);
		// interp:
		double v1 = read(i1, channel);
		double v2 = read(i2, channel);
		return mix(v1, v2, alpha);
	}
	
	inline double lookup(double signal, long channel=0) const {
		channel = channel_clamp(channel);
		double index = signal2index(signal);
		// interp:
		long i1 = (long)index;
		long i2 = i1+1;
		double alpha = index - (double)i1;
		// bound:
		i1 = index_clamp(i1);
		i2 = index_clamp(i2);
		// interp:
		double v1 = read(i1, channel);
		double v2 = read(i2, channel);
		return mix(v1, v2, alpha);
	}
	
	inline void poke(double value, double index, long channel=0) {
		const long i = (long)index;
		if (!(index_oob(i) || channel_oob(channel))) {
			write(fixdenorm(value), i, channel);
		}
	}
	
	inline void splat_adding(double value, double phase, long channel=0) {
		const double valuef = fixdenorm(value);
		channel = channel_clamp(channel);
		double index = phase2index(phase);
		// interp:
		long i1 = (long)index;
		long i2 = i1+1;
		const double alpha = index - (double)i1;
		// bound:
		i1 = index_wrap(i1);
		i2 = index_wrap(i2);
		// interp:
		overdub(valuef*(1.-alpha), i1, channel);
		overdub(valuef*alpha,      i2, channel);
	}
	
	inline void splat(double value, double phase, long channel=0) {
		const double valuef = fixdenorm(value);
		channel = channel_clamp(channel);
		double index = phase2index(phase);
		// interp:
		long i1 = (long)index;
		long i2 = i1+1;
		const double alpha = index - (double)i1;
		// bound:
		i1 = index_wrap(i1);
		i2 = index_wrap(i2);
		// interp:
		const double v1 = read(i1, channel);
		const double v2 = read(i2, channel);
		write(v1 + (1.-alpha)*(valuef-v1), i1, channel);
		write(v2 + (alpha)*(valuef-v2), i2, channel);
	}
};

// DATA_MAXIMUM_ELEMENTS * 8 bytes = 256 mb limit
#define DATA_MAXIMUM_ELEMENTS	(33554432)

struct Data : public DataInterface<double> {
	Data() : DataInterface<double>()  {}
	~Data() {
		if (mData) genlib_sysmem_freeptr(mData);
		mData = 0;
	}
	
	void reset(long s, long c) {
		mData=0;
		resize(s, c);
	}
	
	void resize(long s, long c) {
		if (s * c > DATA_MAXIMUM_ELEMENTS) {
			s = DATA_MAXIMUM_ELEMENTS/c;
			genlib_report_message("warning: resizing data to < 256MB");
		}
		if (mData) {
			genlib_sysmem_resizeptr(mData, sizeof(double) * s * c);
		} else {
			mData = (double *)genlib_sysmem_newptr(sizeof(double) * s * c);
		}
		if (!mData) {
			genlib_report_error("out of memory");
			resize(512, 1);
			return;
		} else {
			dim = s;
			channels = c;
		}
		genlib_set_zero64(mData, dim * channels);
	}
};

struct SineData : public Data {
	SineData() : Data() {
		const int costable_size = 1 << 14;	// 14 bit index (noise floor at around -156 dB)
		mData=0;
		resize(costable_size, 1);
		for (int i=0; i<dim; i++) {
			mData[i] = cos(i * AN_PI * 2. / (double)(dim));
		}
	}
};

template<typename T>
inline int dim(const T& data) { return data.dim; }

template<typename T>
inline int channels(const T& data) { return data.channels; }

// used by cycle when no buffer/data is specified:
struct SineCycle {
	
	uint32_t phasei, pincr;
	double f2i;
	
	void reset(double samplerate, double init = 0) {
		phasei = uint32_t(init * 4294967296.0);
		pincr = 0;
		f2i = 4294967296.0 / samplerate;
	}
	
	inline void freq(double f) {
		pincr = uint32_t(f * f2i);
	}
	
	inline void phase(double f) {
		phasei = uint32_t(f * 4294967296.0);
	}
	
	inline double phase() const {
		return phasei * 0.232830643653869629e-9;
	}
	
	template<typename T>
	inline double operator()(const DataInterface<T>& buf) {
		T * data = buf.mData;
		// divide uint32_t range down to buffer size (32-bit to 14-bit)
		uint32_t idx = phasei >> 18;
		// compute fractional portion and divide by 18-bit range
		const double frac = (phasei & 262143) * 3.81471181759574e-6;
		// index safely in 14-bit range:
		const double y0 = data[idx];
		const double y1 = data[(idx+1) & 16383];
		const double y = linear_interp(frac, y0, y1);
		phasei += pincr;
		return y;
	}
};


struct AmbiDomain {
	double w, x, y, z;
};

// The Reverb struct contains all the Reverb and procedures for the gendsp kernel
typedef struct Reverb {
	genDelay m_delay_10;
	genDelay m_delay_8;
	genDelay m_delay_7;
	genDelay m_delay_6;
	genDelay m_delay_9;
	genDelay m_delay_11;
	genDelay m_delay_14;
	genDelay m_delay_12;
	genDelay m_delay_5;
	genDelay m_delay_16;
	genDelay m_delay_15;
	genDelay m_delay_13;
	double m_damping_17;
	double m_tail_18;
	double m_spread_20;
	double m_early_19;
	double m_roomsize_22;
	double m_history_4;
	double samplerate;
	double m_revtime_21;
	double m_history_1;
	double m_history_3;
	double m_history_2;

	// re-initialize all member variables;
	inline void reset(double __sr) {
		samplerate = __sr;
		m_history_1 = 0;
		m_history_2 = 0;
		m_history_3 = 0;
		m_history_4 = 0;
		m_delay_5.reset("m_delay_5", 7000);
		m_delay_6.reset("m_delay_6", 5000);
		m_delay_7.reset("m_delay_7", 16000);
		m_delay_8.reset("m_delay_8", 15000);
		m_delay_9.reset("m_delay_9", 6000);
		m_delay_10.reset("m_delay_10", 48000);
		m_delay_11.reset("m_delay_11", 12000);
		m_delay_12.reset("m_delay_12", 10000);
		m_delay_13.reset("m_delay_13", 48000);
		m_delay_14.reset("m_delay_14", 48000);
		m_delay_15.reset("m_delay_15", 48000);
		m_delay_16.reset("m_delay_16", 48000);
		m_damping_17 = 0.9;
		m_tail_18 = 0.1;
		m_early_19 = 0.1;
		m_spread_20 = 133;
		m_revtime_21 = 150;
		m_roomsize_22 = 1000;
	};
	// the signal processing routine;
	inline void perform(const int frames, AmbiDomain * ambi_bus) {
		double expr_22855 = safepow(0.001, safediv(1, (m_revtime_21 * 44100)));
		double expr_22856 = safediv((m_roomsize_22 * 44100), 340);
		double mul_22820 = (expr_22856 * 0.81649);
		double expr_22849 = (-safepow(expr_22855, mul_22820));
		double mul_22821 = (expr_22856 * 1);
		double expr_22854 = (-safepow(expr_22855, mul_22821));
		double mul_22819 = (expr_22856 * 0.7071);
		double expr_22848 = (-safepow(expr_22855, mul_22819));
		double mul_22815 = (expr_22856 * 0.000527);
		int int_22814 = int(mul_22815);
		double mul_22818 = (expr_22856 * 0.63245);
		double expr_22847 = (-safepow(expr_22855, mul_22818));
		double mul_22783 = (m_spread_20 * 0.376623);
		double add_22782 = (mul_22783 + 931);
		double rsub_22779 = (1341 - add_22782);
		double mul_22790 = (int_22814 * rsub_22779);
		double mul_22755 = (m_spread_20 * -0.380445);
		double add_22754 = (mul_22755 + 931);
		double rsub_22751 = (1341 - add_22754);
		double mul_22764 = (int_22814 * rsub_22751);
		double add_22744 = (expr_22856 + 5);
		double expr_22850 = safepow(expr_22855, add_22744);
		double mul_22750 = (expr_22856 * 0.41);
		double add_22747 = (mul_22750 + 5);
		double mul_22749 = (expr_22856 * 0.3);
		double add_22746 = (mul_22749 + 5);
		double mul_22748 = (expr_22856 * 0.155);
		double add_22745 = (mul_22748 + 5);
		double expr_22852 = safepow(expr_22855, add_22746);
		double expr_22853 = safepow(expr_22855, add_22747);
		double expr_22851 = safepow(expr_22855, add_22745);
		double mul_22813 = (expr_22856 * 0.110732);
		double mul_22799 = (m_spread_20 * 0.125541);
		double add_22781 = (mul_22799 + 369);
		double rsub_22780 = (add_22782 - add_22781);
		double mul_22797 = (int_22814 * rsub_22780);
		double mul_22757 = (m_spread_20 * -0.568366);
		double add_22753 = (mul_22757 + 369);
		double rsub_22752 = (add_22754 - add_22753);
		double mul_22771 = (int_22814 * rsub_22752);
		double add_22798 = (mul_22799 + 159);
		double mul_22806 = (int_22814 * add_22798);
		double add_22756 = (mul_22757 + 159);
		double mul_22778 = (int_22814 * add_22756);
		// the main sample loop;
		for (int n = 0; n<frames; n++) {
			const double in1 = ambi_bus[n].w;

			double tap_22730 = m_delay_16.read_linear(mul_22820);
			double mul_22726 = (tap_22730 * expr_22849);
			double mix_22873 = (mul_22726 + (m_damping_17 * (m_history_4 - mul_22726)));
			double mix_22728 = mix_22873;
			double tap_22826 = m_delay_15.read_linear(mul_22821);
			double mul_22817 = (tap_22826 * expr_22854);
			double mix_22874 = (mul_22817 + (m_damping_17 * (m_history_3 - mul_22817)));
			double mix_22824 = mix_22874;
			double tap_22724 = m_delay_14.read_linear(mul_22819);
			double mul_22720 = (tap_22724 * expr_22848);
			double mix_22875 = (mul_22720 + (m_damping_17 * (m_history_2 - mul_22720)));
			double mix_22722 = mix_22875;
			double tap_22718 = m_delay_13.read_linear(mul_22818);
			double mul_22714 = (tap_22718 * expr_22847);
			double mix_22876 = (mul_22714 + (m_damping_17 * (m_history_1 - mul_22714)));
			double mix_22716 = mix_22876;
			double tap_22789 = m_delay_12.read_linear(mul_22790);
			double tap_22763 = m_delay_11.read_linear(mul_22764);
			double sub_22707 = (mix_22824 - mix_22728);
			double sub_22704 = (mix_22722 - mix_22716);
			double sub_22703 = (sub_22707 - sub_22704);
			double mul_22687 = (sub_22703 * 0.5);
			double add_22708 = (mix_22824 + mix_22728);
			double add_22706 = (mix_22722 + mix_22716);
			double sub_22705 = (add_22708 - add_22706);
			double mul_22688 = (sub_22705 * 0.5);
			double add_22702 = (sub_22707 + sub_22704);
			double rsub_22700 = (0 - add_22702);
			double mul_22686 = (rsub_22700 * 0.5);
			double mul_22787 = (tap_22789 * 0.625);
			double add_22701 = (add_22708 + add_22706);
			double mul_22685 = (add_22701 * 0.5);
			double mul_22761 = (tap_22763 * 0.625);
			double tap_22732 = m_delay_10.read_linear(add_22747);
			double tap_22733 = m_delay_10.read_linear(add_22746);
			double tap_22734 = m_delay_10.read_linear(add_22745);
			double tap_22735 = m_delay_10.read_linear(add_22744);
			double mul_22740 = (tap_22733 * expr_22852);
			double add_22711 = (mul_22687 + mul_22740);
			double mul_22742 = (tap_22732 * expr_22853);
			double add_22712 = (mul_22688 + mul_22742);
			double mul_22738 = (tap_22734 * expr_22851);
			double add_22710 = (mul_22686 + mul_22738);
			double mul_22736 = (tap_22735 * expr_22850);
			double add_22709 = (mul_22685 + mul_22736);
			double tap_22812 = m_delay_9.read_linear(mul_22813);
			double tap_22796 = m_delay_8.read_linear(mul_22797);
			double tap_22770 = m_delay_7.read_linear(mul_22771);
			double mul_22810 = (tap_22812 * 0.75);
			double tap_22805 = m_delay_6.read_linear(mul_22806);
			double mul_22794 = (tap_22796 * 0.625);
			double tap_22777 = m_delay_5.read_linear(mul_22778);
			double mul_22768 = (tap_22770 * 0.625);
			double mul_22679 = in1;
			double sub_22809 = (mul_22679 - mul_22810);
			double mul_22808 = (sub_22809 * 0.75);
			double add_22807 = (mul_22808 + tap_22812);
			double mul_22803 = (tap_22805 * 0.75);
			double mul_22775 = (tap_22777 * 0.75);
			double mul_22699 = (mul_22688 * m_tail_18);
			double mul_22697 = (mul_22686 * m_tail_18);
			double add_22684 = (mul_22699 + mul_22697);
			double mul_22698 = (mul_22687 * m_tail_18);
			double mul_22696 = (mul_22685 * m_tail_18);
			double add_22683 = (mul_22698 + mul_22696);
			double sub_22691 = (add_22684 - add_22683);
			double mul_22695 = (mul_22742 * m_early_19);
			double mul_22693 = (mul_22738 * m_early_19);
			double add_22682 = (mul_22695 + mul_22693);
			double mul_22694 = (mul_22740 * m_early_19);
			double mul_22692 = (mul_22736 * m_early_19);
			double add_22681 = (mul_22694 + mul_22692);
			double sub_22690 = (add_22682 - add_22681);
			double add_22678 = (sub_22691 + sub_22690);
			double add_22689 = (add_22678 + ambi_bus[n].y);
			double sub_22802 = (add_22689 - mul_22803);
			double mul_22801 = (sub_22802 * 0.75);
			double add_22800 = (mul_22801 + tap_22805);
			double sub_22793 = (add_22800 - mul_22794);
			double mul_22792 = (sub_22793 * 0.625);
			double add_22791 = (mul_22792 + tap_22796);
			double sub_22786 = (add_22791 - mul_22787);
			double mul_22785 = (sub_22786 * 0.625);
			double add_22680 = (add_22678 + ambi_bus[n].x);
			double sub_22774 = (add_22680 - mul_22775);
			double mul_22773 = (sub_22774 * 0.75);
			double add_22772 = (mul_22773 + tap_22777);
			double sub_22767 = (add_22772 - mul_22768);
			double mul_22766 = (sub_22767 * 0.625);
			double add_22765 = (mul_22766 + tap_22770);
			double sub_22760 = (add_22765 - mul_22761);
			double mul_22759 = (sub_22760 * 0.625);

			double outy = (mul_22785 + tap_22789);
			double outx = (mul_22759 + tap_22763);

			double history_22727_next_22869 = mix_22728;
			double history_22823_next_22870 = mix_22824;
			double history_22721_next_22871 = mix_22722;
			double history_22715_next_22872 = mix_22716;
			m_delay_16.write(add_22711);
			m_delay_15.write(add_22712);
			m_delay_14.write(add_22710);
			m_delay_13.write(add_22709);
			m_delay_12.write(sub_22786);
			m_delay_11.write(sub_22760);
			m_delay_10.write(add_22807);
			m_delay_9.write(sub_22809);
			m_delay_8.write(sub_22793);
			m_delay_7.write(sub_22767);
			m_delay_6.write(sub_22802);
			m_delay_5.write(sub_22774);
			m_history_4 = history_22727_next_22869;
			m_history_3 = history_22823_next_22870;
			m_history_2 = history_22721_next_22871;
			m_history_1 = history_22715_next_22872;
			m_delay_5.step();
			m_delay_6.step();
			m_delay_7.step();
			m_delay_8.step();
			m_delay_9.step();
			m_delay_10.step();
			m_delay_11.step();
			m_delay_12.step();
			m_delay_13.step();
			m_delay_14.step();
			m_delay_15.step();
			m_delay_16.step();
			// assign results to output buffer;
			ambi_bus[n].x += outx * 0.2;
			ambi_bus[n].z += outy * 0.2;

		};
	};
	inline void set_damping(double local_value) {
		m_damping_17 = (local_value < 0 ? 0 : (local_value > 1 ? 1 : local_value));
	};
	inline void set_tail(double local_value) {
		m_tail_18 = (local_value < 0 ? 0 : (local_value > 1 ? 1 : local_value));
	};
	inline void set_early(double local_value) {
		m_early_19 = (local_value < 0 ? 0 : (local_value > 1 ? 1 : local_value));
	};
	inline void set_spread(double local_value) {
		m_spread_20 = (local_value < 0 ? 0 : (local_value > 100 ? 100 : local_value));
	};
	inline void set_revtime(double local_value) {
		m_revtime_21 = (local_value < 0.1 ? 0.1 : (local_value > 1 ? 1 : local_value));
	};
	inline void set_roomsize(double local_value) {
		m_roomsize_22 = (local_value < 0.1 ? 0.1 : (local_value > 300 ? 300 : local_value));
	};

} Reverb;

#endif // GENLIB_COMMON_H

