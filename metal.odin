#+build darwin
/* Metal GPU backend: quantized GEMV kernels for Apple Silicon */

package main

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:os"
import MTL "vendor:darwin/Metal"

PAGE_SIZE :: 16384 // Apple Silicon page size (for no-copy buffer alignment)

metal_enabled: bool
m_device: ^MTL.Device
m_queue: ^MTL.CommandQueue
m_weights: ^MTL.Buffer
m_mmap_base: uintptr

m_pso_f32: ^MTL.ComputePipelineState
m_pso_f16: ^MTL.ComputePipelineState
m_pso_q8_0: ^MTL.ComputePipelineState
m_pso_q4_0: ^MTL.ComputePipelineState
m_pso_q4_1: ^MTL.ComputePipelineState
m_pso_q5_0: ^MTL.ComputePipelineState
m_pso_q5_1: ^MTL.ComputePipelineState
m_pso_q4_k: ^MTL.ComputePipelineState
m_pso_q6_k: ^MTL.ComputePipelineState

m_pso_rmsnorm: ^MTL.ComputePipelineState
m_pso_rope: ^MTL.ComputePipelineState
m_pso_attn: ^MTL.ComputePipelineState
m_pso_swiglu: ^MTL.ComputePipelineState
m_pso_residual: ^MTL.ComputePipelineState

// Stage B: resident run-state + KV cache on the GPU (shared memory).
m_b_x: ^MTL.Buffer
m_b_xb: ^MTL.Buffer
m_b_xb2: ^MTL.Buffer
m_b_xb3: ^MTL.Buffer
m_b_q: ^MTL.Buffer
m_b_hb: ^MTL.Buffer
m_b_hb2: ^MTL.Buffer
m_b_att: ^MTL.Buffer
m_b_logits: ^MTL.Buffer
m_b_kc: ^MTL.Buffer
m_b_vc: ^MTL.Buffer
m_cfg: Config

MSL_SRC := `
#include <metal_stdlib>
using namespace metal;

struct Dims { uint n; uint d; };

inline void get_scale_min_k4(int j, device const uchar *q, thread uchar &d, thread uchar &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j + 0] >> 6) << 4);
    }
}

kernel void gemv_f32(device const uchar *wb [[buffer(0)]],
                     device const float *x  [[buffer(1)]],
                     device float *out      [[buffer(2)]],
                     constant Dims &dim     [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]]) {
    if (row >= dim.d) return;
    device const float *w = (device const float*)wb + (ulong)row * dim.n;
    float sum = 0.0;
    for (uint j = tid; j < dim.n; j += 32) sum += w[j] * x[j];
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}

kernel void gemv_f16(device const uchar *wb [[buffer(0)]],
                     device const float *x  [[buffer(1)]],
                     device float *out      [[buffer(2)]],
                     constant Dims &dim     [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]]) {
    if (row >= dim.d) return;
    device const half *w = (device const half*)wb + (ulong)row * dim.n;
    float sum = 0.0;
    for (uint j = tid; j < dim.n; j += 32) sum += (float)w[j] * x[j];
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}

kernel void gemv_q8_0(device const uchar *wb [[buffer(0)]],
                      device const float *x  [[buffer(1)]],
                      device float *out      [[buffer(2)]],
                      constant Dims &dim     [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    if (row >= dim.d) return;
    uint nb = dim.n / 32;
    device const uchar *wr = wb + (ulong)row * nb * 34;
    float sum = 0.0;
    for (uint b = tid; b < nb; b += 32) {
        device const uchar *blk = wr + b * 34;
        float dd = (float)(*(device const half*)blk);
        device const char *qs = (device const char*)(blk + 2);
        uint base = b * 32;
        for (uint j = 0; j < 32; ++j) sum += dd * (float)qs[j] * x[base + j];
    }
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}

kernel void gemv_q4_0(device const uchar *wb [[buffer(0)]],
                      device const float *x  [[buffer(1)]],
                      device float *out      [[buffer(2)]],
                      constant Dims &dim     [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    if (row >= dim.d) return;
    uint nb = dim.n / 32;
    device const uchar *wr = wb + (ulong)row * nb * 18;
    float sum = 0.0;
    for (uint b = tid; b < nb; b += 32) {
        device const uchar *blk = wr + b * 18;
        float dd = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2;
        uint base = b * 32;
        for (uint j = 0; j < 16; ++j) {
            float x0 = (float)((int)(qs[j] & 0xF) - 8);
            float x1 = (float)((int)(qs[j] >> 4) - 8);
            sum += x0 * dd * x[base + j];
            sum += x1 * dd * x[base + j + 16];
        }
    }
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}

kernel void gemv_q4_1(device const uchar *wb [[buffer(0)]],
                      device const float *x  [[buffer(1)]],
                      device float *out      [[buffer(2)]],
                      constant Dims &dim     [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    if (row >= dim.d) return;
    uint nb = dim.n / 32;
    device const uchar *wr = wb + (ulong)row * nb * 20;
    float sum = 0.0;
    for (uint b = tid; b < nb; b += 32) {
        device const uchar *blk = wr + b * 20;
        float dd = (float)(*(device const half*)blk);
        float mm = (float)(*(device const half*)(blk + 2));
        device const uchar *qs = blk + 4;
        uint base = b * 32;
        for (uint j = 0; j < 16; ++j) {
            sum += ((float)(qs[j] & 0xF) * dd + mm) * x[base + j];
            sum += ((float)(qs[j] >> 4)  * dd + mm) * x[base + j + 16];
        }
    }
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}

kernel void gemv_q5_0(device const uchar *wb [[buffer(0)]],
                      device const float *x  [[buffer(1)]],
                      device float *out      [[buffer(2)]],
                      constant Dims &dim     [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    if (row >= dim.d) return;
    uint nb = dim.n / 32;
    device const uchar *wr = wb + (ulong)row * nb * 22;
    float sum = 0.0;
    for (uint b = tid; b < nb; b += 32) {
        device const uchar *blk = wr + b * 22;
        float dd = (float)(*(device const half*)blk);
        uint qh = (uint)blk[2] | ((uint)blk[3] << 8) | ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
        device const uchar *qs = blk + 6;
        uint base = b * 32;
        for (uint j = 0; j < 16; ++j) {
            uchar xh0 = (uchar)((qh >> j) << 4) & 0x10;
            uchar xh1 = (uchar)(qh >> (j + 12)) & 0x10;
            float x0 = (float)((int)((qs[j] & 0xF) | xh0) - 16);
            float x1 = (float)((int)((qs[j] >> 4)  | xh1) - 16);
            sum += x0 * dd * x[base + j];
            sum += x1 * dd * x[base + j + 16];
        }
    }
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}

kernel void gemv_q5_1(device const uchar *wb [[buffer(0)]],
                      device const float *x  [[buffer(1)]],
                      device float *out      [[buffer(2)]],
                      constant Dims &dim     [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    if (row >= dim.d) return;
    uint nb = dim.n / 32;
    device const uchar *wr = wb + (ulong)row * nb * 24;
    float sum = 0.0;
    for (uint b = tid; b < nb; b += 32) {
        device const uchar *blk = wr + b * 24;
        float dd = (float)(*(device const half*)blk);
        float mm = (float)(*(device const half*)(blk + 2));
        uint qh = (uint)blk[4] | ((uint)blk[5] << 8) | ((uint)blk[6] << 16) | ((uint)blk[7] << 24);
        device const uchar *qs = blk + 8;
        uint base = b * 32;
        for (uint j = 0; j < 16; ++j) {
            uchar xh0 = (uchar)((qh >> j) << 4) & 0x10;
            uchar xh1 = (uchar)(qh >> (j + 12)) & 0x10;
            float x0 = (float)((qs[j] & 0xF) | xh0);
            float x1 = (float)((qs[j] >> 4)  | xh1);
            sum += (x0 * dd + mm) * x[base + j];
            sum += (x1 * dd + mm) * x[base + j + 16];
        }
    }
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}

kernel void gemv_q4_k(device const uchar *wb [[buffer(0)]],
                      device const float *x  [[buffer(1)]],
                      device float *out      [[buffer(2)]],
                      constant Dims &dim     [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    if (row >= dim.d) return;
    uint nsb = dim.n / 256;
    device const uchar *wr = wb + (ulong)row * nsb * 144;
    float sum = 0.0;
    for (uint sb = tid; sb < nsb; sb += 32) {
        device const uchar *blk = wr + sb * 144;
        float dall = (float)(*(device const half*)blk);
        float dmin = (float)(*(device const half*)(blk + 2));
        device const uchar *scales = blk + 4;
        device const uchar *qs = blk + 16;
        uint ybase = sb * 256;
        int is = 0;
        uint qoff = 0;
        for (uint j = 0; j < 256; j += 64) {
            uchar sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            float d1 = dall * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = dall * sc, m2 = dmin * m;
            for (uint l = 0; l < 32; ++l)
                sum += (d1 * (float)(qs[qoff + l] & 0xF) - m1) * x[ybase + j + l];
            for (uint l = 0; l < 32; ++l)
                sum += (d2 * (float)(qs[qoff + l] >> 4) - m2) * x[ybase + j + 32 + l];
            qoff += 32;
            is += 2;
        }
    }
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}

kernel void gemv_q6_k(device const uchar *wb [[buffer(0)]],
                      device const float *x  [[buffer(1)]],
                      device float *out      [[buffer(2)]],
                      constant Dims &dim     [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    if (row >= dim.d) return;
    uint nsb = dim.n / 256;
    device const uchar *wr = wb + (ulong)row * nsb * 210;
    float sum = 0.0;
    for (uint sb = tid; sb < nsb; sb += 32) {
        device const uchar *blk = wr + sb * 210;
        device const uchar *ql = blk;
        device const uchar *qh = blk + 128;
        device const char  *sc = (device const char*)(blk + 192);
        float dall = (float)(*(device const half*)(blk + 208));
        uint ybase = sb * 256;
        for (int hf = 0; hf < 2; ++hf) {
            uint qlo = hf * 64, qho = hf * 32, sco = hf * 8, yo = ybase + hf * 128;
            for (int l = 0; l < 32; ++l) {
                int isc = l / 16;
                int q1 = (int)((ql[qlo + l]      & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[qlo + l]      >> 4)  | (((qh[qho + l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[qlo + l + 32] >> 4)  | (((qh[qho + l] >> 6) & 3) << 4)) - 32;
                sum += dall * (float)sc[sco + isc + 0] * (float)q1 * x[yo + l];
                sum += dall * (float)sc[sco + isc + 2] * (float)q2 * x[yo + l + 32];
                sum += dall * (float)sc[sco + isc + 4] * (float)q3 * x[yo + l + 64];
                sum += dall * (float)sc[sco + isc + 6] * (float)q4 * x[yo + l + 96];
            }
        }
    }
    sum = simd_sum(sum);
    if (tid == 0) out[row] = sum;
}

// ---- Stage B elementwise / attention kernels ----

struct NormP { uint size; float eps; };
struct RopeP { uint head_dim; uint pos; float rope_freq; };
struct AttnP { uint head_dim; uint kv_dim; uint kv_mul; uint pos; uint seq_len; };

// One threadgroup (32 lanes) per vector; grp selects the vector slice.
kernel void rmsnorm(device const float *inp [[buffer(0)]],
                    device float *outp       [[buffer(1)]],
                    device const float *w    [[buffer(2)]],
                    constant NormP &P        [[buffer(3)]],
                    uint grp [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]]) {
    device const float *xi = inp + (ulong)grp * P.size;
    device float *oi = outp + (ulong)grp * P.size;
    float ss = 0.0;
    for (uint j = tid; j < P.size; j += 32) ss += xi[j] * xi[j];
    ss = simd_sum(ss);
    ss = ss / (float)P.size + P.eps;
    ss = rsqrt(ss);
    for (uint j = tid; j < P.size; j += 32) oi[j] = w[j] * (ss * xi[j]);
}

// RoPE over vec[h*head_dim + i]; grid = nheads * (head_dim/2).
kernel void rope(device float *vec   [[buffer(0)]],
                 constant RopeP &P   [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    uint half_d = P.head_dim / 2;
    uint h = gid / half_d;
    uint i = gid % half_d;
    device float *q = vec + (ulong)h * P.head_dim;
    float freq = 1.0 / pow(P.rope_freq, (float)i / (float)half_d);
    float c = cos((float)P.pos * freq);
    float s = sin((float)P.pos * freq);
    float x0 = q[i];
    float y0 = q[i + half_d];
    q[i] = x0 * c - y0 * s;
    q[i + half_d] = x0 * s + y0 * c;
}

// One threadgroup (128 lanes) per head. kc/vc are bound at the layer base.
kernel void attention(device const float *q   [[buffer(0)]],
                      device const float *kc  [[buffer(1)]],
                      device const float *vc  [[buffer(2)]],
                      device float *att       [[buffer(3)]],
                      device float *xb3       [[buffer(4)]],
                      constant AttnP &P       [[buffer(5)]],
                      uint h   [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    const uint NT = 128;
    threadgroup float tmp[NT];
    device const float *qh = q + (ulong)h * P.head_dim;
    device float *atth = att + (ulong)h * P.seq_len;
    uint kvh = h / P.kv_mul;
    float scale = rsqrt((float)P.head_dim);

    for (uint t = tid; t <= P.pos; t += NT) {
        device const float *k = kc + (ulong)t * P.kv_dim + (ulong)kvh * P.head_dim;
        float s = 0.0;
        for (uint i = 0; i < P.head_dim; ++i) s += qh[i] * k[i];
        atth[t] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    float lm = -INFINITY;
    for (uint t = tid; t <= P.pos; t += NT) lm = max(lm, atth[t]);
    tmp[tid] = lm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = NT / 2; s > 0; s >>= 1) {
        if (tid < s) tmp[tid] = max(tmp[tid], tmp[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = tmp[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ls = 0.0;
    for (uint t = tid; t <= P.pos; t += NT) {
        float e = exp(atth[t] - m);
        atth[t] = e;
        ls += e;
    }
    tmp[tid] = ls;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint s = NT / 2; s > 0; s >>= 1) {
        if (tid < s) tmp[tid] = tmp[tid] + tmp[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0 / tmp[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < P.head_dim; i += NT) {
        float acc = 0.0;
        for (uint t = 0; t <= P.pos; ++t) {
            device const float *v = vc + (ulong)t * P.kv_dim + (ulong)kvh * P.head_dim;
            acc += atth[t] * v[i];
        }
        xb3[(ulong)h * P.head_dim + i] = acc * inv;
    }
}

kernel void swiglu(device float *hb        [[buffer(0)]],
                   device const float *hb2 [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    float v = hb[gid];
    v *= 1.0 / (1.0 + exp(-v));
    hb[gid] = v * hb2[gid];
}

kernel void residual(device float *x       [[buffer(0)]],
                     device const float *y [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    x[gid] += y[gid];
}
`

@(private = "file")
make_pso :: proc(lib: ^MTL.Library, name: string) -> ^MTL.ComputePipelineState {
	ns_name := NS.String.alloc()->initWithOdinString(name)
	defer ns_name->release()
	fn := lib->newFunctionWithName(ns_name)
	if fn == nil {
		fmt.eprintfln("metal: kernel '%s' not found", name)
		os.exit(1)
	}
	defer fn->release()
	pso, err := m_device->newComputePipelineStateWithFunction(fn)
	if err != nil {
		fmt.eprintfln("metal: pipeline '%s' failed: %s", name, err->localizedDescription()->odinString())
		os.exit(1)
	}
	return pso
}

@(private = "file")
new_shared :: proc(n_floats: int) -> ^MTL.Buffer {
	return m_device->newBufferWithLength(NS.UInteger(n_floats * size_of(f32)), MTL.ResourceStorageModeShared)
}

metal_init :: proc(t: ^Transformer) -> bool {
	g := &t.gguf
	c := t.config
	m_cfg = c

	m_device = MTL.CreateSystemDefaultDevice()
	if m_device == nil {
		fmt.eprintln("metal: no Metal device available")
		return false
	}
	fmt.printf("metal: %s\n", m_device->name()->odinString())

	m_queue = m_device->newCommandQueue()

	src := NS.String.alloc()->initWithOdinString(MSL_SRC)
	defer src->release()
	lib, err := m_device->newLibraryWithSource(src, nil)
	if err != nil {
		fmt.eprintfln("metal: shader compile failed: %s", err->localizedDescription()->odinString())
		return false
	}
	defer lib->release()

	m_pso_f32 = make_pso(lib, "gemv_f32")
	m_pso_f16 = make_pso(lib, "gemv_f16")
	m_pso_q8_0 = make_pso(lib, "gemv_q8_0")
	m_pso_q4_0 = make_pso(lib, "gemv_q4_0")
	m_pso_q4_1 = make_pso(lib, "gemv_q4_1")
	m_pso_q5_0 = make_pso(lib, "gemv_q5_0")
	m_pso_q5_1 = make_pso(lib, "gemv_q5_1")
	m_pso_q4_k = make_pso(lib, "gemv_q4_k")
	m_pso_q6_k = make_pso(lib, "gemv_q6_k")
	m_pso_rmsnorm = make_pso(lib, "rmsnorm")
	m_pso_rope = make_pso(lib, "rope")
	m_pso_attn = make_pso(lib, "attention")
	m_pso_swiglu = make_pso(lib, "swiglu")
	m_pso_residual = make_pso(lib, "residual")

	// Zero-copy wrap the mmap'd weights. Length rounded up to a page (the
	// mapping already covers whole pages, so the extra bytes are valid).
	base := raw_data(g.mmap)
	m_mmap_base = uintptr(base)
	rounded := align_up(len(g.mmap), PAGE_SIZE)
	whole := ([^]u8)(base)[:rounded]
	m_weights = m_device->newBufferWithBytesNoCopy(whole, MTL.ResourceStorageModeShared, nil)
	if m_weights == nil {
		fmt.eprintln("metal: failed to create no-copy weight buffer")
		return false
	}

	att_head_dim := c.n_heads * c.head_dim
	kv_dim := c.n_kv_heads * c.head_dim

	m_b_x = new_shared(c.dim)
	m_b_xb = new_shared(c.dim)
	m_b_xb2 = new_shared(c.dim)
	m_b_xb3 = new_shared(att_head_dim)
	m_b_q = new_shared(att_head_dim)
	m_b_hb = new_shared(c.hidden_dim)
	m_b_hb2 = new_shared(c.hidden_dim)
	m_b_att = new_shared(c.n_heads * c.seq_len)
	m_b_logits = new_shared(c.vocab_size)
	m_b_kc = new_shared(c.n_layers * c.seq_len * kv_dim)
	m_b_vc = new_shared(c.n_layers * c.seq_len * kv_dim)

	metal_enabled = true
	return true
}

metal_destroy :: proc() {
	if !metal_enabled do return
	for b in ([]^MTL.Buffer{m_b_x, m_b_xb, m_b_xb2, m_b_xb3, m_b_q, m_b_hb, m_b_hb2, m_b_att, m_b_logits, m_b_kc, m_b_vc}) {
		b->release()
	}
	m_weights->release()
	m_queue->release()
	m_device->release()
	metal_enabled = false
}

@(private = "file")
pso_for :: proc(k: GGML_Type) -> ^MTL.ComputePipelineState {
	#partial switch k {
	case .F32:
		return m_pso_f32
	case .F16:
		return m_pso_f16
	case .Q8_0:
		return m_pso_q8_0
	case .Q4_0:
		return m_pso_q4_0
	case .Q4_1:
		return m_pso_q4_1
	case .Q5_0:
		return m_pso_q5_0
	case .Q5_1:
		return m_pso_q5_1
	case .Q4_K:
		return m_pso_q4_k
	case .Q6_K:
		return m_pso_q6_k
	}
	return nil
}

@(private = "file")
bytes_of :: proc(p: rawptr, n: int) -> []u8 {
	return ([^]u8)(p)[:n]
}

@(private = "file")
woff :: proc(p: rawptr) -> NS.UInteger {
	return NS.UInteger(uintptr(p) - m_mmap_base)
}

// Encode xout[d] = W[d,n] * x[n]. x/out are GPU buffers (byte offsets given).
@(private = "file")
enc_gemv :: proc(
	enc: ^MTL.ComputeCommandEncoder,
	kind: GGML_Type,
	w_off: NS.UInteger,
	xb: ^MTL.Buffer,
	x_off: NS.UInteger,
	ob: ^MTL.Buffer,
	o_off: NS.UInteger,
	n, d: int,
) {
	enc->setComputePipelineState(pso_for(kind))
	enc->setBuffer(m_weights, w_off, 0)
	enc->setBuffer(xb, x_off, 1)
	enc->setBuffer(ob, o_off, 2)
	dims := [2]u32{u32(n), u32(d)}
	enc->setBytes(bytes_of(&dims, size_of(dims)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(d), 1, 1}, MTL.Size{32, 1, 1})
}

@(private = "file")
enc_rmsnorm :: proc(
	enc: ^MTL.ComputeCommandEncoder,
	inp: ^MTL.Buffer,
	outp: ^MTL.Buffer,
	w: rawptr,
	size, count: int,
	eps: f32,
) {
	enc->setComputePipelineState(m_pso_rmsnorm)
	enc->setBuffer(inp, 0, 0)
	enc->setBuffer(outp, 0, 1)
	enc->setBuffer(m_weights, woff(w), 2)
	P := struct {
		size: u32,
		eps:  f32,
	}{u32(size), eps}
	enc->setBytes(bytes_of(&P, size_of(P)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(count), 1, 1}, MTL.Size{32, 1, 1})
}

@(private = "file")
enc_elementwise :: proc(
	enc: ^MTL.ComputeCommandEncoder,
	pso: ^MTL.ComputePipelineState,
	a: ^MTL.Buffer,
	b: ^MTL.Buffer,
	count: int,
) {
	enc->setComputePipelineState(pso)
	enc->setBuffer(a, 0, 0)
	enc->setBuffer(b, 0, 1)
	tg := min(count, 256)
	enc->dispatchThreads(MTL.Size{NS.Integer(count), 1, 1}, MTL.Size{NS.Integer(tg), 1, 1})
}

// Full Qwen3 forward for one token, entirely on the GPU in a single command
// buffer. Returns logits (shared memory) after completion.
forward_gpu :: proc(transformer: ^Transformer, token: int, pos: int) -> []f32 {
	NS.scoped_autoreleasepool()

	p := &transformer.config
	if pos < 0 || pos >= p.seq_len {
		fmt.eprintf("forward_gpu: pos=%d out of range (seq_len=%d)\n", pos, p.seq_len)
		os.exit(1)
	}
	if token < 0 || token >= p.vocab_size {
		fmt.eprintf("forward_gpu: token=%d out of range (vocab=%d)\n", token, p.vocab_size)
		os.exit(1)
	}

	w := &transformer.weights

	dim := p.dim
	hidden_dim := p.hidden_dim
	head_dim := p.head_dim
	n_heads := p.n_heads
	n_kv_heads := p.n_kv_heads
	kv_dim := n_kv_heads * head_dim
	kv_mul := n_heads / n_kv_heads
	att_head_dim := n_heads * head_dim
	seq_len := p.seq_len
	eps := p.rms_eps

	// Embedding lookup on the CPU into the shared x buffer (cheap, once/token).
	x_slice := m_b_x->contentsAsSlice([]f32)
	get_embedding_row(&w.token_embedding, token, dim, x_slice[:dim])

	cmd := m_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()

	for l in 0 ..< p.n_layers {
		lw := &w.layers[l]
		loff_bytes := NS.UInteger(l * seq_len * kv_dim * size_of(f32))
		k_off := loff_bytes + NS.UInteger(pos * kv_dim * size_of(f32))

		enc_rmsnorm(enc, m_b_x, m_b_xb, raw_data(lw.attn_norm), dim, 1, eps)

		enc_gemv(enc, lw.wq.kind, woff(raw_data(lw.wq.data)), m_b_xb, 0, m_b_q, 0, dim, att_head_dim)
		enc_gemv(enc, lw.wk.kind, woff(raw_data(lw.wk.data)), m_b_xb, 0, m_b_kc, k_off, dim, kv_dim)
		enc_gemv(enc, lw.wv.kind, woff(raw_data(lw.wv.data)), m_b_xb, 0, m_b_vc, k_off, dim, kv_dim)

		// Per-head Q/K RMSNorm (in place), then RoPE.
		enc_rmsnorm(enc, m_b_q, m_b_q, raw_data(lw.q_norm), head_dim, n_heads, eps)
		enc->setComputePipelineState(m_pso_rmsnorm)
		enc->setBuffer(m_b_kc, k_off, 0)
		enc->setBuffer(m_b_kc, k_off, 1)
		enc->setBuffer(m_weights, woff(raw_data(lw.k_norm)), 2)
		{
			P := struct {
				size: u32,
				eps:  f32,
			}{u32(head_dim), eps}
			enc->setBytes(bytes_of(&P, size_of(P)), 3)
		}
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_kv_heads), 1, 1}, MTL.Size{32, 1, 1})

		// RoPE on q (n_heads) and k (n_kv_heads).
		rope_q := struct {
			head_dim:  u32,
			pos:       u32,
			rope_freq: f32,
		}{u32(head_dim), u32(pos), p.rope_freq}
		enc->setComputePipelineState(m_pso_rope)
		enc->setBuffer(m_b_q, 0, 0)
		enc->setBytes(bytes_of(&rope_q, size_of(rope_q)), 1)
		enc->dispatchThreads(
			MTL.Size{NS.Integer(n_heads * head_dim / 2), 1, 1},
			MTL.Size{NS.Integer(min(head_dim / 2, 64)), 1, 1},
		)
		enc->setComputePipelineState(m_pso_rope)
		enc->setBuffer(m_b_kc, k_off, 0)
		enc->setBytes(bytes_of(&rope_q, size_of(rope_q)), 1)
		enc->dispatchThreads(
			MTL.Size{NS.Integer(n_kv_heads * head_dim / 2), 1, 1},
			MTL.Size{NS.Integer(min(head_dim / 2, 64)), 1, 1},
		)

		// Attention: one threadgroup of 128 lanes per head.
		enc->setComputePipelineState(m_pso_attn)
		enc->setBuffer(m_b_q, 0, 0)
		enc->setBuffer(m_b_kc, loff_bytes, 1)
		enc->setBuffer(m_b_vc, loff_bytes, 2)
		enc->setBuffer(m_b_att, 0, 3)
		enc->setBuffer(m_b_xb3, 0, 4)
		attn_p := struct {
			head_dim: u32,
			kv_dim:   u32,
			kv_mul:   u32,
			pos:      u32,
			seq_len:  u32,
		}{u32(head_dim), u32(kv_dim), u32(kv_mul), u32(pos), u32(seq_len)}
		enc->setBytes(bytes_of(&attn_p, size_of(attn_p)), 5)
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(n_heads), 1, 1}, MTL.Size{128, 1, 1})

		enc_gemv(enc, lw.wo.kind, woff(raw_data(lw.wo.data)), m_b_xb3, 0, m_b_xb2, 0, att_head_dim, dim)
		enc_elementwise(enc, m_pso_residual, m_b_x, m_b_xb2, dim)

		enc_rmsnorm(enc, m_b_x, m_b_xb, raw_data(lw.ffn_norm), dim, 1, eps)
		enc_gemv(enc, lw.w1.kind, woff(raw_data(lw.w1.data)), m_b_xb, 0, m_b_hb, 0, dim, hidden_dim)
		enc_gemv(enc, lw.w3.kind, woff(raw_data(lw.w3.data)), m_b_xb, 0, m_b_hb2, 0, dim, hidden_dim)
		enc_elementwise(enc, m_pso_swiglu, m_b_hb, m_b_hb2, hidden_dim)
		enc_gemv(enc, lw.w2.kind, woff(raw_data(lw.w2.data)), m_b_hb, 0, m_b_xb, 0, hidden_dim, dim)
		enc_elementwise(enc, m_pso_residual, m_b_x, m_b_xb, dim)
	}

	enc_rmsnorm(enc, m_b_x, m_b_x, raw_data(w.output_norm), dim, 1, eps)
	enc_gemv(enc, w.output.kind, woff(raw_data(w.output.data)), m_b_x, 0, m_b_logits, 0, dim, p.vocab_size)

	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()

	return m_b_logits->contentsAsSlice([]f32)[:p.vocab_size]
}
