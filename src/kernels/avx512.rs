//! AVX-512 + VNNI kernels (Ice Lake / Tiger Lake / Sapphire Rapids / Zen 4+).
//!
//! The int8 GEMM microkernel computes a 16-row x NT-token output tile:
//!
//! * one 64-byte weight load feeds NT `vpdpbusd` (u8 weights x broadcast s8 activations),
//! * per 32-column block the int32 sums are scaled once:
//!   `acc += w_scale[16] * (dot * x_scale - 128 * x_scale * sum(x))`.
//!
//! Decoding (NT = 1) is purely memory bound; prefill uses NT = 8 so every weight
//! byte is reused 8 times from registers, turning the problem compute bound.

#![allow(clippy::missing_safety_doc)]

use super::{QActs, QMat};
use crate::quant::{BLOCK, DType, GROUP, IQ4NL};
use std::arch::x86_64::*;

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub unsafe fn quantize_row(x: &[f32], q: &mut [i8], d: &mut [f32], m: &mut [f32]) {
    let nb = x.len() / BLOCK;
    let xp = x.as_ptr();
    let qp = q.as_mut_ptr();
    unsafe {
        for b in 0..nb {
            let v0 = _mm512_loadu_ps(xp.add(b * 32));
            let v1 = _mm512_loadu_ps(xp.add(b * 32 + 16));
            let amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
            let db = amax / 127.0;
            let id = if db > 0.0 { 1.0 / db } else { 0.0 };
            let vid = _mm512_set1_ps(id);
            // cvtps uses MXCSR rounding (round-to-nearest-even), matching the scalar path
            let i0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, vid));
            let i1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, vid));
            let s = _mm512_reduce_add_epi32(_mm512_add_epi32(i0, i1));
            _mm_storeu_si128(qp.add(b * 32) as *mut __m128i, _mm512_cvtepi32_epi8(i0));
            _mm_storeu_si128(qp.add(b * 32 + 16) as *mut __m128i, _mm512_cvtepi32_epi8(i1));
            d[b] = db;
            m[b] = 128.0 * db * s as f32;
        }
    }
}

#[inline(always)]
unsafe fn bcast4(p: *const i8) -> __m512i {
    unsafe { _mm512_set1_epi32((p as *const i32).read_unaligned()) }
}

/// Codebook as unsigned bytes (value + 128), replicated in all four 128-bit lanes for vpshufb.
static IQ4_LUT: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = (IQ4NL[i % 16] as i16 + 128) as u8;
        i += 1;
    }
    t
};

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn tile_q8r<const NT: usize>(
    gq: *const u8,
    gs: *const u8,
    nb: usize,
    xq: *const i8,
    k: usize,
    xd: *const f32,
    xm: *const f32,
    out: *mut f32,
    ldo: usize,
) {
    unsafe {
        let mut acc = [_mm512_setzero_ps(); NT];
        for b in 0..nb {
            let dw = _mm512_cvtph_ps(_mm256_loadu_si256(gs.add(b * 32) as *const __m256i));
            let qb = gq.add(b * 512);
            let xb = xq.add(b * 32);
            let mut ia = [_mm512_setzero_si512(); NT];
            if NT == 1 {
                // two independent dependency chains to hide vpdpbusd latency
                let mut ib = _mm512_setzero_si512();
                for j in (0..8).step_by(2) {
                    let w0 = _mm512_loadu_si512(qb.add(j * 64) as *const _);
                    let w1 = _mm512_loadu_si512(qb.add(j * 64 + 64) as *const _);
                    ia[0] = _mm512_dpbusd_epi32(ia[0], w0, bcast4(xb.add(j * 4)));
                    ib = _mm512_dpbusd_epi32(ib, w1, bcast4(xb.add(j * 4 + 4)));
                }
                ia[0] = _mm512_add_epi32(ia[0], ib);
            } else {
                for j in 0..8 {
                    let wv = _mm512_loadu_si512(qb.add(j * 64) as *const _);
                    for t in 0..NT {
                        ia[t] = _mm512_dpbusd_epi32(ia[t], wv, bcast4(xb.add(t * k + j * 4)));
                    }
                }
            }
            for t in 0..NT {
                let s = _mm512_fmsub_ps(
                    _mm512_cvtepi32_ps(ia[t]),
                    _mm512_set1_ps(*xd.add(t * nb + b)),
                    _mm512_set1_ps(*xm.add(t * nb + b)),
                );
                acc[t] = _mm512_fmadd_ps(s, dw, acc[t]);
            }
        }
        for t in 0..NT {
            _mm512_storeu_ps(out.add(t * ldo), acc[t]);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn tile_q4r<const NT: usize>(
    gq: *const u8,
    gs: *const u8,
    nb: usize,
    xq: *const i8,
    k: usize,
    xd: *const f32,
    xm: *const f32,
    out: *mut f32,
    ldo: usize,
) {
    unsafe {
        let lut = _mm512_loadu_si512(IQ4_LUT.as_ptr() as *const _);
        let mask = _mm512_set1_epi8(0x0F);
        let mut acc = [_mm512_setzero_ps(); NT];
        for b in 0..nb {
            let dw = _mm512_cvtph_ps(_mm256_loadu_si256(gs.add(b * 32) as *const __m256i));
            let qb = gq.add(b * 256);
            let xb = xq.add(b * 32);
            let mut ia = [_mm512_setzero_si512(); NT];
            let mut ib = [_mm512_setzero_si512(); NT];
            for p in 0..4 {
                let v = _mm512_loadu_si512(qb.add(p * 64) as *const _);
                let lo = _mm512_shuffle_epi8(lut, _mm512_and_si512(v, mask));
                let hi = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(v), mask));
                for t in 0..NT {
                    ia[t] = _mm512_dpbusd_epi32(ia[t], lo, bcast4(xb.add(t * k + p * 4)));
                    ib[t] = _mm512_dpbusd_epi32(ib[t], hi, bcast4(xb.add(t * k + p * 4 + 16)));
                }
            }
            for t in 0..NT {
                let s = _mm512_fmsub_ps(
                    _mm512_cvtepi32_ps(_mm512_add_epi32(ia[t], ib[t])),
                    _mm512_set1_ps(*xd.add(t * nb + b)),
                    _mm512_set1_ps(*xm.add(t * nb + b)),
                );
                acc[t] = _mm512_fmadd_ps(s, dw, acc[t]);
            }
        }
        for t in 0..NT {
            _mm512_storeu_ps(out.add(t * ldo), acc[t]);
        }
    }
}

type TileFn = unsafe fn(*const u8, *const u8, usize, *const i8, usize, *const f32, *const f32, *mut f32, usize);

fn tiles(dt: DType) -> [TileFn; 9] {
    macro_rules! t {
        ($f:ident) => {
            [$f::<1>, $f::<1>, $f::<2>, $f::<3>, $f::<4>, $f::<5>, $f::<6>, $f::<7>, $f::<8>]
        };
    }
    match dt {
        DType::Q8R => t!(tile_q8r),
        DType::Q4R => t!(tile_q4r),
        DType::F32 | DType::U8 => unreachable!(),
    }
}

/// Compute output row-groups [glo, ghi) for every token in `qa`.
pub unsafe fn gemm(w: &QMat, qa: &QActs, glo: usize, ghi: usize, out: *mut f32) {
    let k = w.k;
    let nb = k / BLOCK;
    let gb = w.dtype.group_bytes(k);
    let qbytes = if w.dtype == DType::Q8R { 512 } else { 256 };
    let tf = tiles(w.dtype);
    let tt = qa.t;
    // Block tokens so the activation slice stays L2-resident while the weight
    // group (a few KB-100KB) is reused for every token tile.
    let tchunk = if tt <= 8 { tt } else { ((384 * 1024) / k).clamp(16, 256) / 8 * 8 };
    let mut t0 = 0;
    while t0 < tt {
        let t1 = (t0 + tchunk).min(tt);
        for g in glo..ghi {
            unsafe {
                let gq = w.ptr.add(g * gb);
                let gs = gq.add(nb * qbytes);
                let mut t = t0;
                while t < t1 {
                    let n = (t1 - t).min(8);
                    (tf[n])(
                        gq,
                        gs,
                        nb,
                        qa.qs.as_ptr().add(t * k),
                        k,
                        qa.d.as_ptr().add(t * nb),
                        qa.m.as_ptr().add(t * nb),
                        out.add(t * w.rows + g * GROUP),
                        w.rows,
                    );
                    t += n;
                }
            }
        }
        t0 = t1;
    }
}

// ---------------------------------------------------------------------------
// f16 helpers for the KV cache

#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32_f16(q: &[f32], k: &[u16]) -> f32 {
    let n = q.len();
    debug_assert!(n.is_multiple_of(16) && k.len() >= n);
    unsafe {
        let mut a0 = _mm512_setzero_ps();
        let mut a1 = _mm512_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            let k0 = _mm512_cvtph_ps(_mm256_loadu_si256(k.as_ptr().add(i) as *const __m256i));
            let k1 = _mm512_cvtph_ps(_mm256_loadu_si256(k.as_ptr().add(i + 16) as *const __m256i));
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q.as_ptr().add(i)), k0, a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q.as_ptr().add(i + 16)), k1, a1);
            i += 32;
        }
        while i < n {
            let k0 = _mm512_cvtph_ps(_mm256_loadu_si256(k.as_ptr().add(i) as *const __m256i));
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q.as_ptr().add(i)), k0, a0);
            i += 16;
        }
        _mm512_reduce_add_ps(_mm512_add_ps(a0, a1))
    }
}

/// out += p * v
#[target_feature(enable = "avx512f")]
pub unsafe fn axpy_f16(out: &mut [f32], p: f32, v: &[u16]) {
    let n = out.len();
    debug_assert!(n.is_multiple_of(16) && v.len() >= n);
    unsafe {
        let vp = _mm512_set1_ps(p);
        let mut i = 0;
        while i < n {
            let x = _mm512_cvtph_ps(_mm256_loadu_si256(v.as_ptr().add(i) as *const __m256i));
            let o = _mm512_loadu_ps(out.as_ptr().add(i));
            _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_fmadd_ps(vp, x, o));
            i += 16;
        }
    }
}

#[target_feature(enable = "avx512f")]
pub unsafe fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    debug_assert!(n.is_multiple_of(16));
    unsafe {
        let mut i = 0;
        while i < n {
            let h = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm512_loadu_ps(src.as_ptr().add(i)));
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, h);
            i += 16;
        }
    }
}

// ---------------------------------------------------------------------------
// f32 GEMM (reference precision path): 4 weight rows x 4 tokens register tile.

#[target_feature(enable = "avx512f")]
unsafe fn f32_tile<const R: usize, const T: usize>(w: *const f32, x: *const f32, k: usize, out: *mut f32, ldo: usize) {
    unsafe {
        let mut acc = [[_mm512_setzero_ps(); T]; R];
        let mut i = 0;
        while i + 16 <= k {
            let mut wv = [_mm512_setzero_ps(); R];
            for r in 0..R {
                wv[r] = _mm512_loadu_ps(w.add(r * k + i));
            }
            for t in 0..T {
                let xv = _mm512_loadu_ps(x.add(t * k + i));
                for r in 0..R {
                    acc[r][t] = _mm512_fmadd_ps(wv[r], xv, acc[r][t]);
                }
            }
            i += 16;
        }
        for r in 0..R {
            for t in 0..T {
                let mut s = _mm512_reduce_add_ps(acc[r][t]);
                for j in i..k {
                    s += *w.add(r * k + j) * *x.add(t * k + j);
                }
                *out.add(t * ldo + r) = s;
            }
        }
    }
}

/// out[t, r] for rows [lo, hi) of a row-major f32 matrix.
pub unsafe fn gemm_f32(w: *const f32, k: usize, lo: usize, hi: usize, x: *const f32, t: usize, out: *mut f32, ldo: usize) {
    unsafe {
        let mut r = lo;
        while r < hi {
            let nr = (hi - r).min(4);
            let mut tt = 0;
            while tt < t {
                let nt = (t - tt).min(4);
                let wp = w.add(r * k);
                let xp = x.add(tt * k);
                let op = out.add(tt * ldo + r);
                match (nr, nt) {
                    (4, 4) => f32_tile::<4, 4>(wp, xp, k, op, ldo),
                    (4, 1) => f32_tile::<4, 1>(wp, xp, k, op, ldo),
                    (4, 2) => f32_tile::<4, 2>(wp, xp, k, op, ldo),
                    (4, 3) => f32_tile::<4, 3>(wp, xp, k, op, ldo),
                    (_, 4) => f32_tile::<1, 4>(wp, xp, k, op, ldo),
                    _ => f32_tile::<1, 1>(wp, xp, k, op, ldo),
                }
                if nr < 4 && nt == 4 {
                    // f32_tile::<1,4> handled one row; finish the rest of this row block
                    for rr in 1..nr {
                        f32_tile::<1, 4>(wp.add(rr * k), xp, k, op.add(rr), ldo);
                    }
                } else if nr < 4 {
                    for rr in 1..nr {
                        for t2 in 0..nt {
                            f32_tile::<1, 1>(wp.add(rr * k), xp.add(t2 * k), k, op.add(t2 * ldo + rr), ldo);
                        }
                    }
                    for t2 in 1..nt {
                        f32_tile::<1, 1>(wp, xp.add(t2 * k), k, op.add(t2 * ldo), ldo);
                    }
                }
                tt += nt;
            }
            r += nr;
        }
    }
}
