//! Tiled ("flash attention" style) causal attention for prefill batches.
//!
//! The decode path processes one query against the KV cache as matrix-vector
//! work. For a prefill chunk that re-reads and re-converts every f16 key for
//! every query. Here each task owns one KV head and a block of BQ query tokens
//! (BQ x heads-per-group query rows). Keys and values are converted from f16
//! once per 64-key tile and reused by every query row, scores are a small
//! register-tiled GEMM against the transposed key tile, and the softmax is
//! computed online so nothing of size O(context) is materialized.

use crate::ops;
use crate::threadpool::{SyncPtr, ThreadPool};
use half::f16;
use half::slice::HalfFloatSliceExt;

const BQ: usize = 16;
const BK: usize = 64;

fn to_f32(src: &[u16], dst: &mut [f32]) {
    // u16 storage of the KV cache has the same layout as half::f16
    let h: &[f16] = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f16, src.len()) };
    h.convert_to_f32_slice(dst);
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::BK;
    use std::arch::x86_64::*;

    /// 4 query rows x 64 keys: 16 zmm accumulators, each key-tile load reused 4x.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn scores4(q: *const f32, kt: *const f32, s: *mut f32, hd: usize) {
        unsafe {
            let mut acc = [[_mm512_setzero_ps(); 4]; 4];
            for d in 0..hd {
                let k = [
                    _mm512_loadu_ps(kt.add(d * BK)),
                    _mm512_loadu_ps(kt.add(d * BK + 16)),
                    _mm512_loadu_ps(kt.add(d * BK + 32)),
                    _mm512_loadu_ps(kt.add(d * BK + 48)),
                ];
                for r in 0..4 {
                    let x = _mm512_set1_ps(*q.add(r * hd + d));
                    for c in 0..4 {
                        acc[r][c] = _mm512_fmadd_ps(x, k[c], acc[r][c]);
                    }
                }
            }
            for r in 0..4 {
                for c in 0..4 {
                    _mm512_storeu_ps(s.add(r * BK + 16 * c), acc[r][c]);
                }
            }
        }
    }

    /// o[r][dc..dc+64] += sum_j p[r][j] v[j][dc..dc+64] for 4 rows.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn pv4(p: *const f32, vt: *const f32, o: *mut f32, hd: usize, nk: usize) {
        unsafe {
            for dc in (0..hd).step_by(64) {
                let mut acc = [[_mm512_setzero_ps(); 4]; 4];
                for r in 0..4 {
                    for c in 0..4 {
                        acc[r][c] = _mm512_loadu_ps(o.add(r * hd + dc + 16 * c));
                    }
                }
                for j in 0..nk {
                    let vp = vt.add(j * hd + dc);
                    let v = [
                        _mm512_loadu_ps(vp),
                        _mm512_loadu_ps(vp.add(16)),
                        _mm512_loadu_ps(vp.add(32)),
                        _mm512_loadu_ps(vp.add(48)),
                    ];
                    for r in 0..4 {
                        let x = _mm512_set1_ps(*p.add(r * BK + j));
                        for c in 0..4 {
                            acc[r][c] = _mm512_fmadd_ps(x, v[c], acc[r][c]);
                        }
                    }
                }
                for r in 0..4 {
                    for c in 0..4 {
                        _mm512_storeu_ps(o.add(r * hd + dc + 16 * c), acc[r][c]);
                    }
                }
            }
        }
    }
}

fn fast() -> bool {
    crate::kernels::isa() == crate::kernels::Isa::Avx512Vnni
}

/// s[r][0..BK] = q[r] . kt[:, j] for every query row (kt is [hd][BK]).
fn scores_tile(q: &[f32], kt: &[f32], s: &mut [f32], nq: usize, hd: usize) {
    let mut r = 0;
    if fast() {
        while r + 4 <= nq {
            unsafe { simd::scores4(q.as_ptr().add(r * hd), kt.as_ptr(), s.as_mut_ptr().add(r * BK), hd) };
            r += 4;
        }
    }
    while r + 2 <= nq {
        let mut a0 = [0f32; BK];
        let mut a1 = [0f32; BK];
        let (q0, q1) = (&q[r * hd..(r + 1) * hd], &q[(r + 1) * hd..(r + 2) * hd]);
        for d in 0..hd {
            let k: &[f32; BK] = kt[d * BK..(d + 1) * BK].try_into().unwrap();
            let (x0, x1) = (q0[d], q1[d]);
            for j in 0..BK {
                a0[j] += x0 * k[j];
                a1[j] += x1 * k[j];
            }
        }
        s[r * BK..(r + 1) * BK].copy_from_slice(&a0);
        s[(r + 1) * BK..(r + 2) * BK].copy_from_slice(&a1);
        r += 2;
    }
    if r < nq {
        let mut a0 = [0f32; BK];
        let q0 = &q[r * hd..(r + 1) * hd];
        for d in 0..hd {
            let k: &[f32; BK] = kt[d * BK..(d + 1) * BK].try_into().unwrap();
            for j in 0..BK {
                a0[j] += q0[d] * k[j];
            }
        }
        s[r * BK..(r + 1) * BK].copy_from_slice(&a0);
    }
}

/// o[r] += sum_j p[r][j] * vt[j] for the first `nk` keys (vt is [BK][hd]).
fn pv_tile(p: &[f32], vt: &[f32], o: &mut [f32], nq: usize, hd: usize, nk: usize) {
    const W: usize = 64;
    let mut r = 0;
    if fast() {
        while r + 4 <= nq {
            unsafe { simd::pv4(p.as_ptr().add(r * BK), vt.as_ptr(), o.as_mut_ptr().add(r * hd), hd, nk) };
            r += 4;
        }
    }
    while r + 2 <= nq {
        for dc in (0..hd).step_by(W) {
            let mut a0: [f32; W] = o[r * hd + dc..r * hd + dc + W].try_into().unwrap();
            let mut a1: [f32; W] = o[(r + 1) * hd + dc..(r + 1) * hd + dc + W].try_into().unwrap();
            for j in 0..nk {
                let v: &[f32; W] = vt[j * hd + dc..j * hd + dc + W].try_into().unwrap();
                let (p0, p1) = (p[r * BK + j], p[(r + 1) * BK + j]);
                for x in 0..W {
                    a0[x] += p0 * v[x];
                    a1[x] += p1 * v[x];
                }
            }
            o[r * hd + dc..r * hd + dc + W].copy_from_slice(&a0);
            o[(r + 1) * hd + dc..(r + 1) * hd + dc + W].copy_from_slice(&a1);
        }
        r += 2;
    }
    if r < nq {
        for dc in (0..hd).step_by(W) {
            let mut a0: [f32; W] = o[r * hd + dc..r * hd + dc + W].try_into().unwrap();
            for j in 0..nk {
                let v: &[f32; W] = vt[j * hd + dc..j * hd + dc + W].try_into().unwrap();
                let p0 = p[r * BK + j];
                for x in 0..W {
                    a0[x] += p0 * v[x];
                }
            }
            o[r * hd + dc..r * hd + dc + W].copy_from_slice(&a0);
        }
    }
}

/// Causal attention for `t` new tokens at positions pos0..pos0+t whose K/V are
/// already in the cache. Queries are read from `proj` rows (stride `rows`, q of
/// head h at `qoff + h*hd`, output gate at `goff + h*hd`); the result goes to
/// `core` laid out [t][nh][hd].
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill(
    pool: &ThreadPool,
    proj: &[f32],
    rows: usize,
    qoff: usize,
    goff: usize,
    gated: bool,
    kc: &[u16],
    vc: &[u16],
    max_ctx: usize,
    core: &mut [f32],
    t: usize,
    pos0: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) {
    assert!(hd.is_multiple_of(64));
    let grp = nh / nkv;
    let nqb = t.div_ceil(BQ);
    let scale = 1.0 / (hd as f32).sqrt();
    let cp = SyncPtr::new(core);
    // later query blocks see more keys: schedule them first for better balance
    pool.par_tasks(nkv * nqb, |task| {
        let g = task % nkv;
        let qb = nqb - 1 - task / nkv;
        let (t0, t1) = (qb * BQ, ((qb + 1) * BQ).min(t));
        let nq = (t1 - t0) * grp;
        let mut q = vec![0f32; nq * hd];
        for tok in t0..t1 {
            for qi in 0..grp {
                let hh = g * grp + qi;
                let src = &proj[tok * rows + qoff + hh * hd..tok * rows + qoff + (hh + 1) * hd];
                let r = (tok - t0) * grp + qi;
                for d in 0..hd {
                    q[r * hd + d] = src[d] * scale;
                }
            }
        }
        let mut o = vec![0f32; nq * hd];
        let mut m = vec![f32::NEG_INFINITY; nq];
        let mut l = vec![0f32; nq];
        let mut kt = vec![0f32; hd * BK];
        let mut vt = vec![0f32; BK * hd];
        let mut s = vec![0f32; nq * BK];
        let mut krow = vec![0f32; hd];
        let key_end = pos0 + t1;
        let mut kb = 0;
        while kb < key_end {
            let nk = (key_end - kb).min(BK);
            for j in 0..nk {
                let off = (g * max_ctx + kb + j) * hd;
                to_f32(&kc[off..off + hd], &mut krow);
                for d in 0..hd {
                    kt[d * BK + j] = krow[d];
                }
                to_f32(&vc[off..off + hd], &mut vt[j * hd..(j + 1) * hd]);
            }
            if nk < BK {
                for d in 0..hd {
                    kt[d * BK + nk..(d + 1) * BK].iter_mut().for_each(|x| *x = 0.0);
                }
            }
            scores_tile(&q, &kt, &mut s, nq, hd);
            for r in 0..nq {
                let limit = pos0 + t0 + r / grp; // last key this query may see
                let srow = &mut s[r * BK..(r + 1) * BK];
                let valid = if kb > limit { 0 } else { (limit - kb + 1).min(nk) };
                if valid == 0 {
                    srow.iter_mut().for_each(|x| *x = 0.0);
                    continue;
                }
                let mb = srow[..valid].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mnew = m[r].max(mb);
                let corr = if m[r] == f32::NEG_INFINITY { 0.0 } else { (m[r] - mnew).exp() };
                let sum = ops::exp_sub_sum(&mut srow[..valid], mnew);
                srow[valid..].iter_mut().for_each(|x| *x = 0.0);
                l[r] = l[r] * corr + sum;
                m[r] = mnew;
                if corr != 1.0 {
                    o[r * hd..(r + 1) * hd].iter_mut().for_each(|x| *x *= corr);
                }
            }
            pv_tile(&s, &vt, &mut o, nq, hd, nk);
            kb += BK;
        }
        for tok in t0..t1 {
            for qi in 0..grp {
                let r = (tok - t0) * grp + qi;
                let hh = g * grp + qi;
                let out = unsafe { cp.slice((tok * nh + hh) * hd, hd) };
                let inv = 1.0 / l[r];
                for d in 0..hd {
                    out[d] = o[r * hd + d] * inv;
                }
                if gated {
                    let gsrc = &proj[tok * rows + goff + hh * hd..tok * rows + goff + (hh + 1) * hd];
                    ops::mul_sigmoid(out, gsrc);
                }
            }
        }
    });
}
