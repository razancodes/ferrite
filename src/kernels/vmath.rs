//! Vectorized transcendental helpers. The C runtime's scalar `expf` on
//! windows-gnu costs ~80ns/call, which made SiLU/softmax dominate the profile;
//! these AVX-512 versions process 16 lanes in a few ns.
//!
//! exp(x) = 2^n * e^r with n = round(x / ln2), |r| <= ln2/2, e^r from a
//! degree-6 Taylor polynomial (rel. error < 2e-7), and 2^n applied with
//! `vscalefps`, which handles the exponent range without bit tricks.

#![allow(clippy::missing_safety_doc)]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn exp512(x: __m512) -> __m512 {
    unsafe {
        let x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-104.0)), _mm512_set1_ps(88.72));
        let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm512_mul_ps(
            x,
            _mm512_set1_ps(std::f32::consts::LOG2_E),
        ));
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693_359_4), x);
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.121_944_4e-4), r);
        let mut p = _mm512_set1_ps(1.0 / 720.0);
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 120.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 24.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 6.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));
        _mm512_scalef_ps(p, n)
    }
}

#[inline(always)]
unsafe fn sigmoid512(x: __m512) -> __m512 {
    unsafe {
        let one = _mm512_set1_ps(1.0);
        let e = exp512(_mm512_sub_ps(_mm512_setzero_ps(), x));
        _mm512_div_ps(one, _mm512_add_ps(one, e))
    }
}

#[inline(always)]
unsafe fn ld(p: *const f32, rem: usize) -> __m512 {
    unsafe {
        if rem >= 16 {
            _mm512_loadu_ps(p)
        } else {
            _mm512_maskz_loadu_ps(((1u32 << rem) - 1) as u16, p)
        }
    }
}

#[inline(always)]
unsafe fn st(p: *mut f32, v: __m512, rem: usize) {
    unsafe {
        if rem >= 16 {
            _mm512_storeu_ps(p, v)
        } else {
            _mm512_mask_storeu_ps(p, ((1u32 << rem) - 1) as u16, v)
        }
    }
}

/// out[i] = silu(g[i]) * u[i]
#[target_feature(enable = "avx512f")]
pub unsafe fn swiglu(g: &[f32], u: &[f32], out: &mut [f32]) {
    let n = out.len();
    let mut i = 0;
    unsafe {
        while i < n {
            let rem = n - i;
            let gv = ld(g.as_ptr().add(i), rem);
            let uv = ld(u.as_ptr().add(i), rem);
            st(out.as_mut_ptr().add(i), _mm512_mul_ps(_mm512_mul_ps(gv, sigmoid512(gv)), uv), rem);
            i += 16;
        }
    }
}

/// x[i] = silu(x[i])
#[target_feature(enable = "avx512f")]
pub unsafe fn silu_inplace(x: &mut [f32]) {
    let n = x.len();
    let mut i = 0;
    unsafe {
        while i < n {
            let rem = n - i;
            let v = ld(x.as_ptr().add(i), rem);
            st(x.as_mut_ptr().add(i), _mm512_mul_ps(v, sigmoid512(v)), rem);
            i += 16;
        }
    }
}

/// x[i] *= silu(z[i])
#[target_feature(enable = "avx512f")]
pub unsafe fn mul_silu(x: &mut [f32], z: &[f32]) {
    let n = x.len();
    let mut i = 0;
    unsafe {
        while i < n {
            let rem = n - i;
            let zv = ld(z.as_ptr().add(i), rem);
            let xv = ld(x.as_ptr().add(i), rem);
            st(x.as_mut_ptr().add(i), _mm512_mul_ps(xv, _mm512_mul_ps(zv, sigmoid512(zv))), rem);
            i += 16;
        }
    }
}

/// x[i] *= sigmoid(g[i])
#[target_feature(enable = "avx512f")]
pub unsafe fn mul_sigmoid(x: &mut [f32], g: &[f32]) {
    let n = x.len();
    let mut i = 0;
    unsafe {
        while i < n {
            let rem = n - i;
            let gv = ld(g.as_ptr().add(i), rem);
            let xv = ld(x.as_ptr().add(i), rem);
            st(x.as_mut_ptr().add(i), _mm512_mul_ps(xv, sigmoid512(gv)), rem);
            i += 16;
        }
    }
}

/// x[i] = exp(x[i] - m); returns the sum.
#[target_feature(enable = "avx512f")]
pub unsafe fn exp_sub_sum(x: &mut [f32], m: f32) -> f32 {
    let n = x.len();
    let mut i = 0;
    unsafe {
        let mv = _mm512_set1_ps(m);
        let mut acc = _mm512_setzero_ps();
        while i < n {
            let rem = n - i;
            let e = exp512(_mm512_sub_ps(ld(x.as_ptr().add(i), rem), mv));
            let e = if rem >= 16 { e } else { _mm512_maskz_mov_ps(((1u32 << rem) - 1) as u16, e) };
            st(x.as_mut_ptr().add(i), e, rem);
            acc = _mm512_add_ps(acc, e);
            i += 16;
        }
        _mm512_reduce_add_ps(acc)
    }
}

/// One gated-delta-rule step for one head with d_v = 128 (state rows held as
/// 8 zmm registers per row; two passes over the d_k x 128 state).
#[target_feature(enable = "avx512f")]
///
/// Reads the state from `src` and writes the updated state to `dst` (they may
/// alias). Writing to a different buffer is how speculative decoding keeps a
/// rollback checkpoint without any extra memory traffic.
pub unsafe fn delta_rule_step_128(src: *const f32, dst: *mut f32, q: &[f32], k: &[f32], v: &[f32], beta: f32, decay: f32, o: &mut [f32]) {
    const DV: usize = 128;
    let dk = k.len();
    unsafe {
        let mut acc = [_mm512_setzero_ps(); 8];
        for i in 0..dk {
            let ki = _mm512_set1_ps(*k.get_unchecked(i));
            let row = src.add(i * DV);
            for c in 0..8 {
                acc[c] = _mm512_fmadd_ps(_mm512_loadu_ps(row.add(16 * c)), ki, acc[c]);
            }
        }
        let vb = _mm512_set1_ps(beta);
        let vd = _mm512_set1_ps(decay);
        let mut delta = [_mm512_setzero_ps(); 8];
        for c in 0..8 {
            let vv = _mm512_loadu_ps(v.as_ptr().add(16 * c));
            delta[c] = _mm512_mul_ps(vb, _mm512_fnmadd_ps(vd, acc[c], vv));
        }
        let mut out = [_mm512_setzero_ps(); 8];
        for i in 0..dk {
            let ki = _mm512_set1_ps(*k.get_unchecked(i));
            let qi = _mm512_set1_ps(*q.get_unchecked(i));
            let row = src.add(i * DV);
            let orow = dst.add(i * DV);
            for c in 0..8 {
                let sv = _mm512_fmadd_ps(_mm512_loadu_ps(row.add(16 * c)), vd, _mm512_mul_ps(ki, delta[c]));
                _mm512_storeu_ps(orow.add(16 * c), sv);
                out[c] = _mm512_fmadd_ps(sv, qi, out[c]);
            }
        }
        for c in 0..8 {
            _mm512_storeu_ps(o.as_mut_ptr().add(16 * c), out[c]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exp_accuracy() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let xs: Vec<f32> = (-2000..2000).map(|i| i as f32 * 0.04).collect();
        let mut v = xs.clone();
        let s = unsafe { exp_sub_sum(&mut v, 0.0) };
        let mut want_sum = 0f64;
        for (x, got) in xs.iter().zip(&v) {
            let want = (*x as f64).exp();
            want_sum += want;
            let rel = ((*got as f64 - want) / want).abs();
            assert!(rel < 1e-6, "exp({x}) = {got}, want {want}, rel {rel}");
        }
        assert!(((s as f64 - want_sum) / want_sum).abs() < 1e-5);
    }
}

/// Causal depthwise conv (kernel 4) + SiLU over channels [lo, hi) for `t` tokens.
/// `rows` points at token 0 of the projection (stride `stride`), conv inputs are
/// replaced in place by outputs; `state` is [3][cd] (oldest first) and updated.
#[target_feature(enable = "avx512f")]
/// If `ck` is non-null, the state after token `ck_tok` is also written there.
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv4_silu(
    rows: *mut f32,
    stride: usize,
    t: usize,
    w: &[f32],
    state: *mut f32,
    cd: usize,
    lo: usize,
    hi: usize,
    ck: *mut f32,
    ck_tok: usize,
) {
    unsafe {
        let mut ch = lo;
        while ch < hi {
            let rem = hi - ch;
            let w0 = ld(w.as_ptr().add(ch), rem);
            let w1 = ld(w.as_ptr().add(cd + ch), rem);
            let w2 = ld(w.as_ptr().add(2 * cd + ch), rem);
            let w3 = ld(w.as_ptr().add(3 * cd + ch), rem);
            let mut s0 = ld(state.add(ch), rem);
            let mut s1 = ld(state.add(cd + ch), rem);
            let mut s2 = ld(state.add(2 * cd + ch), rem);
            for tok in 0..t {
                let p = rows.add(tok * stride + ch);
                let x = ld(p, rem);
                let y = _mm512_fmadd_ps(w0, s0, _mm512_fmadd_ps(w1, s1, _mm512_fmadd_ps(w2, s2, _mm512_mul_ps(w3, x))));
                st(p, _mm512_mul_ps(y, sigmoid512(y)), rem);
                s0 = s1;
                s1 = s2;
                s2 = x;
                if tok == ck_tok && !ck.is_null() {
                    st(ck.add(ch), s0, rem);
                    st(ck.add(cd + ch), s1, rem);
                    st(ck.add(2 * cd + ch), s2, rem);
                }
            }
            st(state.add(ch), s0, rem);
            st(state.add(cd + ch), s1, rem);
            st(state.add(2 * cd + ch), s2, rem);
            ch += 16;
        }
    }
}

#[target_feature(enable = "avx512f")]
pub unsafe fn sum_sq(x: &[f32]) -> f32 {
    let n = x.len();
    let mut i = 0;
    unsafe {
        let mut a0 = _mm512_setzero_ps();
        let mut a1 = _mm512_setzero_ps();
        while i + 32 <= n {
            let v0 = _mm512_loadu_ps(x.as_ptr().add(i));
            let v1 = _mm512_loadu_ps(x.as_ptr().add(i + 16));
            a0 = _mm512_fmadd_ps(v0, v0, a0);
            a1 = _mm512_fmadd_ps(v1, v1, a1);
            i += 32;
        }
        while i < n {
            let v = ld(x.as_ptr().add(i), n - i);
            a0 = _mm512_fmadd_ps(v, v, a0);
            i += 16;
        }
        _mm512_reduce_add_ps(_mm512_add_ps(a0, a1))
    }
}
