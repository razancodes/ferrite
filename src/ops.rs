//! Small element-wise / normalization ops. These are written as simple slice
//! loops that LLVM auto-vectorizes (the build targets the native CPU).

use crate::kernels::{Isa, avx512, isa, vmath};

/// Sum of squares with independent accumulators (a plain iterator sum is a
/// serial dependency chain that LLVM may not reassociate).
#[inline]
pub fn sum_sq(x: &[f32]) -> f32 {
    if isa() == Isa::Avx512Vnni {
        return unsafe { vmath::sum_sq(x) };
    }
    let mut acc = [0f32; 16];
    let chunks = x.len() / 16;
    for c in 0..chunks {
        for j in 0..16 {
            acc[j] += x[c * 16 + j] * x[c * 16 + j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for v in &x[chunks * 16..] {
        s += v * v;
    }
    s
}

#[inline]
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let ss: f32 = sum_sq(x) / n as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv * w[i];
    }
}

#[inline]
pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = sum_sq(x) / n as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * inv * w[i];
    }
}

#[inline]
pub fn l2norm_inplace(x: &mut [f32], eps: f32) {
    let ss: f32 = sum_sq(x);
    let inv = 1.0 / (ss + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch.nn.functional.softplus with beta=1, threshold=20
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// x[i] = exp(x[i] - m), returning the sum
#[inline]
pub fn exp_sub_sum(x: &mut [f32], m: f32) -> f32 {
    if isa() == Isa::Avx512Vnni {
        return unsafe { vmath::exp_sub_sum(x, m) };
    }
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    s
}

/// out = silu(g) * u
#[inline]
pub fn swiglu(g: &[f32], u: &[f32], out: &mut [f32]) {
    if isa() == Isa::Avx512Vnni {
        return unsafe { vmath::swiglu(g, u, out) };
    }
    for i in 0..out.len() {
        out[i] = silu(g[i]) * u[i];
    }
}

#[inline]
pub fn silu_inplace(x: &mut [f32]) {
    if isa() == Isa::Avx512Vnni {
        return unsafe { vmath::silu_inplace(x) };
    }
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

/// x *= silu(z)
#[inline]
pub fn mul_silu(x: &mut [f32], z: &[f32]) {
    if isa() == Isa::Avx512Vnni {
        return unsafe { vmath::mul_silu(x, z) };
    }
    for i in 0..x.len() {
        x[i] *= silu(z[i]);
    }
}

/// x *= sigmoid(g)
#[inline]
pub fn mul_sigmoid(x: &mut [f32], g: &[f32]) {
    if isa() == Isa::Avx512Vnni {
        return unsafe { vmath::mul_sigmoid(x, g) };
    }
    for i in 0..x.len() {
        x[i] *= sigmoid(g[i]);
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let m = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let s = exp_sub_sum(x, m);
    let inv = 1.0 / s;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// NeoX-style ("rotate_half") rotary embedding applied to the first `cos.len()*2`
/// dimensions of `x`; the remaining dimensions pass through (partial rotary).
#[inline]
pub fn rope_neox(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for i in 0..half {
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos[i] - b * sin[i];
        x[i + half] = b * cos[i] + a * sin[i];
    }
}

pub fn rope_tables(pos: usize, inv_freq: &[f32], cos: &mut [f32], sin: &mut [f32]) {
    let p = pos as f32;
    for (i, &f) in inv_freq.iter().enumerate() {
        let a = p * f;
        cos[i] = a.cos();
        sin[i] = a.sin();
    }
}

pub fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
    if isa() == Isa::Avx512Vnni && src.len().is_multiple_of(16) {
        unsafe { avx512::f32_to_f16(src, dst) }
    } else {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = half::f16::from_f32(*s).to_bits();
        }
    }
}

#[inline]
pub fn dot_f32_f16(q: &[f32], k: &[u16]) -> f32 {
    if isa() == Isa::Avx512Vnni && q.len().is_multiple_of(16) {
        unsafe { avx512::dot_f32_f16(q, k) }
    } else {
        q.iter().zip(k).map(|(a, b)| a * half::f16::from_bits(*b).to_f32()).sum()
    }
}

#[inline]
pub fn axpy_f16(out: &mut [f32], p: f32, v: &[u16]) {
    if isa() == Isa::Avx512Vnni && out.len().is_multiple_of(16) {
        unsafe { avx512::axpy_f16(out, p, v) }
    } else {
        for (o, b) in out.iter_mut().zip(v) {
            *o += p * half::f16::from_bits(*b).to_f32();
        }
    }
}
