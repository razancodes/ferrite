//! GPTQ (Frantar et al., 2022) for the row-interleaved 4-bit format: the
//! columns of a weight matrix are rounded one at a time and each column's
//! rounding error is pushed onto the columns not yet rounded, weighted by the
//! inverse of H = X^T X, the covariance of that matrix's calibration inputs.
//! Plain rounding (even importance-weighted, as with an imatrix) treats input
//! columns independently; GPTQ also uses their correlations.
//!
//! Collection: `begin` registers tensor ids, `kernels::matmul` calls `observe`
//! with each registered matrix's input, `take` returns the averaged H.
//! `STOP_AFTER` lets a calibration pass stop after the last layer needed.

use crate::kernels::{QActs, QMat};
use crate::quant::{BLOCK, DType, IQ4NL};
use crate::threadpool::ThreadPool;
use faer::{Accum, Mat, Par, Side};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub static ON: AtomicBool = AtomicBool::new(false);
/// forward passes stop after this layer index (usize::MAX: run all layers)
pub static STOP_AFTER: AtomicUsize = AtomicUsize::new(usize::MAX);

struct Acc {
    h: Vec<f32>,
    scratch: Vec<f32>,
    rows: u64,
}
static STATS: Mutex<Option<HashMap<usize, Acc>>> = Mutex::new(None);

/// Start collecting H for these tensor ids (with their input sizes `k`).
pub fn begin(ids: &[(usize, usize)]) {
    let map = ids.iter().map(|&(id, k)| (id, Acc { h: vec![0f32; k * k], scratch: Vec::new(), rows: 0 })).collect();
    *STATS.lock().unwrap() = Some(map);
    ON.store(true, Ordering::SeqCst);
}

/// Stop collecting; returns H / rows per tensor id.
pub fn take() -> HashMap<usize, Vec<f32>> {
    ON.store(false, Ordering::SeqCst);
    let map = STATS.lock().unwrap().take().unwrap_or_default();
    map.into_iter()
        .map(|(id, mut a)| {
            let s = 1.0 / a.rows.max(1) as f32;
            a.h.iter_mut().for_each(|v| *v *= s);
            (id, a.h)
        })
        .collect()
}

/// H += x^T x for a registered matrix (x: [t, w.k]), via the f32 GEMM:
/// with X^T as both operands, out[i, c] = sum over tokens of x[., c] * x[., i].
pub fn observe(pool: &ThreadPool, w: &QMat, x: &[f32], t: usize) {
    let mut g = STATS.lock().unwrap();
    let Some(acc) = g.as_mut().and_then(|m| m.get_mut(&w.id)) else { return };
    let k = w.k;
    let mut xt = vec![0f32; k * t];
    for tok in 0..t {
        for c in 0..k {
            xt[c * t + tok] = x[tok * k + c];
        }
    }
    let m = QMat { dtype: DType::F32, rows: k, k: t, ptr: xt.as_ptr() as *const u8, len: xt.len() * 4, id: usize::MAX, rot: 0 };
    acc.scratch.resize(k * k, 0.0);
    crate::kernels::matmul(pool, &m, &xt, k, &mut acc.scratch, &mut QActs::default());
    for (h, s) in acc.h.iter_mut().zip(&acc.scratch) {
        *h += *s;
    }
    acc.rows += t as u64;
}

/// Upper-triangular U with H^-1 = U^T U, without forming H^-1: with J the
/// order reversal, chol(J H J) = L gives U = J L^-1 J.
fn inverse_cholesky_upper(h: &[f32], k: usize) -> Mat<f32> {
    let mut damp = 0.01 * (0..k).map(|i| h[i * k + i]).sum::<f32>() / k as f32;
    loop {
        let hr = Mat::<f32>::from_fn(k, k, |i, j| h[(k - 1 - i) * k + (k - 1 - j)] + if i == j { damp } else { 0.0 });
        if let Ok(llt) = hr.llt(Side::Lower) {
            let mut li = Mat::<f32>::zeros(k, k);
            faer::linalg::triangular_inverse::invert_lower_triangular(li.as_mut(), llt.L(), Par::rayon(0));
            return Mat::from_fn(k, k, |i, j| if i <= j { li[(k - 1 - i, k - 1 - j)] } else { 0.0 });
        }
        damp *= 10.0; // not positive definite at this damping: retry with more
    }
}

/// GPTQ-quantize W [rows, k] (row-major) to Q4R given its input covariance H [k, k].
pub fn quantize_q4r(w: &[f32], rows: usize, k: usize, h: &[f32]) -> Vec<u8> {
    assert!(k.is_multiple_of(BLOCK));
    let mut h = h.to_vec();
    let mut wm = Mat::<f32>::from_fn(rows, k, |r, c| w[r * k + c]); // column-major: a column is contiguous
    for i in 0..k {
        if h[i * k + i] <= 0.0 {
            // dead input: nothing to fit, and keeps H invertible
            h[i * k + i] = 1.0;
            wm.col_mut(i).fill(0.0);
        }
    }
    let u = inverse_cholesky_upper(&h, k);
    let nb = k / BLOCK;
    let mut idx = vec![0u8; rows * k];
    let mut scales = vec![0f32; rows * nb];
    let (mut x, mut wts, mut tmp) = ([0f32; BLOCK], [0f32; BLOCK], [0u8; BLOCK]);
    const BS: usize = 128;
    let mut b0 = 0;
    while b0 < k {
        let b1 = (b0 + BS).min(k);
        let mut err = Mat::<f32>::zeros(rows, b1 - b0);
        for c in b0..b1 {
            if c % BLOCK == 0 {
                // block scales from the error-updated weights, importance = input energy
                let b = c / BLOCK;
                for r in 0..rows {
                    for j in 0..BLOCK {
                        x[j] = wm[(r, c + j)];
                    }
                    let s2 = x.iter().map(|v| v * v).sum::<f32>() / BLOCK as f32;
                    for j in 0..BLOCK {
                        wts[j] = h[(c + j) * k + c + j] * (s2 + x[j] * x[j]).sqrt();
                    }
                    scales[r * nb + b] = crate::quant::quantize_block_iq4(&x, &wts, &mut tmp);
                }
            }
            let b = c / BLOCK;
            let d = 1.0 / u[(c, c)];
            for r in 0..rows {
                let s = scales[r * nb + b];
                let v = wm[(r, c)];
                let q = crate::quant::nearest_iq4(if s != 0.0 { v / s } else { 0.0 });
                idx[r * k + c] = q as u8;
                err[(r, c - b0)] = (v - s * IQ4NL[q] as f32) * d;
            }
            // push this column's error onto the rest of the block
            let e = err.col(c - b0).to_owned();
            for j in c + 1..b1 {
                let ucj = u[(c, j)];
                if ucj != 0.0 {
                    let col = wm.col_mut(j);
                    for (wv, ev) in col.iter_mut().zip(e.iter()) {
                        *wv -= ev * ucj;
                    }
                }
            }
        }
        // ... and the whole block's error onto every later column at once
        if b1 < k {
            let rhs = u.as_ref().submatrix(b0, b1, b1 - b0, k - b1);
            faer::linalg::matmul::matmul(wm.as_mut().submatrix_mut(0, b1, rows, k - b1), Accum::Add, err.as_ref(), rhs, -1.0f32, Par::rayon(0));
        }
        b0 = b1;
    }
    crate::quant::pack_q4r_indices(&idx, &scales, rows, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gptq_beats_plain_rounding_on_correlated_inputs() {
        let (rows, k, n) = (64, 256, 2048);
        let mut s = 11u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f32 / (1u64 << 31) as f32 - 0.5
        };
        let w: Vec<f32> = (0..rows * k).map(|_| rnd() * 0.2).collect();
        // correlated inputs: a few shared latent factors plus noise, uneven column scales
        let lat: Vec<f32> = (0..n * 8).map(|_| rnd()).collect();
        let mix: Vec<f32> = (0..8 * k).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..n * k)
            .map(|i| {
                let (t, c) = (i / k, i % k);
                let z: f32 = (0..8).map(|f| lat[t * 8 + f] * mix[f * k + c]).sum();
                (z + 0.3 * rnd()) * (1.0 + (c % 7) as f32)
            })
            .collect();
        let mut h = vec![0f32; k * k];
        for t in 0..n {
            for i in 0..k {
                for j in 0..k {
                    h[i * k + j] += x[t * k + i] * x[t * k + j] / n as f32;
                }
            }
        }
        let diag: Vec<f32> = (0..k).map(|i| h[i * k + i]).collect();
        let deq = |data: &[u8]| {
            let mut out = vec![0f32; rows * k];
            for r in 0..rows {
                crate::quant::dequantize_row(data, DType::Q4R, k, r, &mut out[r * k..(r + 1) * k]);
            }
            out
        };
        // output error on the calibration inputs: sum over t of ||(W - Wq) x_t||^2
        let out_err = |wq: &[f32]| -> f64 {
            let mut e = 0f64;
            for t in 0..n {
                for r in 0..rows {
                    let d: f32 = (0..k).map(|c| (w[r * k + c] - wq[r * k + c]) * x[t * k + c]).sum();
                    e += (d * d) as f64;
                }
            }
            e
        };
        let plain = out_err(&deq(&crate::quant::quantize_matrix(&w, rows, k, DType::Q4R, Some(&diag))));
        let gptq = out_err(&deq(&quantize_q4r(&w, rows, k, &h)));
        assert!(gptq < 0.85 * plain, "gptq {gptq:.3} vs plain {plain:.3}");
    }
}
