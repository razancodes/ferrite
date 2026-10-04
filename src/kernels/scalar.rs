//! Portable reference kernels (also used to cross-check the SIMD versions).

use super::{QActs, QMat};
use crate::quant::{BLOCK, GROUP, qvalue};
use half::f16;

pub fn quantize_row(x: &[f32], q: &mut [i8], d: &mut [f32], m: &mut [f32]) {
    for (b, blk) in x.as_chunks::<BLOCK>().0.iter().enumerate() {
        let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let db = amax / 127.0;
        let id = if db > 0.0 { 1.0 / db } else { 0.0 };
        let mut sum = 0i32;
        for (i, &v) in blk.iter().enumerate() {
            let qi = (v * id).round_ties_even() as i32;
            q[b * BLOCK + i] = qi as i8;
            sum += qi;
        }
        d[b] = db;
        m[b] = 128.0 * db * sum as f32;
    }
}

/// # Safety
/// `out` must be valid for writes of `qa.t * w.rows` floats.
pub unsafe fn gemm(w: &QMat, qa: &QActs, glo: usize, ghi: usize, out: *mut f32) {
    let k = w.k;
    let nb = k / BLOCK;
    let gb = w.dtype.group_bytes(k);
    let data = w.bytes();
    let qbytes = w.dtype.block_qbytes();
    for g in glo..ghi {
        let gd = &data[g * gb..(g + 1) * gb];
        let scales = &gd[nb * qbytes..];
        for t in 0..qa.t {
            let xq = &qa.qs[t * k..(t + 1) * k];
            let mut acc = [0f32; GROUP];
            for b in 0..nb {
                for r in 0..GROUP {
                    let so = (b * GROUP + r) * 2;
                    let dw = f16::from_bits(u16::from_le_bytes([scales[so], scales[so + 1]])).to_f32();
                    let mut dot = 0i32;
                    for c in 0..BLOCK {
                        // same "u8 = value + 128" arithmetic as the SIMD kernels
                        let wu = qvalue(w.dtype, gd, b, r, c) + 128;
                        dot += wu * xq[b * BLOCK + c] as i32;
                    }
                    let s = dot as f32 * qa.d[t * nb + b] - qa.m[t * nb + b];
                    acc[r] += dw * s;
                }
            }
            for r in 0..GROUP {
                unsafe { *out.add(t * w.rows + g * GROUP + r) = acc[r] };
            }
        }
    }
}
