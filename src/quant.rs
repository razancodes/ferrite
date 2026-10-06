//! Weight storage formats and quantizers.
//!
//! All quantized formats are *row-interleaved*: 16 consecutive output rows are
//! packed together so that a single AVX-512 VNNI `vpdpbusd` instruction computes
//! 4-element dot products for 16 rows at once (one row per int32 lane). The
//! activation is broadcast, so the kernels never need horizontal reductions.
//!
//! Group layout (16 rows x K columns, nb = K/32 blocks of 32 columns):
//!
//! ```text
//! Q8R  quants: nb x 512 B, block b: [j in 0..8][row in 0..16][4 bytes]
//!      (u8 = int8 weight + 128) for columns 32b+4j..32b+4j+4
//!      scales: nb x 16 x f16
//! Q4R  quants: nb x 256 B, block b: [p in 0..4][row in 0..16][4 bytes]
//!      low nibble -> column group p, high nibble -> column group p+4,
//!      nibble indexes the IQ4_NL non-linear codebook
//!      scales: nb x 16 x f16
//! Q6R  quants: nb x 384 B, block b: low plane (256 B, laid out like Q4R) +
//!      high plane (2 x 64 B: [h in 0..2][row in 0..16][4 bytes], 2 bits per
//!      value, column group 4h+jj at bits 2jj..2jj+1); u6 = value + 32
//!      scales: nb x 16 x f16
//! ```
//!
//! Both formats store "u8 = 128 + signed value" so kernels share one correction
//! term (128 * sum(activation block)).

use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    Q8R,
    Q4R,
    Q6R,
    /// raw bytes (embedded blobs such as the tokenizer)
    U8,
    /// 2-bit indices into `Q2CB` (2.5 bits/weight). Only for the proxy LM head,
    /// whose job is ranking candidates that the exact head then scores.
    Q2R,
}

pub const GROUP: usize = 16;
pub const BLOCK: usize = 32;

/// Non-linear 4-bit codebook (same values as llama.cpp's IQ4_NL). Denser near
/// zero where most weights live, giving lower error than uniform Q4 at the same size.
pub const IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// 2-bit codebook (symmetric, outer/inner ratio ~3 as for a Gaussian's 4-level Lloyd-Max).
pub const Q2CB: [i8; 4] = [-127, -40, 40, 127];

impl DType {
    /// Bytes of packed quants per 32-column block of a 16-row group.
    pub fn block_qbytes(self) -> usize {
        match self {
            DType::Q8R => 512,
            DType::Q4R => 256,
            DType::Q6R => 384,
            DType::Q2R => 128,
            _ => 0,
        }
    }

    pub fn group_bytes(self, k: usize) -> usize {
        let nb = k / BLOCK;
        match self {
            DType::F32 => GROUP * k * 4,
            DType::Q8R => nb * (512 + 32),
            DType::Q4R => nb * (256 + 32),
            DType::Q6R => nb * (384 + 32),
            DType::Q2R => nb * (128 + 32),
            DType::U8 => GROUP * k,
        }
    }
    pub fn matrix_bytes(self, rows: usize, k: usize) -> usize {
        match self {
            DType::F32 => rows * k * 4,
            DType::U8 => rows * k,
            _ => (rows / GROUP) * self.group_bytes(k),
        }
    }
    pub fn bits_per_weight(self) -> f64 {
        match self {
            DType::F32 => 32.0,
            DType::Q8R => 8.5,
            DType::Q4R => 4.5,
            DType::Q6R => 6.5,
            DType::Q2R => 2.5,
            DType::U8 => 8.0,
        }
    }
    pub fn parse(s: &str) -> Option<DType> {
        match s.to_ascii_lowercase().as_str() {
            "f32" => Some(DType::F32),
            "q8" | "q8r" => Some(DType::Q8R),
            "q4" | "q4r" => Some(DType::Q4R),
            "q6" | "q6r" => Some(DType::Q6R),
            "q2" | "q2r" => Some(DType::Q2R),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Randomized Hadamard rotation ("incoherence processing", as in QuIP#/QuaRot).
//
// Weight rows and activations are both transformed by R = H_b * D (D: fixed
// pseudo-random signs, H_b: orthonormal Walsh-Hadamard on blocks of b columns).
// Since R is orthogonal, (W R^T)(R x) = W x, but outlier weights get spread
// across their block, so 4-bit rounding error drops sharply. R^T = D H_b, and
// D H_b applied to a weight row equals R applied to it because H_b is symmetric,
// so the very same transform is used on both sides.

/// Rotation block size for a reduction dimension k (largest power of two <= 256 dividing k).
pub fn rot_block(k: usize) -> usize {
    let mut b = 256;
    while b > 1 && !k.is_multiple_of(b) {
        b /= 2;
    }
    b
}

fn rot_signs() -> &'static [f32] {
    static SIGNS: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    SIGNS.get_or_init(|| {
        (0..65536u64)
            .map(|c| {
                let mut z = c.wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                if (z ^ (z >> 31)) & 1 == 1 { -1.0 } else { 1.0 }
            })
            .collect()
    })
}

/// In-place normalized fast Walsh-Hadamard transform (len must be a power of two).
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut len = 1;
    while len < n {
        for i in (0..n).step_by(2 * len) {
            let (a, b) = x[i..i + 2 * len].split_at_mut(len);
            for j in 0..len {
                let (u, v) = (a[j], b[j]);
                a[j] = u + v;
                b[j] = u - v;
            }
        }
        len *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= s);
}

/// x <- H_b D x for a vector of length k (block b).
pub fn rotate(x: &mut [f32], b: usize) {
    let signs = rot_signs();
    assert!(x.len() <= signs.len());
    if b == 256 && crate::kernels::isa() == crate::kernels::Isa::Avx512Vnni {
        unsafe { crate::kernels::vmath::rotate256(x, signs) };
        return;
    }
    for (v, s) in x.iter_mut().zip(signs) {
        *v *= s;
    }
    for blk in x.chunks_exact_mut(b) {
        fwht(blk);
    }
}

/// Inverse of `rotate`: x <- D H_b x (used to read embedding rows back out of a
/// rotated tied embedding / LM head).
pub fn unrotate(x: &mut [f32], b: usize) {
    for blk in x.chunks_exact_mut(b) {
        fwht(blk);
    }
    for (v, s) in x.iter_mut().zip(rot_signs()) {
        *v *= s;
    }
}

/// Quantize a row-major f32 matrix [rows, k] into `dt` (rows % 16 == 0, k % 32 == 0).
/// `importance` (length k) optionally weights the per-column error (from an imatrix).
pub fn quantize_matrix(w: &[f32], rows: usize, k: usize, dt: DType, importance: Option<&[f32]>) -> Vec<u8> {
    assert_eq!(w.len(), rows * k);
    if dt == DType::F32 {
        let mut out = Vec::with_capacity(rows * k * 4);
        for x in w {
            out.extend_from_slice(&x.to_le_bytes());
        }
        return out;
    }
    assert!(rows.is_multiple_of(GROUP), "rows {rows} not a multiple of {GROUP}");
    assert!(k.is_multiple_of(BLOCK), "k {k} not a multiple of {BLOCK}");
    let gb = dt.group_bytes(k);
    let ngroups = rows / GROUP;
    let mut out = vec![0u8; ngroups * gb];
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let per = ngroups.div_ceil(nthreads);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * gb).enumerate() {
            s.spawn(move || {
                for (gi, gout) in chunk.chunks_mut(gb).enumerate() {
                    let g = ci * per + gi;
                    let rows_w = &w[g * GROUP * k..(g + 1) * GROUP * k];
                    match dt {
                        DType::Q8R => pack_q8r_group(rows_w, k, gout),
                        DType::Q4R => pack_q4r_group(rows_w, k, gout, importance),
                        DType::Q6R => pack_q6r_group(rows_w, k, gout, importance),
                        DType::Q2R => pack_q2r_group(rows_w, k, gout, importance),
                        DType::F32 | DType::U8 => unreachable!(),
                    }
                }
            });
        }
    });
    out
}

fn pack_q8r_group(w: &[f32], k: usize, out: &mut [u8]) {
    let nb = k / BLOCK;
    let (quants, scales) = out.split_at_mut(nb * 512);
    for r in 0..GROUP {
        let row = &w[r * k..(r + 1) * k];
        for b in 0..nb {
            let x = &row[b * BLOCK..(b + 1) * BLOCK];
            let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let d16 = f16::from_f32(d);
            scales[(b * GROUP + r) * 2..(b * GROUP + r) * 2 + 2].copy_from_slice(&d16.to_bits().to_le_bytes());
            for (c, &v) in x.iter().enumerate() {
                let q = (v * id).round_ties_even().clamp(-127.0, 127.0) as i32;
                let j = c / 4;
                let m = c % 4;
                quants[b * 512 + j * 64 + r * 4 + m] = (q + 128) as u8;
            }
        }
    }
}

/// `quantize_block_iq4` for a small arbitrary codebook (sorted ascending).
fn quantize_block_cb(x: &[f32], wts: &[f32], cb: &[i8], idx: &mut [u8]) -> f32 {
    let near = |v: f32| (0..cb.len()).min_by(|&a, &b| (v - cb[a] as f32).abs().total_cmp(&(v - cb[b] as f32).abs())).unwrap();
    let vmax = x.iter().fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
    if vmax.abs() < 1e-30 {
        idx.iter_mut().for_each(|i| *i = 0);
        return 0.0;
    }
    let lo = cb[0] as f32;
    let fit = |id: f32, idx: &mut [u8]| {
        let (mut sqx, mut sq2) = (0f32, 0f32);
        for (j, &v) in x.iter().enumerate() {
            idx[j] = near(id * v) as u8;
            let q = cb[idx[j] as usize] as f32;
            sqx += wts[j] * q * v;
            sq2 += wts[j] * q * q;
        }
        (sqx, sq2)
    };
    let (mut best_d, mut best) = (vmax / lo, -1f32);
    for itry in -10..=10 {
        let (sqx, sq2) = fit((lo + 4.0 * itry as f32) / vmax, idx);
        if sq2 > 0.0 && (best < 0.0 || sqx * sqx > best * sq2) {
            best_d = sqx / sq2;
            best = best_d * sqx;
        }
    }
    let (sqx, sq2) = fit(if best_d != 0.0 { 1.0 / best_d } else { 0.0 }, idx);
    if sq2 > 0.0 { sqx / sq2 } else { best_d }
}

/// 2-bit layout per 32-column block: two 64-byte chunks of [row][4 bytes];
/// column 4*j+m sits in chunk j%2, bits 2*(j/2).. of byte (row, m).
fn pack_q2r_group(w: &[f32], k: usize, out: &mut [u8], importance: Option<&[f32]>) {
    let nb = k / BLOCK;
    let (quants, scales) = out.split_at_mut(nb * 128);
    let mut idx = [0u8; BLOCK];
    let mut wts = [0f32; BLOCK];
    for r in 0..GROUP {
        let row = &w[r * k..(r + 1) * k];
        for b in 0..nb {
            let x = &row[b * BLOCK..(b + 1) * BLOCK];
            block_weights(x, importance, b * BLOCK, &mut wts);
            let d = quantize_block_cb(x, &wts, &Q2CB, &mut idx);
            scales[(b * GROUP + r) * 2..(b * GROUP + r) * 2 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            for v in 0..2 {
                for m in 0..4 {
                    let mut byte = 0u8;
                    for f in 0..4 {
                        byte |= idx[4 * (v + 2 * f) + m] << (2 * f);
                    }
                    quants[b * 128 + v * 64 + r * 4 + m] = byte;
                }
            }
        }
    }
}

#[inline]
fn nearest_iq4(x: f32) -> usize {
    // binary search in the sorted codebook
    if x <= IQ4NL[0] as f32 {
        return 0;
    }
    if x >= IQ4NL[15] as f32 {
        return 15;
    }
    let (mut lo, mut hi) = (0usize, 15usize);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x < IQ4NL[mid] as f32 { hi = mid } else { lo = mid }
    }
    if x - IQ4NL[lo] as f32 <= IQ4NL[hi] as f32 - x { lo } else { hi }
}

/// Find the scale and codebook indices for one block of 32 weights that
/// minimize the importance-weighted squared error (search over candidate
/// scales, each followed by a closed-form least-squares refit).
pub fn quantize_block_iq4(x: &[f32], wts: &[f32], idx: &mut [u8]) -> f32 {
    let mut amax = 0f32;
    let mut vmax = 0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            vmax = v;
        }
    }
    if amax < 1e-30 {
        idx.iter_mut().for_each(|i| *i = 8);
        return 0.0;
    }
    let mut best_d = vmax / IQ4NL[0] as f32;
    let mut best = -1f32;
    let ntry = 7;
    for itry in -ntry..=ntry {
        let id = (itry as f32 + IQ4NL[0] as f32) / vmax;
        let (mut sqx, mut sq2) = (0f32, 0f32);
        for (j, &v) in x.iter().enumerate() {
            let q = IQ4NL[nearest_iq4(id * v)] as f32;
            sqx += wts[j] * q * v;
            sq2 += wts[j] * q * q;
        }
        if sq2 > 0.0 && (best < 0.0 || sqx * sqx > best * sq2) {
            best_d = sqx / sq2;
            best = best_d * sqx;
        }
    }
    let id = if best_d != 0.0 { 1.0 / best_d } else { 0.0 };
    for (j, &v) in x.iter().enumerate() {
        idx[j] = nearest_iq4(id * v) as u8;
    }
    // one more least-squares refit of the scale for the final assignment
    let (mut sqx, mut sq2) = (0f32, 0f32);
    for (j, &v) in x.iter().enumerate() {
        let q = IQ4NL[idx[j] as usize] as f32;
        sqx += wts[j] * q * v;
        sq2 += wts[j] * q * q;
    }
    if sq2 > 0.0 { sqx / sq2 } else { best_d }
}

fn pack_q4r_group(w: &[f32], k: usize, out: &mut [u8], importance: Option<&[f32]>) {
    let nb = k / BLOCK;
    let (quants, scales) = out.split_at_mut(nb * 256);
    let mut idx = [0u8; BLOCK];
    let mut wts = [0f32; BLOCK];
    for r in 0..GROUP {
        let row = &w[r * k..(r + 1) * k];
        for b in 0..nb {
            let x = &row[b * BLOCK..(b + 1) * BLOCK];
            let sigma2 = x.iter().map(|v| v * v).sum::<f32>() / BLOCK as f32;
            for j in 0..BLOCK {
                wts[j] = match importance {
                    Some(imp) => imp[b * BLOCK + j] * (sigma2 + x[j] * x[j]).sqrt(),
                    None => x[j] * x[j] + sigma2 * 0.25,
                };
            }
            let d = quantize_block_iq4(x, &wts, &mut idx);
            let d16 = f16::from_f32(d);
            scales[(b * GROUP + r) * 2..(b * GROUP + r) * 2 + 2].copy_from_slice(&d16.to_bits().to_le_bytes());
            for p in 0..4 {
                for m in 0..4 {
                    let lo = idx[4 * p + m];
                    let hi = idx[4 * (p + 4) + m];
                    quants[b * 256 + p * 64 + r * 4 + m] = lo | (hi << 4);
                }
            }
        }
    }
}

/// Best scale for a block quantized to integers in [qmin, qmax]: candidate
/// scales around amax/qmax, each refit by weighted least squares; returns the
/// scale and writes the integers to `q`.
pub fn quantize_block_uniform(x: &[f32], wts: &[f32], qmin: i32, qmax: i32, q: &mut [i32]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax < 1e-30 {
        q.iter_mut().for_each(|v| *v = 0);
        return 0.0;
    }
    let assign = |id: f32, q: &mut [i32]| {
        for (j, &v) in x.iter().enumerate() {
            q[j] = ((v * id).round_ties_even() as i32).clamp(qmin, qmax);
        }
    };
    let fit = |q: &[i32]| {
        let (mut sqx, mut sq2) = (0f32, 0f32);
        for (j, &v) in x.iter().enumerate() {
            let qf = q[j] as f32;
            sqx += wts[j] * qf * v;
            sq2 += wts[j] * qf * qf;
        }
        (sqx, sq2)
    };
    let (mut best_d, mut best) = (amax / qmax as f32, -1f32);
    for itry in -8..=8 {
        let id = (qmax as f32 + 0.125 * itry as f32) / amax;
        assign(id, q);
        let (sqx, sq2) = fit(q);
        if sq2 > 0.0 && (best < 0.0 || sqx * sqx > best * sq2) {
            best_d = sqx / sq2;
            best = best_d * sqx;
        }
    }
    assign(if best_d != 0.0 { 1.0 / best_d } else { 0.0 }, q);
    let (sqx, sq2) = fit(q);
    if sq2 > 0.0 { sqx / sq2 } else { best_d }
}

fn block_weights(x: &[f32], importance: Option<&[f32]>, col0: usize, wts: &mut [f32]) {
    let sigma2 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    for j in 0..x.len() {
        wts[j] = match importance {
            Some(imp) => imp[col0 + j] * (sigma2 + x[j] * x[j]).sqrt(),
            None => x[j] * x[j] + sigma2 * 0.25,
        };
    }
}

fn pack_q6r_group(w: &[f32], k: usize, out: &mut [u8], importance: Option<&[f32]>) {
    let nb = k / BLOCK;
    let (quants, scales) = out.split_at_mut(nb * 384);
    let mut q = [0i32; BLOCK];
    let mut wts = [0f32; BLOCK];
    for r in 0..GROUP {
        let row = &w[r * k..(r + 1) * k];
        for b in 0..nb {
            let x = &row[b * BLOCK..(b + 1) * BLOCK];
            block_weights(x, importance, b * BLOCK, &mut wts);
            let d = quantize_block_uniform(x, &wts, -32, 31, &mut q);
            let d16 = f16::from_f32(d);
            scales[(b * GROUP + r) * 2..(b * GROUP + r) * 2 + 2].copy_from_slice(&d16.to_bits().to_le_bytes());
            let u: Vec<u8> = q.iter().map(|v| (v + 32) as u8).collect();
            let qb = &mut quants[b * 384..(b + 1) * 384];
            for p in 0..4 {
                for m in 0..4 {
                    qb[p * 64 + r * 4 + m] = (u[4 * p + m] & 15) | ((u[4 * (p + 4) + m] & 15) << 4);
                }
            }
            for h in 0..2 {
                for m in 0..4 {
                    let mut byte = 0u8;
                    for jj in 0..4 {
                        byte |= (u[4 * (4 * h + jj) + m] >> 4) << (2 * jj);
                    }
                    qb[256 + h * 64 + r * 4 + m] = byte;
                }
            }
        }
    }
}

/// Signed integer stored for (row lane `r`, column `c`) of block `b` in a
/// group's packed quants (value * scale = weight).
#[inline]
pub fn qvalue(dt: DType, gd: &[u8], b: usize, r: usize, c: usize) -> i32 {
    let (j, m) = (c / 4, c % 4);
    match dt {
        DType::Q8R => gd[b * 512 + j * 64 + r * 4 + m] as i32 - 128,
        DType::Q4R => {
            let byte = gd[b * 256 + (j % 4) * 64 + r * 4 + m];
            let nib = if j < 4 { byte & 15 } else { byte >> 4 };
            IQ4NL[nib as usize] as i32
        }
        DType::Q6R => {
            let byte = gd[b * 384 + (j % 4) * 64 + r * 4 + m];
            let lo = if j < 4 { byte & 15 } else { byte >> 4 };
            let hb = gd[b * 384 + 256 + (j / 4) * 64 + r * 4 + m];
            let hi = (hb >> (2 * (j % 4))) & 3;
            (lo | (hi << 4)) as i32 - 32
        }
        DType::Q2R => {
            let byte = gd[b * 128 + (j % 2) * 64 + r * 4 + m];
            Q2CB[((byte >> (2 * (j / 2))) & 3) as usize] as i32
        }
        _ => unreachable!(),
    }
}

/// Build a new matrix (same format) from a subset of rows (`ids.len()` must be
/// a multiple of 16). Used to carve a small "draft" LM head out of the full one.
pub fn gather_rows(data: &[u8], dt: DType, k: usize, ids: &[u32]) -> Vec<u8> {
    assert!(ids.len().is_multiple_of(GROUP));
    match dt {
        DType::F32 => {
            let mut out = Vec::with_capacity(ids.len() * k * 4);
            for &r in ids {
                out.extend_from_slice(&data[r as usize * k * 4..(r as usize + 1) * k * 4]);
            }
            out
        }
        DType::Q8R | DType::Q4R | DType::Q6R | DType::Q2R => {
            let nb = k / BLOCK;
            let gb = dt.group_bytes(k);
            // every format stores its quants as 64-byte chunks of [row][4 bytes]
            let qbytes = dt.block_qbytes();
            let chunks = qbytes / 64;
            let mut out = vec![0u8; ids.len() / GROUP * gb];
            for (i, &r) in ids.iter().enumerate() {
                let (sg, sl) = (r as usize / GROUP, r as usize % GROUP);
                let (dg, dl) = (i / GROUP, i % GROUP);
                let src = &data[sg * gb..(sg + 1) * gb];
                let dst = &mut out[dg * gb..(dg + 1) * gb];
                for b in 0..nb {
                    for j in 0..chunks {
                        let so = b * qbytes + j * 64 + sl * 4;
                        let d = b * qbytes + j * 64 + dl * 4;
                        dst[d..d + 4].copy_from_slice(&src[so..so + 4]);
                    }
                    let so = nb * qbytes + (b * GROUP + sl) * 2;
                    let d = nb * qbytes + (b * GROUP + dl) * 2;
                    dst[d..d + 2].copy_from_slice(&src[so..so + 2]);
                }
            }
            out
        }
        DType::U8 => panic!("not a matrix"),
    }
}

/// Dequantize a single row of a stored matrix (used for embedding lookups and tests).
pub fn dequantize_row(data: &[u8], dt: DType, k: usize, row: usize, out: &mut [f32]) {
    match dt {
        DType::U8 => panic!("U8 tensors are not matrices"),
        DType::F32 => {
            let base = row * k * 4;
            for c in 0..k {
                let o = base + c * 4;
                out[c] = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
            }
        }
        DType::Q8R | DType::Q4R | DType::Q6R | DType::Q2R => {
            let nb = k / BLOCK;
            let g = row / GROUP;
            let r = row % GROUP;
            let gdata = &data[g * dt.group_bytes(k)..(g + 1) * dt.group_bytes(k)];
            let scales = &gdata[nb * dt.block_qbytes()..];
            for b in 0..nb {
                let so = (b * GROUP + r) * 2;
                let d = f16::from_bits(u16::from_le_bytes([scales[so], scales[so + 1]])).to_f32();
                for c in 0..BLOCK {
                    out[b * BLOCK + c] = d * qvalue(dt, gdata, b, r, c) as f32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rand_mat(rows: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..rows * k)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.2
            })
            .collect()
    }
    #[test]
    fn rotation_is_orthogonal() {
        let k = 768;
        let b = rot_block(k);
        assert_eq!(b, 256);
        let w = rand_mat(1, k, 3);
        let x = rand_mat(1, k, 4);
        let dot = |a: &[f32], c: &[f32]| a.iter().zip(c).map(|(p, q)| p * q).sum::<f32>();
        let (mut wr, mut xr) = (w.clone(), x.clone());
        rotate(&mut wr, b);
        rotate(&mut xr, b);
        assert!((dot(&w, &x) - dot(&wr, &xr)).abs() < 1e-4);
        assert!((dot(&w, &w) - dot(&wr, &wr)).abs() < 1e-4);
        let mut back = wr.clone();
        unrotate(&mut back, b);
        for (p, q) in back.iter().zip(&w) {
            assert!((p - q).abs() < 1e-5);
        }
        // the SIMD path must match the scalar definition
        let mut a = x.clone();
        for (v, s) in a.iter_mut().zip(rot_signs()) {
            *v *= s;
        }
        for blk in a.chunks_exact_mut(256) {
            fwht(blk);
        }
        for (p, q) in a.iter().zip(&xr) {
            assert!((p - q).abs() < 1e-5, "simd rotate mismatch {p} vs {q}");
        }
    }

    #[test]
    fn roundtrip_error_is_small() {
        let (rows, k) = (32, 128);
        let w = rand_mat(rows, k, 7);
        for (dt, tol) in [(DType::Q8R, 0.01), (DType::Q6R, 0.035), (DType::Q4R, 0.12)] {
            let q = quantize_matrix(&w, rows, k, dt, None);
            assert_eq!(q.len(), dt.matrix_bytes(rows, k));
            let mut row = vec![0f32; k];
            let (mut err, mut norm) = (0f64, 0f64);
            for r in 0..rows {
                dequantize_row(&q, dt, k, r, &mut row);
                for c in 0..k {
                    err += ((row[c] - w[r * k + c]) as f64).powi(2);
                    norm += (w[r * k + c] as f64).powi(2);
                }
            }
            let rel = (err / norm).sqrt();
            assert!(rel < tol, "{dt:?} relative error {rel}");
        }
    }
}
