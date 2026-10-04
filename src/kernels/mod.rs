//! Matrix multiplication front-end: activation quantization, CPU feature
//! dispatch and multi-threaded work splitting.

pub mod avx512;
pub mod scalar;
pub mod vmath;

use crate::quant::{BLOCK, DType, GROUP};
use crate::threadpool::{SyncPtr, ThreadPool};
use std::sync::OnceLock;

/// A weight matrix [rows, k] living in the memory-mapped model file.
#[derive(Clone, Copy)]
pub struct QMat {
    pub dtype: DType,
    pub rows: usize,
    pub k: usize,
    pub ptr: *const u8,
    pub len: usize,
    /// index of the tensor in the model file (for imatrix collection), or usize::MAX
    pub id: usize,
    /// Hadamard rotation block of the input dimension (0 = none): activations
    /// are rotated the same way before quantization
    pub rot: usize,
}
unsafe impl Send for QMat {}
unsafe impl Sync for QMat {}

impl QMat {
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) {
        crate::quant::dequantize_row(self.bytes(), self.dtype, self.k, row, out);
    }
}

/// Activations quantized to int8 in blocks of 32 (per token):
/// value ~= d[blk] * qs[i]; `m[blk] = 128 * d[blk] * sum(qs in blk)` is the
/// correction term for the "u8 = w + 128" weight encoding.
#[derive(Default)]
pub struct QActs {
    pub t: usize,
    pub k: usize,
    pub qs: Vec<i8>,
    pub d: Vec<f32>,
    pub m: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    Avx512Vnni,
    Scalar,
}

pub fn isa() -> Isa {
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(|| {
        if std::env::var("FERRITE_FORCE_SCALAR").is_ok() {
            return Isa::Scalar;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vnni")
            {
                return Isa::Avx512Vnni;
            }
        }
        Isa::Scalar
    })
}

pub fn quantize_acts_into(pool: &ThreadPool, x: &[f32], t: usize, k: usize, rot: usize, qa: &mut QActs) {
    assert!(k.is_multiple_of(BLOCK));
    assert!(x.len() >= t * k);
    let nb = k / BLOCK;
    qa.t = t;
    qa.k = k;
    qa.qs.resize(t * k, 0);
    qa.d.resize(t * nb, 0.0);
    qa.m.resize(t * nb, 0.0);
    let qs = SyncPtr::new(&mut qa.qs);
    let dp = SyncPtr::new(&mut qa.d);
    let mp = SyncPtr::new(&mut qa.m);
    let work = |lo: usize, hi: usize| {
        let mut tmp = if rot > 0 { vec![0f32; k] } else { Vec::new() };
        for tt in lo..hi {
            let mut xr = &x[tt * k..(tt + 1) * k];
            if rot > 0 {
                tmp.copy_from_slice(xr);
                crate::quant::rotate(&mut tmp, rot);
                xr = &tmp;
            }
            let q = unsafe { qs.slice(tt * k, k) };
            let d = unsafe { dp.slice(tt * nb, nb) };
            let m = unsafe { mp.slice(tt * nb, nb) };
            match isa() {
                Isa::Avx512Vnni => unsafe { avx512::quantize_row(xr, q, d, m) },
                Isa::Scalar => scalar::quantize_row(xr, q, d, m),
            }
        }
    };
    if t * k < 16384 {
        work(0, t);
    } else {
        pool.par_chunks(t, 1, work);
    }
}

/// Activation statistics collector for importance-matrix generation:
/// per weight tensor id, the running sum of x^2 per input column and the row count.
pub type ImatrixStats = std::collections::HashMap<usize, (Vec<f64>, u64)>;
pub static IMATRIX: std::sync::Mutex<Option<ImatrixStats>> = std::sync::Mutex::new(None);
pub static IMATRIX_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn collect_imatrix(w: &QMat, x: &[f32], t: usize) {
    let mut g = IMATRIX.lock().unwrap();
    let map = g.get_or_insert_with(Default::default);
    let e = map.entry(w.id).or_insert_with(|| (vec![0f64; w.k], 0));
    let mut tmp = vec![0f32; w.k];
    for tt in 0..t {
        // statistics are collected in the space the weights live in (rotated or not)
        tmp.copy_from_slice(&x[tt * w.k..(tt + 1) * w.k]);
        if w.rot > 0 {
            crate::quant::rotate(&mut tmp, w.rot);
        }
        for (c, v) in tmp.iter().enumerate() {
            e.0[c] += (*v as f64) * (*v as f64);
        }
    }
    e.1 += t as u64;
}

/// out[t, rows] = x[t, k] @ W^T. `qa` is scratch space for quantized activations.
pub fn matmul(pool: &ThreadPool, w: &QMat, x: &[f32], t: usize, out: &mut [f32], qa: &mut QActs) {
    assert!(out.len() >= t * w.rows, "matmul out too small");
    if IMATRIX_ON.load(std::sync::atomic::Ordering::Relaxed) && w.id != usize::MAX {
        collect_imatrix(w, x, t);
    }
    match w.dtype {
        DType::F32 => matmul_f32(pool, w, x, t, out),
        DType::U8 => panic!("U8 tensor used as a matrix"),
        DType::Q8R | DType::Q4R => {
            quantize_acts_into(pool, x, t, w.k, w.rot, qa);
            matmul_q(pool, w, qa, out);
        }
    }
}

/// Matmul with pre-quantized activations (lets several weights share one quantization).
pub fn matmul_q(pool: &ThreadPool, w: &QMat, qa: &QActs, out: &mut [f32]) {
    let ngroups = w.rows / GROUP;
    let op = SyncPtr::new(out);
    let isa = isa();
    // Spread groups over threads; keep a few groups per chunk for locality.
    pool.par_chunks(ngroups, 1, |glo, ghi| unsafe {
        match isa {
            Isa::Avx512Vnni => avx512::gemm(w, qa, glo, ghi, op.get()),
            Isa::Scalar => scalar::gemm(w, qa, glo, ghi, op.get()),
        }
    });
}

fn matmul_f32(pool: &ThreadPool, w: &QMat, x: &[f32], t: usize, out: &mut [f32]) {
    let k = w.k;
    let rows = w.rows;
    let wf: &[f32] = unsafe { std::slice::from_raw_parts(w.ptr as *const f32, rows * k) };
    let op = SyncPtr::new(out);
    // rows outer: each weight row is streamed from memory once and reused
    // (from L1) for every token
    let fast = isa() == Isa::Avx512Vnni;
    pool.par_chunks(rows, 16, |lo, hi| {
        if fast {
            unsafe { avx512::gemm_f32(wf.as_ptr(), k, lo, hi, x.as_ptr(), t, op.get(), rows) };
            return;
        }
        for r in lo..hi {
            let wr = &wf[r * k..(r + 1) * k];
            for tt in 0..t {
                let v = dot_f32(wr, &x[tt * k..(tt + 1) * k]);
                unsafe { *op.get().add(tt * rows + r) = v };
            }
        }
    });
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // 8 independent accumulators so LLVM vectorizes without a serial dependency.
    let n = a.len().min(b.len());
    let mut acc = [0f32; 16];
    let chunks = n / 16;
    for c in 0..chunks {
        for j in 0..16 {
            acc[j] += a[c * 16 + j] * b[c * 16 + j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 16..n {
        s += a[i] * b[i];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::quantize_matrix;

    fn rand_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale
            })
            .collect()
    }

    #[test]
    fn quantized_matmul_matches_reference() {
        let pool = ThreadPool::new(4);
        let (rows, k, t) = (64, 256, 7);
        let w = rand_vec(rows * k, 1, 0.1);
        let x = rand_vec(t * k, 2, 2.0);
        let mut want = vec![0f32; t * rows];
        for tt in 0..t {
            for r in 0..rows {
                want[tt * rows + r] = dot_f32(&w[r * k..(r + 1) * k], &x[tt * k..(tt + 1) * k]);
            }
        }
        for dt in [DType::F32, DType::Q8R, DType::Q4R] {
            let data = quantize_matrix(&w, rows, k, dt, None);
            let m = QMat { dtype: dt, rows, k, ptr: data.as_ptr(), len: data.len(), id: usize::MAX, rot: 0 };
            // dequantized reference: isolates kernel bugs from quantization error
            let mut deq = vec![0f32; rows * k];
            for r in 0..rows {
                m.dequantize_row(r, &mut deq[r * k..(r + 1) * k]);
            }
            let mut out = vec![0f32; t * rows];
            let mut qa = QActs::default();
            matmul(&pool, &m, &x, t, &mut out, &mut qa);
            let mut scalar_out = vec![0f32; t * rows];
            if dt != DType::F32 {
                unsafe { scalar::gemm(&m, &qa, 0, rows / GROUP, scalar_out.as_mut_ptr()) };
            }
            for tt in 0..t {
                for r in 0..rows {
                    let exact = dot_f32(&deq[r * k..(r + 1) * k], &x[tt * k..(tt + 1) * k]);
                    let got = out[tt * rows + r];
                    let tol = 0.02 * exact.abs().max(1.0);
                    assert!((got - exact).abs() < tol, "{dt:?} t{tt} r{r}: got {got} want {exact}");
                    if dt != DType::F32 {
                        let s = scalar_out[tt * rows + r];
                        assert!((got - s).abs() < 1e-3 * s.abs().max(1.0), "{dt:?} simd {got} vs scalar {s}");
                    }
                }
            }
            let _ = &want;
        }
    }
}
