//! Microbenchmarks: raw VNNI throughput vs the GEMM kernels on L2-resident data.
use ferrite::kernels::{QActs, QMat, matmul};
use ferrite::quant::{DType, quantize_matrix};
use ferrite::threadpool::ThreadPool;
use std::arch::x86_64::*;
use std::time::Instant;

#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn vnni_peak(iters: usize) -> i32 {
    {
        let a = _mm512_set1_epi8(3);
        let b = _mm512_set1_epi8(5);
        let mut acc = [_mm512_setzero_si512(); 12];
        for _ in 0..iters {
            for j in 0..12 {
                acc[j] = _mm512_dpbusd_epi32(acc[j], a, b);
            }
        }
        let mut s = acc[0];
        for j in 1..12 {
            s = _mm512_add_epi32(s, acc[j]);
        }
        _mm512_reduce_add_epi32(s)
    }
}

fn main() {
    let iters = 20_000_000;
    for threads in [1usize, 4] {
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| std::hint::black_box(unsafe { vnni_peak(iters) }));
            }
        });
        let secs = t0.elapsed().as_secs_f64();
        let macs = (threads * iters * 12 * 64) as f64;
        println!("raw vpdpbusd peak, {threads} thread(s): {:.1} GMAC/s ({:.2} GHz-equivalent per core)", macs / secs / 1e9, macs / secs / 1e9 / 64.0 / threads as f64);
    }
    // A realistic decode stream: cycle through 8 different 16384 x 2048 matrices
    // (8 x 17-35 MB, far larger than the 8 MB L3) so every read comes from DRAM.
    let (rows, k) = (16384usize, 2048usize);
    let w: Vec<f32> = (0..rows * k).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    for dt in [DType::Q8R, DType::Q6R, DType::Q4R] {
        let mats: Vec<Vec<u8>> = (0..8).map(|_| quantize_matrix(&w, rows, k, dt, None)).collect();
        let qm: Vec<QMat> = mats
            .iter()
            .map(|d| QMat { dtype: dt, rows, k, ptr: d.as_ptr(), len: d.len(), id: usize::MAX, rot: 0 })
            .collect();
        for (threads, t) in [(1usize, 1usize), (2, 1), (4, 1), (4, 4)] {
            let pool = ThreadPool::new(threads);
            let x: Vec<f32> = (0..t * k).map(|i| ((i * 31) % 97) as f32 / 97.0 - 0.5).collect();
            let mut out = vec![0f32; t * rows];
            let mut qa = QActs::default();
            for m in &qm {
                matmul(&pool, m, &x, t, &mut out, &mut qa);
            }
            let reps = 6;
            let t0 = Instant::now();
            for _ in 0..reps {
                for m in &qm {
                    matmul(&pool, m, &x, t, &mut out, &mut qa);
                }
            }
            let secs = t0.elapsed().as_secs_f64() / (reps * qm.len()) as f64;
            println!(
                "{dt:?} tokens={t} threads={threads}: {:8.1} us/matrix  {:5.1} GB/s",
                secs * 1e6,
                mats[0].len() as f64 / secs / 1e9
            );
        }
    }
}
