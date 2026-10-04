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
    let (rows, k) = (2048usize, 1024usize);
    let w: Vec<f32> = (0..rows * k).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    for dt in [DType::Q8R, DType::Q4R] {
        let data = quantize_matrix(&w, rows, k, dt, None);
        let m = QMat { dtype: dt, rows, k, ptr: data.as_ptr(), len: data.len(), id: usize::MAX, rot: 0 };
        for (threads, t) in [(1usize, 1usize), (4, 1), (1, 64), (4, 256)] {
            let pool = ThreadPool::new(threads);
            let x: Vec<f32> = (0..t * k).map(|i| ((i * 31) % 97) as f32 / 97.0 - 0.5).collect();
            let mut out = vec![0f32; t * rows];
            let mut qa = QActs::default();
            matmul(&pool, &m, &x, t, &mut out, &mut qa);
            let reps = (2000 / t).max(4);
            let t0 = Instant::now();
            for _ in 0..reps {
                matmul(&pool, &m, &x, t, &mut out, &mut qa);
            }
            let secs = t0.elapsed().as_secs_f64() / reps as f64;
            println!(
                "{dt:?} {rows}x{k} tokens={t:<4} threads={threads}: {:8.1} us  {:6.1} GMAC/s  {:5.1} GB/s weights",
                secs * 1e6,
                (rows * k * t) as f64 / secs / 1e9,
                data.len() as f64 / secs / 1e9
            );
        }
    }
}
