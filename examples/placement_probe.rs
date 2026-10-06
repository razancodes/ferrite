//! Does memory placement vary at a granularity the engine could exploit?
//! Allocates 32 chunks of 64 MiB (fresh pages each) and measures each chunk's
//! 4-thread streaming read bandwidth separately. On this laptop's flex-mode
//! memory (4 GB soldered + 16 GB DIMM: 8 GB dual-channel, the rest single) a
//! bimodal spread means chunks could be tested and the slow ones discarded.
use std::arch::x86_64::*;
use std::time::Instant;

#[target_feature(enable = "avx512f")]
unsafe fn sum_stream(p: *const f32, n: usize) -> f32 {
    unsafe {
        let mut a = [_mm512_setzero_ps(); 8];
        let mut i = 0;
        while i + 128 <= n {
            for j in 0..8 {
                a[j] = _mm512_add_ps(a[j], _mm512_loadu_ps(p.add(i + 16 * j)));
            }
            i += 128;
        }
        let mut s = a[0];
        for j in 1..8 {
            s = _mm512_add_ps(s, a[j]);
        }
        _mm512_reduce_add_ps(s)
    }
}

fn chunk_gbs(v: &[f32], threads: usize, reps: usize) -> f64 {
    let n = v.len();
    let t0 = Instant::now();
    for _ in 0..reps {
        let per = n / threads;
        std::thread::scope(|s| {
            for k in 0..threads {
                let p = unsafe { v.as_ptr().add(k * per) } as usize;
                s.spawn(move || std::hint::black_box(unsafe { sum_stream(p as *const f32, per) }));
            }
        });
    }
    (reps * n * 4) as f64 / t0.elapsed().as_secs_f64() / 1e9
}

fn main() {
    let chunk = 64usize << 20;
    let chunks: Vec<Vec<f32>> = (0..32).map(|i| vec![i as f32; chunk / 4]).collect();
    let mut g: Vec<f64> = chunks.iter().map(|c| (0..3).map(|_| chunk_gbs(c, 4, 3)).fold(0.0, f64::max)).collect();
    for (i, x) in g.iter().enumerate() {
        print!("{x:5.1}{}", if i % 8 == 7 { "\n" } else { " " });
    }
    g.sort_by(|a, b| a.total_cmp(b));
    println!("min {:.1}  p25 {:.1}  median {:.1}  p75 {:.1}  max {:.1} GB/s", g[0], g[8], g[16], g[24], g[31]);
}
