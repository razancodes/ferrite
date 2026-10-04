//! Memory bandwidth probe: streaming AVX-512 reads over large buffers with
//! 1..8 threads, repeated over several independent allocations to expose
//! placement effects (e.g. Intel "flex mode" with mismatched DIMMs, where only
//! part of the address space is dual-channel interleaved).
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

fn main() {
    let bytes = 1usize << 30; // 1 GiB per buffer
    let n = bytes / 4;
    for b in 0..3 {
        let v: Vec<f32> = vec![1.0; n];
        print!("buffer {b}:");
        for threads in [1usize, 2, 4, 8] {
            let reps = 4;
            let t0 = Instant::now();
            for _ in 0..reps {
                let chunk = n / threads;
                std::thread::scope(|s| {
                    for k in 0..threads {
                        let p = unsafe { v.as_ptr().add(k * chunk) } as usize;
                        s.spawn(move || std::hint::black_box(unsafe { sum_stream(p as *const f32, chunk) }));
                    }
                });
            }
            let gbs = (reps * bytes) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            print!("  {threads}T {gbs:5.1} GB/s");
        }
        println!();
        std::mem::forget(v); // keep it allocated so the next buffer lands elsewhere
    }
}
