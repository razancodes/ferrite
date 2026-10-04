//! A tiny persistent thread pool tuned for LLM decoding.
//!
//! Decoding a single token issues a few hundred small parallel kernels (one per
//! matmul / attention / recurrence op). General-purpose pools (rayon, std::thread
//! scoped spawns) cost 10-50us per dispatch, which is a large fraction of a ~20ms
//! token. Here workers spin on one atomic word so a dispatch costs roughly one
//! cache-line round trip.
//!
//! The pool also has an adjustable *width*: memory-bound decoding saturates DRAM
//! with 2-4 threads, while compute-bound prefill wants every hyper-thread. Workers
//! outside the current width sleep on a condvar (spinning siblings would steal
//! the laptop's shared turbo budget), and the dispatcher only touches the mutex
//! when a sleeping worker is actually needed.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Job = *const (dyn Fn(usize, usize) + Sync);

struct Shared {
    /// (generation << 8) | width of the current job
    word: AtomicU64,
    done: AtomicUsize,
    quit: AtomicBool,
    /// bit i set while worker i is parked on the condvar
    sleep_mask: AtomicU64,
    job: std::cell::UnsafeCell<Option<Job>>,
    lock: Mutex<()>,
    cv: Condvar,
}

// The job pointer is written by the dispatching thread before publishing a new
// generation (SeqCst) and read by workers after observing it (Acquire).
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct ThreadPool {
    n: usize,
    width: AtomicUsize,
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

/// How long a worker busy-waits for the next job before going to sleep.
const SPIN_ITERS: usize = 1 << 17;

impl ThreadPool {
    pub fn new(n_threads: usize) -> Self {
        let n = n_threads.clamp(1, 64);
        let shared = Arc::new(Shared {
            word: AtomicU64::new(n as u64),
            done: AtomicUsize::new(0),
            quit: AtomicBool::new(false),
            sleep_mask: AtomicU64::new(0),
            job: std::cell::UnsafeCell::new(None),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        });
        let mut handles = Vec::new();
        for tid in 1..n {
            let sh = shared.clone();
            handles.push(
                std::thread::Builder::new()
                    .name(format!("ferrite-worker-{tid}"))
                    .spawn(move || worker_loop(sh, tid))
                    .expect("spawn worker"),
            );
        }
        ThreadPool { n, width: AtomicUsize::new(n), shared, handles }
    }

    /// Threads that take part in parallel jobs right now.
    pub fn n_threads(&self) -> usize {
        self.width.load(Ordering::Relaxed)
    }

    pub fn max_threads(&self) -> usize {
        self.n
    }

    /// Restrict jobs to the first `n` threads; the others go to sleep.
    pub fn set_active(&self, n: usize) {
        self.width.store(n.clamp(1, self.n), Ordering::Relaxed);
    }

    /// Run `f(thread_id, n_threads)` on every active thread (the caller acts as
    /// thread 0) and wait for all of them to finish.
    pub fn run<F: Fn(usize, usize) + Sync>(&self, f: F) {
        let width = self.n_threads();
        if width == 1 {
            f(0, 1);
            return;
        }
        let dynf: &(dyn Fn(usize, usize) + Sync) = &f;
        // SAFETY: we block below until every participating worker has finished,
        // so the erased lifetime never outlives `f`.
        let ptr: Job = unsafe { std::mem::transmute(dynf as *const (dyn Fn(usize, usize) + Sync)) };
        let sh = &*self.shared;
        unsafe { *sh.job.get() = Some(ptr) };
        sh.done.store(0, Ordering::Relaxed);
        let generation = (sh.word.load(Ordering::Relaxed) >> 8) + 1;
        sh.word.store((generation << 8) | width as u64, Ordering::SeqCst);
        let needed = ((1u64 << width) - 1) & !1;
        if sh.sleep_mask.load(Ordering::SeqCst) & needed != 0 {
            let _g = sh.lock.lock().unwrap();
            sh.cv.notify_all();
        }
        f(0, width);
        let need = width - 1;
        let mut spins = 0usize;
        while sh.done.load(Ordering::Acquire) != need {
            spins += 1;
            if spins < (1 << 22) {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Split `0..n` into contiguous chunks (multiples of `align`) and process
    /// each chunk on a different thread.
    pub fn par_chunks<F: Fn(usize, usize) + Sync>(&self, n: usize, align: usize, f: F) {
        let nt = self.n_threads();
        if n == 0 {
            return;
        }
        let units = n.div_ceil(align);
        if nt == 1 || units == 1 {
            f(0, n);
            return;
        }
        self.run(|tid, nt| {
            let (a, b) = split_range(units, nt, tid);
            let lo = (a * align).min(n);
            let hi = (b * align).min(n);
            if lo < hi {
                f(lo, hi);
            }
        });
    }

    /// Dynamic scheduling over `n` independent tasks (work stealing via an atomic
    /// counter). Good for uneven tasks such as causal attention rows.
    pub fn par_tasks<F: Fn(usize) + Sync>(&self, n: usize, f: F) {
        if n == 0 {
            return;
        }
        if self.n_threads() == 1 || n == 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let next = AtomicUsize::new(0);
        self.run(|_, _| loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            f(i);
        });
    }
}

pub fn split_range(n: usize, parts: usize, idx: usize) -> (usize, usize) {
    let base = n / parts;
    let rem = n % parts;
    let lo = idx * base + idx.min(rem);
    let hi = lo + base + usize::from(idx < rem);
    (lo, hi)
}

fn worker_loop(sh: Arc<Shared>, tid: usize) {
    let bit = 1u64 << tid;
    let mut seen = 0u64;
    loop {
        // wait for a job whose width includes this thread
        let mut spins = 0usize;
        let width = loop {
            let w = sh.word.load(Ordering::Acquire);
            let (generation, width) = (w >> 8, (w & 0xff) as usize);
            if generation != seen {
                seen = generation;
                if tid < width {
                    break width;
                }
                spins = SPIN_ITERS; // not part of this job: likely idle for a while
            }
            if sh.quit.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins < SPIN_ITERS {
                std::hint::spin_loop();
                continue;
            }
            // park until a job that needs this thread is published
            let mut guard = sh.lock.lock().unwrap();
            sh.sleep_mask.fetch_or(bit, Ordering::SeqCst);
            loop {
                if sh.quit.load(Ordering::SeqCst) {
                    sh.sleep_mask.fetch_and(!bit, Ordering::SeqCst);
                    return;
                }
                let w = sh.word.load(Ordering::SeqCst);
                if w >> 8 != seen && tid < (w & 0xff) as usize {
                    break;
                }
                guard = sh.cv.wait(guard).unwrap();
            }
            sh.sleep_mask.fetch_and(!bit, Ordering::SeqCst);
            drop(guard);
            spins = 0;
        };
        let job = unsafe { (*sh.job.get()).expect("job set") };
        unsafe { (*job)(tid, width) };
        sh.done.fetch_add(1, Ordering::Release);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::SeqCst);
        {
            let _g = self.shared.lock.lock().unwrap();
            self.shared.cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Raw pointer wrapper that lets disjoint mutable slices be written from
/// multiple pool threads. Callers guarantee non-overlapping access.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub fn new(s: &mut [T]) -> Self {
        SyncPtr(s.as_mut_ptr())
    }
    /// Accessor (closures must capture the whole wrapper, not the raw field).
    #[inline(always)]
    pub fn get(&self) -> *mut T {
        self.0
    }
    /// # Safety
    /// The caller must ensure the range is in bounds and not aliased.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, off: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pool_runs_all_threads_and_resizes() {
        let pool = ThreadPool::new(6);
        for round in 0..3000 {
            let width = [6, 2, 4, 1, 3][round % 5];
            pool.set_active(width);
            let c = AtomicUsize::new(0);
            pool.run(|_, nt| {
                assert_eq!(nt, width);
                c.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(c.load(Ordering::Relaxed), width);
        }
        pool.set_active(6);
        let sum = AtomicUsize::new(0);
        pool.par_chunks(1003, 16, |a, b| {
            sum.fetch_add(b - a, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 1003);
    }
}
