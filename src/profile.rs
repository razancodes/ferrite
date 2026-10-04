//! Lightweight op-level profiler, enabled with FERRITE_PROFILE=1.
//! Ops are dispatched from one thread and every parallel op ends in a barrier,
//! so wall-clock timing around each op is accurate.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static ENABLED: OnceLock<bool> = OnceLock::new();
static TABLE: Mutex<Option<HashMap<&'static str, (Duration, u64)>>> = Mutex::new(None);

#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("FERRITE_PROFILE").map(|v| v != "0").unwrap_or(false))
}

pub struct Scope {
    name: &'static str,
    t0: Option<Instant>,
}

#[inline]
pub fn scope(name: &'static str) -> Scope {
    Scope { name, t0: if enabled() { Some(Instant::now()) } else { None } }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(t0) = self.t0 {
            let d = t0.elapsed();
            let mut g = TABLE.lock().unwrap();
            let e = g.get_or_insert_with(HashMap::new).entry(self.name).or_insert((Duration::ZERO, 0));
            e.0 += d;
            e.1 += 1;
        }
    }
}

pub fn reset() {
    *TABLE.lock().unwrap() = None;
}

pub fn report() {
    if !enabled() {
        return;
    }
    let g = TABLE.lock().unwrap();
    let Some(t) = g.as_ref() else { return };
    let mut v: Vec<_> = t.iter().collect();
    v.sort_by_key(|e| std::cmp::Reverse(e.1.0));
    let total: Duration = v.iter().map(|x| x.1.0).sum();
    eprintln!("\n{:<24} {:>10} {:>8} {:>10} {:>6}", "op", "total ms", "calls", "us/call", "%");
    for (name, (d, n)) in v {
        eprintln!(
            "{:<24} {:>10.1} {:>8} {:>10.1} {:>5.1}%",
            name,
            d.as_secs_f64() * 1e3,
            n,
            d.as_secs_f64() * 1e6 / *n as f64,
            d.as_secs_f64() / total.as_secs_f64() * 100.0
        );
    }
}
