//! Physical-placement probe for flex-mode memory (4 GB soldered + 16 GB DIMM:
//! 8 GB dual-channel, the rest single-channel). Allocates physical pages with
//! AWE (`AllocateUserPhysicalPages`, which reports each page's frame number),
//! groups them by physical address and measures each group's streaming read
//! bandwidth. Needs the "Lock pages in memory" privilege (SeLockMemoryPrivilege).
//!
//! cargo run --release --example awe_probe [MiB]
use std::arch::x86_64::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C)]
struct Luid {
    low: u32,
    high: i32,
}
#[repr(C)]
struct TokenPrivileges {
    count: u32,
    luid: Luid,
    attrs: u32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn GetLastError() -> u32;
    fn AllocateUserPhysicalPages(process: isize, pages: *mut usize, pfns: *mut usize) -> i32;
    fn FreeUserPhysicalPages(process: isize, pages: *mut usize, pfns: *mut usize) -> i32;
    fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
    fn MapUserPhysicalPages(addr: *mut c_void, pages: usize, pfns: *const usize) -> i32;
}
#[link(name = "advapi32")]
unsafe extern "system" {
    fn OpenProcessToken(process: isize, access: u32, token: *mut isize) -> i32;
    fn LookupPrivilegeValueW(system: *const u16, name: *const u16, luid: *mut Luid) -> i32;
    fn AdjustTokenPrivileges(token: isize, disable_all: i32, new: *const TokenPrivileges, len: u32, prev: *mut c_void, ret: *mut u32) -> i32;
}

const PAGE: usize = 4096;

fn enable_lock_memory_privilege() -> Result<(), String> {
    unsafe {
        let mut tok = 0isize;
        if OpenProcessToken(GetCurrentProcess(), 0x0020 | 0x0008, &mut tok) == 0 {
            return Err(format!("OpenProcessToken failed ({})", GetLastError()));
        }
        let name: Vec<u16> = "SeLockMemoryPrivilege\0".encode_utf16().collect();
        let mut luid = Luid { low: 0, high: 0 };
        if LookupPrivilegeValueW(std::ptr::null(), name.as_ptr(), &mut luid) == 0 {
            return Err(format!("LookupPrivilegeValue failed ({})", GetLastError()));
        }
        let tp = TokenPrivileges { count: 1, luid, attrs: 0x2 }; // SE_PRIVILEGE_ENABLED
        AdjustTokenPrivileges(tok, 0, &tp, 0, std::ptr::null_mut(), std::ptr::null_mut());
        match GetLastError() {
            0 => Ok(()),
            1300 => Err("this account does not hold \"Lock pages in memory\" (SeLockMemoryPrivilege)".into()),
            e => Err(format!("AdjustTokenPrivileges failed ({e})")),
        }
    }
}

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

/// Map `pfns` into a fresh virtual range and measure 4-thread read bandwidth.
fn bandwidth(pfns: &[usize]) -> f64 {
    let bytes = pfns.len() * PAGE;
    unsafe {
        let va = VirtualAlloc(std::ptr::null_mut(), bytes, 0x2000 | 0x40_0000, 0x04); // MEM_RESERVE|MEM_PHYSICAL, RW
        assert!(!va.is_null(), "VirtualAlloc failed ({})", GetLastError());
        assert!(MapUserPhysicalPages(va, pfns.len(), pfns.as_ptr()) != 0, "MapUserPhysicalPages failed ({})", GetLastError());
        std::ptr::write_bytes(va as *mut u8, 1, bytes);
        let n = bytes / 4;
        let mut best = 0f64;
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..3 {
                let per = n / 4;
                std::thread::scope(|s| {
                    for k in 0..4 {
                        let p = (va as *const f32).add(k * per) as usize;
                        s.spawn(move || std::hint::black_box(sum_stream(p as *const f32, per)));
                    }
                });
            }
            best = best.max((3 * bytes) as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
        MapUserPhysicalPages(va, pfns.len(), std::ptr::null());
        VirtualFree(va, 0, 0x8000);
        best
    }
}

fn main() {
    let mib: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    if let Err(e) = enable_lock_memory_privilege() {
        eprintln!("{e}");
        std::process::exit(1);
    }
    let mut n = mib * 1024 * 1024 / PAGE;
    let mut pfns = vec![0usize; n];
    unsafe {
        assert!(AllocateUserPhysicalPages(GetCurrentProcess(), &mut n, pfns.as_mut_ptr()) != 0, "AllocateUserPhysicalPages failed ({})", GetLastError());
    }
    pfns.truncate(n);
    let mut sorted = pfns.clone();
    sorted.sort_unstable();
    println!("got {} MiB of physical pages, addresses {:.2}-{:.2} GiB", n * PAGE >> 20, (sorted[0] * PAGE) as f64 / (1u64 << 30) as f64, (sorted[n - 1] * PAGE) as f64 / (1u64 << 30) as f64);
    // 1 GiB bins of physical address; bins with at least 64 MiB of our pages are measured
    let bin = |pfn: usize| pfn * PAGE >> 30;
    let mut b = 0;
    while b < sorted.len() {
        let g = bin(sorted[b]);
        let e = sorted[b..].iter().position(|&p| bin(p) != g).map_or(sorted.len(), |x| b + x);
        let pages = &sorted[b..e];
        if pages.len() * PAGE >= 64 << 20 {
            println!("physical {g:2}-{:2} GiB: {:5} MiB  {:5.1} GB/s", g + 1, pages.len() * PAGE >> 20, bandwidth(pages));
        } else {
            println!("physical {g:2}-{:2} GiB: {:5} MiB  (too few to measure)", g + 1, pages.len() * PAGE >> 20);
        }
        b = e;
    }
    unsafe {
        FreeUserPhysicalPages(GetCurrentProcess(), &mut n, pfns.as_mut_ptr());
    }
}
