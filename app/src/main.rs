//! ferrite chat: desktop app (Tauri) for local models on the ferrite CPU engine.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod worker;

use ferrite::chat::Message;
use ferrite::sampler::SamplerConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use tauri::State;
use worker::Cmd;

struct AppState {
    tx: Mutex<Sender<Cmd>>,
    stop: Arc<AtomicBool>,
}

#[derive(Serialize)]
struct ModelEntry {
    name: String,
    path: String,
    size_gb: f64,
    quant: String,
    params_b: f64,
    layers: usize,
}

#[derive(Serialize)]
struct SystemInfo {
    cpu: String,
    logical_cores: usize,
    default_threads: usize,
    isa: String,
}

#[derive(Deserialize)]
struct SamplerOpts {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    presence_penalty: f32,
    seed: u64,
}

fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![PathBuf::from("models")];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.join("models"));
            dirs.push(d.join("../../models")); // target/release -> repo root
        }
    }
    dirs
}

fn describe(p: &Path) -> Option<ModelEntry> {
    let size_gb = std::fs::metadata(p).ok()?.len() as f64 / 1e9;
    let name = p.file_stem()?.to_string_lossy().to_string();
    // read only the header (the file is memory-mapped lazily and closed again)
    let (params_b, layers, recipe) = match ferrite::format::ModelFile::open(p) {
        Ok(f) => (
            f.header.meta.get("params").and_then(|v| v.as_f64()).unwrap_or(0.0) / 1e9,
            f.header.config.n_layers,
            f.header.meta.get("recipe").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        ),
        Err(_) => (0.0, 0, String::new()),
    };
    let quant = if !recipe.is_empty() {
        recipe.split(',').next().unwrap_or("").to_string()
    } else {
        name.rsplit('-').next().unwrap_or("").to_string()
    };
    Some(ModelEntry { name, path: p.display().to_string(), size_gb, quant, params_b, layers })
}

#[tauri::command]
fn list_models() -> Vec<ModelEntry> {
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut out = Vec::new();
    for d in model_dirs() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "ferrite").unwrap_or(false))
            .collect();
        files.sort();
        for p in files {
            let canon = p.canonicalize().unwrap_or(p.clone());
            if seen.contains(&canon) {
                continue;
            }
            seen.push(canon.clone());
            // skip the f32 reference conversions: huge and slow for chat
            if canon.to_string_lossy().contains("-f32") {
                continue;
            }
            if let Some(m) = describe(&canon) {
                out.push(m);
            }
        }
    }
    out
}

#[tauri::command]
fn describe_model(path: String) -> Option<ModelEntry> {
    describe(Path::new(&path))
}

#[tauri::command]
fn system_info() -> SystemInfo {
    SystemInfo {
        cpu: cpu_brand(),
        logical_cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        default_threads: ferrite::engine::default_threads(),
        isa: format!("{:?}", ferrite::kernels::isa()),
    }
}

#[tauri::command]
fn load_model(state: State<AppState>, path: String, threads: Option<usize>, ctx: Option<usize>, spec: bool) {
    let _ = state.tx.lock().unwrap().send(Cmd::Load {
        path: PathBuf::from(path),
        threads: threads.unwrap_or_else(ferrite::engine::default_threads),
        ctx: ctx.unwrap_or(8192),
        spec,
    });
}

#[tauri::command]
fn chat(state: State<AppState>, messages: Vec<Message>, thinking: bool, sampler: SamplerOpts, max_tokens: usize) {
    state.stop.store(false, Ordering::SeqCst);
    let cfg = SamplerConfig {
        temperature: sampler.temperature,
        top_k: sampler.top_k,
        top_p: sampler.top_p,
        min_p: sampler.min_p,
        presence_penalty: sampler.presence_penalty,
        frequency_penalty: 0.0,
        repetition_penalty: 1.0,
        seed: sampler.seed,
    };
    let _ = state.tx.lock().unwrap().send(Cmd::Generate { messages, thinking, sampler: cfg, max_tokens });
}

#[tauri::command]
fn stop(state: State<AppState>) {
    state.stop.store(true, Ordering::SeqCst);
}

#[tauri::command]
fn reset(state: State<AppState>) {
    let _ = state.tx.lock().unwrap().send(Cmd::Reset);
}

#[tauri::command]
fn set_spec(state: State<AppState>, on: bool) {
    let _ = state.tx.lock().unwrap().send(Cmd::SetSpec(on));
}

/// FERRITE_CHAT_AUTOSEND="msg1||msg2": messages the UI sends automatically once a
/// model is ready (scripted demos and testing).
#[tauri::command]
fn autosend() -> Vec<String> {
    std::env::var("FERRITE_CHAT_AUTOSEND")
        .map(|s| s.split("||").map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
        .unwrap_or_default()
}

#[cfg(target_arch = "x86_64")]
fn cpu_brand() -> String {
    use std::arch::x86_64::__cpuid;
    // CPUID leaves 0x80000002..4 hold the 48-byte processor brand string
    let mut bytes = Vec::with_capacity(48);
    for leaf in 0x8000_0002u32..=0x8000_0004 {
        let r = __cpuid(leaf);
        for reg in [r.eax, r.ebx, r.ecx, r.edx] {
            bytes.extend_from_slice(&reg.to_le_bytes());
        }
    }
    let s = String::from_utf8_lossy(&bytes).trim_matches(char::from(0)).trim().to_string();
    s.replace("(R)", "").replace("(TM)", "").replace("  ", " ")
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_brand() -> String {
    "CPU".into()
}

fn main() {
    // run from the repo root when launched from target/release so ./models resolves
    if let Ok(exe) = std::env::current_exe() {
        if let Some(root) = exe.parent().and_then(|d| d.parent()).and_then(|d| d.parent()) {
            if root.join("models").is_dir() && !Path::new("models").is_dir() {
                let _ = std::env::set_current_dir(root);
            }
        }
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            use tauri::Manager;
            let stop = Arc::new(AtomicBool::new(false));
            let tx = worker::spawn(app.handle().clone(), stop.clone());
            app.manage(AppState { tx: Mutex::new(tx), stop });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![list_models, describe_model, system_info, load_model, chat, stop, reset, set_spec, autosend])
        .run(tauri::generate_context!())
        .expect("error while running ferrite chat");
}
