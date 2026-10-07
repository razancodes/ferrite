//! The engine lives on its own thread; the UI talks to it through commands
//! and receives a stream of `engine` events (loading, deltas, stats).

use ferrite::chat::{ChatTemplate, Message};
use ferrite::engine::{DRAFT_VOCAB, Engine, GenStats, StopReason};
use ferrite::sampler::{Sampler, SamplerConfig};
use ferrite::stream::TextRouter;
use ferrite::tokenizer::StreamDecoder;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;
use tauri::{AppHandle, Emitter};

pub enum Cmd {
    Load { path: PathBuf, threads: usize, ctx: usize, spec: bool },
    SetSpec(bool),
    Generate { messages: Vec<Message>, thinking: bool, sampler: SamplerConfig, max_tokens: usize },
    Reset,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub layers: usize,
    pub linear_layers: usize,
    pub attn_layers: usize,
    pub hidden: usize,
    pub params_b: f64,
    pub weights_gb: f64,
    pub recipe: String,
    pub gptq: bool,
    /// the 2-bit proxy LM head shortlists the vocabulary for the exact head
    pub proxy_head: bool,
    pub in_ram: bool,
    pub has_mtp: bool,
    pub spec: bool,
    pub threads: usize,
    pub decode_threads: usize,
    pub ctx: usize,
    pub isa: String,
    pub load_secs: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DoneStats {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub prefill_tps: f64,
    pub gen_tokens: usize,
    pub gen_tps: f64,
    pub drafted: usize,
    pub accepted: usize,
    pub acceptance: f64,
    /// drafts proposed by prompt lookup (the rest came from the MTP head)
    pub lookup: usize,
    pub ttft_ms: f64,
    pub stop: &'static str,
    pub context_used: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Loading { name: String },
    Loaded { info: ModelInfo },
    LoadFailed { error: String },
    SpecChanged { on: bool },
    /// the prompt is being read (`tokens` new tokens, `cached` reused)
    Prefill { tokens: usize },
    Delta { reasoning: bool, text: String, tokens: usize },
    Done { stats: DoneStats },
    Failed { error: String },
}

pub fn spawn(app: AppHandle, stop: Arc<AtomicBool>) -> Sender<Cmd> {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("ferrite-engine".into())
        .spawn(move || run(rx, app, stop))
        .expect("spawn engine thread");
    tx
}

fn run(rx: Receiver<Cmd>, app: AppHandle, stop: Arc<AtomicBool>) {
    let send = |e: Event| {
        let _ = app.emit("engine", e);
    };
    let mut engine: Option<Engine> = None;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Load { path, threads, ctx, spec } => {
                let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                send(Event::Loading { name: name.clone() });
                drop(engine.take()); // release the previous model's memory first
                let t0 = Instant::now();
                match Engine::load(&path, threads, ctx) {
                    Ok(mut e) => {
                        let spec_on = spec && e.enable_spec(DRAFT_VOCAB).unwrap_or(false);
                        let c = &e.model.cfg;
                        let meta = &e.model.file.header.meta;
                        let info = ModelInfo {
                            name,
                            path: path.display().to_string(),
                            layers: c.n_layers,
                            linear_layers: c.n_linear(),
                            attn_layers: c.n_full(),
                            hidden: c.hidden,
                            params_b: meta.get("params").and_then(|v| v.as_f64()).unwrap_or(0.0) / 1e9,
                            weights_gb: e.model.weight_bytes() as f64 / 1e9,
                            recipe: meta.get("recipe").and_then(|v| v.as_str()).unwrap_or("?").to_string(),
                            gptq: meta.get("gptq").and_then(|v| v.as_bool()).unwrap_or(false),
                            proxy_head: e.fast_head && e.model.lm_proxy.is_some(),
                            in_ram: matches!(e.model.file.mmap, ferrite::format::Storage::Heap { .. }),
                            has_mtp: e.model.mtp.is_some(),
                            spec: spec_on,
                            threads: e.pool.max_threads(),
                            decode_threads: e.decode_threads,
                            ctx: e.state.max_ctx,
                            isa: format!("{:?}", ferrite::kernels::isa()),
                            load_secs: t0.elapsed().as_secs_f64(),
                        };
                        engine = Some(e);
                        send(Event::Loaded { info });
                    }
                    Err(err) => send(Event::LoadFailed { error: format!("{err:#}") }),
                }
            }
            Cmd::SetSpec(on) => {
                if let Some(e) = engine.as_mut() {
                    let now = if on {
                        e.enable_spec(DRAFT_VOCAB).unwrap_or(false)
                    } else {
                        e.spec = false;
                        e.reset();
                        false
                    };
                    send(Event::SpecChanged { on: now });
                }
            }
            Cmd::Reset => {
                if let Some(e) = engine.as_mut() {
                    e.reset();
                }
            }
            Cmd::Generate { messages, thinking, sampler, max_tokens } => {
                let Some(e) = engine.as_mut() else {
                    send(Event::Failed { error: "No model is loaded.".into() });
                    continue;
                };
                match generate(e, &messages, thinking, sampler, max_tokens, &stop, &send) {
                    Ok((stats, ttft)) => {
                        let stop_reason = match stats.stop {
                            StopReason::Eos => "eos",
                            StopReason::Length => "length",
                            StopReason::Callback => "stopped",
                            StopReason::Context => "context",
                        };
                        send(Event::Done {
                            stats: DoneStats {
                                prompt_tokens: stats.prompt_tokens,
                                cached_tokens: stats.cached_tokens,
                                prefill_tps: stats.prefill_tps(),
                                gen_tokens: stats.gen_tokens,
                                gen_tps: stats.gen_tps(),
                                drafted: stats.drafted,
                                accepted: stats.accepted,
                                acceptance: stats.acceptance(),
                                lookup: stats.lookup_drafted,
                                ttft_ms: ttft * 1e3,
                                stop: stop_reason,
                                context_used: e.pos(),
                            },
                        })
                    }
                    Err(err) => send(Event::Failed { error: format!("{err:#}") }),
                }
            }
        }
    }
}

fn generate(
    e: &mut Engine,
    messages: &[Message],
    thinking: bool,
    sampler: SamplerConfig,
    max_tokens: usize,
    stop: &AtomicBool,
    send: &dyn Fn(Event),
) -> anyhow::Result<(GenStats, f64)> {
    let t0 = Instant::now();
    let (prompt, head, eos) = {
        let tok = &e.model.tokenizer;
        let tmpl = ChatTemplate::new(tok);
        // snapshot right before the assistant header so the next turn (or a
        // regenerate) resumes from here instead of re-reading the conversation
        let head = tmpl.render(tok, messages, false, thinking).len();
        (tmpl.render(tok, messages, true, thinking), head, tmpl.eos.clone())
    };
    if prompt.len() + 16 >= e.state.max_ctx {
        anyhow::bail!(
            "The conversation is too long for the context window ({} tokens, limit {}). Start a new chat or raise the context size.",
            prompt.len(),
            e.state.max_ctx
        );
    }
    send(Event::Prefill { tokens: prompt.len() });
    let max_new = max_tokens.min(e.state.max_ctx - prompt.len() - 1);
    let mut sampler = Sampler::new(sampler);
    let mut dec = StreamDecoder::default();
    let mut router = TextRouter::new(thinking, Vec::new());
    let mut n = 0usize;
    let mut ttft = None;
    let emit = |kind: &str, text: &str, n: usize| {
        send(Event::Delta { reasoning: kind == "reasoning_content", text: text.to_string(), tokens: n });
        true
    };
    let stats = e.generate(&prompt, Some(head), &mut sampler, max_new, &eos, |eng, t| {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        if ttft.is_none() {
            ttft = Some(t0.elapsed().as_secs_f64());
        }
        n += 1;
        let piece = dec.push(&eng.model.tokenizer, t);
        router.push(&piece, &mut |k, s| emit(k, s, n))
    })?;
    let tail = dec.flush();
    if !tail.is_empty() {
        router.push(&tail, &mut |k, s| emit(k, s, n));
    }
    router.finish(&mut |k, s| emit(k, s, n));
    Ok((stats, ttft.unwrap_or(0.0)))
}
