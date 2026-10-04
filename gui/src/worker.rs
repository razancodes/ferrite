//! The engine runs on its own thread so the UI never blocks: the UI sends
//! commands, the worker streams events back and asks egui to repaint.

use ferrite::chat::{ChatTemplate, Message};
use ferrite::engine::{DRAFT_VOCAB, Engine, GenStats};
use ferrite::sampler::{Sampler, SamplerConfig};
use ferrite::stream::TextRouter;
use ferrite::tokenizer::StreamDecoder;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

pub enum Cmd {
    Load { path: PathBuf, threads: usize, ctx: usize, spec: bool },
    SetSpec(bool),
    Generate { messages: Vec<Message>, thinking: bool, sampler: SamplerConfig, max_tokens: usize },
    Reset,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub layers: usize,
    pub linear_layers: usize,
    pub attn_layers: usize,
    pub hidden: usize,
    pub params_b: f64,
    pub weights_gb: f64,
    pub recipe: String,
    pub has_mtp: bool,
    pub spec: bool,
    pub threads: usize,
    pub decode_threads: usize,
    pub ctx: usize,
    pub isa: String,
    pub load_secs: f64,
}

pub enum Event {
    Loading(String),
    Loaded(ModelInfo),
    LoadFailed(String),
    SpecChanged(bool),
    /// streamed text; `tokens` = tokens generated so far in this reply
    Delta { reasoning: bool, text: String, tokens: usize },
    Done { stats: GenStats, context_used: usize },
    Failed(String),
}

pub fn spawn(ui: eframe::egui::Context, stop: Arc<AtomicBool>) -> (Sender<Cmd>, Receiver<Event>) {
    let (cmd_tx, cmd_rx) = channel();
    let (ev_tx, ev_rx) = channel();
    std::thread::Builder::new()
        .name("ferrite-engine".into())
        .spawn(move || run(cmd_rx, ev_tx, ui, stop))
        .expect("spawn engine thread");
    (cmd_tx, ev_rx)
}

fn run(rx: Receiver<Cmd>, tx: Sender<Event>, ui: eframe::egui::Context, stop: Arc<AtomicBool>) {
    let send = |e: Event| {
        let _ = tx.send(e);
        ui.request_repaint();
    };
    let mut engine: Option<Engine> = None;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Load { path, threads, ctx, spec } => {
                let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                send(Event::Loading(name.clone()));
                drop(engine.take()); // release the previous model's memory first
                let t0 = Instant::now();
                match Engine::load(&path, threads, ctx) {
                    Ok(mut e) => {
                        let spec_on = spec && e.enable_spec(DRAFT_VOCAB).unwrap_or(false);
                        let c = &e.model.cfg;
                        let meta = &e.model.file.header.meta;
                        let info = ModelInfo {
                            name,
                            layers: c.n_layers,
                            linear_layers: c.n_linear(),
                            attn_layers: c.n_full(),
                            hidden: c.hidden,
                            params_b: meta.get("params").and_then(|v| v.as_f64()).unwrap_or(0.0) / 1e9,
                            weights_gb: e.model.weight_bytes() as f64 / 1e9,
                            recipe: meta.get("recipe").and_then(|v| v.as_str()).unwrap_or("?").to_string(),
                            has_mtp: e.model.mtp.is_some(),
                            spec: spec_on,
                            threads: e.pool.max_threads(),
                            decode_threads: e.decode_threads,
                            ctx: e.state.max_ctx,
                            isa: format!("{:?}", ferrite::kernels::isa()),
                            load_secs: t0.elapsed().as_secs_f64(),
                        };
                        engine = Some(e);
                        send(Event::Loaded(info));
                    }
                    Err(err) => send(Event::LoadFailed(format!("{err:#}"))),
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
                    send(Event::SpecChanged(now));
                }
            }
            Cmd::Reset => {
                if let Some(e) = engine.as_mut() {
                    e.reset();
                }
            }
            Cmd::Generate { messages, thinking, sampler, max_tokens } => {
                let Some(e) = engine.as_mut() else {
                    send(Event::Failed("no model loaded".into()));
                    continue;
                };
                match generate(e, &messages, thinking, sampler, max_tokens, &stop, &send) {
                    Ok(stats) => send(Event::Done { stats, context_used: e.pos() }),
                    Err(err) => send(Event::Failed(format!("{err:#}"))),
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
) -> anyhow::Result<GenStats> {
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
            "conversation is too long for the context window ({} tokens, limit {}). Start a new chat or raise the context size.",
            prompt.len(),
            e.state.max_ctx
        );
    }
    let max_new = max_tokens.min(e.state.max_ctx - prompt.len() - 1);
    let mut sampler = Sampler::new(sampler);
    let mut dec = StreamDecoder::default();
    let mut router = TextRouter::new(thinking, Vec::new());
    let mut n = 0usize;
    let emit = |kind: &str, text: &str, n: usize| {
        send(Event::Delta { reasoning: kind == "reasoning_content", text: text.to_string(), tokens: n });
        true
    };
    let stats = e.generate(&prompt, Some(head), &mut sampler, max_new, &eos, |eng, t| {
        if stop.load(Ordering::Relaxed) {
            return false;
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
    Ok(stats)
}
