//! ferrite chat: a desktop chat app for local models running on the ferrite
//! CPU inference engine.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod mathfmt;
mod worker;

use eframe::egui::{self, Color32, RichText};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use ferrite::chat::Message;
use ferrite::sampler::SamplerConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use worker::{Cmd, Event, ModelInfo};

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
struct Settings {
    model_path: String,
    threads: usize,
    ctx: usize,
    spec: bool,
    thinking: bool,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    presence_penalty: f32,
    max_tokens: usize,
    seed: u64,
    system: String,
    show_settings: bool,
}

impl Default for Settings {
    fn default() -> Self {
        let d = SamplerConfig::default();
        Settings {
            model_path: String::new(),
            threads: ferrite::engine::default_threads(),
            ctx: 8192,
            spec: true,
            thinking: false,
            temperature: d.temperature,
            top_p: d.top_p,
            top_k: d.top_k,
            min_p: d.min_p,
            presence_penalty: d.presence_penalty,
            max_tokens: 2048,
            seed: 0,
            system: String::new(),
            show_settings: true,
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Role {
    User,
    Assistant,
}

struct ChatMsg {
    /// unique per message and app session: keys egui widget state (which eframe
    /// persists across restarts) so a new chat never inherits stale UI state
    uid: u64,
    role: Role,
    content: String,
    reasoning: String,
    stats: Option<String>,
    error: Option<String>,
    streaming: bool,
}

impl ChatMsg {
    fn new(role: Role, content: &str) -> Self {
        use std::sync::atomic::AtomicU64;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        let _ = NEXT.compare_exchange(0, seed, Ordering::Relaxed, Ordering::Relaxed);
        let uid = NEXT.fetch_add(1, Ordering::Relaxed);
        ChatMsg { uid, role, content: content.into(), reasoning: String::new(), stats: None, error: None, streaming: false }
    }
}

struct App {
    s: Settings,
    tx: Sender<Cmd>,
    rx: Receiver<Event>,
    stop: Arc<AtomicBool>,
    model: Option<ModelInfo>,
    loading: Option<String>,
    load_error: Option<String>,
    messages: Vec<ChatMsg>,
    input: String,
    generating: bool,
    gen_start: Option<Instant>,
    first_token: Option<Instant>,
    live_tokens: usize,
    context_used: usize,
    md: CommonMarkCache,
    models: Vec<PathBuf>,
    /// FERRITE_CHAT_AUTOSEND: messages (separated by "||") sent automatically one
    /// after another once a model is ready (testing / scripting)
    autosend: std::collections::VecDeque<String>,
}

const EXAMPLES: [&str; 4] = [
    "Explain how a CPU cache works, with an analogy.",
    "Write a Python function that merges overlapping intervals, with tests.",
    "Give me a 3-day itinerary for Lahore.",
    "What's the difference between linear attention and softmax attention?",
];

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

fn scan_models() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for d in model_dirs() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x == "ferrite").unwrap_or(false) {
                let p = p.canonicalize().unwrap_or(p);
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

fn file_label(p: &Path) -> String {
    let size = std::fs::metadata(p).map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);
    format!("{}  ({size:.2} GB)", p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default())
}

fn install_fonts(ctx: &egui::Context) {
    // Segoe UI for text and a CJK fallback when available (egui's built-in
    // fonts cover Latin, Greek, Cyrillic and emoji only).
    let mut fonts = egui::FontDefinitions::default();
    let mut add = |name: &str, path: &str, primary: bool| {
        if let Ok(bytes) = std::fs::read(path) {
            fonts.font_data.insert(name.to_owned(), Arc::new(egui::FontData::from_owned(bytes)));
            let fam = fonts.families.entry(egui::FontFamily::Proportional).or_default();
            if primary { fam.insert(0, name.to_owned()) } else { fam.push(name.to_owned()) }
        }
    };
    add("segoe", "C:/Windows/Fonts/segoeui.ttf", true);
    add("cjk", "C:/Windows/Fonts/msyh.ttc", false);
    ctx.set_fonts(fonts);
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_fonts(&cc.egui_ctx);
        cc.egui_ctx.all_styles_mut(|st| {
            st.spacing.item_spacing = egui::vec2(8.0, 6.0);
            st.text_styles.insert(egui::TextStyle::Body, egui::FontId::proportional(15.0));
            st.text_styles.insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
        });
        let s: Settings = cc.storage.and_then(|st| eframe::get_value(st, eframe::APP_KEY)).unwrap_or_default();
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = worker::spawn(cc.egui_ctx.clone(), stop.clone());
        let models = scan_models();
        let mut app = App {
            s,
            tx,
            rx,
            stop,
            model: None,
            loading: None,
            load_error: None,
            messages: Vec::new(),
            input: String::new(),
            generating: false,
            gen_start: None,
            first_token: None,
            live_tokens: 0,
            context_used: 0,
            md: CommonMarkCache::default(),
            models,
            autosend: std::env::var("FERRITE_CHAT_AUTOSEND")
                .map(|s| s.split("||").map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
                .unwrap_or_default(),
        };
        // reopen the last model, or pick a sensible default (prefer 2B q6)
        if app.s.model_path.is_empty() || !Path::new(&app.s.model_path).exists() {
            let pick = app
                .models
                .iter()
                .find(|p| p.to_string_lossy().contains("2b-q6"))
                .or_else(|| app.models.first())
                .cloned();
            if let Some(p) = pick {
                app.s.model_path = p.display().to_string();
            }
        }
        if !app.s.model_path.is_empty() {
            app.load();
        }
        app
    }

    fn load(&mut self) {
        if self.generating {
            return;
        }
        self.load_error = None;
        self.model = None;
        self.loading = Some(String::new());
        let _ = self.tx.send(Cmd::Load {
            path: PathBuf::from(&self.s.model_path),
            threads: self.s.threads,
            ctx: self.s.ctx,
            spec: self.s.spec,
        });
    }

    fn sampler(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.s.temperature,
            top_k: self.s.top_k,
            top_p: self.s.top_p,
            min_p: self.s.min_p,
            presence_penalty: self.s.presence_penalty,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: self.s.seed,
        }
    }

    /// Ask the model to answer the conversation as it stands.
    fn start_generation(&mut self) {
        let mut history = Vec::new();
        if !self.s.system.trim().is_empty() {
            history.push(Message::new("system", &self.s.system));
        }
        for m in &self.messages {
            let role = if m.role == Role::User { "user" } else { "assistant" };
            history.push(Message::new(role, &m.content));
        }
        let mut reply = ChatMsg::new(Role::Assistant, "");
        reply.streaming = true;
        self.messages.push(reply);
        self.stop.store(false, Ordering::SeqCst);
        self.generating = true;
        self.gen_start = Some(Instant::now());
        self.first_token = None;
        self.live_tokens = 0;
        let _ = self.tx.send(Cmd::Generate {
            messages: history,
            thinking: self.s.thinking,
            sampler: self.sampler(),
            max_tokens: self.s.max_tokens,
        });
    }

    fn send(&mut self, text: String) {
        let mut text = text.trim().to_string();
        // Qwen-style soft switches: "/think ..." or "/nothink ..." toggle thinking mode
        for (cmd, on) in [("/think", true), ("/nothink", false), ("/no_think", false)] {
            if let Some(rest) = text.strip_prefix(cmd) {
                if rest.is_empty() || rest.starts_with(char::is_whitespace) {
                    self.set_thinking(on);
                    text = rest.trim().to_string();
                    break;
                }
            }
        }
        if text.is_empty() || self.generating || self.model.is_none() {
            return;
        }
        self.messages.push(ChatMsg::new(Role::User, &text));
        self.start_generation();
    }

    fn set_thinking(&mut self, on: bool) {
        if self.s.thinking != on {
            self.s.thinking = on;
            let d = if on { SamplerConfig::thinking() } else { SamplerConfig::default() };
            self.s.temperature = d.temperature;
            self.s.top_p = d.top_p;
        }
    }

    fn regenerate(&mut self) {
        if self.generating {
            return;
        }
        if self.messages.last().map(|m| m.role == Role::Assistant).unwrap_or(false) {
            self.messages.pop();
        }
        if self.messages.last().map(|m| m.role == Role::User).unwrap_or(false) {
            self.start_generation();
        }
    }

    fn new_chat(&mut self) {
        if self.generating {
            self.stop.store(true, Ordering::SeqCst);
        }
        self.messages.clear();
        self.context_used = 0;
        let _ = self.tx.send(Cmd::Reset);
    }

    fn poll(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                Event::Loading(name) => self.loading = Some(name),
                Event::Loaded(info) => {
                    self.loading = None;
                    self.s.spec = info.spec || !info.has_mtp && self.s.spec;
                    self.model = Some(info);
                }
                Event::LoadFailed(err) => {
                    self.loading = None;
                    self.load_error = Some(err);
                }
                Event::SpecChanged(on) => {
                    if let Some(m) = self.model.as_mut() {
                        m.spec = on;
                    }
                }
                Event::Delta { reasoning, text, tokens } => {
                    if self.first_token.is_none() {
                        self.first_token = Some(Instant::now());
                    }
                    self.live_tokens = tokens;
                    if let Some(m) = self.messages.last_mut() {
                        if reasoning { m.reasoning.push_str(&text) } else { m.content.push_str(&text) }
                    }
                }
                Event::Done { stats, context_used } => {
                    self.generating = false;
                    self.context_used = context_used;
                    if let Some(m) = self.messages.last_mut() {
                        m.streaming = false;
                        let mut s = format!(
                            "{:.1} tok/s · {} tokens · prompt {} tok",
                            stats.gen_tps(),
                            stats.gen_tokens,
                            stats.prompt_tokens - stats.cached_tokens
                        );
                        if stats.cached_tokens > 0 {
                            s += &format!(" (+{} cached)", stats.cached_tokens);
                        }
                        if stats.prompt_tokens > stats.cached_tokens {
                            s += &format!(" @ {:.0} tok/s", stats.prefill_tps());
                        }
                        if stats.drafted > 0 {
                            s += &format!(" · draft acceptance {:.0}%", stats.acceptance() * 100.0);
                        }
                        if let (Some(a), Some(b)) = (self.gen_start, self.first_token) {
                            s += &format!(" · first token {:.0} ms", (b - a).as_secs_f64() * 1e3);
                        }
                        if self.stop.load(Ordering::SeqCst) {
                            s += " · stopped";
                        }
                        m.stats = Some(s);
                    }
                }
                Event::Failed(err) => {
                    self.generating = false;
                    if let Some(m) = self.messages.last_mut() {
                        m.streaming = false;
                        m.error = Some(err);
                    }
                }
            }
        }
    }

    fn settings_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.heading("Model");
        let current = if self.s.model_path.is_empty() { "choose a model…".to_string() } else { file_label(Path::new(&self.s.model_path)) };
        let mut chosen: Option<String> = None;
        egui::ComboBox::from_id_salt("model").width(ui.available_width()).selected_text(current).show_ui(ui, |ui| {
            for p in &self.models {
                let ps = p.display().to_string();
                if ui.selectable_label(ps == self.s.model_path, file_label(p)).clicked() {
                    chosen = Some(ps);
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Browse…").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("ferrite model", &["ferrite"]).pick_file() {
                    chosen = Some(p.display().to_string());
                }
            }
            if ui.button("⟳ Rescan").on_hover_text("Look for .ferrite files in ./models again").clicked() {
                self.models = scan_models();
            }
        });
        if let Some(p) = chosen {
            if p != self.s.model_path || self.model.is_none() {
                self.s.model_path = p;
                self.messages.clear();
                self.load();
            }
        }
        egui::CollapsingHeader::new("Engine").default_open(false).show(ui, |ui| {
            let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
            ui.add(egui::Slider::new(&mut self.s.threads, 1..=max_threads).text("threads"));
            ui.add(egui::Slider::new(&mut self.s.ctx, 1024..=65536).logarithmic(true).text("context"));
            if ui.add_enabled(!self.generating, egui::Button::new("Reload model")).clicked() {
                self.messages.clear();
                self.load();
            }
        });
        if let Some(name) = &self.loading {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(if name.is_empty() { "Loading…".to_string() } else { format!("Loading {name}…") });
            });
        }
        if let Some(err) = &self.load_error {
            ui.colored_label(Color32::from_rgb(220, 80, 60), format!("Load failed: {err}"));
        }
        if let Some(m) = &self.model {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(&m.name).strong());
                ui.label(
                    RichText::new(format!(
                        "{:.2}B params · {} · {:.2} GB\n{} layers ({} DeltaNet + {} attention), hidden {}\n{} threads ({} for decode) · {} · ctx {}\nloaded in {:.1}s",
                        m.params_b,
                        m.recipe,
                        m.weights_gb,
                        m.layers,
                        m.linear_layers,
                        m.attn_layers,
                        m.hidden,
                        m.threads,
                        m.decode_threads,
                        m.isa,
                        m.ctx,
                        m.load_secs
                    ))
                    .size(12.5)
                    .weak(),
                );
            });
        }

        ui.separator();
        ui.heading("Generation");
        let has_mtp = self.model.as_ref().map(|m| m.has_mtp).unwrap_or(true);
        let mut spec = self.s.spec;
        if ui
            .add_enabled(has_mtp && !self.generating, egui::Checkbox::new(&mut spec, "MTP speculative decoding"))
            .on_hover_text("Drafts several tokens per step with the model's own prediction head and verifies them in one pass. Same output, 1.5-2x faster.")
            .changed()
        {
            self.s.spec = spec;
            let _ = self.tx.send(Cmd::SetSpec(spec));
            self.messages.clear();
        }
        let mut thinking = self.s.thinking;
        if ui
            .checkbox(&mut thinking, "Thinking mode")
            .on_hover_text("Let the model reason step by step before answering (or start a message with /think or /nothink)")
            .changed()
        {
            self.set_thinking(thinking);
        }
        ui.add(egui::Slider::new(&mut self.s.temperature, 0.0..=2.0).text("temperature"));
        ui.add(egui::Slider::new(&mut self.s.top_p, 0.05..=1.0).text("top-p"));
        ui.add(egui::Slider::new(&mut self.s.top_k, 0..=100).text("top-k"));
        ui.add(egui::Slider::new(&mut self.s.min_p, 0.0..=0.5).text("min-p"));
        ui.add(egui::Slider::new(&mut self.s.presence_penalty, 0.0..=2.0).text("presence penalty"));
        ui.add(egui::Slider::new(&mut self.s.max_tokens, 16..=16384).logarithmic(true).text("max tokens"));
        ui.horizontal(|ui| {
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut self.s.seed)).on_hover_text("0 = random");
        });
        if ui.small_button("Reset to Qwen defaults").clicked() {
            let d = if self.s.thinking { SamplerConfig::thinking() } else { SamplerConfig::default() };
            self.s.temperature = d.temperature;
            self.s.top_p = d.top_p;
            self.s.top_k = d.top_k;
            self.s.min_p = d.min_p;
            self.s.presence_penalty = d.presence_penalty;
        }

        ui.separator();
        ui.heading("System prompt");
        ui.add(
            egui::TextEdit::multiline(&mut self.s.system)
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .hint_text("e.g. You are a concise assistant."),
        );
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            egui::widgets::global_theme_preference_buttons(ui);
        });
    }

    fn message_ui(ui: &mut egui::Ui, md: &mut CommonMarkCache, m: &ChatMsg, idx: usize, last: bool, actions: &mut Vec<(usize, &'static str)>) {
        let dark = ui.visuals().dark_mode;
        match m.role {
            Role::User => {
                let max_w = ui.available_width() * 0.75;
                ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                    let fill = if dark { Color32::from_rgb(52, 58, 70) } else { Color32::from_rgb(226, 232, 242) };
                    egui::Frame::new().fill(fill).corner_radius(12.0).inner_margin(egui::Margin::symmetric(12, 8)).show(ui, |ui| {
                        ui.set_max_width(max_w - 24.0);
                        ui.add(egui::Label::new(&m.content).wrap());
                    });
                });
            }
            Role::Assistant => {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("◆").color(Color32::from_rgb(214, 120, 70)));
                    ui.label(RichText::new("Assistant").strong());
                    if m.streaming && m.content.is_empty() && m.reasoning.is_empty() {
                        ui.spinner();
                        ui.label(RichText::new("reading prompt…").weak());
                    }
                });
                if !m.reasoning.is_empty() {
                    let reasoning = mathfmt::display_text(m.reasoning.trim());
                    if m.streaming && m.content.is_empty() {
                        // live reasoning while the model thinks
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(RichText::new("Thinking…").weak());
                        });
                        egui::Frame::new().inner_margin(egui::Margin { left: 12, ..Default::default() }).show(ui, |ui| {
                            ui.add(egui::Label::new(RichText::new(reasoning).weak().size(13.5)).wrap());
                        });
                    } else {
                        // folded away once the answer starts
                        egui::CollapsingHeader::new(RichText::new("Thoughts").weak())
                            .id_salt(("think", m.uid))
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.add(egui::Label::new(RichText::new(reasoning).weak().size(13.5)).wrap());
                            });
                    }
                }
                if !m.content.is_empty() {
                    CommonMarkViewer::new().show(ui, md, &mathfmt::display_text(&m.content));
                }
                if let Some(e) = &m.error {
                    ui.colored_label(Color32::from_rgb(220, 80, 60), e);
                }
                if !m.streaming {
                    ui.horizontal(|ui| {
                        if let Some(s) = &m.stats {
                            ui.label(RichText::new(s).size(12.5).weak());
                        }
                        if ui.small_button("Copy").clicked() {
                            actions.push((idx, "copy"));
                        }
                        if last && ui.small_button("Regenerate").clicked() {
                            actions.push((idx, "regen"));
                        }
                    });
                }
            }
        }
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        if self.model.is_some() && !self.generating {
            if let Some(p) = self.autosend.pop_front() {
                self.send(p);
            }
        }
        if self.generating {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(self.s.show_settings, "☰").on_hover_text("Settings").clicked() {
                    self.s.show_settings = !self.s.show_settings;
                }
                ui.label(RichText::new("ferrite").strong().size(18.0).color(Color32::from_rgb(214, 120, 70)));
                ui.label(RichText::new("chat").size(18.0));
                ui.separator();
                match (&self.model, &self.loading) {
                    (_, Some(_)) => {
                        ui.spinner();
                        ui.label("loading model…");
                    }
                    (Some(m), None) => {
                        ui.label(RichText::new(&m.name).strong());
                        ui.label(RichText::new(format!("{} · {}", m.isa, if m.spec { "MTP speculation" } else { "no speculation" })).weak());
                    }
                    (None, None) => {
                        ui.label(RichText::new("no model loaded").weak());
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("New chat").clicked() {
                        self.new_chat();
                    }
                    if let Some(m) = &self.model {
                        ui.label(RichText::new(format!("context {} / {}", self.context_used, m.ctx)).weak());
                    }
                    if self.generating {
                        if let Some(t) = self.first_token {
                            let secs = t.elapsed().as_secs_f64().max(1e-3);
                            ui.label(RichText::new(format!("{:.1} tok/s", self.live_tokens as f64 / secs)).strong());
                        }
                    }
                });
            });
        });

        if self.s.show_settings {
            egui::Panel::left("settings").resizable(true).default_size(300.0).show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.settings_panel(ui));
            });
        }

        egui::Panel::bottom("composer").show(ui, |ui| {
            ui.add_space(6.0);
            let ready = self.model.is_some() && self.loading.is_none();
            ui.horizontal(|ui| {
                let w = ui.available_width() - 90.0;
                let resp = ui.add_enabled(
                    ready,
                    egui::TextEdit::multiline(&mut self.input)
                        .desired_rows(2)
                        .desired_width(w)
                        .hint_text(if ready { "Message the model…  (Enter to send, Shift+Enter for a new line)" } else { "Load a model to start" }),
                );
                let enter = resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                ui.vertical(|ui| {
                    if self.generating {
                        if ui.add_sized([80.0, 32.0], egui::Button::new("Stop")).clicked() {
                            self.stop.store(true, Ordering::SeqCst);
                        }
                    } else if ui.add_enabled(ready, egui::Button::new("Send").min_size(egui::vec2(80.0, 32.0))).clicked() || enter {
                        let text = std::mem::take(&mut self.input);
                        self.send(text);
                        resp.request_focus();
                    }
                });
            });
            ui.add_space(6.0);
        });

        egui::CentralPanel::default().show(ui, |ui| {
            if self.messages.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() * 0.2);
                    ui.label(RichText::new("Local AI on your CPU").size(28.0).strong());
                    ui.label(RichText::new("Qwen3.5 running on ferrite's AVX-512 kernels — nothing leaves this laptop.").weak());
                    ui.add_space(18.0);
                    for ex in EXAMPLES {
                        if ui.add_enabled(self.model.is_some() && !self.generating, egui::Button::new(ex).min_size(egui::vec2(420.0, 30.0))).clicked() {
                            self.send(ex.to_string());
                        }
                    }
                });
                return;
            }
            let mut actions = Vec::new();
            egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false; 2]).show(ui, |ui| {
                // leave room for egui's floating scrollbar on the right
                ui.set_max_width((ui.available_width() - 18.0).min(920.0));
                let n = self.messages.len();
                for (i, m) in self.messages.iter().enumerate() {
                    App::message_ui(ui, &mut self.md, m, i, i + 1 == n, &mut actions);
                    ui.add_space(10.0);
                }
            });
            for (i, a) in actions {
                match a {
                    "copy" => ctx.copy_text(self.messages[i].content.clone()),
                    "regen" => self.regenerate(),
                    _ => {}
                }
            }
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.s);
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ferrite chat")
            .with_inner_size([1180.0, 820.0])
            .with_min_inner_size([720.0, 480.0]),
        ..Default::default()
    };
    eframe::run_native("ferrite chat", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
