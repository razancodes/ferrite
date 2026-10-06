use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use ferrite::chat::{ChatTemplate, Message};
use ferrite::engine::{Engine, default_threads};
use ferrite::sampler::{Sampler, SamplerConfig};
use ferrite::tokenizer::StreamDecoder;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "ferrite", version, about = "CPU inference engine for Qwen3.5 / Qwen3.8-architecture models")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// path to a .ferrite model file
    #[arg(short, long)]
    model: PathBuf,
    /// worker threads (default: number of physical cores)
    #[arg(short, long)]
    threads: Option<usize>,
    /// maximum context length (tokens)
    #[arg(long, default_value_t = 8192)]
    ctx: usize,
    /// disable self-speculative decoding with the model's MTP head
    /// (on by default when the model has one; output distribution is identical)
    #[arg(long)]
    no_spec: bool,
    /// rows of the draft LM head used for speculation
    #[arg(long, default_value_t = ferrite::engine::DRAFT_VOCAB)]
    draft_vocab: usize,
}

#[derive(Args, Clone)]
struct SampleArgs {
    /// sampling temperature (0 = greedy)
    #[arg(long)]
    temp: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    min_p: Option<f32>,
    #[arg(long)]
    presence_penalty: Option<f32>,
    #[arg(long)]
    repeat_penalty: Option<f32>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

impl SampleArgs {
    fn build(&self, thinking: bool) -> SamplerConfig {
        let mut c = if thinking { SamplerConfig::thinking() } else { SamplerConfig::default() };
        if let Some(v) = self.temp {
            c.temperature = v;
        }
        if let Some(v) = self.top_k {
            c.top_k = v;
        }
        if let Some(v) = self.top_p {
            c.top_p = v;
        }
        if let Some(v) = self.min_p {
            c.min_p = v;
        }
        if let Some(v) = self.presence_penalty {
            c.presence_penalty = v;
        }
        if let Some(v) = self.repeat_penalty {
            c.repetition_penalty = v;
        }
        c.seed = self.seed;
        c
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a Hugging Face checkpoint directory into a .ferrite file
    Convert {
        /// directory with config.json, tokenizer.json and *.safetensors
        #[arg(long)]
        hf: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
        /// f32 | q8 | q4 | q4-fast
        #[arg(short, long, default_value = "q8")]
        quant: String,
        /// importance matrix (from `ferrite imatrix`) used for 4-bit quantization
        #[arg(long)]
        imatrix: Option<PathBuf>,
        /// force Hadamard rotation of weight matrices on (default: on for q8/q6)
        #[arg(long, conflicts_with = "no_rotate")]
        rotate: bool,
        /// force Hadamard rotation off
        #[arg(long)]
        no_rotate: bool,
    },
    /// Generate a completion for one prompt
    Run {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        s: SampleArgs,
        #[arg(short, long)]
        prompt: String,
        #[arg(short = 'n', long, default_value_t = 512)]
        max_tokens: usize,
        /// feed the prompt as raw text (no chat template)
        #[arg(long)]
        raw: bool,
        /// enable the model's thinking mode
        #[arg(long)]
        think: bool,
        #[arg(long)]
        system: Option<String>,
    },
    /// Interactive chat in the terminal
    Chat {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        s: SampleArgs,
        #[arg(long)]
        think: bool,
        #[arg(long)]
        system: Option<String>,
        #[arg(short = 'n', long, default_value_t = 4096)]
        max_tokens: usize,
    },
    /// OpenAI-compatible HTTP server with a built-in web chat UI
    Serve {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// default max tokens per response
        #[arg(short = 'n', long, default_value_t = 4096)]
        max_tokens: usize,
    },
    /// Measure prefill and decode speed
    Bench {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(short, long, default_value_t = 512)]
        prompt_tokens: usize,
        #[arg(short = 'n', long, default_value_t = 128)]
        gen_tokens: usize,
        #[arg(long, default_value_t = 2)]
        reps: usize,
    },
    /// Tokenize text and print the ids
    Tokenize {
        #[command(flatten)]
        m: ModelArgs,
        /// text to tokenize (or use --file)
        #[arg(default_value = "")]
        text: String,
        /// tokenize a whole file and print the ids as a JSON array
        #[arg(short, long)]
        file: Option<PathBuf>,
        #[arg(long)]
        special: bool,
    },
    /// Render a chat (JSON list of {role, content}) with the chat template and print token ids
    Template {
        #[command(flatten)]
        m: ModelArgs,
        /// JSON file with the messages
        #[arg(short, long)]
        file: PathBuf,
        #[arg(long)]
        think: bool,
    },
    /// Dump logits for given token ids (testing against a reference implementation)
    DumpLogits {
        #[command(flatten)]
        m: ModelArgs,
        /// comma-separated token ids
        #[arg(long)]
        tokens: String,
        #[arg(short, long)]
        out: PathBuf,
        /// process tokens one at a time instead of as a batch
        #[arg(long)]
        step: bool,
        /// process tokens in batches of this size (tests prefill continuing an existing state)
        #[arg(long, default_value_t = 0)]
        chunk: usize,
    },
    /// Collect an importance matrix (activation statistics) on calibration text
    Imatrix {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
        /// number of 512-token chunks
        #[arg(long, default_value_t = 64)]
        chunks: usize,
    },
    /// KL divergence of a (quantized) model against a reference model
    Kld {
        #[command(flatten)]
        m: ModelArgs,
        /// reference model (e.g. the f32 or q8 conversion)
        #[arg(long)]
        reference: PathBuf,
        #[arg(short, long)]
        file: PathBuf,
        #[arg(long, default_value_t = 512)]
        window: usize,
        #[arg(long, default_value_t = 8)]
        windows: usize,
    },
    /// Rank tokens by frequency on a corpus and store the list in the model
    /// (used to build the speculative-decoding draft head)
    DraftVocab {
        #[command(flatten)]
        m: ModelArgs,
        /// corpus files
        #[arg(short, long, num_args = 1..)]
        files: Vec<PathBuf>,
        #[arg(long, default_value_t = 65536)]
        keep: usize,
    },
    /// Perplexity of a text file
    Ppl {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(short, long)]
        file: PathBuf,
        /// tokens per evaluation window
        #[arg(long, default_value_t = 512)]
        window: usize,
        /// maximum number of windows
        #[arg(long, default_value_t = 16)]
        windows: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Convert { hf, out, quant, imatrix, rotate, no_rotate } => {
            let rot = if rotate { Some(true) } else if no_rotate { Some(false) } else { None };
            ferrite::convert::convert(&hf, &out, &quant, imatrix.as_deref(), rot)
        }
        Cmd::Run { m, s, prompt, max_tokens, raw, think, system } => cmd_run(m, s, prompt, max_tokens, raw, think, system),
        Cmd::Chat { m, s, think, system, max_tokens } => cmd_chat(m, s, think, system, max_tokens),
        Cmd::Serve { m, host, port, max_tokens } => {
            let e = load_engine_opts(&m, true)?;
            let name = m.model.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "ferrite".into());
            ferrite::server::serve(
                ferrite::server::ServerState { engine: std::sync::Mutex::new(e), model_name: name, default_max_tokens: max_tokens },
                &host,
                port,
            )
        }
        Cmd::Bench { m, prompt_tokens, gen_tokens, reps } => cmd_bench(m, prompt_tokens, gen_tokens, reps),
        Cmd::Tokenize { m, text, file, special } => {
            let model = ferrite::model::Model::load(&m.model)?;
            if let Some(f) = file {
                let t0 = Instant::now();
                let ids = model.tokenizer.encode(&std::fs::read_to_string(&f)?, special);
                eprintln!("{} tokens in {:.2}s", ids.len(), t0.elapsed().as_secs_f64());
                println!("{}", serde_json::to_string(&ids)?);
                return Ok(());
            }
            let ids = model.tokenizer.encode(&text, special);
            println!("{ids:?}");
            for id in ids {
                println!("{id:>8} {:?}", String::from_utf8_lossy(model.tokenizer.token_bytes(id)));
            }
            Ok(())
        }
        Cmd::DumpLogits { m, tokens, out, step, chunk } => cmd_dump(m, tokens, out, if step { 1 } else { chunk }),
        Cmd::Template { m, file, think } => {
            let model = ferrite::model::Model::load(&m.model)?;
            let msgs: Vec<Message> = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
            let ids = ChatTemplate::new(&model.tokenizer).render(&model.tokenizer, &msgs, true, think);
            println!("{}", serde_json::to_string(&ids)?);
            Ok(())
        }
        Cmd::Ppl { m, file, window, windows } => cmd_ppl(m, file, window, windows),
        Cmd::Imatrix { m, file, out, chunks } => cmd_imatrix(m, file, out, chunks),
        Cmd::DraftVocab { m, files, keep } => {
            let mut counts: Vec<u64>;
            {
                let model = ferrite::model::Model::load(&m.model)?;
                counts = vec![0; model.tokenizer.vocab_size()];
                for f in &files {
                    let text = std::fs::read_to_string(f)?;
                    for id in model.tokenizer.encode(&text, false) {
                        counts[id as usize] += 1;
                    }
                }
            } // drop the mmap before rewriting the file
            let mut ids: Vec<u32> = (0..counts.len() as u32).filter(|&i| counts[i as usize] > 0).collect();
            ids.sort_by_key(|&i| (std::cmp::Reverse(counts[i as usize]), i));
            let total: u64 = counts.iter().sum();
            for n in [16384usize, 32768, 65536] {
                let cov: u64 = ids.iter().take(n).map(|&i| counts[i as usize]).sum();
                eprintln!("top {n:>6} tokens cover {:.2}% of the corpus", cov as f64 * 100.0 / total as f64);
            }
            ids.truncate(keep);
            let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
            ferrite::format::append_tensor(&m.model, "draft_vocab", ferrite::quant::DType::U8, &[bytes.len()], &bytes)?;
            println!("stored {} frequency-ranked token ids in {}", ids.len(), m.model.display());
            Ok(())
        }
        Cmd::Kld { m, reference, file, window, windows } => cmd_kld(m, reference, file, window, windows),
    }
}

fn load_engine(m: &ModelArgs) -> Result<Engine> {
    load_engine_opts(m, false)
}

/// `generation`: enable the speculative-decoding machinery (draft head, MTP state).
fn load_engine_opts(m: &ModelArgs, generation: bool) -> Result<Engine> {
    let t0 = Instant::now();
    let threads = m.threads.unwrap_or_else(default_threads);
    let mut e = Engine::load(&m.model, threads, m.ctx)?;
    if generation && !m.no_spec {
        e.enable_spec(m.draft_vocab)?;
    }
    // analysis tools (perplexity, KL, logit dumps, imatrix) need the exact full distribution
    e.fast_head &= generation;
    let c = &e.model.cfg;
    eprintln!(
        "[ferrite] {} | {} layers ({} DeltaNet + {} attn) | {:.2} GB weights | {} threads | {} | MTP speculation {} | loaded in {:.2}s",
        m.model.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
        c.n_layers,
        c.n_linear(),
        c.n_full(),
        e.model.weight_bytes() as f64 / 1e9,
        threads,
        format!("{:?}", ferrite::kernels::isa()),
        if e.spec { "on" } else { "off" },
        t0.elapsed().as_secs_f64()
    );
    Ok(e)
}

fn print_stats(st: &ferrite::engine::GenStats) {
    eprintln!(
        "\n[ferrite] prompt {} tok ({} cached) {:.1} tok/s | generated {} tok {:.2} tok/s | stop: {:?}{}",
        st.prompt_tokens,
        st.cached_tokens,
        st.prefill_tps(),
        st.gen_tokens,
        st.gen_tps(),
        st.stop,
        if st.drafted > 0 {
            format!(" | draft acceptance {:.1}% ({}/{})", st.acceptance() * 100.0, st.accepted, st.drafted)
        } else {
            String::new()
        }
    );
}

fn cmd_run(
    m: ModelArgs,
    s: SampleArgs,
    prompt: String,
    max_tokens: usize,
    raw: bool,
    think: bool,
    system: Option<String>,
) -> Result<()> {
    let mut e = load_engine_opts(&m, true)?;
    let tmpl = ChatTemplate::new(&e.model.tokenizer);
    let ids = if raw {
        e.model.tokenizer.encode(&prompt, true)
    } else {
        let mut msgs = Vec::new();
        if let Some(sys) = &system {
            msgs.push(Message::new("system", sys));
        }
        msgs.push(Message::new("user", &prompt));
        tmpl.render(&e.model.tokenizer, &msgs, true, think)
    };
    let mut sampler = Sampler::new(s.build(think));
    let mut dec = StreamDecoder::default();
    let stop = tmpl.eos.clone();
    let mut out = std::io::stdout();
    let st = e.generate(&ids, None, &mut sampler, max_tokens, &stop, |eng, tok| {
        let piece = dec.push(&eng.model.tokenizer, tok);
        let _ = out.write_all(piece.as_bytes());
        let _ = out.flush();
        true
    })?;
    print!("{}", dec.flush());
    print_stats(&st);
    Ok(())
}

fn cmd_chat(m: ModelArgs, s: SampleArgs, think: bool, system: Option<String>, max_tokens: usize) -> Result<()> {
    let mut e = load_engine_opts(&m, true)?;
    let tmpl = ChatTemplate::new(&e.model.tokenizer);
    let mut thinking = think;
    let mut sampler = Sampler::new(s.build(thinking));
    eprintln!("Type a message. Commands: /reset, /think, /nothink, /stats, /quit");
    let stdin = std::io::stdin();
    let mut first = true;
    let mut out = std::io::stdout();
    loop {
        eprint!("\n\x1b[1;32m>>> \x1b[0m");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end_matches(['\r', '\n']).to_string();
        match line.trim() {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                e.reset();
                first = true;
                eprintln!("(conversation reset)");
                continue;
            }
            "/think" | "/nothink" => {
                thinking = line.trim() == "/think";
                sampler = Sampler::new(s.build(thinking));
                eprintln!("(thinking {})", if thinking { "on" } else { "off" });
                continue;
            }
            "/stats" => {
                eprintln!("(context: {} / {} tokens)", e.pos(), e.state.max_ctx);
                continue;
            }
            _ => {}
        }
        let ids = if first {
            let mut msgs = Vec::new();
            if let Some(sys) = &system {
                msgs.push(Message::new("system", sys));
            }
            msgs.push(Message::new("user", &line));
            tmpl.render(&e.model.tokenizer, &msgs, true, thinking)
        } else {
            // continue from the exact state left by the previous reply
            let mut ids = e.state.tokens.clone();
            ids.extend(tmpl.continuation(&e.model.tokenizer, &line, thinking));
            ids
        };
        first = false;
        let mut dec = StreamDecoder::default();
        let stop = tmpl.eos.clone();
        let st = e.generate(&ids, None, &mut sampler, max_tokens, &stop, |eng, tok| {
            let piece = dec.push(&eng.model.tokenizer, tok);
            let _ = out.write_all(piece.as_bytes());
            let _ = out.flush();
            true
        })?;
        print!("{}", dec.flush());
        let _ = out.flush();
        print_stats(&st);
    }
    Ok(())
}

fn cmd_bench(m: ModelArgs, pp: usize, tg: usize, reps: usize) -> Result<()> {
    let mut e = load_engine_opts(&m, true)?;
    let prompt: Vec<u32> = (0..pp as u32).map(|i| (i * 7919 + 13) % 50000 + 100).collect();
    // decode is measured on real text so that speculative decoding sees
    // realistic draft acceptance
    let tmpl = ChatTemplate::new(&e.model.tokenizer);
    let chat = tmpl.render(
        &e.model.tokenizer,
        &[Message::new("user", "Write a detailed essay about the history of computing, from the abacus to modern AI.")],
        true,
        false,
    );
    let (mut best_pp, mut best_tg, mut acc) = (0f64, 0f64, 0f64);
    for r in 0..reps {
        e.reset();
        let t0 = Instant::now();
        e.feed(&prompt)?;
        let pps = pp as f64 / t0.elapsed().as_secs_f64();
        if r == 0 && ferrite::profile::enabled() {
            eprintln!("--- prefill profile");
            ferrite::profile::report();
            ferrite::profile::reset();
            eprintln!("--- decode profile");
        }
        e.reset();
        let mut sampler = Sampler::new(SamplerConfig::greedy());
        let st = e.generate(&chat, None, &mut sampler, tg, &[], |_, _| true)?;
        let tgs = st.gen_tps();
        if r == 0 {
            ferrite::profile::report();
        }
        ferrite::profile::reset();
        eprintln!(
            "  rep {}: prefill {:.1} tok/s, decode {:.2} tok/s{}",
            r + 1,
            pps,
            tgs,
            if st.drafted > 0 { format!(" (draft acceptance {:.1}%)", st.acceptance() * 100.0) } else { String::new() }
        );
        best_pp = best_pp.max(pps);
        if tgs > best_tg {
            best_tg = tgs;
            acc = st.acceptance();
        }
    }
    let bytes = e.model.decode_bytes() as f64;
    println!(
        "{} | threads {}/{} | prefill (pp{pp}): {:.1} tok/s | decode (tg{tg}): {:.2} tok/s{}",
        m.model.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
        e.pool.max_threads(),
        e.decode_threads,
        best_pp,
        best_tg,
        if e.spec {
            format!(" | spec acceptance {:.1}%", acc * 100.0)
        } else {
            format!(" | {:.1} GB/s weight bandwidth", bytes * best_tg / 1e9)
        }
    );
    Ok(())
}

fn cmd_dump(m: ModelArgs, tokens: String, out: PathBuf, chunk: usize) -> Result<()> {
    let mut e = load_engine(&m)?;
    let ids: Vec<u32> = tokens.split(',').map(|s| s.trim().parse::<u32>().context("token id")).collect::<Result<_>>()?;
    let v = e.model.vocab_rows();
    let mut all = Vec::with_capacity(ids.len() * v);
    if chunk == 1 {
        for &id in &ids {
            all.extend_from_slice(e.feed(&[id])?);
        }
    } else if chunk > 1 {
        for c in ids.chunks(chunk) {
            e.model.forward(&e.pool, &mut e.state, &mut e.sc, c)?;
            e.model.logits(&e.pool, &mut e.sc, 0, c.len());
            all.extend_from_slice(&e.sc.logits[..c.len() * v]);
        }
    } else {
        e.model.forward(&e.pool, &mut e.state, &mut e.sc, &ids)?;
        e.model.logits(&e.pool, &mut e.sc, 0, ids.len());
        all.extend_from_slice(&e.sc.logits[..ids.len() * v]);
    }
    let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&out, bytes)?;
    eprintln!("wrote {} x {} logits to {}", ids.len(), v, out.display());
    Ok(())
}

fn cmd_ppl(m: ModelArgs, file: PathBuf, window: usize, windows: usize) -> Result<()> {
    let mut e = load_engine(&m)?;
    let text = std::fs::read_to_string(&file)?;
    let ids = e.model.tokenizer.encode(&text, false);
    if ids.len() < window + 1 {
        bail!("text too short: {} tokens", ids.len());
    }
    let v = e.model.vocab_rows();
    let (mut nll, mut count) = (0f64, 0usize);
    let t0 = Instant::now();
    for w in 0..windows {
        let start = w * window;
        if start + window + 1 > ids.len() {
            break;
        }
        e.reset();
        let seg = &ids[start..start + window];
        // score only the second half of each window so every token has context
        let half = window / 2;
        e.feed(&seg[..half])?;
        let rest = &seg[half..];
        for (ci, chunk) in rest.chunks(64).enumerate() {
            e.model.forward(&e.pool, &mut e.state, &mut e.sc, chunk)?;
            e.model.logits(&e.pool, &mut e.sc, 0, chunk.len());
            for i in 0..chunk.len() {
                let gi = half + ci * 64 + i; // index within seg of this position
                // targets stay inside the window, as in llama.cpp's llama-perplexity, so the
                // two report perplexity over exactly the same tokens
                if gi + 1 >= seg.len() {
                    continue;
                }
                let target = seg[gi + 1];
                let row = &e.sc.logits[i * v..(i + 1) * v];
                let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let lse = row.iter().map(|&x| ((x - mx) as f64).exp()).sum::<f64>().ln() + mx as f64;
                nll += lse - row[target as usize] as f64;
                count += 1;
            }
        }
        eprint!("\r  window {}/{}  ppl so far {:.4}", w + 1, windows, (nll / count as f64).exp());
    }
    eprintln!();
    println!(
        "perplexity: {:.4} over {} tokens ({:.1}s)",
        (nll / count as f64).exp(),
        count,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_imatrix(m: ModelArgs, file: PathBuf, out: PathBuf, chunks: usize) -> Result<()> {
    use std::sync::atomic::Ordering;
    let mut e = load_engine(&m)?;
    let text = std::fs::read_to_string(&file)?;
    let ids = e.model.tokenizer.encode(&text, false);
    let window = 512;
    ferrite::kernels::IMATRIX_ON.store(true, Ordering::SeqCst);
    let t0 = Instant::now();
    let n = chunks.min(ids.len() / window);
    for c in 0..n {
        e.reset();
        e.feed(&ids[c * window..(c + 1) * window])?;
        eprint!("\r  chunk {}/{} ({:.0}s)", c + 1, n, t0.elapsed().as_secs_f64());
    }
    eprintln!();
    ferrite::kernels::IMATRIX_ON.store(false, Ordering::SeqCst);
    let g = ferrite::kernels::IMATRIX.lock().unwrap();
    let (mut res, mut raw) = (std::collections::HashMap::new(), std::collections::HashMap::new());
    if let Some(map) = g.as_ref() {
        for (&id, (sums, raw_sums, count)) in map {
            let name = e.model.file.header.tensors[id].name.clone();
            res.insert(name.clone(), sums.iter().map(|s| (s / *count as f64) as f32).collect::<Vec<f32>>());
            raw.insert(name, raw_sums.iter().map(|s| (s / *count as f64) as f32).collect::<Vec<f32>>());
        }
    }
    let rotated = e.model.file.header.tensors.iter().any(|t| t.rot > 0);
    ferrite::imatrix::save(&out, &res, rotated)?;
    println!("saved importance matrix for {} tensors to {}", res.len(), out.display());
    if rotated {
        // unrotated statistics too, for formats that are better left unrotated (q4)
        let raw_path = out.with_extension("raw.json");
        ferrite::imatrix::save(&raw_path, &raw, false)?;
        println!("saved unrotated importance matrix to {}", raw_path.display());
    }
    Ok(())
}

/// Compare one logits row against a reference row:
/// returns (KL(ref||q), argmax equal, nll_ref, nll_q) for `target`.
/// Uses KL = sum_i p_i (r_i - q_i) + lse(q) - lse(r) so only one exp pass is needed.
fn row_stats(r: &[f32], q: &[f32], target: usize, tmp: &mut Vec<f32>) -> (f64, bool, f64, f64) {
    let argmax = |x: &[f32]| x.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    let (ar, mr) = argmax(r);
    let (aq, mq) = argmax(q);
    tmp.clear();
    tmp.extend_from_slice(q);
    let sq = ferrite::ops::exp_sub_sum(tmp, mq) as f64;
    tmp.clear();
    tmp.extend_from_slice(r);
    let sr = ferrite::ops::exp_sub_sum(tmp, mr) as f64;
    let (lr, lq) = (sr.ln() + mr as f64, sq.ln() + mq as f64);
    let mut acc = 0f64;
    for (blk, (eb, (rb, qb))) in tmp.chunks(1024).zip(r.chunks(1024).zip(q.chunks(1024))).enumerate() {
        let _ = blk;
        let mut a = 0f32;
        for i in 0..eb.len() {
            a += eb[i] * (rb[i] - qb[i]);
        }
        acc += a as f64;
    }
    let kl = (acc / sr + lq - lr).max(0.0);
    (kl, ar == aq, lr - r[target] as f64, lq - q[target] as f64)
}

fn cmd_kld(m: ModelArgs, reference: PathBuf, file: PathBuf, window: usize, windows: usize) -> Result<()> {
    let mut e = load_engine(&m)?;
    let mut r = load_engine(&ModelArgs { model: reference, ..m.clone() })?;
    let text = std::fs::read_to_string(&file)?;
    let ids = e.model.tokenizer.encode(&text, false);
    let v = e.model.vocab_rows();
    let vr = e.model.tokenizer.vocab_size().min(v); // exclude padded (masked) rows
    let mut tmp = Vec::new();
    let (mut kl, mut agree, mut nr, mut nq, mut n) = (0f64, 0usize, 0f64, 0f64, 0usize);
    let mut kls = Vec::new();
    for w in 0..windows {
        let seg = &ids[w * window..(w + 1) * window + 1];
        e.reset();
        r.reset();
        for c in (0..window).step_by(64) {
            let chunk = &seg[c..(c + 64).min(window)];
            e.model.forward(&e.pool, &mut e.state, &mut e.sc, chunk)?;
            e.model.logits(&e.pool, &mut e.sc, 0, chunk.len());
            r.model.forward(&r.pool, &mut r.state, &mut r.sc, chunk)?;
            r.model.logits(&r.pool, &mut r.sc, 0, chunk.len());
            for i in 0..chunk.len() {
                let pos = c + i;
                if pos < window / 4 {
                    continue; // skip positions with little context
                }
                let (k, a, lr, lq) =
                    row_stats(&r.sc.logits[i * v..i * v + vr], &e.sc.logits[i * v..i * v + vr], seg[pos + 1] as usize, &mut tmp);
                kl += k;
                kls.push(k);
                agree += a as usize;
                nr += lr;
                nq += lq;
                n += 1;
            }
        }
        eprint!("\r  window {}/{}  mean KL {:.5}", w + 1, windows, kl / n as f64);
    }
    eprintln!();
    ferrite::profile::report();
    kls.sort_by(|a, b| a.total_cmp(b));
    println!(
        "tokens {n} | mean KL {:.5} | p99 KL {:.4} | top-1 agree {:.2}% | ppl ref {:.4} | ppl model {:.4} ({:+.2}%)",
        kl / n as f64,
        kls[(kls.len() * 99) / 100],
        agree as f64 * 100.0 / n as f64,
        (nr / n as f64).exp(),
        (nq / n as f64).exp(),
        ((nq - nr) / n as f64).exp_m1() * 100.0
    );
    Ok(())
}
