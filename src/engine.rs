//! High-level inference engine: owns the model, thread pool, sequence state
//! and scratch buffers; implements chunked prefill, prompt-prefix reuse,
//! cheap state snapshots, the token generation loop and MTP self-speculative
//! decoding.

use crate::model::{Model, Scratch, State};
use crate::ops::rmsnorm;
use crate::sampler::Sampler;
use crate::threadpool::ThreadPool;
use anyhow::Result;
use std::path::Path;
use std::time::Instant;

pub struct Engine {
    pub model: Model,
    pub pool: ThreadPool,
    pub state: State,
    pub sc: Scratch,
    sc_mtp: Scratch,
    /// tokens per prefill chunk (bounds scratch memory)
    pub chunk: usize,
    /// draft one token per step with the model's MTP head and verify it in the
    /// same forward pass as the current token
    pub spec: bool,
    /// threads used for (memory-bound) decoding; prefill uses the whole pool
    pub decode_threads: usize,
    /// maximum tokens drafted per speculative step (MTP head chained on itself)
    pub draft_k: usize,
    /// keep chaining drafts only while the MTP head's top probability is at least this
    pub draft_conf: f32,
    /// generation scores the vocabulary with the proxy LM head (when the model
    /// has one) and only the shortlist within `head_delta` logits of the top
    /// exactly (at most `head_cap` 16-token groups per position); tools that
    /// need the exact full distribution (perplexity, logit dumps) turn this off
    pub fast_head: bool,
    pub head_delta: f32,
    pub head_cap: usize,
    last_logits: Vec<f32>,
    /// `last_logits` belong to the current end of the sequence
    logits_fresh: bool,
    snapshot: Option<Snapshot>,
}

/// Everything needed to rewind the sequence to an earlier position. KV-cache
/// entries before `pos` are never overwritten by later tokens, so only the
/// constant-size recurrent state has to be copied.
pub struct Snapshot {
    pos: usize,
    tokens: Vec<u32>,
    conv: Vec<Vec<f32>>,
    ssm: Vec<Vec<f32>>,
    logits: Vec<f32>,
    mtp_pos: usize,
    mtp_hidden: Vec<f32>,
}

#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub prefill_secs: f64,
    pub gen_tokens: usize,
    pub gen_secs: f64,
    pub stop: StopReason,
    pub drafted: usize,
    pub accepted: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    #[default]
    Eos,
    Length,
    Callback,
    Context,
}

impl GenStats {
    pub fn prefill_tps(&self) -> f64 {
        (self.prompt_tokens - self.cached_tokens) as f64 / self.prefill_secs.max(1e-9)
    }
    pub fn gen_tps(&self) -> f64 {
        self.gen_tokens as f64 / self.gen_secs.max(1e-9)
    }
    pub fn acceptance(&self) -> f64 {
        self.accepted as f64 / self.drafted.max(1) as f64
    }
}

/// Pool size: every logical core (prefill is compute bound and benefits from
/// hyper-threading).
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Decoding is DRAM-bandwidth bound; on a 4-core laptop two or three threads
/// already saturate memory, and more only add synchronization and contention.
/// Override with FERRITE_DECODE_THREADS.
pub fn default_decode_threads(pool: usize) -> usize {
    if let Some(n) = std::env::var("FERRITE_DECODE_THREADS").ok().and_then(|v| v.parse().ok()) {
        return n;
    }
    (pool / 2).clamp(1, 4).min(pool)
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Default number of rows in the draft LM head.
pub const DRAFT_VOCAB: usize = 32768;

impl Engine {
    pub fn load(path: &Path, threads: usize, ctx: usize) -> Result<Engine> {
        let model = Model::load(path)?;
        let pool = ThreadPool::new(threads);
        model.prefault(&pool);
        let ctx = ctx.min(model.cfg.max_position);
        let state = model.new_state(ctx);
        Ok(Engine {
            model,
            pool,
            state,
            sc: Scratch::default(),
            sc_mtp: Scratch::default(),
            chunk: 256,
            spec: false,
            decode_threads: default_decode_threads(threads),
            draft_k: env_or("FERRITE_DRAFT_K", 4usize),
            draft_conf: env_or("FERRITE_DRAFT_CONF", 0.3f32),
            fast_head: env_or("FERRITE_FAST_HEAD", 1u8) != 0,
            head_delta: env_or("FERRITE_HEAD_DELTA", 8.0f32),
            head_cap: env_or("FERRITE_HEAD_CAP", 256usize),
            last_logits: Vec::new(),
            logits_fresh: false,
            snapshot: None,
        })
    }

    /// Turn on MTP self-speculative decoding (no-op if the model has no MTP head).
    pub fn enable_spec(&mut self, draft_vocab: usize) -> Result<bool> {
        if self.model.mtp.is_none() {
            return Ok(false);
        }
        if self.model.draft.is_none() {
            self.model.build_draft_head(draft_vocab)?;
        }
        self.spec = true;
        self.reset();
        Ok(true)
    }

    pub fn reset(&mut self) {
        self.state = self.model.new_state(self.state.max_ctx);
        self.last_logits.clear();
        self.logits_fresh = false;
    }

    pub fn pos(&self) -> usize {
        self.state.pos
    }

    fn hidden(&self) -> usize {
        self.model.cfg.hidden
    }

    /// Append the final-normed hidden states of the last forward (rows
    /// `first..first+n` of `sc.x`) to the MTP input queue.
    fn push_mtp_hidden(&mut self, first: usize, n: usize) {
        let h = self.hidden();
        let base = self.state.mtp_hidden.len();
        self.state.mtp_hidden.resize(base + n * h, 0.0);
        for i in 0..n {
            rmsnorm(
                &self.sc.x[(first + i) * h..(first + i + 1) * h],
                self.model.final_norm,
                self.model.cfg.rms_eps,
                &mut self.state.mtp_hidden[base + i * h..base + (i + 1) * h],
            );
        }
    }

    /// Argmax of the draft head's logits and its softmax probability.
    fn draft_argmax(&mut self) -> (u32, f32) {
        let draft = self.model.draft.as_ref().unwrap();
        let n = draft.mat.rows;
        let l = &mut self.sc_mtp.logits[..n];
        let (best, mx) = l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        let id = draft.ids[best];
        let sum = crate::ops::exp_sub_sum(l, mx); // logits are scratch from here on
        (id, 1.0 / sum)
    }

    /// Draft up to `k` tokens: the first from the main model's hidden state,
    /// the following ones by feeding the MTP head its own output, stopping
    /// early once the head is unsure. The MTP position is rewound afterwards so
    /// the speculative cache entries get recomputed from the verified states.
    fn draft_chain(&mut self, next: u32, k: usize) -> Result<Vec<u32>> {
        let h = self.hidden();
        let (first, mut conf) = self.mtp_catchup(Some(next))?.expect("draft");
        let base = self.state.mtp_pos;
        let mut drafts = vec![first];
        while drafts.len() < k && conf >= self.draft_conf {
            let hid = self.sc_mtp.xn[..h].to_vec();
            let prev = *drafts.last().unwrap();
            self.model.mtp_forward(&self.pool, &mut self.state, &mut self.sc_mtp, &[prev], &hid, true)?;
            let (d, c) = self.draft_argmax();
            drafts.push(d);
            conf = c;
        }
        self.state.mtp_pos = base;
        Ok(drafts)
    }

    /// Run the MTP head over every queued position whose next token is known.
    /// With `next` (the token at the current position), the final queued
    /// position is processed too and a draft token (with its probability) is returned.
    fn mtp_catchup(&mut self, next: Option<u32>) -> Result<Option<(u32, f32)>> {
        let h = self.hidden();
        let st = &mut self.state;
        let queued = st.mtp_hidden.len() / h;
        let mut toks = Vec::with_capacity(queued);
        for j in 0..queued {
            let p = st.mtp_pos + j;
            if p + 1 < st.pos {
                toks.push(st.tokens[p + 1]);
            } else if let Some(nx) = next {
                toks.push(nx);
            } else {
                break;
            }
        }
        if toks.is_empty() {
            return Ok(None);
        }
        let n = toks.len();
        let hid = std::mem::take(&mut st.mtp_hidden);
        let want = next.is_some() && n == queued;
        self.model.mtp_forward(&self.pool, st, &mut self.sc_mtp, &toks, &hid[..n * h], want)?;
        st.mtp_hidden = hid;
        st.mtp_hidden.drain(..n * h);
        if !want {
            return Ok(None);
        }
        Ok(Some(self.draft_argmax()))
    }

    fn spec_ready(&self) -> bool {
        self.spec && self.state.mtp_hidden.len() / self.hidden() + self.state.mtp_pos == self.state.pos
    }

    /// Use all threads for batches (compute bound) and `decode_threads` for
    /// single-token steps (memory bound).
    fn set_width(&self, batch: usize) {
        self.pool.set_active(if batch <= 2 { self.decode_threads } else { self.pool.max_threads() });
    }

    /// LM head over `n` hidden rows from `first` (see `fast_head`).
    fn head_logits(&mut self, first: usize, n: usize) {
        if self.fast_head {
            self.model.logits_shortlist(&self.pool, &mut self.sc, first, n, self.head_delta, self.head_cap);
        } else {
            self.model.logits(&self.pool, &mut self.sc, first, n);
        }
    }

    /// Feed tokens and return the logits after the last one.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.set_width(tokens.len());
        let spec = self.spec_ready();
        for c in tokens.chunks(self.chunk) {
            self.model.forward(&self.pool, &mut self.state, &mut self.sc, c)?;
            if spec {
                self.push_mtp_hidden(0, c.len());
            }
            if c.as_ptr_range().end == tokens.as_ptr_range().end {
                self.head_logits(c.len() - 1, 1);
            }
            if spec {
                self.mtp_catchup(None)?;
            }
        }
        let v = self.model.vocab_rows();
        self.last_logits.clear();
        self.last_logits.extend_from_slice(&self.sc.logits[..v]);
        self.logits_fresh = true;
        Ok(&self.last_logits)
    }

    pub fn last_logits(&self) -> &[f32] {
        &self.last_logits
    }

    pub fn save_snapshot(&mut self) {
        self.snapshot = Some(Snapshot {
            pos: self.state.pos,
            tokens: self.state.tokens.clone(),
            conv: self.state.conv.clone(),
            ssm: self.state.ssm.clone(),
            logits: self.last_logits.clone(),
            mtp_pos: self.state.mtp_pos,
            mtp_hidden: self.state.mtp_hidden.clone(),
        });
    }

    fn restore_snapshot(&mut self) -> bool {
        let Some(s) = &self.snapshot else { return false };
        if s.pos > self.state.pos || self.state.tokens[..s.pos] != s.tokens[..] || s.mtp_pos > self.state.mtp_pos {
            return false; // the caches below s.pos no longer match
        }
        self.state.pos = s.pos;
        self.state.tokens.truncate(s.pos);
        for (d, src) in self.state.conv.iter_mut().zip(&s.conv) {
            d.copy_from_slice(src);
        }
        for (d, src) in self.state.ssm.iter_mut().zip(&s.ssm) {
            d.copy_from_slice(src);
        }
        self.state.mtp_pos = s.mtp_pos;
        self.state.mtp_hidden.clone_from(&s.mtp_hidden);
        self.last_logits.clone_from(&s.logits);
        self.logits_fresh = true;
        true
    }

    /// Bring the state to exactly `prompt`, reusing as much previous work as
    /// possible. If `snapshot_at` is given, a snapshot is taken after that many
    /// prompt tokens (e.g. right before the assistant header) so the next
    /// request in the same conversation can resume from there.
    /// Returns the number of reused tokens.
    pub fn prime(&mut self, prompt: &[u32], snapshot_at: Option<usize>) -> Result<usize> {
        assert!(!prompt.is_empty());
        let st = &self.state.tokens;
        let prefix = st.len() <= prompt.len() && prompt[..st.len()] == st[..];
        let reused = if prefix && !st.is_empty() && (st.len() < prompt.len() || self.logits_fresh) {
            st.len()
        } else if self
            .snapshot
            .as_ref()
            .map(|s| s.tokens.len() <= prompt.len() && prompt[..s.tokens.len()] == s.tokens[..])
            .unwrap_or(false)
            && self.restore_snapshot()
        {
            self.state.pos
        } else {
            self.reset();
            0
        };
        if reused == prompt.len() {
            return Ok(reused);
        }
        let rest = &prompt[reused..];
        match snapshot_at {
            Some(at) if at > reused && at < prompt.len() => {
                let (a, b) = rest.split_at(at - reused);
                self.feed(a)?;
                self.save_snapshot();
                self.feed(b)?;
            }
            _ => {
                self.feed(rest)?;
            }
        }
        Ok(reused)
    }

    /// Generate up to `max_new` tokens after `prompt`. `on_token` receives each
    /// new token id and returns false to stop early.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        snapshot_at: Option<usize>,
        sampler: &mut Sampler,
        max_new: usize,
        stop: &[u32],
        mut on_token: impl FnMut(&Engine, u32) -> bool,
    ) -> Result<GenStats> {
        let t0 = Instant::now();
        let reused = self.prime(prompt, snapshot_at)?;
        let mut stats = GenStats {
            prompt_tokens: prompt.len(),
            cached_tokens: reused,
            prefill_secs: t0.elapsed().as_secs_f64(),
            ..Default::default()
        };
        let t1 = Instant::now();
        sampler.reset();
        let mut logits = self.last_logits.clone();
        stats.stop = StopReason::Length;
        if self.spec_ready() {
            self.generate_spec(sampler, max_new, stop, &mut on_token, &mut stats, logits)?;
            stats.gen_secs = t1.elapsed().as_secs_f64();
            return Ok(stats);
        }
        for _ in 0..max_new {
            let tok = sampler.sample(&mut logits);
            sampler.accept(tok);
            stats.gen_tokens += 1;
            if stop.contains(&tok) {
                stats.stop = StopReason::Eos;
                break;
            }
            if !on_token(self, tok) {
                stats.stop = StopReason::Callback;
                break;
            }
            if self.state.pos + 1 >= self.state.max_ctx {
                stats.stop = StopReason::Context;
                break;
            }
            logits.clear();
            logits.extend_from_slice(self.feed(&[tok])?);
        }
        stats.gen_secs = t1.elapsed().as_secs_f64();
        Ok(stats)
    }

    /// Self-speculative decoding. Each step:
    ///   1. the MTP head drafts up to `draft_k` tokens d1..dn (chained on its own
    ///      hidden state, gated by confidence),
    ///   2. the main model runs [next, d1..dn] as one batch (memory-bound: about
    ///      the cost of one token) with a state checkpoint after every token,
    ///   3. drafts are accepted left to right with probability p(d) (exact
    ///      speculative sampling for a deterministic draft); at the first rejection
    ///      the recurrent state is rolled back by swapping in the matching
    ///      checkpoint and the corrected token is used.
    fn generate_spec(
        &mut self,
        sampler: &mut Sampler,
        max_new: usize,
        stop: &[u32],
        on_token: &mut impl FnMut(&Engine, u32) -> bool,
        stats: &mut GenStats,
        mut logits: Vec<f32>,
    ) -> Result<()> {
        let v = self.model.vocab_rows();
        let h = self.hidden();
        let mut next = sampler.sample(&mut logits);
        sampler.accept(next);
        'outer: loop {
            stats.gen_tokens += 1;
            if stop.contains(&next) {
                stats.stop = StopReason::Eos;
                break;
            }
            if !on_token(self, next) {
                stats.stop = StopReason::Callback;
                break;
            }
            if stats.gen_tokens >= max_new {
                stats.stop = StopReason::Length;
                break;
            }
            let k = self.draft_k.clamp(1, max_new - stats.gen_tokens);
            if self.state.pos + k + 1 >= self.state.max_ctx {
                stats.stop = StopReason::Context;
                break;
            }
            self.set_width(2);
            let drafts = self.draft_chain(next, k)?;
            let n = drafts.len();
            let p = self.state.pos;
            let mut batch = Vec::with_capacity(n + 1);
            batch.push(next);
            batch.extend_from_slice(&drafts);
            self.model.forward_ckpt(&self.pool, &mut self.state, &mut self.sc, &batch, n)?;
            self.push_mtp_hidden(0, n + 1);
            self.head_logits(0, n + 1);
            let mut accepted = 0;
            let mut corrected = None;
            let mut halt = None;
            for (i, &d) in drafts.iter().enumerate() {
                let (y, acc) = sampler.sample_spec(&mut self.sc.logits[i * v..(i + 1) * v], d);
                stats.drafted += 1;
                if !acc || stop.contains(&d) {
                    corrected = Some(y);
                    break;
                }
                stats.accepted += 1;
                sampler.accept(d);
                stats.gen_tokens += 1;
                accepted += 1;
                if !on_token(self, d) {
                    halt = Some(StopReason::Callback);
                    break;
                }
                if stats.gen_tokens >= max_new {
                    halt = Some(StopReason::Length);
                    break;
                }
            }
            if accepted < n {
                // undo the unaccepted drafts: recurrent state <- checkpoint, KV length <- p+1+accepted
                self.model.rollback(&self.pool, &mut self.state, p + 1 + accepted, accepted);
                self.state.mtp_hidden.truncate((accepted + 1) * h);
            }
            if let Some(r) = halt {
                stats.stop = r;
                break 'outer;
            }
            next = match corrected {
                Some(y) => y,
                None => sampler.sample(&mut self.sc.logits[n * v..(n + 1) * v]),
            };
            sampler.accept(next);
        }
        // the state has moved past the logits computed during verification
        self.logits_fresh = false;
        Ok(())
    }
}
