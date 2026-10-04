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
    last_logits: Vec<f32>,
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
            last_logits: Vec::new(),
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

    /// Run the MTP head over every queued position whose next token is known.
    /// With `next` (the token at the current position), the final queued
    /// position is processed too and a draft token is returned.
    fn mtp_catchup(&mut self, next: Option<u32>) -> Result<Option<u32>> {
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
        let draft = self.model.draft.as_ref().unwrap();
        let l = &self.sc_mtp.logits[..draft.mat.rows];
        let best = l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
        Ok(Some(draft.ids[best]))
    }

    fn spec_ready(&self) -> bool {
        self.spec && self.state.mtp_hidden.len() / self.hidden() + self.state.mtp_pos == self.state.pos
    }

    /// Use all threads for batches (compute bound) and `decode_threads` for
    /// single-token steps (memory bound).
    fn set_width(&self, batch: usize) {
        self.pool.set_active(if batch <= 2 { self.decode_threads } else { self.pool.max_threads() });
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
                self.model.logits(&self.pool, &mut self.sc, c.len() - 1, 1);
            }
            if spec {
                self.mtp_catchup(None)?;
            }
        }
        let v = self.model.vocab_rows();
        self.last_logits.clear();
        self.last_logits.extend_from_slice(&self.sc.logits[..v]);
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
        let reused = if st.len() <= prompt.len() && prompt[..st.len()] == st[..] && !self.last_logits.is_empty() {
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
    ///   1. the MTP head drafts token d for position P+1 from (embed(next), h[P-1]),
    ///   2. the main model runs [next, d] as one batch (memory-bound: ~cost of one
    ///      token) with a state checkpoint after `next`,
    ///   3. d is accepted with probability p(d) (exact speculative sampling for a
    ///      deterministic draft); on rejection the recurrent state is rolled back
    ///      by swapping in the checkpoint and the corrected token is used.
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
        loop {
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
            if self.state.pos + 2 >= self.state.max_ctx {
                stats.stop = StopReason::Context;
                break;
            }
            self.set_width(2);
            let draft = self.mtp_catchup(Some(next))?.expect("draft");
            let p = self.state.pos;
            self.model.forward_ckpt(&self.pool, &mut self.state, &mut self.sc, &[next, draft], Some(0))?;
            self.push_mtp_hidden(0, 2);
            self.model.logits(&self.pool, &mut self.sc, 0, 2);
            stats.drafted += 1;
            let mut l0 = self.sc.logits[..v].to_vec();
            let (y, acc) = sampler.sample_spec(&mut l0, draft);
            if acc && !stop.contains(&draft) {
                stats.accepted += 1;
                sampler.accept(draft);
                stats.gen_tokens += 1;
                if !on_token(self, draft) {
                    stats.stop = StopReason::Callback;
                    break;
                }
                if stats.gen_tokens >= max_new {
                    stats.stop = StopReason::Length;
                    break;
                }
                logits.clear();
                logits.extend_from_slice(&self.sc.logits[v..2 * v]);
                next = sampler.sample(&mut logits);
                sampler.accept(next);
            } else {
                // undo the draft token: recurrent state <- checkpoint, KV length <- p+1
                self.model.rollback(&mut self.state, p + 1);
                let keep = self.state.mtp_hidden.len() - h;
                self.state.mtp_hidden.truncate(keep);
                next = y;
                sampler.accept(next);
            }
        }
        // leave last_logits consistent for prompt reuse (not used after generation)
        self.last_logits.clear();
        self.last_logits.extend_from_slice(&self.sc.logits[..v]);
        Ok(())
    }
}
