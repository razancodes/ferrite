//! Qwen3.5 / Qwen3.8-architecture forward pass (also handles dense Qwen3).
//!
//! Layer stack (e.g. Qwen3.5-0.8B/2B: 24 layers = 6 x [3 DeltaNet + 1 attention]):
//!
//! * Gated DeltaNet ("linear attention"): fused in-projection -> causal depthwise
//!   conv(4)+SiLU -> L2-normalised q/k -> gated delta rule recurrence on a fixed
//!   [d_k x d_v] state per head -> gated RMSNorm (x SiLU(z)) -> out-projection.
//!   Memory per sequence is constant: no KV cache.
//! * Gated attention: GQA with per-head QK RMSNorm, partial RoPE (25% of dims),
//!   f16 KV cache, sigmoid output gate.
//! * SwiGLU MLP everywhere.

use crate::config::{Config, LayerKind};
use crate::format::ModelFile;
use crate::kernels::{QActs, QMat, matmul, matmul_q};
use crate::profile::scope;
use crate::ops::{self, l2norm_inplace, rmsnorm, rmsnorm_inplace, rope_neox, sigmoid, softplus};
use crate::quant::DType;
use crate::threadpool::{SyncPtr, ThreadPool};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, bail};
use std::path::Path;

type FSlice = &'static [f32];

pub struct LinearW {
    pub in_proj: QMat,
    pub in_ba: QMat,
    pub conv_w: FSlice,
    pub neg_a: FSlice,
    pub dt_bias: FSlice,
    pub gnorm: FSlice,
    pub out_proj: QMat,
}

pub struct AttnW {
    pub qkv: QMat,
    pub q_norm: FSlice,
    pub k_norm: FSlice,
    pub o_proj: QMat,
}

pub enum Mixer {
    Linear(LinearW),
    Full(AttnW),
}

pub struct LayerW {
    pub attn_norm: FSlice,
    pub mlp_norm: FSlice,
    pub mixer: Mixer,
    pub gate_up: QMat,
    pub down: QMat,
}

pub struct MtpW {
    pub fc: QMat,
    pub norm_e: FSlice,
    pub norm_h: FSlice,
    pub final_norm: FSlice,
    pub layer: LayerW,
}

/// A reduced-vocabulary copy of the LM head used to draft tokens cheaply
/// (frequency-ranked tokens; ~97% coverage at 1/4 of the cost).
pub struct DraftHead {
    pub mat: QMat,
    pub ids: Vec<u32>,
    _buf: Vec<u8>,
}

pub struct Model {
    pub draft: Option<DraftHead>,
    pub cfg: Config,
    pub file: ModelFile,
    pub lm_head: QMat,
    pub embed: Option<QMat>,
    pub final_norm: FSlice,
    pub layers: Vec<LayerW>,
    pub mtp: Option<MtpW>,
    pub tokenizer: Tokenizer,
    pub inv_freq: Vec<f32>,
    /// for each layer: index into the DeltaNet-state or KV-cache arrays
    pub slot: Vec<usize>,
}

/// Per-sequence inference state.
pub struct State {
    pub pos: usize,
    pub max_ctx: usize,
    pub k_cache: Vec<Vec<u16>>,
    pub v_cache: Vec<Vec<u16>>,
    pub conv: Vec<Vec<f32>>,
    pub ssm: Vec<Vec<f32>>,
    /// token history (everything processed so far)
    pub tokens: Vec<u32>,
    /// Speculative-decoding rollback support for the last checkpointed batch:
    /// `ckpt_conv[c][layer]` is the conv state after token c; `ssm_alt[layer]`
    /// holds the DeltaNet state from *before* the batch and `replay[layer]` the
    /// per-token rank-1 updates, so any intermediate state can be rebuilt.
    pub ckpt_conv: Vec<Vec<Vec<f32>>>,
    pub ssm_alt: Vec<Vec<f32>>,
    pub replay: Vec<Vec<f32>>,
    /// MTP head: KV cache, number of positions it has processed, and the main
    /// model's final-normed hidden states for positions [mtp_pos, pos)
    pub mtp_k: Vec<u16>,
    pub mtp_v: Vec<u16>,
    pub mtp_pos: usize,
    pub mtp_hidden: Vec<f32>,
}

/// Reusable activation buffers.
#[derive(Default)]
pub struct Scratch {
    pub x: Vec<f32>,
    pub xn: Vec<f32>,
    proj: Vec<f32>,
    ba: Vec<f32>,
    core: Vec<f32>,
    mix: Vec<f32>,
    gu: Vec<f32>,
    act: Vec<f32>,
    part: Vec<f32>,
    cat: Vec<f32>,
    qa: QActs,
    pub logits: Vec<f32>,
}

fn resize(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

impl Model {
    pub fn load(path: &Path) -> Result<Model> {
        let file = ModelFile::open(path)?;
        let cfg = file.header.config.clone();
        let mat = |name: &str| -> Result<QMat> {
            let t = file.info(name).with_context(|| format!("missing tensor {name}"))?;
            if t.shape.len() != 2 {
                bail!("{name} is not a matrix");
            }
            let b = file.bytes(name)?;
            Ok(QMat {
                dtype: t.dtype,
                rows: t.shape[0],
                k: t.shape[1],
                ptr: b.as_ptr(),
                len: b.len(),
                id: file.index_of(name).unwrap_or(usize::MAX),
                rot: t.rot,
            })
        };
        let vecf = |name: &str| -> Result<FSlice> {
            let t = file.info(name).with_context(|| format!("missing tensor {name}"))?;
            if t.dtype != DType::F32 {
                bail!("{name} should be f32");
            }
            let b = file.bytes(name)?;
            // SAFETY: tensors are 64-byte aligned inside the mmap, which lives as long as Model.
            Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) })
        };
        let attn = |p: &str| -> Result<AttnW> {
            Ok(AttnW {
                qkv: mat(&format!("{p}.qkv"))?,
                q_norm: vecf(&format!("{p}.q_norm"))?,
                k_norm: vecf(&format!("{p}.k_norm"))?,
                o_proj: mat(&format!("{p}.o_proj"))?,
            })
        };
        let mut layers = Vec::new();
        let mut slot = Vec::new();
        let (mut nl, mut nf) = (0, 0);
        for (i, kind) in cfg.layer_kinds.iter().enumerate() {
            let p = format!("L{i}");
            let mixer = match kind {
                LayerKind::Linear => {
                    slot.push(nl);
                    nl += 1;
                    Mixer::Linear(LinearW {
                        in_proj: mat(&format!("{p}.in_proj"))?,
                        in_ba: mat(&format!("{p}.in_ba"))?,
                        conv_w: vecf(&format!("{p}.conv_w"))?,
                        neg_a: vecf(&format!("{p}.neg_a"))?,
                        dt_bias: vecf(&format!("{p}.dt_bias"))?,
                        gnorm: vecf(&format!("{p}.gnorm"))?,
                        out_proj: mat(&format!("{p}.out_proj"))?,
                    })
                }
                LayerKind::Full => {
                    slot.push(nf);
                    nf += 1;
                    Mixer::Full(attn(&p)?)
                }
            };
            layers.push(LayerW {
                attn_norm: vecf(&format!("{p}.attn_norm"))?,
                mlp_norm: vecf(&format!("{p}.mlp_norm"))?,
                mixer,
                gate_up: mat(&format!("{p}.gate_up"))?,
                down: mat(&format!("{p}.down"))?,
            });
        }
        let mtp = if file.info("mtp.fc").is_some() {
            Some(MtpW {
                fc: mat("mtp.fc")?,
                norm_e: vecf("mtp.norm_e")?,
                norm_h: vecf("mtp.norm_h")?,
                final_norm: vecf("mtp.final_norm")?,
                layer: LayerW {
                    attn_norm: vecf("mtp.L0.attn_norm")?,
                    mlp_norm: vecf("mtp.L0.mlp_norm")?,
                    mixer: Mixer::Full(attn("mtp.L0")?),
                    gate_up: mat("mtp.L0.gate_up")?,
                    down: mat("mtp.L0.down")?,
                },
            })
        } else {
            None
        };
        let tok_bytes = file.bytes("tokenizer.json")?;
        let tokenizer = Tokenizer::from_json_str(std::str::from_utf8(tok_bytes)?)?;
        let half = cfg.rotary_dim / 2;
        let inv_freq =
            (0..half).map(|i| 1.0 / cfg.rope_theta.powf((2 * i) as f32 / cfg.rotary_dim as f32)).collect();
        Ok(Model {
            draft: None,
            lm_head: mat("lm_head")?,
            embed: if file.info("embed").is_some() { Some(mat("embed")?) } else { None },
            final_norm: vecf("final_norm")?,
            layers,
            mtp,
            tokenizer,
            inv_freq,
            slot,
            cfg,
            file,
        })
    }

    /// Touch every page of the weights so the first token does not pay for page faults.
    pub fn prefault(&self, pool: &ThreadPool) {
        let m = &self.file.mmap;
        let n = m.len();
        let p = SyncPtr(m.as_ptr() as *mut u8);
        let sum = std::sync::atomic::AtomicUsize::new(0);
        pool.par_chunks(n, 4096, |lo, hi| {
            let mut s = 0usize;
            let mut i = lo;
            while i < hi {
                s = s.wrapping_add(unsafe { *p.get().add(i) } as usize);
                i += 4096;
            }
            sum.fetch_add(s, std::sync::atomic::Ordering::Relaxed);
        });
        std::hint::black_box(sum.load(std::sync::atomic::Ordering::Relaxed));
    }

    pub fn weight_bytes(&self) -> usize {
        self.file.header.tensors.iter().filter(|t| t.name != "tokenizer.json").map(|t| t.nbytes as usize).sum()
    }

    /// Bytes streamed from memory for one decoded token (all weights except the
    /// embedding table and the MTP head).
    pub fn decode_bytes(&self) -> usize {
        self.file
            .header
            .tensors
            .iter()
            .filter(|t| t.name != "tokenizer.json" && t.name != "embed" && !t.name.starts_with("mtp."))
            .map(|t| t.nbytes as usize)
            .sum()
    }

    pub fn new_state(&self, max_ctx: usize) -> State {
        let c = &self.cfg;
        let nf = c.n_full();
        let nl = c.n_linear();
        let kv = c.n_kv_heads * max_ctx * c.head_dim;
        let conv_n = (c.conv_kernel.max(1) - 1) * c.conv_dim();
        let ssm_n = c.lin_v_heads * c.lin_k_dim * c.lin_v_dim;
        let mtp_kv = if self.mtp.is_some() { kv } else { 0 };
        State {
            pos: 0,
            max_ctx,
            k_cache: (0..nf).map(|_| vec![0u16; kv]).collect(),
            v_cache: (0..nf).map(|_| vec![0u16; kv]).collect(),
            conv: (0..nl).map(|_| vec![0f32; conv_n]).collect(),
            ssm: (0..nl).map(|_| vec![0f32; ssm_n]).collect(),
            tokens: Vec::new(),
            ckpt_conv: Vec::new(),
            ssm_alt: Vec::new(),
            replay: Vec::new(),
            mtp_k: vec![0u16; mtp_kv],
            mtp_v: vec![0u16; mtp_kv],
            mtp_pos: 0,
            mtp_hidden: Vec::new(),
        }
    }

    /// Build the reduced-vocabulary draft head (`n` rows, rounded down to 16).
    /// Uses the frequency ranking stored at conversion time if present,
    /// otherwise the lowest token ids (BPE ids roughly follow frequency).
    pub fn build_draft_head(&mut self, n: usize) -> Result<()> {
        let vocab = self.tokenizer.vocab_size().min(self.lm_head.rows);
        let mut ids: Vec<u32> = match self.file.bytes("draft_vocab") {
            Ok(b) => b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Err(_) => (0..vocab as u32).collect(),
        };
        // always allow drafting the chat-control tokens
        let mut extra: Vec<u32> = ["<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<|im_start|>"]
            .iter()
            .filter_map(|s| self.tokenizer.token_id(s))
            .collect();
        let n = (n.min(ids.len() + extra.len()) / 16 * 16).max(16);
        ids.retain(|i| !extra.contains(i) && (*i as usize) < vocab);
        ids.truncate(n - extra.len());
        ids.append(&mut extra);
        let buf = crate::quant::gather_rows(self.lm_head.bytes(), self.lm_head.dtype, self.lm_head.k, &ids);
        let mat = QMat {
            dtype: self.lm_head.dtype,
            rows: ids.len(),
            k: self.lm_head.k,
            ptr: buf.as_ptr(),
            len: buf.len(),
            id: usize::MAX,
            rot: self.lm_head.rot,
        };
        self.draft = Some(DraftHead { mat, ids, _buf: buf });
        Ok(())
    }

    /// Allocate rollback buffers for batches with `n` checkpoints.
    pub fn ensure_ckpts(&self, st: &mut State, n: usize) {
        while st.ckpt_conv.len() < n {
            st.ckpt_conv.push(st.conv.iter().map(|v| vec![0f32; v.len()]).collect());
        }
        if n > 0 && st.ssm_alt.is_empty() {
            st.ssm_alt = st.ssm.iter().map(|v| vec![0f32; v.len()]).collect();
            st.replay = st.ssm.iter().map(|_| Vec::new()).collect();
        }
    }

    fn replay_stride(&self) -> usize {
        self.cfg.lin_k_dim + self.cfg.lin_v_dim + 1
    }

    /// Undo the tail of a forward pass run with checkpoints: the sequence goes
    /// back to `pos` and the recurrent states to "after token c" of that batch.
    /// Conv states are swapped in; DeltaNet states are rebuilt by replaying the
    /// logged rank-1 updates of tokens 0..=c on top of the pre-batch state
    /// (one read + one write of the state, bitwise identical to the forward pass).
    pub fn rollback(&self, pool: &ThreadPool, st: &mut State, pos: usize, c: usize) {
        let cfg = &self.cfg;
        let (nv, dk, dv) = (cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim);
        let stride = self.replay_stride();
        let fast = dv == 128 && crate::kernels::isa() == crate::kernels::Isa::Avx512Vnni;
        for i in 0..st.conv.len() {
            std::mem::swap(&mut st.conv[i], &mut st.ckpt_conv[c][i]);
            let dst = SyncPtr::new(&mut st.ssm[i]);
            let s0 = &st.ssm_alt[i];
            let log = &st.replay[i];
            pool.par_tasks(nv, |hd| {
                let base = hd * dk * dv;
                // log layout: [token][head][entry]
                let entry0 = unsafe { log.as_ptr().add(hd * stride) };
                if fast {
                    unsafe { crate::kernels::vmath::delta_replay_128(s0.as_ptr().add(base), dst.get().add(base), dk, entry0, nv * stride, c + 1) };
                } else {
                    for ii in 0..dk {
                        for j in 0..dv {
                            let mut x = s0[base + ii * dv + j];
                            for t in 0..=c {
                                let e = &log[(t * nv + hd) * stride..(t * nv + hd + 1) * stride];
                                x = e[dk + dv] * x + e[ii] * e[dk + j];
                            }
                            unsafe { *dst.get().add(base + ii * dv + j) = x };
                        }
                    }
                }
            });
        }
        st.pos = pos;
        st.tokens.truncate(pos);
    }

    /// Run the multi-token-prediction block over positions `st.mtp_pos..+t`.
    /// Inputs per position p: the embedding of token p+1 and the main model's
    /// final-normed hidden state at p. With `logits`, the draft-head logits of
    /// the last position are left in `sc.logits[..draft.rows]`.
    pub fn mtp_forward(&self, pool: &ThreadPool, st: &mut State, sc: &mut Scratch, next: &[u32], hid: &[f32], logits: bool) -> Result<()> {
        let _p = scope("mtp");
        let mtp = self.mtp.as_ref().context("model has no MTP head")?;
        let t = next.len();
        let h = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let pos0 = st.mtp_pos;
        if t == 0 {
            return Ok(());
        }
        if pos0 + t > st.max_ctx {
            bail!("context length exceeded (mtp)");
        }
        resize(&mut sc.cat, t * 2 * h);
        resize(&mut sc.x, t * h);
        resize(&mut sc.xn, t * h);
        for i in 0..t {
            let row = &mut sc.cat[i * 2 * h..(i + 1) * 2 * h];
            self.embed_row(next[i], &mut row[..h]);
            rmsnorm_inplace(&mut row[..h], mtp.norm_e, eps);
            rmsnorm(&hid[i * h..(i + 1) * h], mtp.norm_h, eps, &mut row[h..]);
        }
        let cat = std::mem::take(&mut sc.cat);
        matmul(pool, &mtp.fc, &cat, t, &mut sc.x, &mut sc.qa);
        sc.cat = cat;
        par_rmsnorm(pool, &sc.x, mtp.layer.attn_norm, eps, &mut sc.xn, t, h);
        if let Mixer::Full(aw) = &mtp.layer.mixer {
            self.attention(pool, aw, &mut st.mtp_k, &mut st.mtp_v, st.max_ctx, sc, t, pos0);
        }
        for i in 0..t * h {
            sc.x[i] += sc.mix[i];
        }
        self.mlp(pool, &mtp.layer, sc, t);
        st.mtp_pos += t;
        if logits {
            let draft = self.draft.as_ref().context("draft head not built")?;
            rmsnorm(&sc.x[(t - 1) * h..t * h], mtp.final_norm, eps, &mut sc.xn[..h]);
            resize(&mut sc.logits, draft.mat.rows);
            let xn = std::mem::take(&mut sc.xn);
            matmul(pool, &draft.mat, &xn, 1, &mut sc.logits, &mut sc.qa);
            sc.xn = xn;
        }
        Ok(())
    }

    pub fn embed_row(&self, tok: u32, out: &mut [f32]) {
        let m = self.embed.as_ref().unwrap_or(&self.lm_head);
        m.dequantize_row(tok as usize, out);
    }

    /// Run the transformer over `tokens` (appending to `st`). On return
    /// `sc.x[..t*hidden]` holds the final hidden states *before* the final norm.
    pub fn forward(&self, pool: &ThreadPool, st: &mut State, sc: &mut Scratch, tokens: &[u32]) -> Result<()> {
        self.forward_ckpt(pool, st, sc, tokens, 0)
    }

    /// Like `forward`, additionally capturing the recurrent state after each of
    /// the first `nckpt` tokens so `rollback` can undo the tokens after them.
    pub fn forward_ckpt(&self, pool: &ThreadPool, st: &mut State, sc: &mut Scratch, tokens: &[u32], nckpt: usize) -> Result<()> {
        let t = tokens.len();
        let ckpt = nckpt.min(t);
        self.ensure_ckpts(st, ckpt);
        if t == 0 {
            return Ok(());
        }
        if st.pos + t > st.max_ctx {
            bail!("context length exceeded ({} + {} > {})", st.pos, t, st.max_ctx);
        }
        let h = self.cfg.hidden;
        resize(&mut sc.x, t * h);
        resize(&mut sc.xn, t * h);
        {
            let _p = scope("embed");
            for (i, &tok) in tokens.iter().enumerate() {
                self.embed_row(tok, &mut sc.x[i * h..(i + 1) * h]);
            }
        }
        for (li, layer) in self.layers.iter().enumerate() {
            self.layer_forward(pool, layer, li, st, sc, t, st.pos, ckpt)?;
        }
        st.pos += t;
        st.tokens.extend_from_slice(tokens);
        Ok(())
    }

    fn layer_forward(
        &self,
        pool: &ThreadPool,
        layer: &LayerW,
        li: usize,
        st: &mut State,
        sc: &mut Scratch,
        t: usize,
        pos0: usize,
        ckpt: usize,
    ) -> Result<()> {
        let h = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        {
            let _p = scope("norm+residual");
            par_rmsnorm(pool, &sc.x, layer.attn_norm, eps, &mut sc.xn, t, h);
        }
        match &layer.mixer {
            Mixer::Linear(lw) => self.deltanet(pool, lw, self.slot[li], st, sc, t, ckpt),
            Mixer::Full(aw) => {
                let fi = self.slot[li];
                let (kc, vc) = (&mut st.k_cache[fi], &mut st.v_cache[fi]);
                self.attention(pool, aw, kc, vc, st.max_ctx, sc, t, pos0)
            }
        }
        {
            let _p = scope("norm+residual");
            for i in 0..t * h {
                sc.x[i] += sc.mix[i];
            }
        }
        self.mlp(pool, layer, sc, t);
        Ok(())
    }

    fn mlp(&self, pool: &ThreadPool, layer: &LayerW, sc: &mut Scratch, t: usize) {
        let h = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let inter = layer.down.k;
        {
            let _p = scope("norm+residual");
            par_rmsnorm(pool, &sc.x, layer.mlp_norm, eps, &mut sc.xn, t, h);
        }
        resize(&mut sc.gu, t * 2 * inter);
        resize(&mut sc.act, t * inter);
        {
            let _p = scope("mlp.gate_up");
            matmul(pool, &layer.gate_up, &sc.xn, t, &mut sc.gu, &mut sc.qa);
        }
        let _p = scope("mlp.act");
        let gu = &sc.gu;
        let ap = SyncPtr::new(&mut sc.act);
        pool.par_chunks(t, 1, |lo, hi| {
            for i in lo..hi {
                let g = &gu[i * 2 * inter..i * 2 * inter + inter];
                let u = &gu[i * 2 * inter + inter..(i + 1) * 2 * inter];
                let a = unsafe { ap.slice(i * inter, inter) };
                ops::swiglu(g, u, a);
            }
        });
        drop(_p);
        resize(&mut sc.mix, t * h);
        {
            let _p = scope("mlp.down");
            matmul(pool, &layer.down, &sc.act, t, &mut sc.mix, &mut sc.qa);
        }
        let _p = scope("norm+residual");
        for i in 0..t * h {
            sc.x[i] += sc.mix[i];
        }
    }

    /// Gated DeltaNet token mixer. Leaves its output in `sc.mix`.
    #[allow(clippy::too_many_arguments)]
    fn deltanet(&self, pool: &ThreadPool, lw: &LinearW, slot: usize, st: &mut State, sc: &mut Scratch, t: usize, ckpt: usize) {
        let c = &self.cfg;
        let (nk, nv, dk, dv) = (c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim);
        let kd = nk * dk;
        let vd = nv * dv;
        let cd = c.conv_dim();
        let rows = lw.in_proj.rows; // cd + vd
        let kk = c.conv_kernel;
        resize(&mut sc.proj, t * rows);
        resize(&mut sc.ba, t * lw.in_ba.rows);
        {
            let _p = scope("dn.in_proj");
            matmul(pool, &lw.in_proj, &sc.xn, t, &mut sc.proj, &mut sc.qa);
        }
        let _p = scope("dn.in_ba");
        if lw.in_ba.dtype != DType::F32 && lw.in_proj.dtype != DType::F32 && lw.in_ba.rot == lw.in_proj.rot {
            matmul_q(pool, &lw.in_ba, &sc.qa, &mut sc.ba);
        } else {
            matmul(pool, &lw.in_ba, &sc.xn, t, &mut sc.ba, &mut sc.qa);
        }

        drop(_p);
        // causal depthwise conv + SiLU over the q/k/v channels, sequential in time
        {
            let _p = scope("dn.conv");
            let conv_w = lw.conv_w;
            let pp = SyncPtr::new(&mut sc.proj);
            let sp = SyncPtr::new(&mut st.conv[slot]);
            let ckps: Vec<SyncPtr<f32>> = (0..ckpt).map(|c| SyncPtr::new(&mut st.ckpt_conv[c][slot])).collect();
            let fast = kk == 4 && crate::kernels::isa() == crate::kernels::Isa::Avx512Vnni;
            pool.par_chunks(cd, 64, |lo, hi| {
                if fast {
                    let cks: Vec<*mut f32> = ckps.iter().map(|p| p.get()).collect();
                    unsafe { crate::kernels::vmath::conv4_silu(pp.get(), rows, t, conv_w, sp.get(), cd, lo, hi, &cks) };
                    return;
                }
                let state = unsafe { sp.slice(0, (kk - 1) * cd) };
                for tok in 0..t {
                    let row = unsafe { pp.slice(tok * rows, cd) };
                    for ch in lo..hi {
                        let xin = row[ch];
                        let mut y = conv_w[(kk - 1) * cd + ch] * xin;
                        for j in 0..kk - 1 {
                            y += conv_w[j * cd + ch] * state[j * cd + ch];
                        }
                        for j in 0..kk.saturating_sub(2) {
                            state[j * cd + ch] = state[(j + 1) * cd + ch];
                        }
                        state[(kk - 2) * cd + ch] = xin;
                        row[ch] = y;
                    }
                    ops::silu_inplace(&mut row[lo..hi]);
                    if let Some(ckp) = ckps.get(tok) {
                        let ckst = unsafe { ckp.slice(0, (kk - 1) * cd) };
                        for j in 0..kk - 1 {
                            ckst[j * cd + lo..j * cd + hi].copy_from_slice(&state[j * cd + lo..j * cd + hi]);
                        }
                    }
                }
            });
        }

        // gated delta rule, one task per value head (sequential over tokens)
        resize(&mut sc.core, t * vd);
        {
            let _p = scope("dn.recurrence");
            let proj = &sc.proj;
            let ba = &sc.ba;
            let nba = lw.in_ba.rows;
            let ssm = SyncPtr::new(&mut st.ssm[slot]);
            // checkpointed batch: the state streams main -> alt once (kept in L2
            // between tokens) and each token's update is logged for rollback
            let stride = self.replay_stride();
            let (alt, log) = if ckpt > 0 {
                st.replay[slot].resize(t * nv * stride, 0.0);
                (SyncPtr::new(&mut st.ssm_alt[slot]), SyncPtr::new(&mut st.replay[slot]))
            } else {
                (SyncPtr(std::ptr::null_mut()), SyncPtr(std::ptr::null_mut()))
            };
            let core = SyncPtr::new(&mut sc.core);
            let scale = 1.0 / (dk as f32).sqrt();
            let ratio = nv / nk;
            let eps = c.rms_eps;
            let fast = crate::kernels::isa() == crate::kernels::Isa::Avx512Vnni;
            pool.par_tasks(nv, |hd| {
                let kh = hd / ratio;
                let main = unsafe { ssm.get().add(hd * dk * dv) };
                let altp = if ckpt > 0 { unsafe { alt.get().add(hd * dk * dv) } } else { main };
                let mut q = vec![0f32; dk];
                let mut k = vec![0f32; dk];
                let mut kv = vec![0f32; dv];
                let mut o = vec![0f32; dv];
                for tok in 0..t {
                    let row = &proj[tok * rows..(tok + 1) * rows];
                    q.copy_from_slice(&row[kh * dk..(kh + 1) * dk]);
                    k.copy_from_slice(&row[kd + kh * dk..kd + (kh + 1) * dk]);
                    let v = &row[2 * kd + hd * dv..2 * kd + (hd + 1) * dv];
                    l2norm_inplace(&mut q, 1e-6);
                    l2norm_inplace(&mut k, 1e-6);
                    for x in q.iter_mut() {
                        *x *= scale;
                    }
                    let beta = sigmoid(ba[tok * nba + hd]);
                    let g = lw.neg_a[hd] * softplus(ba[tok * nba + nv + hd] + lw.dt_bias[hd]);
                    let decay = g.exp();
                    let src = if tok == 0 { main as *const f32 } else { altp as *const f32 };
                    let dst = altp;
                    let entry = if ckpt > 0 { unsafe { log.get().add((tok * nv + hd) * stride) } } else { std::ptr::null_mut() };
                    if !entry.is_null() {
                        unsafe {
                            std::ptr::copy_nonoverlapping(k.as_ptr(), entry, dk);
                            *entry.add(dk + dv) = decay;
                        }
                    }
                    if dv == 128 && fast {
                        let dl = if entry.is_null() { entry } else { unsafe { entry.add(dk) } };
                        unsafe { crate::kernels::vmath::delta_rule_step_128(src, dst, &q, &k, v, beta, decay, &mut o, dl) };
                    } else {
                        unsafe { delta_rule_step(src, dst, &q, &k, v, beta, decay, &mut kv, &mut o, dk, dv) };
                        if !entry.is_null() {
                            unsafe { std::ptr::copy_nonoverlapping(kv.as_ptr(), entry.add(dk), dv) };
                        }
                    }
                    // gated RMSNorm: rmsnorm(o) * w * silu(z)
                    let z = &row[cd + hd * dv..cd + (hd + 1) * dv];
                    let out = unsafe { core.slice(tok * vd + hd * dv, dv) };
                    rmsnorm(&o, lw.gnorm, eps, out);
                    ops::mul_silu(out, z);
                }
            });
            if ckpt > 0 {
                // live state = after the batch; alt keeps the pre-batch state for replay
                std::mem::swap(&mut st.ssm[slot], &mut st.ssm_alt[slot]);
            }
        }
        resize(&mut sc.mix, t * c.hidden);
        let _p = scope("dn.out_proj");
        matmul(pool, &lw.out_proj, &sc.core, t, &mut sc.mix, &mut sc.qa);
    }

    /// Gated softmax attention. Leaves its output in `sc.mix`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        pool: &ThreadPool,
        aw: &AttnW,
        kc: &mut [u16],
        vc: &mut [u16],
        max_ctx: usize,
        sc: &mut Scratch,
        t: usize,
        pos0: usize,
    ) {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let rows = aw.qkv.rows;
        let qoff = 0;
        let goff = nh * hd;
        let koff = if c.attn_output_gate { 2 * nh * hd } else { nh * hd };
        let voff = koff + nkv * hd;
        let eps = c.rms_eps;
        resize(&mut sc.proj, t * rows);
        {
            let _p = scope("attn.qkv");
            matmul(pool, &aw.qkv, &sc.xn, t, &mut sc.proj, &mut sc.qa);
        }

        // QK-norm, RoPE, append K/V to the cache
        {
            let _p = scope("attn.norm_rope");
            let pp = SyncPtr::new(&mut sc.proj);
            let kp = SyncPtr::new(kc);
            let vp = SyncPtr::new(vc);
            let inv_freq = &self.inv_freq;
            pool.par_chunks(t, 1, |lo, hi| {
                let half = inv_freq.len();
                let mut cos = vec![0f32; half];
                let mut sin = vec![0f32; half];
                for tok in lo..hi {
                    let pos = pos0 + tok;
                    ops::rope_tables(pos, inv_freq, &mut cos, &mut sin);
                    let row = unsafe { pp.slice(tok * rows, rows) };
                    for hh in 0..nh {
                        let q = &mut row[qoff + hh * hd..qoff + (hh + 1) * hd];
                        rmsnorm_inplace(q, aw.q_norm, eps);
                        rope_neox(q, &cos, &sin);
                    }
                    for g in 0..nkv {
                        let k = &mut row[koff + g * hd..koff + (g + 1) * hd];
                        rmsnorm_inplace(k, aw.k_norm, eps);
                        rope_neox(k, &cos, &sin);
                        let dst = unsafe { kp.slice((g * max_ctx + pos) * hd, hd) };
                        ops::f32_to_f16(k, dst);
                        let v = &row[voff + g * hd..voff + (g + 1) * hd];
                        let dst = unsafe { vp.slice((g * max_ctx + pos) * hd, hd) };
                        ops::f32_to_f16(v, dst);
                    }
                }
            });
        }

        // attention: tasks = token x kv-head x key-split (flash-decoding style for long contexts)
        let grp = nh / nkv;
        let n_keys_max = pos0 + t;
        let nsplit = if t == 1 { (n_keys_max / 256).clamp(1, (2 * pool.n_threads()).div_ceil(nkv).max(1)) } else { 1 };
        let part_stride = hd + 2; // [m, l, acc...]
        resize(&mut sc.part, t * nkv * nsplit * grp * part_stride);
        resize(&mut sc.core, t * nh * hd);
        {
            let _p = scope("attn.softmax_attn");
            let proj = &sc.proj;
            let kc: &[u16] = kc;
            let vc: &[u16] = vc;
            let part = SyncPtr::new(&mut sc.part);
            let scale = 1.0 / (hd as f32).sqrt();
            pool.par_tasks(t * nkv * nsplit, |task| {
                let s = task % nsplit;
                let g = (task / nsplit) % nkv;
                let tok = task / (nsplit * nkv);
                let n_keys = pos0 + tok + 1;
                let chunk = n_keys.div_ceil(nsplit);
                let k0 = (s * chunk).min(n_keys);
                let k1 = ((s + 1) * chunk).min(n_keys);
                let len = k1 - k0;
                let row = &proj[tok * rows..(tok + 1) * rows];
                let mut scores = vec![0f32; grp * len.max(1)];
                for j in 0..len {
                    let kr = &kc[(g * max_ctx + k0 + j) * hd..(g * max_ctx + k0 + j + 1) * hd];
                    for qi in 0..grp {
                        let hh = g * grp + qi;
                        let q = &row[qoff + hh * hd..qoff + (hh + 1) * hd];
                        scores[qi * len + j] = ops::dot_f32_f16(q, kr) * scale;
                    }
                }
                for qi in 0..grp {
                    let base = ((tok * nkv + g) * nsplit + s) * grp + qi;
                    let out = unsafe { part.slice(base * part_stride, part_stride) };
                    let sl = &mut scores[qi * len..(qi + 1) * len];
                    let m = sl.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let l = if sl.is_empty() { 0.0 } else { ops::exp_sub_sum(sl, m) };
                    out[0] = m;
                    out[1] = l;
                    out[2..].iter_mut().for_each(|x| *x = 0.0);
                }
                for j in 0..len {
                    let vr = &vc[(g * max_ctx + k0 + j) * hd..(g * max_ctx + k0 + j + 1) * hd];
                    for qi in 0..grp {
                        let base = ((tok * nkv + g) * nsplit + s) * grp + qi;
                        let out = unsafe { part.slice(base * part_stride + 2, hd) };
                        ops::axpy_f16(out, scores[qi * len + j], vr);
                    }
                }
            });
            // merge partial softmaxes and apply the output gate
            let part = &sc.part;
            let core = SyncPtr::new(&mut sc.core);
            let gated = c.attn_output_gate;
            pool.par_chunks(t * nh, 1, |lo, hi| {
                for idx in lo..hi {
                    let tok = idx / nh;
                    let hh = idx % nh;
                    let g = hh / grp;
                    let qi = hh % grp;
                    let out = unsafe { core.slice((tok * nh + hh) * hd, hd) };
                    let mut mmax = f32::NEG_INFINITY;
                    for s in 0..nsplit {
                        let b = (((tok * nkv + g) * nsplit + s) * grp + qi) * part_stride;
                        if part[b + 1] > 0.0 {
                            mmax = mmax.max(part[b]);
                        }
                    }
                    out.iter_mut().for_each(|x| *x = 0.0);
                    let mut denom = 0f32;
                    for s in 0..nsplit {
                        let b = (((tok * nkv + g) * nsplit + s) * grp + qi) * part_stride;
                        if part[b + 1] <= 0.0 {
                            continue;
                        }
                        let w = (part[b] - mmax).exp();
                        denom += w * part[b + 1];
                        for d in 0..hd {
                            out[d] += w * part[b + 2 + d];
                        }
                    }
                    let inv = 1.0 / denom;
                    let row = &proj[tok * rows..(tok + 1) * rows];
                    out.iter_mut().for_each(|x| *x *= inv);
                    if gated {
                        ops::mul_sigmoid(out, &row[goff + hh * hd..goff + (hh + 1) * hd]);
                    }
                }
            });
        }
        resize(&mut sc.mix, t * c.hidden);
        let _p = scope("attn.o_proj");
        matmul(pool, &aw.o_proj, &sc.core, t, &mut sc.mix, &mut sc.qa);
    }

    /// Final norm + LM head for `n` hidden rows starting at row `first` of `sc.x`.
    /// Logits land in `sc.logits[..n * vocab_rows]`.
    pub fn logits(&self, pool: &ThreadPool, sc: &mut Scratch, first: usize, n: usize) {
        let h = self.cfg.hidden;
        resize(&mut sc.xn, n * h);
        for i in 0..n {
            let src = &sc.x[(first + i) * h..(first + i + 1) * h];
            rmsnorm(src, self.final_norm, self.cfg.rms_eps, &mut sc.xn[i * h..(i + 1) * h]);
        }
        let v = self.lm_head.rows;
        resize(&mut sc.logits, n * v);
        let xn = std::mem::take(&mut sc.xn);
        let _p = scope("lm_head");
        matmul(pool, &self.lm_head, &xn, n, &mut sc.logits, &mut sc.qa);
        sc.xn = xn;
        // mask padded vocabulary rows that the tokenizer never produces
        let tv = self.tokenizer.vocab_size().min(v);
        for i in 0..n {
            sc.logits[i * v + tv..(i + 1) * v].iter_mut().for_each(|x| *x = f32::NEG_INFINITY);
        }
    }

    pub fn vocab_rows(&self) -> usize {
        self.lm_head.rows
    }
}

fn par_rmsnorm(pool: &ThreadPool, x: &[f32], w: &[f32], eps: f32, out: &mut [f32], t: usize, h: usize) {
    if t < 8 {
        for i in 0..t {
            rmsnorm(&x[i * h..(i + 1) * h], w, eps, &mut out[i * h..(i + 1) * h]);
        }
        return;
    }
    let op = SyncPtr::new(out);
    pool.par_chunks(t, 1, |lo, hi| {
        for i in lo..hi {
            rmsnorm(&x[i * h..(i + 1) * h], w, eps, unsafe { op.slice(i * h, h) });
        }
    });
}

/// # Safety
/// `src` and `dst` must point to `dk * dv` floats (they may be the same buffer).
///
/// One step of the gated delta rule for a single head (state S is [dk][dv], row-major):
///   S <- decay * S;  delta = beta * (v - S^T k);  S <- S + k delta^T;  o = S^T q
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn delta_rule_step(
    src: *const f32,
    dst: *mut f32,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    kv: &mut [f32],
    o: &mut [f32],
    dk: usize,
    dv: usize,
) {
    let srcs = unsafe { std::slice::from_raw_parts(src, dk * dv) };
    kv.iter_mut().for_each(|x| *x = 0.0);
    for i in 0..dk {
        let ki = k[i];
        let srow = &srcs[i * dv..(i + 1) * dv];
        for j in 0..dv {
            kv[j] += srow[j] * ki;
        }
    }
    for j in 0..dv {
        kv[j] = beta * (v[j] - decay * kv[j]); // kv now holds delta
    }
    o.iter_mut().for_each(|x| *x = 0.0);
    for i in 0..dk {
        let ki = k[i];
        let qi = q[i];
        for j in 0..dv {
            // read before write: src and dst may alias
            let x = decay * unsafe { *src.add(i * dv + j) } + ki * kv[j];
            unsafe { *dst.add(i * dv + j) = x };
            o[j] += x * qi;
        }
    }
}
