//! Hugging Face checkpoint -> `.ferrite` converter.
//!
//! Besides quantizing, conversion pre-bakes everything the runtime would
//! otherwise recompute: `(1 + w)` RMSNorm offsets are folded, `-exp(A_log)` is
//! precomputed, projections that read the same input are fused into one matrix
//! (q/k/v/z for DeltaNet, q/gate/k/v for attention, gate/up for the MLP), the
//! attention output gate is de-interleaved and conv1d weights are transposed.

use crate::config::{Config, LayerKind};
use crate::format::Writer;
use crate::quant::{DType, quantize_matrix};
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Which storage format each matrix gets. Matrices are identified by their
/// class (the suffix of the tensor name: `in_proj`, `in_ba`, `out_proj`, `qkv`,
/// `o_proj`, `gate_up`, `down`, `lm_head`, `embed`, `fc`), optionally with a
/// layer index (`down@3`).
#[derive(Clone, Debug)]
pub struct Recipe {
    pub default: DType,
    pub overrides: Vec<(String, DType)>,
    /// Hadamard-rotate matrices by default (measured on Qwen3.5-0.8B vs f32:
    /// rotation lowers KL by 30-40% for 8/6-bit but raises it for 4-bit, where
    /// imatrix-weighted rounding of salient channels matters more)
    pub rotate: bool,
    /// Also store a low-bit copy of the LM head (`lm_proxy`) that generation
    /// reads for every token, scoring only its shortlist with the exact head.
    /// At 0.8B the 8-bit head is ~47% of the bytes per token; the 2-bit proxy is 80 MB.
    pub proxy: Option<DType>,
}

impl Recipe {
    /// `base[,class=dtype,...]`, e.g. `q4` or `q4,down=q8,lm_head=q4`.
    pub fn parse(spec: &str) -> Result<Recipe> {
        let mut parts = spec.split(',');
        let base = parts.next().unwrap_or("q8").trim();
        let q8 = |c: &str| (c.to_string(), DType::Q8R);
        let (default, mut overrides, rotate): (DType, Vec<(String, DType)>, bool) = match base {
            "f32" => (DType::F32, vec![], false),
            "q8" => (DType::Q8R, vec![], true),
            // 6-bit body, 8-bit tiny gate projections
            "q6" => (DType::Q6R, vec![q8("in_ba")], true),
            // 4-bit body; the tied embedding/LM head, the tiny DeltaNet gate
            // projections and the attention/DeltaNet output projections (small
            // but the most quantization-sensitive per byte) stay 8-bit
            "q4" => (DType::Q4R, vec![q8("lm_head"), q8("embed"), q8("in_ba"), q8("out_proj"), q8("o_proj")], false),
            // everything 4-bit except the gate projections: smallest/fastest, lossy
            "q4-fast" => (DType::Q4R, vec![q8("in_ba")], false),
            _ => bail!("unknown quantization recipe '{base}' (f32 | q8 | q6 | q4 | q4-fast)"),
        };
        // every quantized recipe gets a 2-bit proxy head unless its head is already 4-bit
        // (0.8B q4: 4-bit proxy +23% decode, 2-bit +19% more; greedy output unchanged)
        let mut proxy = (base != "f32" && base != "q4-fast").then_some(DType::Q2R);
        for p in parts {
            let (k, v) = p.split_once('=').with_context(|| format!("bad override '{p}'"))?;
            if k.trim() == "lm_proxy" {
                proxy = if v.trim() == "none" { None } else { Some(DType::parse(v.trim()).with_context(|| format!("bad dtype '{v}'"))?) };
                continue;
            }
            let dt = DType::parse(v.trim()).with_context(|| format!("bad dtype '{v}'"))?;
            overrides.retain(|(c, _)| c != k.trim());
            overrides.push((k.trim().to_string(), dt));
        }
        Ok(Recipe { default, overrides, rotate, proxy })
    }

    /// Keys: a class (`down`), a class in one layer (`down@3`), the whole MTP
    /// block (`mtp`) or one class inside it (`mtp.qkv`). Later overrides win.
    pub fn dtype_for(&self, name: &str) -> DType {
        // name like "L3.down", "mtp.L0.qkv", "mtp.fc", "lm_head"
        let class = name.rsplit('.').next().unwrap_or(name);
        let is_mtp = name.starts_with("mtp.");
        let layer = name.split('.').find_map(|p| p.strip_prefix('L')).and_then(|n| n.parse::<usize>().ok());
        let mut dt = self.default;
        for (k, v) in &self.overrides {
            let hit = if k == "mtp" {
                is_mtp
            } else if let Some(c) = k.strip_prefix("mtp.") {
                is_mtp && c == class
            } else {
                match k.split_once('@') {
                    Some((c, l)) => c == class && !is_mtp && layer.map(|x| x.to_string()) == Some(l.to_string()),
                    None => k == class,
                }
            };
            if hit {
                dt = *v;
            }
        }
        dt
    }
}

/// Calibration source for GPTQ: an already converted (8-bit) model of the same
/// checkpoint and calibration tokens. Input covariances are collected a few
/// layers at a time (as many as fit in `budget` bytes) when first needed.
pub struct Gptq {
    e: crate::engine::Engine,
    calib: Vec<u32>,
    chunks: usize,
    budget: usize,
    cache: HashMap<String, Vec<f32>>,
}

impl Gptq {
    pub fn new(model: &Path, calib: &Path, chunks: usize) -> Result<Gptq> {
        let mut e = crate::engine::Engine::load(model, crate::engine::default_threads(), 1024)?;
        e.fast_head = false;
        let ids = e.model.tokenizer.encode(&std::fs::read_to_string(calib).with_context(|| format!("read {}", calib.display()))?, false);
        let chunks = chunks.min(ids.len() / 512);
        let budget = std::env::var("FERRITE_GPTQ_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(3072usize) << 20;
        Ok(Gptq { e, calib: ids, chunks, budget, cache: HashMap::new() })
    }

    /// Input covariance of the per-layer matrix `name` (e.g. "L3.down"), or None.
    fn hessian(&mut self, name: &str) -> Result<Option<Vec<f32>>> {
        if let Some(h) = self.cache.remove(name) {
            return Ok(Some(h));
        }
        let layer = |n: &str| n.strip_prefix('L').and_then(|s| s.split('.').next()).and_then(|s| s.parse::<usize>().ok());
        let Some(first) = layer(name) else { return Ok(None) };
        let hdr = &self.e.model.file.header;
        let mut want = Vec::new();
        let (mut bytes, mut last) = (0usize, first);
        for li in first..hdr.config.n_layers {
            let mats: Vec<(usize, usize)> = hdr
                .tensors
                .iter()
                .enumerate()
                .filter(|(_, t)| t.shape.len() == 2 && layer(&t.name) == Some(li) && t.name.starts_with('L'))
                .map(|(id, t)| (id, t.shape[1]))
                .collect();
            let b: usize = mats.iter().map(|(_, k)| 2 * k * k * 4).sum(); // H + scratch
            if li > first && bytes + b > self.budget {
                break;
            }
            want.extend(mats);
            bytes += b;
            last = li;
        }
        let t0 = Instant::now();
        crate::gptq::begin(&want);
        crate::gptq::STOP_AFTER.store(last, std::sync::atomic::Ordering::SeqCst);
        let run = (0..self.chunks).try_for_each(|c| {
            self.e.reset();
            self.e.feed(&self.calib[c * 512..(c + 1) * 512]).map(|_| ())
        });
        crate::gptq::STOP_AFTER.store(usize::MAX, std::sync::atomic::Ordering::SeqCst);
        let hs = crate::gptq::take();
        run?;
        for (id, h) in hs {
            self.cache.insert(self.e.model.file.header.tensors[id].name.clone(), h);
        }
        eprint!("\r  gptq: input covariances of layers {first}-{last} ({:.0}s)            \n", t0.elapsed().as_secs_f64());
        Ok(self.cache.remove(name))
    }
}

struct Ctx<'a> {
    st: &'a SafeTensors,
    w: Writer,
    total_params: usize,
    total_bytes: usize,
    imatrix: Option<HashMap<String, Vec<f32>>>,
    /// imatrix statistics were collected in rotated space
    imatrix_rotated: bool,
    /// the matching unrotated statistics (`<imatrix>.raw.json`), for matrices
    /// left unrotated such as the 4-bit proxy head
    imatrix_raw: Option<HashMap<String, Vec<f32>>>,
    recipe: Recipe,
    rotate: bool,
    gptq: Option<Gptq>,
}

impl Ctx<'_> {
    fn load(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let t = self.st.get(name)?;
        Ok((t.to_f32(), t.shape.clone()))
    }

    /// Quantize and write a matrix built by stacking `parts` (each [rows_i, k]).
    /// Takes ownership so a single part is used without copying (the f32
    /// embedding of a 4B model alone is 2.5 GB).
    fn matrix(&mut self, out_name: &str, parts: Vec<Vec<f32>>, k: usize) -> Result<()> {
        let dt = self.recipe.dtype_for(out_name);
        self.matrix_as(out_name, parts, k, dt, self.rotate, out_name)
    }

    /// `matrix` with an explicit dtype, rotation choice and imatrix entry.
    fn matrix_as(&mut self, out_name: &str, mut parts: Vec<Vec<f32>>, k: usize, dt: DType, rotate: bool, imp_key: &str) -> Result<()> {
        for p in &parts {
            assert_eq!(p.len() % k, 0);
        }
        let w = if parts.len() == 1 {
            parts.pop().unwrap()
        } else {
            let mut w = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
            for p in parts.drain(..) {
                w.extend_from_slice(&p);
            }
            w
        };
        let nrows = w.len() / k;
        let mut w = w;
        // Hadamard-rotate the input dimension of every quantized matrix (rows of
        // a rotated embedding are un-rotated when looked up)
        let rot = if rotate && dt != DType::F32 {
            crate::quant::rot_block(k)
        } else {
            0
        };
        if rot > 1 {
            for row in w.chunks_exact_mut(k) {
                crate::quant::rotate(row, rot);
            }
        }
        if !nrows.is_multiple_of(16) {
            bail!("{out_name}: {nrows} rows is not a multiple of 16");
        }
        let mut imp: Option<Vec<f32>> = self.imatrix.as_ref().and_then(|m| m.get(imp_key)).cloned();
        if let Some(v) = imp.as_mut() {
            if rot > 1 && !self.imatrix_rotated {
                // stats from an unrotated model: in rotated space each column of a
                // block sees (in expectation) the block's mean activation energy
                for blk in v.chunks_exact_mut(rot) {
                    let m = blk.iter().sum::<f32>() / rot as f32;
                    blk.iter_mut().for_each(|x| *x = m);
                }
            } else if rot <= 1 && self.imatrix_rotated {
                // rotated stats do not apply to an unrotated matrix; use the raw ones if any
                imp = self.imatrix_raw.as_ref().and_then(|m| m.get(imp_key)).cloned();
            }
        }
        // GPTQ for unrotated 4-bit matrices when a calibration model is given
        let h = match self.gptq.as_mut() {
            Some(g) if dt == DType::Q4R && rot <= 1 => g.hessian(out_name)?,
            _ => None,
        };
        let q = match h {
            Some(h) => crate::gptq::quantize_q4r(&w, nrows, k, &h),
            None => quantize_matrix(&w, nrows, k, dt, imp.as_deref()),
        };
        if out_name != "lm_proxy" {
            self.total_params += nrows * k; // the proxy duplicates the head's parameters
        }
        self.total_bytes += q.len();
        self.w.add(out_name, dt, &[nrows, k], &q)?;
        if rot > 1 {
            self.w.set_rot(rot);
        }
        Ok(())
    }

    fn vector(&mut self, out_name: &str, v: &[f32]) -> Result<()> {
        self.total_params += v.len();
        self.total_bytes += v.len() * 4;
        self.w.add_f32(out_name, v)
    }

    fn norm(&mut self, out_name: &str, src: &str, offset: bool) -> Result<()> {
        let (mut v, _) = self.load(src)?;
        if offset {
            v.iter_mut().for_each(|x| *x += 1.0);
        }
        self.vector(out_name, &v)
    }
}

/// Convert an attention block (`self_attn.*`) of the qwen3_5 / qwen3 family.
fn convert_attention(cx: &mut Ctx, cfg: &Config, src: &str, dst: &str, offset: bool) -> Result<()> {
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let (q, _) = cx.load(&format!("{src}.q_proj.weight"))?;
    let (k, _) = cx.load(&format!("{src}.k_proj.weight"))?;
    let (v, _) = cx.load(&format!("{src}.v_proj.weight"))?;
    let mut parts = Vec::new();
    if cfg.attn_output_gate {
        // q_proj rows are [head][query(hd) | gate(hd)] -> split into [all q][all gates]
        let mut qq = Vec::with_capacity(cfg.n_heads * hd * h);
        let mut gg = Vec::with_capacity(cfg.n_heads * hd * h);
        for head in 0..cfg.n_heads {
            let base = head * 2 * hd * h;
            qq.extend_from_slice(&q[base..base + hd * h]);
            gg.extend_from_slice(&q[base + hd * h..base + 2 * hd * h]);
        }
        parts.push(qq);
        parts.push(gg);
    } else {
        parts.push(q);
    }
    parts.push(k);
    parts.push(v);
    cx.matrix(&format!("{dst}.qkv"), parts, h)?;
    cx.norm(&format!("{dst}.q_norm"), &format!("{src}.q_norm.weight"), offset)?;
    cx.norm(&format!("{dst}.k_norm"), &format!("{src}.k_norm.weight"), offset)?;
    let (o, _) = cx.load(&format!("{src}.o_proj.weight"))?;
    cx.matrix(&format!("{dst}.o_proj"), vec![o], cfg.n_heads * hd)
}

fn convert_mlp(cx: &mut Ctx, cfg: &Config, src: &str, dst: &str) -> Result<()> {
    let (g, _) = cx.load(&format!("{src}.gate_proj.weight"))?;
    let (u, _) = cx.load(&format!("{src}.up_proj.weight"))?;
    cx.matrix(&format!("{dst}.gate_up"), vec![g, u], cfg.hidden)?;
    let (d, _) = cx.load(&format!("{src}.down_proj.weight"))?;
    cx.matrix(&format!("{dst}.down"), vec![d], cfg.intermediate)
}

/// `rotate`: override the recipe's default Hadamard rotation.
pub fn convert(hf_dir: &Path, out: &Path, recipe_name: &str, imatrix: Option<&Path>, rotate: Option<bool>, gptq: Option<Gptq>) -> Result<()> {
    let imatrix_path = imatrix;
    let t0 = Instant::now();
    let recipe = Recipe::parse(recipe_name)?;
    let rotate = rotate.unwrap_or(recipe.rotate);
    let cfg_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(hf_dir.join("config.json")).context("read config.json")?,
    )?;
    let cfg = Config::from_hf(&cfg_json)?;
    let tok_json = std::fs::read_to_string(hf_dir.join("tokenizer.json")).context("read tokenizer.json")?;
    let st = SafeTensors::open_dir(hf_dir)?;
    let (imatrix, imatrix_rotated) = match imatrix {
        Some(p) => {
            let (m, r) = crate::imatrix::load(p)?;
            (Some(m), r)
        }
        None => (None, false),
    };
    let raw_path = imatrix.as_ref().and(imatrix_path.map(|p| p.with_extension("raw.json")));
    let imatrix_raw = match raw_path.filter(|p| imatrix_rotated && p.exists()) {
        Some(p) => Some(crate::imatrix::load(&p)?.0),
        None => None,
    };
    let prefix = if st.contains("model.language_model.embed_tokens.weight") {
        "model.language_model"
    } else if st.contains("model.embed_tokens.weight") {
        "model"
    } else {
        bail!("cannot find embed_tokens in checkpoint");
    };
    let offset = cfg.arch == "qwen3_5"; // Qwen3.5 RMSNorm computes x * (1 + w)
    eprintln!(
        "converting {} ({} layers: {} DeltaNet + {} attention, hidden {}, vocab {}) recipe={recipe_name}",
        hf_dir.display(),
        cfg.n_layers,
        cfg.n_linear(),
        cfg.n_full(),
        cfg.hidden,
        cfg.vocab
    );

    let mut cx = Ctx { st: &st, w: Writer::create(out)?, total_params: 0, total_bytes: 0, imatrix, imatrix_rotated, imatrix_raw, recipe: recipe.clone(), rotate, gptq };
    let h = cfg.hidden;

    // Embedding / LM head
    let (emb, eshape) = cx.load(&format!("{prefix}.embed_tokens.weight"))?;
    let vocab_rows = eshape[0];
    let head_src = if cfg.tie_embeddings { format!("{prefix}.embed_tokens.weight") } else { "lm_head.weight".to_string() };
    if cfg.tie_embeddings {
        cx.matrix("lm_head", vec![emb], h)?;
    } else {
        cx.matrix("embed", vec![emb], h)?;
        let (lm, _) = cx.load("lm_head.weight")?;
        cx.matrix("lm_head", vec![lm], h)?;
    }
    if let Some(pdt) = recipe.proxy.filter(|&p| p != recipe.dtype_for("lm_head")) {
        // unrotated: rotation hurts at 4 bits (see `Recipe::rotate`)
        let (w, _) = cx.load(&head_src)?;
        cx.matrix_as("lm_proxy", vec![w], h, pdt, false, "lm_head")?;
    }
    cx.norm("final_norm", &format!("{prefix}.norm.weight"), offset)?;

    for (i, kind) in cfg.layer_kinds.clone().iter().enumerate() {
        let src = format!("{prefix}.layers.{i}");
        let dst = format!("L{i}");
        cx.norm(&format!("{dst}.attn_norm"), &format!("{src}.input_layernorm.weight"), offset)?;
        cx.norm(&format!("{dst}.mlp_norm"), &format!("{src}.post_attention_layernorm.weight"), offset)?;
        match kind {
            LayerKind::Linear => {
                let la = format!("{src}.linear_attn");
                let (qkv, _) = cx.load(&format!("{la}.in_proj_qkv.weight"))?;
                let (z, _) = cx.load(&format!("{la}.in_proj_z.weight"))?;
                cx.matrix(&format!("{dst}.in_proj"), vec![qkv, z], h)?;
                let (b, _) = cx.load(&format!("{la}.in_proj_b.weight"))?;
                let (a, _) = cx.load(&format!("{la}.in_proj_a.weight"))?;
                cx.matrix(&format!("{dst}.in_ba"), vec![b, a], h)?;
                // conv1d.weight [C, 1, K] -> [K][C] so the runtime vectorizes over channels
                let (cw, cs) = cx.load(&format!("{la}.conv1d.weight"))?;
                let (c, kk) = (cs[0], cs[2]);
                let mut ct = vec![0f32; c * kk];
                for ch in 0..c {
                    for j in 0..kk {
                        ct[j * c + ch] = cw[ch * kk + j];
                    }
                }
                cx.vector(&format!("{dst}.conv_w"), &ct)?;
                let (alog, _) = cx.load(&format!("{la}.A_log"))?;
                let neg_a: Vec<f32> = alog.iter().map(|x| -x.exp()).collect();
                cx.vector(&format!("{dst}.neg_a"), &neg_a)?;
                let (dtb, _) = cx.load(&format!("{la}.dt_bias"))?;
                cx.vector(&format!("{dst}.dt_bias"), &dtb)?;
                cx.norm(&format!("{dst}.gnorm"), &format!("{la}.norm.weight"), false)?;
                let (o, _) = cx.load(&format!("{la}.out_proj.weight"))?;
                cx.matrix(&format!("{dst}.out_proj"), vec![o], cfg.lin_value_dim())?;
            }
            LayerKind::Full => {
                convert_attention(&mut cx, &cfg, &format!("{src}.self_attn"), &dst, offset)?;
            }
        }
        convert_mlp(&mut cx, &cfg, &format!("{src}.mlp"), &dst)?;
        eprint!("\r  layer {}/{}  ({:.1}s)", i + 1, cfg.n_layers, t0.elapsed().as_secs_f32());
    }
    eprintln!();

    // Multi-token-prediction head (used for self-speculative decoding)
    let mut has_mtp = false;
    if cfg.mtp_layers > 0 && st.contains("mtp.fc.weight") {
        let (fc, _) = cx.load("mtp.fc.weight")?;
        cx.matrix("mtp.fc", vec![fc], 2 * h)?;
        cx.norm("mtp.norm_e", "mtp.pre_fc_norm_embedding.weight", offset)?;
        cx.norm("mtp.norm_h", "mtp.pre_fc_norm_hidden.weight", offset)?;
        cx.norm("mtp.final_norm", "mtp.norm.weight", offset)?;
        cx.norm("mtp.L0.attn_norm", "mtp.layers.0.input_layernorm.weight", offset)?;
        cx.norm("mtp.L0.mlp_norm", "mtp.layers.0.post_attention_layernorm.weight", offset)?;
        convert_attention(&mut cx, &cfg, "mtp.layers.0.self_attn", "mtp.L0", offset)?;
        convert_mlp(&mut cx, &cfg, "mtp.layers.0.mlp", "mtp.L0")?;
        has_mtp = true;
    }

    // Embed the tokenizer so the model file is self-contained.
    cx.w.add("tokenizer.json", DType::U8, &[tok_json.len()], tok_json.as_bytes())?;

    let mut meta = HashMap::new();
    meta.insert("source".into(), serde_json::json!(hf_dir.file_name().map(|s| s.to_string_lossy().to_string())));
    meta.insert("recipe".into(), serde_json::json!(recipe_name));
    meta.insert("rotated".into(), serde_json::json!(rotate));
    meta.insert("gptq".into(), serde_json::json!(cx.gptq.is_some()));
    meta.insert("has_mtp".into(), serde_json::json!(has_mtp));
    meta.insert("vocab_rows".into(), serde_json::json!(vocab_rows));
    meta.insert("params".into(), serde_json::json!(cx.total_params));
    let (params, bytes) = (cx.total_params, cx.total_bytes);
    cx.w.finish(cfg, meta)?;
    eprintln!(
        "wrote {} : {:.2}B params, {:.2} GB, {:.2} bits/param, {:.1}s",
        out.display(),
        params as f64 / 1e9,
        bytes as f64 / 1e9,
        bytes as f64 * 8.0 / params as f64,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
