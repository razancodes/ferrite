//! Model hyper-parameters, parsed from a Hugging Face `config.json` and stored
//! verbatim in the ferrite file header.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    /// Gated DeltaNet linear attention (recurrent state, no KV cache).
    Linear,
    /// Softmax attention with GQA, QK-norm, partial RoPE and an optional output gate.
    Full,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    /// "qwen3_5" (hybrid DeltaNet + gated attention; also Qwen3.6/3.8) or "qwen3" (dense).
    pub arch: String,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_layers: usize,
    pub layer_kinds: Vec<LayerKind>,
    // softmax attention
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub attn_output_gate: bool,
    // gated delta net
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    // misc
    pub rms_eps: f32,
    pub vocab: usize,
    pub tie_embeddings: bool,
    pub mtp_layers: usize,
    pub max_position: usize,
}

impl Config {
    pub fn from_hf(v: &Value) -> Result<Config> {
        // Multimodal checkpoints nest the language model config.
        let t = if v.get("text_config").is_some() { &v["text_config"] } else { v };
        let mt = t["model_type"].as_str().or(v["model_type"].as_str()).unwrap_or("");
        let gu = |k: &str| -> Result<usize> {
            t[k].as_u64().map(|x| x as usize).with_context(|| format!("config missing {k}"))
        };
        let hidden = gu("hidden_size")?;
        let n_layers = gu("num_hidden_layers")?;
        let n_heads = gu("num_attention_heads")?;
        let head_dim = t["head_dim"].as_u64().map(|x| x as usize).unwrap_or(hidden / n_heads);
        let tie = t["tie_word_embeddings"].as_bool().or(v["tie_word_embeddings"].as_bool()).unwrap_or(false);
        let rms_eps = t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;

        if mt.starts_with("qwen3_5") || mt.starts_with("qwen3_6") || mt.starts_with("qwen3_8") || mt == "qwen3_next" {
            let rp = &t["rope_parameters"];
            let rope_theta = rp["rope_theta"].as_f64().or(t["rope_theta"].as_f64()).unwrap_or(1e7) as f32;
            let prf = rp["partial_rotary_factor"].as_f64().or(t["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
            let layer_kinds: Vec<LayerKind> = match t["layer_types"].as_array() {
                Some(a) => a
                    .iter()
                    .map(|x| match x.as_str() {
                        Some("linear_attention") => Ok(LayerKind::Linear),
                        Some("full_attention") => Ok(LayerKind::Full),
                        o => bail!("unknown layer type {o:?}"),
                    })
                    .collect::<Result<_>>()?,
                None => {
                    let every = gu("full_attention_interval")?;
                    (0..n_layers)
                        .map(|i| if (i + 1) % every == 0 { LayerKind::Full } else { LayerKind::Linear })
                        .collect()
                }
            };
            if t["num_experts"].as_u64().unwrap_or(0) > 0 {
                bail!("mixture-of-experts checkpoints are not supported yet");
            }
            Ok(Config {
                arch: "qwen3_5".into(),
                hidden,
                intermediate: gu("intermediate_size")?,
                n_layers,
                layer_kinds,
                n_heads,
                n_kv_heads: gu("num_key_value_heads")?,
                head_dim,
                rotary_dim: (head_dim as f64 * prf) as usize,
                rope_theta,
                attn_output_gate: t["attn_output_gate"].as_bool().unwrap_or(true),
                lin_k_heads: gu("linear_num_key_heads")?,
                lin_v_heads: gu("linear_num_value_heads")?,
                lin_k_dim: gu("linear_key_head_dim")?,
                lin_v_dim: gu("linear_value_head_dim")?,
                conv_kernel: gu("linear_conv_kernel_dim")?,
                rms_eps,
                vocab: gu("vocab_size")?,
                tie_embeddings: tie,
                mtp_layers: t["mtp_num_hidden_layers"].as_u64().unwrap_or(0) as usize,
                max_position: gu("max_position_embeddings").unwrap_or(262144),
            })
        } else if mt == "qwen3" {
            Ok(Config {
                arch: "qwen3".into(),
                hidden,
                intermediate: gu("intermediate_size")?,
                n_layers,
                layer_kinds: vec![LayerKind::Full; n_layers],
                n_heads,
                n_kv_heads: gu("num_key_value_heads")?,
                head_dim,
                rotary_dim: head_dim,
                rope_theta: t["rope_theta"].as_f64().unwrap_or(1e6) as f32,
                attn_output_gate: false,
                lin_k_heads: 0,
                lin_v_heads: 0,
                lin_k_dim: 0,
                lin_v_dim: 0,
                conv_kernel: 0,
                rms_eps,
                vocab: gu("vocab_size")?,
                tie_embeddings: tie,
                mtp_layers: 0,
                max_position: gu("max_position_embeddings").unwrap_or(40960),
            })
        } else {
            bail!("unsupported model_type '{mt}' (supported: qwen3_5 family incl. Qwen3.5/3.6/3.8, qwen3)")
        }
    }

    pub fn n_linear(&self) -> usize {
        self.layer_kinds.iter().filter(|k| **k == LayerKind::Linear).count()
    }
    pub fn n_full(&self) -> usize {
        self.layer_kinds.iter().filter(|k| **k == LayerKind::Full).count()
    }
    pub fn lin_key_dim(&self) -> usize {
        self.lin_k_heads * self.lin_k_dim
    }
    pub fn lin_value_dim(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }
    /// Channels that go through the short causal convolution (q, k and v).
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_key_dim() + self.lin_value_dim()
    }
    /// Rows of the fused full-attention input projection: q (+gate), k, v.
    pub fn qkv_rows(&self) -> usize {
        let q = self.n_heads * self.head_dim * if self.attn_output_gate { 2 } else { 1 };
        q + 2 * self.n_kv_heads * self.head_dim
    }
}
