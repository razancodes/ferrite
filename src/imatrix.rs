//! Importance matrix ("imatrix") support: per-input-column activation
//! statistics collected on calibration text. 4-bit quantization uses them to
//! weight the rounding error of each column by how much it actually matters.
//!
//! File format: JSON object { matrix name: [mean of x^2 per input column] },
//! plus `"__rotated": [1]` when collected from a Hadamard-rotated model (the
//! statistics then describe rotated activations).

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

const ROT_KEY: &str = "__rotated";

/// Returns the statistics and whether they were collected in rotated space.
pub fn load(path: &Path) -> Result<(HashMap<String, Vec<f32>>, bool)> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut m: HashMap<String, Vec<f32>> = serde_json::from_str(&s).context("parse imatrix json")?;
    let rotated = m.remove(ROT_KEY).map(|v| v.first().copied().unwrap_or(0.0) > 0.0).unwrap_or(false);
    Ok((m, rotated))
}

pub fn save(path: &Path, m: &HashMap<String, Vec<f32>>, rotated: bool) -> Result<()> {
    let mut m = m.clone();
    m.insert(ROT_KEY.to_string(), vec![if rotated { 1.0 } else { 0.0 }]);
    std::fs::write(path, serde_json::to_vec(&m)?)?;
    Ok(())
}
