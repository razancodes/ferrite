//! Importance matrix ("imatrix") support: per-input-channel activation
//! statistics collected on calibration text. 4-bit quantization uses them to
//! weight the rounding error of each column by how much it actually matters.
//!
//! File format: JSON object { matrix name: [mean of x^2 per input column] }.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub fn load(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&s).context("parse imatrix json")
}

pub fn save(path: &Path, m: &HashMap<String, Vec<f32>>) -> Result<()> {
    std::fs::write(path, serde_json::to_vec(m)?)?;
    Ok(())
}
