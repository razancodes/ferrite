//! Minimal zero-copy safetensors reader (memory mapped).
//!
//! Format: u64 little-endian header length, JSON header mapping tensor name to
//! {dtype, shape, data_offsets:[begin,end]}, then the raw tensor bytes.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SDType {
    F32,
    F16,
    BF16,
}

pub struct TensorRef<'a> {
    pub dtype: SDType,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl TensorRef<'_> {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    /// Convert to f32 (bf16/f16 are widened exactly).
    pub fn to_f32(&self) -> Vec<f32> {
        let n = self.numel();
        let mut out = vec![0f32; n];
        match self.dtype {
            SDType::F32 => {
                for (i, c) in self.data.as_chunks::<4>().0.iter().enumerate() {
                    out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            SDType::BF16 => {
                for (i, c) in self.data.as_chunks::<2>().0.iter().enumerate() {
                    out[i] = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                }
            }
            SDType::F16 => {
                for (i, c) in self.data.as_chunks::<2>().0.iter().enumerate() {
                    out[i] = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                }
            }
        }
        out
    }
}

struct Entry {
    file: usize,
    dtype: SDType,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

/// A (possibly sharded) safetensors checkpoint.
pub struct SafeTensors {
    maps: Vec<Mmap>,
    entries: HashMap<String, Entry>,
}

impl SafeTensors {
    /// Open every `*.safetensors` file inside `dir`.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        let mut maps = Vec::new();
        let mut entries = HashMap::new();
        for (fi, path) in files.iter().enumerate() {
            let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
            let m = unsafe { Mmap::map(&f) }.context("mmap")?;
            if m.len() < 8 {
                bail!("{} too small", path.display());
            }
            let hlen = u64::from_le_bytes(m[0..8].try_into().unwrap()) as usize;
            let header: serde_json::Value =
                serde_json::from_slice(&m[8..8 + hlen]).context("safetensors header json")?;
            let base = 8 + hlen;
            for (name, v) in header.as_object().context("header object")? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = match v["dtype"].as_str().unwrap_or("") {
                    "F32" => SDType::F32,
                    "F16" => SDType::F16,
                    "BF16" => SDType::BF16,
                    other => bail!("unsupported dtype {other} for {name}"),
                };
                let shape: Vec<usize> = v["shape"]
                    .as_array()
                    .context("shape")?
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect();
                let off = v["data_offsets"].as_array().context("offsets")?;
                let begin = base + off[0].as_u64().unwrap() as usize;
                let end = base + off[1].as_u64().unwrap() as usize;
                entries.insert(name.clone(), Entry { file: fi, dtype, shape, begin, end });
            }
            maps.push(m);
        }
        Ok(SafeTensors { maps, entries })
    }

    pub fn get(&self, name: &str) -> Result<TensorRef<'_>> {
        let e = self.entries.get(name).with_context(|| format!("tensor not found: {name}"))?;
        Ok(TensorRef { dtype: e.dtype, shape: e.shape.clone(), data: &self.maps[e.file][e.begin..e.end] })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }
}
