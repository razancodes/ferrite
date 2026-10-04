//! The `.ferrite` single-file model format.
//!
//! ```text
//! [8B magic "FERRITE1"][tensor data, each tensor 64-byte aligned ...]
//! [JSON header][u64 header offset][u64 header length][8B magic "FERRITE1"]
//! ```
//! The header is written last so tensors can be streamed to disk during
//! conversion (important for models larger than RAM). Loading memory-maps the
//! file and points directly into it: no parsing or copying of weights.

use crate::config::Config;
use crate::quant::DType;
use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"FERRITE1";
const ALIGN: u64 = 64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    /// [rows, cols] for matrices, [n] for vectors
    pub shape: Vec<usize>,
    pub offset: u64,
    pub nbytes: u64,
    /// Hadamard rotation block applied to the input dimension (0 = none)
    #[serde(default)]
    pub rot: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Header {
    pub config: Config,
    pub tensors: Vec<TensorInfo>,
    /// arbitrary metadata (source repo, recipe, tokenizer special ids, ...)
    pub meta: HashMap<String, serde_json::Value>,
}

pub struct Writer {
    f: BufWriter<File>,
    pos: u64,
    tensors: Vec<TensorInfo>,
}

impl Writer {
    pub fn create(path: &Path) -> Result<Writer> {
        let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
        let mut w = Writer { f: BufWriter::with_capacity(1 << 22, f), pos: 0, tensors: Vec::new() };
        w.f.write_all(MAGIC)?;
        w.pos = MAGIC.len() as u64;
        Ok(w)
    }

    fn pad(&mut self) -> Result<()> {
        let rem = self.pos % ALIGN;
        if rem != 0 {
            let n = (ALIGN - rem) as usize;
            self.f.write_all(&vec![0u8; n])?;
            self.pos += n as u64;
        }
        Ok(())
    }

    pub fn add(&mut self, name: &str, dtype: DType, shape: &[usize], data: &[u8]) -> Result<()> {
        self.pad()?;
        self.tensors.push(TensorInfo {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            offset: self.pos,
            nbytes: data.len() as u64,
            rot: 0,
        });
        self.f.write_all(data)?;
        self.pos += data.len() as u64;
        Ok(())
    }

    /// Mark the most recently added tensor as Hadamard-rotated with block `b`.
    pub fn set_rot(&mut self, b: usize) {
        if let Some(t) = self.tensors.last_mut() {
            t.rot = b;
        }
    }

    pub fn add_f32(&mut self, name: &str, data: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add(name, DType::F32, &[data.len()], &bytes)
    }

    pub fn finish(mut self, config: Config, meta: HashMap<String, serde_json::Value>) -> Result<()> {
        self.pad()?;
        let header = Header { config, tensors: self.tensors, meta };
        let js = serde_json::to_vec(&header)?;
        let off = self.pos;
        self.f.write_all(&js)?;
        self.f.write_all(&off.to_le_bytes())?;
        self.f.write_all(&(js.len() as u64).to_le_bytes())?;
        self.f.write_all(MAGIC)?;
        self.f.flush()?;
        let _ = self.f.get_mut().stream_position();
        Ok(())
    }
}

/// Append (or replace) a raw tensor in an existing model file without
/// rewriting the weights: the new bytes go where the old header was, followed
/// by a new header and footer. The file must not be memory-mapped elsewhere.
pub fn append_tensor(path: &Path, name: &str, dtype: DType, shape: &[usize], data: &[u8]) -> Result<()> {
    use std::io::{Read, SeekFrom};
    let mut f = std::fs::OpenOptions::new().read(true).write(true).open(path).with_context(|| format!("open {}", path.display()))?;
    let n = f.metadata()?.len();
    let mut footer = [0u8; 24];
    f.seek(SeekFrom::Start(n - 24))?;
    f.read_exact(&mut footer)?;
    if &footer[16..] != MAGIC {
        bail!("{} is not a ferrite model file", path.display());
    }
    let off = u64::from_le_bytes(footer[0..8].try_into().unwrap());
    let len = u64::from_le_bytes(footer[8..16].try_into().unwrap()) as usize;
    let mut hj = vec![0u8; len];
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(&mut hj)?;
    let mut header: Header = serde_json::from_slice(&hj)?;
    header.tensors.retain(|t| t.name != name);
    let pos = off.div_ceil(ALIGN) * ALIGN;
    f.seek(SeekFrom::Start(off))?;
    f.write_all(&vec![0u8; (pos - off) as usize])?;
    f.write_all(data)?;
    header.tensors.push(TensorInfo { name: name.into(), dtype, shape: shape.to_vec(), offset: pos, nbytes: data.len() as u64, rot: 0 });
    let hpos = (pos + data.len() as u64).div_ceil(ALIGN) * ALIGN;
    f.write_all(&vec![0u8; (hpos - pos - data.len() as u64) as usize])?;
    let js = serde_json::to_vec(&header)?;
    f.write_all(&js)?;
    f.write_all(&hpos.to_le_bytes())?;
    f.write_all(&(js.len() as u64).to_le_bytes())?;
    f.write_all(MAGIC)?;
    let end = f.stream_position()?;
    f.set_len(end)?;
    Ok(())
}

pub struct ModelFile {
    pub mmap: Mmap,
    pub header: Header,
    index: HashMap<String, usize>,
}

impl ModelFile {
    pub fn open(path: &Path) -> Result<ModelFile> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f) }.context("mmap model")?;
        let n = mmap.len();
        if n < 32 || &mmap[0..8] != MAGIC || &mmap[n - 8..] != MAGIC {
            bail!("{} is not a ferrite model file", path.display());
        }
        let off = u64::from_le_bytes(mmap[n - 24..n - 16].try_into().unwrap()) as usize;
        let len = u64::from_le_bytes(mmap[n - 16..n - 8].try_into().unwrap()) as usize;
        let header: Header = serde_json::from_slice(&mmap[off..off + len]).context("parse header")?;
        let index = header.tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(ModelFile { mmap, header, index })
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.header.tensors[i])
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.info(name).with_context(|| format!("missing tensor {name}"))?;
        Ok(&self.mmap[t.offset as usize..(t.offset + t.nbytes) as usize])
    }
}
