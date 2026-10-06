//! ferrite: a from-scratch CPU inference engine for the Qwen3.5 / Qwen3.8
//! hybrid (Gated DeltaNet + gated attention) architecture and dense Qwen3.

// SIMD kernels index register arrays in loops and take many raw-pointer
// arguments on purpose.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod attn;
pub mod chat;
pub mod config;
pub mod convert;
pub mod engine;
pub mod format;
pub mod gptq;
pub mod imatrix;
pub mod kernels;
pub mod model;
pub mod ops;
pub mod profile;
pub mod quant;
pub mod safetensors;
pub mod sampler;
pub mod server;
pub mod stream;
pub mod threadpool;
pub mod tokenizer;
