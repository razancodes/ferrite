//! OpenAI-compatible HTTP server (std::net only, no async runtime).
//!
//! Endpoints:
//!   GET  /                       built-in web chat UI
//!   GET  /health
//!   GET  /v1/models
//!   POST /v1/chat/completions    (stream or not; thinking via `enable_thinking`
//!                                 or `chat_template_kwargs.enable_thinking`)
//!   POST /v1/completions         (raw prompt)
//!
//! Requests are served one at a time on a shared engine. Multi-turn chats reuse
//! work: the recurrent state is snapshotted right before each assistant header,
//! so the next turn only prefills the new tokens.

use crate::chat::{ChatTemplate, Message};
use crate::engine::{Engine, StopReason};
use crate::sampler::{Sampler, SamplerConfig};
use crate::stream::TextRouter;
use crate::tokenizer::StreamDecoder;
use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const INDEX_HTML: &str = include_str!("../web/index.html");

pub struct ServerState {
    pub engine: Mutex<Engine>,
    pub model_name: String,
    pub default_max_tokens: usize,
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<Request> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or_else(|| anyhow!("empty request"))?.to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':')
            && k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
    }
    if content_length > 64 << 20 {
        return Err(anyhow!("request too large"));
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(Request { method, path, body })
}

fn respond(stream: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    let status = match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let head = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn json_err(stream: &mut TcpStream, code: u16, msg: &str) {
    let _ = respond(stream, code, "application/json", json!({"error": {"message": msg}}).to_string().as_bytes());
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn sampler_from(body: &Value, thinking: bool) -> SamplerConfig {
    let mut c = if thinking { SamplerConfig::thinking() } else { SamplerConfig::default() };
    let f = |k: &str| body.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
    if let Some(v) = f("temperature") {
        c.temperature = v;
    }
    if let Some(v) = f("top_p") {
        c.top_p = v;
    }
    if let Some(v) = body.get("top_k").and_then(|v| v.as_u64()) {
        c.top_k = v as usize;
    }
    if let Some(v) = f("min_p") {
        c.min_p = v;
    }
    if let Some(v) = f("presence_penalty") {
        c.presence_penalty = v;
    }
    if let Some(v) = f("frequency_penalty") {
        c.frequency_penalty = v;
    }
    if let Some(v) = f("repetition_penalty") {
        c.repetition_penalty = v;
    }
    if let Some(v) = body.get("seed").and_then(|v| v.as_u64()) {
        c.seed = v;
    }
    c
}

fn sse(stream: &mut TcpStream, v: &Value) -> bool {
    stream.write_all(format!("data: {v}\n\n").as_bytes()).and_then(|_| stream.flush()).is_ok()
}

fn handle_completion(state: &ServerState, stream: &mut TcpStream, body: &Value, chat: bool) -> Result<()> {
    let thinking = body
        .get("enable_thinking")
        .or_else(|| body.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let streaming = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(state.default_max_tokens);
    let stops: Vec<String> = match body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => vec![],
    };
    let mut engine = state.engine.lock().unwrap_or_else(|e| e.into_inner());
    let tok = &engine.model.tokenizer;
    let tmpl = ChatTemplate::new(tok);
    let (prompt, snapshot_at) = if chat {
        let msgs: Vec<Message> = body
            .get("messages")
            .and_then(|m| m.as_array())
            .ok_or_else(|| anyhow!("missing messages"))?
            .iter()
            .map(|m| Message {
                role: m.get("role").and_then(|r| r.as_str()).unwrap_or("user").to_string(),
                content: content_text(m.get("content").unwrap_or(&Value::Null)),
            })
            .collect();
        if msgs.is_empty() {
            return Err(anyhow!("messages is empty"));
        }
        let head = tmpl.render(tok, &msgs, false, thinking).len();
        (tmpl.render(tok, &msgs, true, thinking), Some(head))
    } else {
        let p = body.get("prompt").map(content_text).unwrap_or_default();
        (tok.encode(&p, true), None)
    };
    if prompt.is_empty() {
        return Err(anyhow!("empty prompt"));
    }
    if prompt.len() + 1 >= engine.state.max_ctx {
        return Err(anyhow!("prompt ({} tokens) exceeds the context size {}", prompt.len(), engine.state.max_ctx));
    }
    let id = format!("chatcmpl-{:x}", now_secs() ^ (prompt.len() as u64) << 32);
    let created = now_secs();
    let model = state.model_name.clone();
    let object = if chat { "chat.completion.chunk" } else { "text_completion" };
    if streaming {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        stream.write_all(head.as_bytes())?;
        if chat {
            sse(stream, &json!({"id": id, "object": object, "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}));
        }
    }
    let mut sampler = Sampler::new(sampler_from(body, thinking));
    let mut dec = StreamDecoder::default();
    let mut router = TextRouter::new(chat && thinking, stops);
    let stop_ids = tmpl.eos.clone();
    let t0 = Instant::now();
    let mut ttft = None;
    let st = {
        let mut emit = |kind: &str, text: &str| -> bool {
            if !streaming {
                return true;
            }
            let v = if chat {
                json!({"id": id, "object": object, "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {kind: text}, "finish_reason": null}]})
            } else {
                json!({"id": id, "object": object, "created": created, "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": null}]})
            };
            sse(stream, &v)
        };
        let st = engine.generate(&prompt, snapshot_at, &mut sampler, max_tokens, &stop_ids, |eng, t| {
            if ttft.is_none() {
                ttft = Some(t0.elapsed().as_secs_f64());
            }
            let piece = dec.push(&eng.model.tokenizer, t);
            router.push(&piece, &mut emit)
        })?;
        let tail = dec.flush();
        if !tail.is_empty() {
            router.push(&tail, &mut emit);
        }
        router.finish(&mut emit);
        st
    };
    let finish = match st.stop {
        StopReason::Eos => "stop",
        StopReason::Callback if router.hit_stop => "stop",
        StopReason::Callback => "stop",
        StopReason::Length | StopReason::Context => "length",
    };
    let usage = json!({"prompt_tokens": st.prompt_tokens, "completion_tokens": st.gen_tokens,
        "total_tokens": st.prompt_tokens + st.gen_tokens});
    let timings = json!({"prompt_n": st.prompt_tokens - st.cached_tokens, "cached_n": st.cached_tokens,
        "prompt_per_second": st.prefill_tps(), "predicted_n": st.gen_tokens, "predicted_per_second": st.gen_tps(),
        "time_to_first_token_ms": ttft.unwrap_or(0.0) * 1e3, "draft_n": st.drafted, "draft_n_accepted": st.accepted});
    eprintln!(
        "[server] prompt {} ({} cached) {:.1} tok/s | gen {} {:.2} tok/s | {}",
        st.prompt_tokens,
        st.cached_tokens,
        st.prefill_tps(),
        st.gen_tokens,
        st.gen_tps(),
        finish
    );
    if streaming {
        let fin = if chat {
            json!({"id": id, "object": object, "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage, "timings": timings})
        } else {
            json!({"id": id, "object": object, "created": created, "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": finish}], "usage": usage, "timings": timings})
        };
        sse(stream, &fin);
        let _ = stream.write_all(b"data: [DONE]\n\n");
        let _ = stream.flush();
    } else {
        let v = if chat {
            let mut msg = json!({"role": "assistant", "content": router.content});
            if !router.reasoning.is_empty() {
                msg["reasoning_content"] = json!(router.reasoning);
            }
            json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage, "timings": timings})
        } else {
            json!({"id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": router.content, "finish_reason": finish}], "usage": usage, "timings": timings})
        };
        respond(stream, 200, "application/json", v.to_string().as_bytes())?;
    }
    Ok(())
}

fn handle(state: &ServerState, mut stream: TcpStream) {
    let req = match read_request(&mut stream) {
        Ok(r) => r,
        Err(e) => return json_err(&mut stream, 400, &e.to_string()),
    };
    let path = req.path.split('?').next().unwrap_or("/").to_string();
    match (req.method.as_str(), path.as_str()) {
        ("OPTIONS", _) => {
            let _ = respond(&mut stream, 204, "text/plain", b"");
        }
        ("GET", "/") | ("GET", "/index.html") => {
            let _ = respond(&mut stream, 200, "text/html; charset=utf-8", INDEX_HTML.as_bytes());
        }
        ("GET", "/health") => {
            let _ = respond(&mut stream, 200, "application/json", br#"{"status":"ok"}"#);
        }
        ("GET", "/v1/models") => {
            let v = json!({"object": "list", "data": [{"id": state.model_name, "object": "model", "owned_by": "ferrite"}]});
            let _ = respond(&mut stream, 200, "application/json", v.to_string().as_bytes());
        }
        ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {
            let body: Value = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return json_err(&mut stream, 400, &format!("invalid JSON: {e}")),
            };
            if let Err(e) = handle_completion(state, &mut stream, &body, path.ends_with("chat/completions")) {
                json_err(&mut stream, 400, &e.to_string());
            }
        }
        _ => json_err(&mut stream, 404, "not found"),
    }
}

pub fn serve(state: ServerState, host: &str, port: u16) -> Result<()> {
    let listener = TcpListener::bind((host, port))?;
    eprintln!("[server] listening on http://{host}:{port}  (web UI at /, OpenAI API at /v1)");
    let state = Arc::new(state);
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let st = state.clone();
        std::thread::spawn(move || handle(&st, stream));
    }
    Ok(())
}
