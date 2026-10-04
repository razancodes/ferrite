//! Qwen3.5 chat template (a faithful re-implementation of the Jinja template
//! shipped with the model for text-only conversations).

use crate::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Message { role: role.into(), content: content.into() }
    }
}

enum Seg {
    Special(u32),
    Text(String),
}

pub struct ChatTemplate {
    pub im_start: u32,
    pub im_end: u32,
    pub eos: Vec<u32>,
}

impl ChatTemplate {
    pub fn new(tok: &Tokenizer) -> Self {
        let im_start = tok.token_id("<|im_start|>").unwrap_or(0);
        let im_end = tok.token_id("<|im_end|>").unwrap_or(0);
        let mut eos = vec![im_end];
        if let Some(e) = tok.token_id("<|endoftext|>") {
            eos.push(e);
        }
        ChatTemplate { im_start, im_end, eos }
    }

    /// Split an assistant message into (reasoning, answer) like the template does.
    fn split_reasoning(content: &str) -> (String, String) {
        if let Some(idx) = content.find("</think>") {
            let before = &content[..idx];
            let reasoning = before.rsplit("<think>").next().unwrap_or("").trim_end_matches('\n');
            let reasoning = reasoning.trim_start_matches('\n');
            let answer = content[idx + "</think>".len()..].trim_start_matches('\n');
            (reasoning.to_string(), answer.to_string())
        } else {
            (String::new(), content.to_string())
        }
    }

    pub fn render(&self, tok: &Tokenizer, msgs: &[Message], add_generation_prompt: bool, thinking: bool) -> Vec<u32> {
        let mut segs: Vec<Seg> = Vec::new();
        let text = |segs: &mut Vec<Seg>, s: &str| {
            if let Some(Seg::Text(t)) = segs.last_mut() {
                t.push_str(s);
            } else {
                segs.push(Seg::Text(s.to_string()));
            }
        };
        let mut start = 0;
        if let Some(m) = msgs.first()
            && m.role == "system" {
                segs.push(Seg::Special(self.im_start));
                text(&mut segs, &format!("system\n{}", m.content.trim()));
                segs.push(Seg::Special(self.im_end));
                text(&mut segs, "\n");
                start = 1;
            }
        let last_query = msgs.iter().rposition(|m| m.role == "user").unwrap_or(0);
        for (i, m) in msgs.iter().enumerate().skip(start) {
            let content = m.content.trim();
            match m.role.as_str() {
                "assistant" => {
                    let (reasoning, answer) = Self::split_reasoning(content);
                    segs.push(Seg::Special(self.im_start));
                    if i > last_query {
                        text(&mut segs, &format!("assistant\n<think>\n{}\n</think>\n\n{}", reasoning.trim(), answer));
                    } else {
                        text(&mut segs, &format!("assistant\n{answer}"));
                    }
                    segs.push(Seg::Special(self.im_end));
                    text(&mut segs, "\n");
                }
                "tool" => {
                    let prev_tool = i > 0 && msgs[i - 1].role == "tool";
                    if !prev_tool {
                        segs.push(Seg::Special(self.im_start));
                        text(&mut segs, "user");
                    }
                    text(&mut segs, &format!("\n<tool_response>\n{content}\n</tool_response>"));
                    let next_tool = msgs.get(i + 1).map(|n| n.role == "tool").unwrap_or(false);
                    if !next_tool {
                        segs.push(Seg::Special(self.im_end));
                        text(&mut segs, "\n");
                    }
                }
                role => {
                    segs.push(Seg::Special(self.im_start));
                    text(&mut segs, &format!("{role}\n{content}"));
                    segs.push(Seg::Special(self.im_end));
                    text(&mut segs, "\n");
                }
            }
        }
        if add_generation_prompt {
            segs.push(Seg::Special(self.im_start));
            text(&mut segs, if thinking { "assistant\n<think>\n" } else { "assistant\n<think>\n\n</think>\n\n" });
        }
        let mut ids = Vec::new();
        for s in segs {
            match s {
                Seg::Special(id) => ids.push(id),
                Seg::Text(t) => ids.extend(tok.encode(&t, false)),
            }
        }
        ids
    }

    /// Tokens that continue an existing conversation whose state already ends
    /// with an assistant reply: close the reply, add a user turn, open a new reply.
    pub fn continuation(&self, tok: &Tokenizer, user: &str, thinking: bool) -> Vec<u32> {
        let mut ids = vec![self.im_end];
        ids.extend(tok.encode("\n", false));
        ids.push(self.im_start);
        ids.extend(tok.encode(&format!("user\n{}", user.trim()), false));
        ids.push(self.im_end);
        ids.extend(tok.encode("\n", false));
        ids.push(self.im_start);
        ids.extend(tok.encode(if thinking { "assistant\n<think>\n" } else { "assistant\n<think>\n\n</think>\n\n" }, false));
        ids
    }
}
