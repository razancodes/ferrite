//! Byte-level BPE tokenizer compatible with Hugging Face `tokenizer.json` files
//! used by the Qwen2/Qwen3/Qwen3.5 family.
//!
//! Pipeline (identical to HF `tokenizers`):
//!   1. split out added/special tokens (leftmost-longest),
//!   2. NFC-normalize the remaining text,
//!   3. pre-tokenize with the Qwen regex (re-implemented here as a hand written
//!      matcher, including its backtracking quirks such as `\s+(?!\S)`),
//!   4. map bytes to the GPT-2 printable alphabet and apply BPE merges by rank.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;
use unicode_general_category::{GeneralCategory as GC, get_general_category};
use unicode_normalization::UnicodeNormalization;

pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
}

pub struct Tokenizer {
    /// id -> raw bytes of the token
    id_to_bytes: Vec<Vec<u8>>,
    /// byte-level string -> id
    vocab: HashMap<String, u32>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_to_id: [u32; 256],
    byte_to_char: [char; 256],
    pub added: Vec<AddedToken>,
    is_special: Vec<bool>,
    cache: Mutex<HashMap<String, Vec<u32>>>,
}

fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    table
}

#[inline]
fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    matches!(
        get_general_category(c),
        GC::UppercaseLetter | GC::LowercaseLetter | GC::TitlecaseLetter | GC::ModifierLetter | GC::OtherLetter
    )
}
#[inline]
fn is_mark(c: char) -> bool {
    if c.is_ascii() {
        return false;
    }
    matches!(get_general_category(c), GC::NonspacingMark | GC::SpacingMark | GC::EnclosingMark)
}
#[inline]
fn is_number(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    matches!(get_general_category(c), GC::DecimalNumber | GC::LetterNumber | GC::OtherNumber)
}
#[inline]
fn is_ws(c: char) -> bool {
    c.is_whitespace()
}
#[inline]
fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}
#[inline]
fn is_lm(c: char) -> bool {
    is_letter(c) || is_mark(c)
}

/// Implements
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
/// returning the end index (in chars) of the match starting at `i`.
fn match_at(c: &[char], i: usize) -> usize {
    let n = c.len();
    // 1. contractions (case-insensitive)
    if c[i] == '\'' && i + 1 < n {
        let fold = |x: char| if x == 'ſ' { 's' } else { x.to_ascii_lowercase() };
        let a = fold(c[i + 1]);
        if matches!(a, 's' | 't' | 'm' | 'd') {
            return i + 2;
        }
        if i + 2 < n {
            let b = fold(c[i + 2]);
            if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') {
                return i + 3;
            }
        }
    }
    // 2. optional non-letter/number prefix followed by letters/marks
    {
        let prefix_ok = !is_nl(c[i]) && !is_letter(c[i]) && !is_number(c[i]);
        let start = if prefix_ok && i + 1 < n && is_lm(c[i + 1]) {
            Some(i + 1)
        } else if is_lm(c[i]) {
            Some(i)
        } else {
            None
        };
        if let Some(s) = start {
            let mut e = s;
            while e < n && is_lm(c[e]) {
                e += 1;
            }
            return e;
        }
    }
    // 3. a single number
    if is_number(c[i]) {
        return i + 1;
    }
    // 4. optional space, punctuation/symbol run, trailing newlines
    {
        let p = |x: char| !is_ws(x) && !is_letter(x) && !is_mark(x) && !is_number(x);
        let j = if c[i] == ' ' && i + 1 < n && p(c[i + 1]) { i + 1 } else { i };
        if p(c[j]) {
            let mut e = j;
            while e < n && p(c[e]) {
                e += 1;
            }
            while e < n && is_nl(c[e]) {
                e += 1;
            }
            return e;
        }
    }
    // whitespace alternatives
    if is_ws(c[i]) {
        let mut r = i;
        while r < n && is_ws(c[r]) {
            r += 1;
        }
        // 5. \s*[\r\n]+  -> ends right after the last newline inside the run
        if let Some(p) = (i..r).rev().find(|&k| is_nl(c[k])) {
            return p + 1;
        }
        // 6. \s+(?!\S)
        if r == n {
            return r;
        }
        if r - i >= 2 {
            return r - 1;
        }
        // 7. \s+
        return r;
    }
    i + 1
}

/// Split text into pre-tokens (as &str slices).
pub fn pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<char> = text.chars().collect();
    let mut offs: Vec<usize> = Vec::with_capacity(chars.len() + 1);
    let mut o = 0;
    for ch in &chars {
        offs.push(o);
        o += ch.len_utf8();
    }
    offs.push(o);
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let e = match_at(&chars, i).max(i + 1);
        out.push(&text[offs[i]..offs[e]]);
        i = e;
    }
    out
}

impl Tokenizer {
    pub fn from_json_str(s: &str) -> Result<Tokenizer> {
        let v: Value = serde_json::from_str(s).context("parse tokenizer.json")?;
        let model = &v["model"];
        let byte_to_char = bytes_to_unicode();
        let mut char_to_byte = HashMap::new();
        for (b, ch) in byte_to_char.iter().enumerate() {
            char_to_byte.insert(*ch, b as u8);
        }
        let vocab_obj = model["vocab"].as_object().context("vocab")?;
        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0u32;
        for (k, id) in vocab_obj {
            let id = id.as_u64().unwrap() as u32;
            vocab.insert(k.clone(), id);
            max_id = max_id.max(id);
        }
        let mut added = Vec::new();
        for a in v["added_tokens"].as_array().context("added_tokens")? {
            let id = a["id"].as_u64().unwrap() as u32;
            max_id = max_id.max(id);
            added.push(AddedToken {
                id,
                content: a["content"].as_str().unwrap().to_string(),
                special: a["special"].as_bool().unwrap_or(false),
            });
        }
        let mut id_to_bytes = vec![Vec::new(); max_id as usize + 1];
        for (k, &id) in &vocab {
            id_to_bytes[id as usize] = k.chars().map(|c| *char_to_byte.get(&c).unwrap_or(&b'?')).collect();
        }
        let mut is_special = vec![false; max_id as usize + 1];
        for a in &added {
            id_to_bytes[a.id as usize] = a.content.as_bytes().to_vec();
            is_special[a.id as usize] = a.special;
        }
        let mut byte_to_id = [0u32; 256];
        for b in 0..256 {
            byte_to_id[b] = *vocab.get(&byte_to_char[b].to_string()).context("byte token missing from vocab")?;
        }
        let mut merges = HashMap::new();
        for (rank, m) in model["merges"].as_array().context("merges")?.iter().enumerate() {
            let (a, b) = if let Some(s) = m.as_str() {
                let mut it = s.splitn(2, ' ');
                (it.next().unwrap().to_string(), it.next().unwrap_or("").to_string())
            } else {
                let arr = m.as_array().unwrap();
                (arr[0].as_str().unwrap().to_string(), arr[1].as_str().unwrap().to_string())
            };
            let (Some(&ia), Some(&ib)) = (vocab.get(&a), vocab.get(&b)) else { continue };
            let Some(&im) = vocab.get(&format!("{a}{b}")) else { continue };
            merges.entry((ia, ib)).or_insert((rank as u32, im));
        }
        // longest content first so that matching is leftmost-longest
        added.sort_by(|a, b| b.content.len().cmp(&a.content.len()));
        Ok(Tokenizer {
            id_to_bytes,
            vocab,
            merges,
            byte_to_id,
            byte_to_char,
            added,
            is_special,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.added.iter().find(|a| a.content == s).map(|a| a.id).or_else(|| {
            let bl: String = s.bytes().map(|b| self.byte_to_char[b as usize]).collect();
            self.vocab.get(&bl).copied()
        })
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    fn bpe_word(&self, word: &str, out: &mut Vec<u32>) {
        if let Some(hit) = self.cache.lock().unwrap().get(word) {
            out.extend_from_slice(hit);
            return;
        }
        let mut syms: Vec<u32> = word.bytes().map(|b| self.byte_to_id[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(syms[i], syms[i + 1]))
                    && best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, i, merged));
                    }
            }
            let Some((rank, _, _)) = best else { break };
            // merge every occurrence of this pair, left to right
            let mut j = 0;
            let mut merged_syms = Vec::with_capacity(syms.len());
            while j < syms.len() {
                if j + 1 < syms.len()
                    && let Some(&(r, m)) = self.merges.get(&(syms[j], syms[j + 1]))
                        && r == rank {
                            merged_syms.push(m);
                            j += 2;
                            continue;
                        }
                merged_syms.push(syms[j]);
                j += 1;
            }
            syms = merged_syms;
        }
        if word.len() < 64 {
            let mut c = self.cache.lock().unwrap();
            if c.len() > 200_000 {
                c.clear();
            }
            c.insert(word.to_string(), syms.clone());
        }
        out.extend_from_slice(&syms);
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let norm: String = text.nfc().collect();
        for piece in pretokenize(&norm) {
            self.bpe_word(piece, out);
        }
    }

    /// Encode text. If `parse_special` is false, special tokens such as
    /// `<|im_end|>` appearing in the text are treated as plain text (safer for
    /// user input); non-special added tokens like `<think>` are always matched,
    /// mirroring HF `split_special_tokens=True`.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let mut seg_start = 0;
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'<' {
                let rest = &text[i..];
                if let Some(a) = self.added.iter().find(|a| (parse_special || !a.special) && rest.starts_with(&a.content)) {
                    self.encode_plain(&text[seg_start..i], &mut out);
                    out.push(a.id);
                    i += a.content.len();
                    seg_start = i;
                    continue;
                }
            }
            i += 1;
        }
        self.encode_plain(&text[seg_start..], &mut out);
        out
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.id_to_bytes.get(id as usize).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut b = Vec::new();
        for &id in ids {
            b.extend_from_slice(self.token_bytes(id));
        }
        String::from_utf8_lossy(&b).into_owned()
    }
}

/// Incremental detokenizer: emits only complete UTF-8 sequences, buffering
/// partial multi-byte characters split across tokens.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        self.pending.extend_from_slice(tok.token_bytes(id));
        self.drain(false)
    }
    pub fn flush(&mut self) -> String {
        self.drain(true)
    }
    fn drain(&mut self, all: bool) -> String {
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // An invalid (not merely incomplete) sequence: emit lossily.
                if e.error_len().is_some() || all || self.pending.len() - valid > 4 {
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return s;
                }
                let s = std::str::from_utf8(&self.pending[..valid]).unwrap().to_string();
                self.pending.drain(..valid);
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pretokenizer_matches_regex_semantics() {
        assert_eq!(pretokenize("Hello world"), vec!["Hello", " world"]);
        assert_eq!(pretokenize("it's 2024!"), vec!["it", "'s", " ", "2", "0", "2", "4", "!"]);
        assert_eq!(pretokenize("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pretokenize("x\n\n  y"), vec!["x", "\n\n", " ", " y"]);
        assert_eq!(pretokenize("end   "), vec!["end", "   "]);
        assert_eq!(pretokenize(" ...\n"), vec![" ...\n"]);
    }
}
