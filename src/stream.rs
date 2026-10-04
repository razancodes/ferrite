//! Routing of streamed model output: splits `<think>...</think>` reasoning from
//! the answer and holds back text that might be the start of a stop string, so
//! clients (HTTP server, GUI) never display a partial stop sequence or tag.

/// Splits generated text into reasoning (`<think>...</think>`) and answer,
/// holds back text that might be the start of a stop string, and streams the
/// safe part through `emit(kind, text)`.
pub struct TextRouter {
    text: String,
    emitted: usize,
    in_think: bool,
    stops: Vec<String>,
    pub hit_stop: bool,
    pub reasoning: String,
    pub content: String,
}

impl TextRouter {
    pub fn new(in_think: bool, stops: Vec<String>) -> Self {
        TextRouter { text: String::new(), emitted: 0, in_think, stops, hit_stop: false, reasoning: String::new(), content: String::new() }
    }

    /// Feed newly decoded text; `emit(kind, text)` receives the safe-to-show part
    /// (`kind` is "reasoning_content" or "content"). Returns false to stop.
    pub fn push(&mut self, piece: &str, emit: &mut dyn FnMut(&str, &str) -> bool) -> bool {
        self.text.push_str(piece);
        let mut limit = self.text.len();
        for s in &self.stops {
            let mut from = self.emitted.saturating_sub(s.len());
            while !self.text.is_char_boundary(from) {
                from -= 1;
            }
            if let Some(i) = self.text[from..].find(s.as_str()) {
                let at = from + i;
                limit = limit.min(at);
                self.hit_stop = true;
            }
        }
        let hold = if self.hit_stop {
            0
        } else {
            // longest suffix that is a prefix of a stop string or of a think tag
            let mut hold = 0;
            let tags = ["<think>", "</think>"];
            for s in self.stops.iter().map(|s| s.as_str()).chain(tags) {
                for l in (1..s.len()).rev() {
                    if self.text.len() >= l && self.text.is_char_boundary(self.text.len() - l) && s.starts_with(&self.text[self.text.len() - l..]) {
                        hold = hold.max(l);
                        break;
                    }
                }
            }
            hold
        };
        let mut end = limit.saturating_sub(hold).max(self.emitted);
        while !self.text.is_char_boundary(end) {
            end -= 1;
        }
        let ok = self.route(end, emit);
        ok && !self.hit_stop
    }

    fn route(&mut self, end: usize, emit: &mut dyn FnMut(&str, &str) -> bool) -> bool {
        let mut ok = true;
        while self.emitted < end {
            let rest = &self.text[self.emitted..end];
            let tag = if self.in_think { "</think>" } else { "<think>" };
            match rest.find(tag) {
                Some(i) => {
                    let chunk = rest[..i].to_string();
                    ok &= self.out(&chunk, emit);
                    self.emitted += i + tag.len();
                    self.in_think = !self.in_think;
                    // drop the newline(s) the template places after the tags
                    while self.emitted < end && self.text.as_bytes()[self.emitted] == b'\n' {
                        self.emitted += 1;
                    }
                }
                None => {
                    let chunk = rest.to_string();
                    ok &= self.out(&chunk, emit);
                    self.emitted = end;
                }
            }
        }
        ok
    }

    fn out(&mut self, s: &str, emit: &mut dyn FnMut(&str, &str) -> bool) -> bool {
        if s.is_empty() {
            return true;
        }
        if self.in_think {
            self.reasoning.push_str(s);
            emit("reasoning_content", s)
        } else {
            self.content.push_str(s);
            emit("content", s)
        }
    }

    /// Flush everything still held back (call once generation has ended).
    pub fn finish(&mut self, emit: &mut dyn FnMut(&str, &str) -> bool) {
        if !self.hit_stop {
            let end = self.text.len();
            self.route(end, emit);
        }
    }
}

