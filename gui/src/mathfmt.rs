//! Display-only cleanup of the LaTeX math that models like to emit
//! (`$13 \times 17$`), since the markdown renderer does not typeset math.
//! Code spans and fenced code blocks are left untouched.

const SYMBOLS: &[(&str, &str)] = &[
    ("\\times", "×"),
    ("\\cdot", "·"),
    ("\\div", "÷"),
    ("\\pm", "±"),
    ("\\leq", "≤"),
    ("\\le", "≤"),
    ("\\geq", "≥"),
    ("\\ge", "≥"),
    ("\\neq", "≠"),
    ("\\ne", "≠"),
    ("\\approx", "≈"),
    ("\\infty", "∞"),
    ("\\rightarrow", "→"),
    ("\\Rightarrow", "⇒"),
    ("\\to", "→"),
    ("\\in", "∈"),
    ("\\sum", "∑"),
    ("\\prod", "∏"),
    ("\\pi", "π"),
    ("\\theta", "θ"),
    ("\\alpha", "α"),
    ("\\beta", "β"),
    ("\\lambda", "λ"),
    ("\\mu", "μ"),
    ("\\sigma", "σ"),
    ("\\Delta", "Δ"),
    ("\\ldots", "…"),
    ("\\dots", "…"),
    ("\\%", "%"),
    ("\\,", " "),
    ("\\;", " "),
    ("\\quad", "  "),
    ("\\left", ""),
    ("\\right", ""),
];

/// Parse one balanced `{...}` group at the start of `t`: (contents, bytes used).
fn group(t: &str) -> Option<(&str, usize)> {
    if !t.starts_with('{') {
        return None;
    }
    let mut depth = 0;
    for (i, c) in t.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&t[1..i], i + 1));
                }
            }
            _ => {}
        }
    }
    None
}

/// Rewrite the LaTeX command at the start of `tail` (which begins with a backslash).
fn rewrite_one(tail: &str) -> Option<(String, usize)> {
    if let Some(t) = tail.strip_prefix("\\frac") {
        let (a, n1) = group(t)?;
        let (b, n2) = group(&t[n1..])?;
        return Some((format!("({})/({})", rewrite_commands(a), rewrite_commands(b)), 5 + n1 + n2));
    }
    if let Some(t) = tail.strip_prefix("\\sqrt") {
        let (a, n) = group(t)?;
        return Some((format!("√({})", rewrite_commands(a)), 5 + n));
    }
    for cmd in ["\\text", "\\mathrm", "\\mathbf", "\\textbf"] {
        if let Some(t) = tail.strip_prefix(cmd) {
            if let Some((a, n)) = group(t) {
                return Some((a.to_string(), cmd.len() + n));
            }
        }
    }
    for (k, v) in SYMBOLS {
        if let Some(after) = tail.strip_prefix(k) {
            // `\in` must not match the start of `\int`
            let alpha = k[1..].starts_with(|c: char| c.is_ascii_alphabetic());
            if !alpha || !after.starts_with(|c: char| c.is_ascii_alphabetic()) {
                return Some((v.to_string(), k.len()));
            }
        }
    }
    None
}

/// Rewrite the LaTeX commands in a math span.
fn rewrite_commands(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('\\') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        match rewrite_one(tail) {
            Some((text, used)) => {
                out.push_str(&text);
                rest = &tail[used..];
            }
            None => {
                out.push('\\');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn prettify_math(seg: &str) -> String {
    // strip $$...$$ and $...$ delimiters, rewriting the math inside
    let mut out = String::with_capacity(seg.len());
    let mut rest = seg;
    while let Some(i) = rest.find('$') {
        let display = rest[i..].starts_with("$$");
        let open = if display { 2 } else { 1 };
        let body = &rest[i + open..];
        let close = if display { body.find("$$") } else { body.find('$') };
        match close {
            // inline math must not span lines or start with a space (likely a price)
            // (pandoc rules: no space just inside the delimiters, no digit right after)
            Some(j)
                if display
                    || (j > 0
                        && !body[..j].contains('\n')
                        && !body.starts_with(' ')
                        && !body[..j].ends_with(' ')
                        && !body[j + 1..].starts_with(|c: char| c.is_ascii_digit())) =>
            {
                out.push_str(&rest[..i]);
                out.push_str(&rewrite_commands(body[..j].trim()));
                rest = &body[j + open..];
            }
            _ => {
                out.push_str(&rest[..i + open]);
                rest = body;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Prettify math outside of code spans / fenced blocks.
pub fn display_text(md: &str) -> String {
    if !md.contains('$') && !md.contains('\\') {
        return md.to_string();
    }
    let mut out = String::with_capacity(md.len());
    for (i, block) in md.split("```").enumerate() {
        if i > 0 {
            out.push_str("```");
        }
        if i % 2 == 1 {
            out.push_str(block); // fenced code
            continue;
        }
        for (j, span) in block.split('`').enumerate() {
            if j > 0 {
                out.push('`');
            }
            if j % 2 == 1 {
                out.push_str(span);
            } else {
                out.push_str(&prettify_math(span));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::display_text;
    #[test]
    fn math_is_prettified_outside_code() {
        assert_eq!(display_text("It is $13 \\times 17$."), "It is 13 × 17.");
        assert_eq!(display_text("$$\\frac{a}{b} \\le \\sqrt{2}$$"), "(a)/(b) ≤ √(2)");
        assert_eq!(display_text("costs $5 and $6"), "costs $5 and $6");
        assert_eq!(display_text("`$x \\times y$`"), "`$x \\times y$`");
        assert_eq!(display_text("```\n$a \\times b$\n```"), "```\n$a \\times b$\n```");
        assert_eq!(display_text("$\\text{area} = \\pi r^2$"), "area = π r^2");
        assert_eq!(display_text("$\\int_0^1 x \\in A$"), "\\int_0^1 x ∈ A");
    }
}
