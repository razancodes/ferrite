// Markdown → HTML for model output. Model text is untrusted: raw HTML is escaped,
// links are restricted to http(s) and opened externally, math renders with KaTeX
// (as marked extensions, so it never fires inside code), code gets highlight.js.
(function () {
  "use strict";

  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  function tex(src, display) {
    try {
      return katex.renderToString(src, { displayMode: display, throwOnError: false, output: "html", strict: "ignore", trust: false });
    } catch (_) {
      return `<code>${esc(src)}</code>`;
    }
  }

  const mathBlock = {
    name: "mathBlock",
    level: "block",
    start(src) {
      const m = src.match(/\$\$|\\\[/);
      return m ? m.index : undefined;
    },
    tokenizer(src) {
      const m = /^\$\$([\s\S]+?)\$\$/.exec(src) || /^\\\[([\s\S]+?)\\\]/.exec(src);
      if (m) return { type: "mathBlock", raw: m[0], text: m[1].trim() };
    },
    renderer(tok) {
      return `<div class="math">${tex(tok.text, true)}</div>`;
    },
  };

  const mathInline = {
    name: "mathInline",
    level: "inline",
    start(src) {
      const m = src.match(/\$|\\\(/);
      return m ? m.index : undefined;
    },
    tokenizer(src) {
      // $$...$$ inline, \( \), and pandoc-style $...$ (no space inside the delimiters,
      // no digit right after the closing $ — so "$5 and $6" stays plain text)
      let m = /^\$\$([^$]+?)\$\$/.exec(src);
      if (m) return { type: "mathInline", raw: m[0], text: m[1], display: true };
      m = /^\\\(([\s\S]+?)\\\)/.exec(src);
      if (m) return { type: "mathInline", raw: m[0], text: m[1], display: false };
      m = /^\$(?!\s)((?:\\\$|[^$\n])+?)(?<!\s)\$(?!\d)/.exec(src);
      if (m) return { type: "mathInline", raw: m[0], text: m[1], display: false };
    },
    renderer(tok) {
      return tex(tok.text, tok.display);
    },
  };

  const md = new marked.Marked({ gfm: true, breaks: false });
  md.use({
    extensions: [mathBlock, mathInline],
    renderer: {
      html(tok) {
        return esc(tok.text || tok.raw || "");
      },
      link(tok) {
        const text = this.parser.parseInline(tok.tokens);
        const href = String(tok.href || "");
        if (!/^https?:\/\//i.test(href)) return text;
        return `<a data-href="${esc(href)}" title="${esc(href)}">${text}</a>`;
      },
      image(tok) {
        return `<span class="label">[image: ${esc(tok.text || tok.href || "")}]</span>`;
      },
      code(tok) {
        const lang = String(tok.lang || "").trim().split(/\s+/)[0].toLowerCase();
        let body;
        if (highlight && lang && window.hljs && hljs.getLanguage(lang)) {
          try {
            body = hljs.highlight(tok.text, { language: lang, ignoreIllegals: true }).value;
          } catch (_) {
            body = esc(tok.text);
          }
        } else {
          body = esc(tok.text);
        }
        return `<figure class="code"><div class="cap"><span class="label">${esc(lang || "text")}</span><button type="button" data-copy>copy</button></div><pre><code>${body}</code></pre></figure>`;
      },
    },
  });

  // `opts.highlight`: run highlight.js on code (skipped while streaming: re-highlighting
  // a growing answer on every update would steal CPU from inference)
  let highlight = true;
  window.renderMarkdown = function (src, opts = {}) {
    highlight = opts.highlight !== false;
    try {
      return md.parse(src || "");
    } catch (e) {
      return `<p>${esc(src)}</p>`;
    }
  };
  window.escapeHtml = esc;
})();
