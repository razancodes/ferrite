// ferrite chat — UI logic. The engine runs in Rust on its own thread; this file
// sends commands (invoke) and renders the stream of `engine` events.
(function () {
  "use strict";

  const T = window.__TAURI__;
  const invoke = T.core.invoke;
  const listen = T.event.listen;
  const $ = (id) => document.getElementById(id);
  const esc = window.escapeHtml;

  // ───────────────────────── settings (per-machine conveniences) ─────────────────────────
  const QWEN = {
    chat: { temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 1.5 },
    think: { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0, presence_penalty: 1.5 },
  };
  const load = (k, d) => {
    try {
      const v = localStorage.getItem("ferrite." + k);
      return v == null ? d : JSON.parse(v);
    } catch (_) {
      return d;
    }
  };
  const S = Object.assign(
    { theme: "ink", motion: true, spec: true, think: false, modelPath: null, system: "", maxTokens: 2048, seed: 0, ctx: 8192, threads: 0, sampler: { ...QWEN.chat } },
    load("settings", {})
  );
  const save = () => {
    try {
      localStorage.setItem("ferrite.settings", JSON.stringify(S));
    } catch (_) {}
    markPreset();
  };
  // one-click configurations for the loaded model (the model itself is never swapped)
  const PRESETS = {
    think: { think: true, spec: true, maxTokens: 32768, ctx: 32768, sampler: QWEN.think },
    speed: { think: false, spec: true, motion: false, maxTokens: 2048, sampler: QWEN.chat },
  };
  const matches = (p) => Object.keys(p).every((k) => (k === "sampler" ? Object.keys(p.sampler).every((j) => S.sampler[j] === p.sampler[j]) : S[k] === p[k]));
  function markPreset() {
    for (const name in PRESETS) $("p-" + name).classList.toggle("on", matches(PRESETS[name]));
  }

  // ───────────────────────── state ─────────────────────────
  let models = [];
  let info = null; // loaded model
  let loadingName = null;
  let sys = null;
  const msgs = []; // {role, content, reasoning, stats, error, el...}
  let generating = false;
  let live = { tokens: 0, first: 0, start: 0, prefill: 0, samples: [] };
  let lastSamples = [];
  let lastAccept = "";
  let renderQueued = false;
  let contextUsed = 0;
  let autoQueue = null; // scripted messages (FERRITE_CHAT_AUTOSEND)
  function autoNext() {
    if (autoQueue && autoQueue.length && info && !generating) setTimeout(() => send(autoQueue.shift()), 400);
  }
  // a faint full-window stipple, and the loud framed figure on the empty state
  const field = window.DitherField ? window.DitherField.init($("field"), { line: 0.05, fill: 0, amp: 0.16 }) : null;
  const fig = window.DitherField ? window.DitherField.init($("fig0"), { line: 0.09, fill: 0.16, amp: 1.0, local: true }) : null;

  // ───────────────────────── helpers ─────────────────────────
  const fmt = (n, d = 1) => (Number.isFinite(n) ? n.toFixed(d) : "—");
  const thousands = (n) => Number(n || 0).toLocaleString("en-US");
  const pad2 = (n) => String(n).padStart(2, "0");
  const clock = () => {
    const d = new Date();
    return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
  };
  function toast(text, ms = 3200) {
    const t = $("toast");
    t.textContent = text;
    t.classList.add("show");
    clearTimeout(toast.h);
    toast.h = setTimeout(() => t.classList.remove("show"), ms);
  }
  function prettyName(name) {
    // qwen3.5-2b-q6 -> "Qwen3.5 <em>2B</em>"
    const parts = name.split("-");
    const quant = /^q\d/i.test(parts[parts.length - 1]) ? parts.pop() : "";
    const size = parts.length > 1 && /^\d/.test(parts[parts.length - 1]) ? parts.pop() : "";
    const base = parts.join("-").replace(/^qwen/i, "Qwen");
    return { html: `${esc(base)}${size ? ` <em>${esc(size.toUpperCase())}</em>` : ""}`, quant: quant.toUpperCase(), plain: `${base} ${size.toUpperCase()}`.trim() };
  }
  function setTog(id, value) {
    for (const i of $(id).querySelectorAll("i")) i.classList.toggle("on", i.dataset.v === value);
  }
  function nearBottom() {
    const s = $("scroll");
    return s.scrollHeight - s.scrollTop - s.clientHeight < 160;
  }
  function toBottom(force) {
    const s = $("scroll");
    if (force || nearBottom()) s.scrollTop = s.scrollHeight;
  }

  // ───────────────────────── theme / motion / toggles ─────────────────────────
  function applyTheme() {
    document.documentElement.dataset.theme = S.theme;
    setTog("t-theme", S.theme);
    if (field) field.colors();
    if (fig) fig.colors();
    drawSpark();
  }
  function applyMotion() {
    document.documentElement.dataset.motion = S.motion ? "on" : "off";
    setTog("t-motion", S.motion ? "on" : "off");
    if (field) field.setMotion(S.motion);
    if (fig) fig.setMotion(S.motion);
  }
  function applyThink() {
    setTog("t-think", S.think ? "on" : "off");
  }
  function applySpec() {
    setTog("t-spec", S.spec ? "on" : "off");
    $("t-spec").disabled = !!(info && !info.has_mtp);
    markPreset(); // the engine can flip spec on load
  }
  function setThinking(on) {
    if (S.think === on) return;
    S.think = on;
    S.sampler = { ...S.sampler, ...(on ? { temperature: QWEN.think.temperature, top_p: QWEN.think.top_p } : { temperature: QWEN.chat.temperature, top_p: QWEN.chat.top_p }) };
    applyThink();
    renderSampling();
    save();
  }
  function applyPreset(name) {
    if (generating || loadingName) return toast("Wait for the current reply to finish");
    const p = PRESETS[name];
    // a bigger context only takes effect on load: reload the current model if it changes
    const reload = info && p.ctx && p.ctx !== info.ctx ? info.path : null;
    Object.assign(S, p, { sampler: { ...p.sampler } });
    if (info && !info.has_mtp) S.spec = false;
    applyThink();
    applySpec();
    applyMotion();
    renderSampling();
    save();
    if (reload) loadModel(reload);
    else if (info && info.spec !== S.spec) invoke("set_spec", { on: S.spec });
    toast(
      name === "think"
        ? `Max think — reasoning on, thinking sampler, up to ${thousands(p.maxTokens)} tokens${reload ? " · reloading with a 32K context" : ""}`
        : "Max speed — thinking off, MTP speculation on, animation off"
    );
  }
  function fieldMode() {
    if (fig) fig.setVisible(msgs.length === 0);
    if (!field) return;
    field.setMode(msgs.length === 0 ? "hero" : generating ? "busy" : "chat");
  }

  // ───────────────────────── sidebar: models (a dropdown) ─────────────────────────
  const quantOf = (m) => esc((m.quant || prettyName(m.name).quant || "?").toUpperCase());
  function renderModels() {
    $("model-count").textContent = models.length ? `${models.length} found` : "none";
    const curPath = (info ? info.path : S.modelPath || "").toLowerCase();
    const ci = models.findIndex((m) => m.path.toLowerCase() === curPath);
    const cur = models[ci];
    const name = cur ? cur.name : loadingName;

    // the trigger: the current model as the "selected label"
    const sum = $("mcur");
    sum.classList.toggle("loading", !!loadingName);
    const caret = `<span class="go" aria-hidden="true">${loadingName ? "loading" : "▾"}</span>`;
    if (name) {
      const meta = cur ? `<span class="tag">${quantOf(cur)}</span><span class="label">${fmt(cur.size_gb, 2)} GB</span>${cur.params_b ? `<span class="label">· ${fmt(cur.params_b, 2)}B</span>` : ""}` : "";
      sum.innerHTML = `<span class="idx">M/${ci >= 0 ? pad2(ci + 1) : "··"}</span><span class="nm">${prettyName(name).html}</span>${caret}<span class="meta">${meta}</span>`;
    } else {
      sum.innerHTML = `<span class="idx">M/··</span><span class="nm">${models.length ? "Choose a model" : "No models"}</span>${caret}<span class="meta"><span class="label">${models.length ? "click to pick" : "none in ./models — use Browse"}</span></span>`;
    }
    sum.insertAdjacentHTML("beforeend", `<span class="h1"></span><span class="h2"></span>`);

    // the list
    const box = $("models");
    box.innerHTML = "";
    models.forEach((m, i) => {
      const n = prettyName(m.name);
      const row = document.createElement("button");
      row.type = "button";
      // a heavier rule where the model size changes (0.8B | 2B | 4B)
      row.className = "mopt" + (i === ci ? " cur" : "") + (i && prettyName(models[i - 1].name).plain !== n.plain ? " grp" : "");
      row.title = m.path;
      row.innerHTML = `<span class="nm">${n.html}</span><span class="tag">${quantOf(m)}</span><span class="label">${fmt(m.size_gb, 2)} GB</span>`;
      row.addEventListener("click", () => {
        $("msel").open = false;
        if (generating || loadingName) return toast("Wait for the current reply to finish");
        if (i !== ci || !info) loadModel(m.path);
      });
      box.appendChild(row);
    });
  }

  async function refreshModels() {
    try {
      models = await invoke("list_models");
    } catch (e) {
      models = [];
    }
    if (S.modelPath && !models.some((m) => m.path.toLowerCase() === S.modelPath.toLowerCase())) {
      try {
        const extra = await invoke("describe_model", { path: S.modelPath });
        if (extra) models.push(extra);
      } catch (_) {}
    }
    renderModels();
  }

  function loadModel(path) {
    S.modelPath = path;
    save();
    const m = models.find((x) => x.path === path);
    loadingName = m ? m.name : path.split(/[\\/]/).pop().replace(/\.ferrite$/, "");
    info = null;
    renderModels();
    renderStatus();
    invoke("load_model", { path, threads: S.threads || null, ctx: S.ctx, spec: S.spec });
  }

  // ───────────────────────── sidebar: sampling ─────────────────────────
  const SLIDERS = [
    { k: "temperature", label: "Temperature", min: 0, max: 2, step: 0.05, d: 2 },
    { k: "top_p", label: "Top-p", min: 0.05, max: 1, step: 0.01, d: 2 },
    { k: "top_k", label: "Top-k", min: 0, max: 100, step: 1, d: 0 },
    { k: "min_p", label: "Min-p", min: 0, max: 0.5, step: 0.01, d: 2 },
    { k: "presence_penalty", label: "Presence penalty", min: 0, max: 2, step: 0.05, d: 2 },
  ];
  function renderSampling() {
    const box = $("sampling");
    box.innerHTML = "";
    for (const s of SLIDERS) {
      const v = S.sampler[s.k];
      const row = document.createElement("div");
      row.className = "set";
      row.innerHTML = `<div class="top"><span class="label">${s.label}</span><span class="val">${Number(v).toFixed(s.d)}</span></div>
        <input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${v}" aria-label="${s.label}">`;
      const input = row.querySelector("input");
      const val = row.querySelector(".val");
      const paint = () => input.style.setProperty("--p", `${((input.value - s.min) / (s.max - s.min)) * 100}%`);
      paint();
      input.addEventListener("input", () => {
        S.sampler[s.k] = s.d ? Number(input.value) : parseInt(input.value, 10);
        val.textContent = Number(input.value).toFixed(s.d);
        paint();
        save();
      });
      box.appendChild(row);
    }
    const nums = [
      { k: "maxTokens", label: "Max tokens", min: 16, max: 32768 },
      { k: "seed", label: "Seed (0 = random)", min: 0, max: 2 ** 31 },
      { k: "ctx", label: "Context (on load)", min: 1024, max: 65536 },
      { k: "threads", label: "Threads (0 = auto)", min: 0, max: 64 },
    ];
    for (const n of nums) {
      const row = document.createElement("div");
      row.className = "set row2";
      row.innerHTML = `<span class="label">${n.label}</span><input type="number" min="${n.min}" max="${n.max}" value="${S[n.k]}" aria-label="${n.label}">`;
      row.querySelector("input").addEventListener("change", (e) => {
        const v = Math.max(n.min, Math.min(n.max, parseInt(e.target.value || "0", 10)));
        e.target.value = v;
        S[n.k] = v;
        save();
        if ((n.k === "ctx" || n.k === "threads") && info) toast("Applies when a model is (re)loaded — click the model to reload.");
      });
      box.appendChild(row);
    }
  }

  // ───────────────────────── sidebar: machine ─────────────────────────
  function renderMachine() {
    const kv = $("kv");
    const rows = [];
    if (sys) rows.push(["CPU", sys.cpu], ["Threads", info ? `${info.threads} · ${info.decode_threads} decode` : `${sys.default_threads}`]);
    if (info) {
      rows.push(
        ["Model", `${fmt(info.params_b, 2)}B · ${info.recipe}`],
        ["Layers", `${info.layers} · ${info.linear_layers} DeltaNet + ${info.attn_layers} attn`],
        ["Weights", `${fmt(info.weights_gb, 2)} GB · mmap`],
        ["Context", `${thousands(contextUsed)} / ${thousands(info.ctx)}`],
        ["Speculation", info.spec ? "MTP · on" : info.has_mtp ? "off" : "n/a"]
      );
    }
    kv.innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd title="${esc(v)}">${esc(v)}</dd>`).join("");
    $("isa").textContent = sys ? sys.isa.replace("Avx512Vnni", "AVX-512 VNNI") : "";
    $("ctx-use").textContent = info ? `context ${thousands(contextUsed)} / ${thousands(info.ctx)}` : "";
  }

  // Fig. 1: throughput as a Bayer-dithered bar chart (ink dots; the newest bar in ember)
  const BAYER = (() => {
    const b = [[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22], [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25], [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]];
    return (x, y) => (b[y & 7][x & 7] + 0.5) / 64;
  })();
  function drawSpark() {
    const c = $("spark");
    const ctx = c.getContext("2d");
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const cs = getComputedStyle(document.documentElement);
    const ink = cs.getPropertyValue("--fg").trim();
    const acc = cs.getPropertyValue("--accent").trim();
    const rule = cs.getPropertyValue("--rule").trim();
    const data = generating ? live.samples : lastSamples;
    // baseline
    ctx.fillStyle = rule;
    for (let x = 0; x < W; x += 2) ctx.fillRect(x, H - 1, 1, 1);
    if (!data.length) return;
    const cols = Math.floor(W / 4);
    const series = data.slice(-cols);
    const max = Math.max(4, ...series) * 1.12;
    series.forEach((v, i) => {
      const x0 = W - (series.length - i) * 4;
      const h = Math.max(1, Math.round((v / max) * (H - 3)));
      ctx.fillStyle = i === series.length - 1 && generating ? acc : ink;
      for (let y = H - 2; y > H - 2 - h; y--) {
        // density ramps up toward the top of each bar
        const dens = 0.35 + 0.65 * ((H - 2 - y) / Math.max(1, h));
        for (let x = x0; x < x0 + 3; x++) if (BAYER(x, y) < dens) ctx.fillRect(x, y, 1, 1);
      }
    });
  }
  function setBigTps(v, isLive) {
    $("tps").textContent = Number.isFinite(v) && v > 0 ? fmt(v, 1) : "—";
    $("machine").classList.toggle("live", !!isLive);
  }

  // ───────────────────────── hero ─────────────────────────
  const PROMPTS = [
    "Explain how a CPU cache works, with an analogy.",
    "Write a Python function that merges overlapping intervals, with tests.",
    "What is the difference between linear attention and softmax attention?",
    "Give me a three-day itinerary for Lahore.",
  ];
  function renderPrompts() {
    const box = $("prompts");
    box.innerHTML = "";
    PROMPTS.forEach((p, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "prow";
      b.disabled = !info;
      b.innerHTML = `<span class="idx">P/${pad2(i + 1)}</span><span class="t">${esc(p)}</span><span class="ar">→</span>`;
      b.addEventListener("click", () => send(p));
      box.appendChild(b);
    });
  }
  function renderStatus() {
    const st = $("hero-status");
    if (loadingName) st.innerHTML = `<span class="chip live">loading ${esc(loadingName)}</span>`;
    else if (info) st.innerHTML = `<span class="chip">ready — ${esc(prettyName(info.name).plain)} · ${esc(info.recipe)}</span><span class="label">${info.spec ? "MTP speculation" : "standard decoding"} · loaded in ${fmt(info.load_secs, 1)} s</span>`;
    else st.innerHTML = `<span class="chip dim">no model</span><span class="label">pick one on the left</span>`;
    $("crumb-model").textContent = info ? prettyName(info.name).plain : loadingName ? "loading…" : "chat";
    $("input").disabled = !info;
    $("send").disabled = !info && !generating;
    renderPrompts();
    renderMachine();
    applySpec();
  }

  // ───────────────────────── conversation ─────────────────────────
  function turnHeader(i, who, extra = "") {
    return `<div class="turn-h"><span class="idx">T/${pad2(i)}</span><span class="who">${esc(who)}</span><span class="r"></span>${extra}<span class="t">${clock()}</span></div>`;
  }
  function addUserTurn(m) {
    const li = document.createElement("li");
    li.className = "turn user";
    li.innerHTML = turnHeader(msgs.length, "You") + `<div class="body">${esc(m.content)}</div>`;
    $("log").appendChild(li);
    m.el = li;
  }
  function addModelTurn(m) {
    const li = document.createElement("li");
    li.className = "turn model";
    const who = info ? `${prettyName(info.name).plain} · ${info.recipe}` : "Model";
    li.innerHTML =
      turnHeader(msgs.length, who, `<span class="chip live" data-chip>reading</span>`) +
      `<div class="reason" hidden><button type="button" aria-expanded="false"><span class="caret">▸</span><span class="label" data-rl>reasoning</span></button><div class="rbody"></div></div>
       <div class="waiting" data-wait><span class="dots"><i></i><i></i><i></i></span><span class="label" data-wl>reading prompt</span></div>
       <div class="md"></div>
       <div class="turn-f" hidden><span class="stats"></span><span class="acts"><button type="button" class="btn-rule" data-act="copy">Copy</button><button type="button" class="btn-rule" data-act="regen">Regenerate</button></span></div>`;
    $("log").appendChild(li);
    m.el = li;
    const r = li.querySelector(".reason");
    r.querySelector("button").addEventListener("click", () => {
      r.classList.toggle("open");
      r.querySelector("button").setAttribute("aria-expanded", r.classList.contains("open"));
    });
    li.querySelector('[data-act="copy"]').addEventListener("click", () => copy(m.content));
    li.querySelector('[data-act="regen"]').addEventListener("click", regenerate);
  }

  function renderStreaming() {
    renderQueued = false;
    const m = msgs[msgs.length - 1];
    if (!m || m.role !== "assistant" || !m.el) return;
    const stick = nearBottom();
    const li = m.el;
    if (m.reasoning) {
      const r = li.querySelector(".reason");
      r.hidden = false;
      const thinkingNow = m.streaming && !m.content;
      if (thinkingNow) r.classList.add("open");
      li.querySelector("[data-rl]").textContent = thinkingNow ? `reasoning — ${thousands(live.tokens)} tokens` : "reasoning";
      const rb = r.querySelector(".rbody");
      rb.textContent = m.reasoning.trim();
      if (thinkingNow) rb.scrollTop = rb.scrollHeight;
    }
    if (m.content || m.reasoning) li.querySelector("[data-wait]").hidden = true;
    const md = li.querySelector(".md");
    md.innerHTML = m.streaming ? streamingHtml(m) : window.renderMarkdown(m.content);
    md.classList.toggle("cursor-blink", !!m.streaming && !!m.content);
    if (stick) toBottom(true);
  }
  // While streaming, everything before the last blank line (outside a code fence) is
  // finished: render it once and cache it, and re-render only the growing tail.
  function stableSplit(text) {
    let idx = -1, inFence = false, m;
    const re = /```|\n\n/g;
    while ((m = re.exec(text))) {
      if (m[0] === "```") inFence = !inFence;
      else if (!inFence) idx = m.index;
    }
    return idx;
  }
  function streamingHtml(m) {
    const cut = stableSplit(m.content);
    if (cut > 0 && cut !== m.stableLen) {
      m.stableLen = cut;
      m.stableHtml = window.renderMarkdown(m.content.slice(0, cut), { highlight: false });
    }
    const head = cut > 0 ? m.stableHtml : "";
    return head + window.renderMarkdown(cut > 0 ? m.content.slice(cut) : m.content, { highlight: false });
  }

  // streaming re-renders are throttled to ~8 per second: the webview's CPU time comes
  // out of the same cores the model is running on
  let lastRender = 0;
  function queueRender() {
    if (renderQueued) return;
    renderQueued = true;
    const wait = Math.max(0, 120 - (performance.now() - lastRender));
    setTimeout(() => {
      lastRender = performance.now();
      requestAnimationFrame(renderStreaming);
    }, wait);
  }

  function finishTurn(m, stats) {
    const li = m.el;
    if (!li) return;
    m.streaming = false;
    renderStreaming();
    const chip = li.querySelector("[data-chip]");
    if (chip) chip.remove();
    li.querySelector("[data-wait]").hidden = true;
    // fold the reasoning away once the answer is there
    const r = li.querySelector(".reason");
    if (m.reasoning) {
      r.classList.remove("open");
      li.querySelector("[data-rl]").textContent = `reasoning — ${thousands(m.reasoningTokens || 0)} tokens`;
    }
    const f = li.querySelector(".turn-f");
    f.hidden = false;
    if (stats) {
      const parts = [
        `<b>${fmt(stats.gen_tps, 1)}</b> tok/s`,
        `${thousands(stats.gen_tokens)} tokens`,
        `prompt ${thousands(stats.prompt_tokens - stats.cached_tokens)}${stats.cached_tokens ? ` <span class="acc">+${thousands(stats.cached_tokens)} cached</span>` : ""}`,
      ];
      if (stats.drafted) parts.push(`accept <b>${Math.round(stats.acceptance * 100)}%</b>`);
      parts.push(`first token ${Math.round(stats.ttft_ms)} ms`);
      if (stats.stop === "stopped") parts.push(`<span class="acc">stopped</span>`);
      if (stats.stop === "length") parts.push(`<span class="acc">max tokens</span>`);
      f.querySelector(".stats").innerHTML = parts.join(" · ");
    }
    if (m.error) {
      const e = document.createElement("div");
      e.className = "err";
      e.textContent = m.error;
      li.querySelector(".md").after(e);
    }
    // only the latest answer can be regenerated
    for (const b of document.querySelectorAll('[data-act="regen"]')) b.hidden = true;
    li.querySelector('[data-act="regen"]').hidden = false;
  }

  function setGenerating(on) {
    generating = on;
    const comp = $("composer");
    comp.classList.toggle("busy", on);
    const btn = $("send");
    btn.innerHTML = on ? `Stop <span class="arrow">■</span>` : `Send <span class="arrow arrow-r">→</span>`;
    btn.disabled = !info && !on;
    $("fig-state").textContent = on ? "generating" : "idle";
    $("spark-l").textContent = on ? "live" : "last reply";
    if (field) field.setActivity(on ? 1 : 0);
    fieldMode();
    if (!on) {
      const p = $("progress");
      p.classList.remove("reading");
      p.querySelector("span").style.width = "0";
    }
  }

  function history() {
    const out = [];
    if (S.system.trim()) out.push({ role: "system", content: S.system });
    for (const m of msgs) if (!m.streaming) out.push({ role: m.role, content: m.content });
    return out;
  }

  function startGeneration() {
    const m = { role: "assistant", content: "", reasoning: "", streaming: true, reasoningTokens: 0 };
    const hist = history();
    msgs.push(m);
    addModelTurn(m);
    $("hero").hidden = true;
    live = { tokens: 0, first: 0, start: performance.now(), prefill: 0, samples: [] };
    setGenerating(true);
    toBottom(true);
    invoke("chat", {
      messages: hist,
      thinking: S.think,
      sampler: { ...S.sampler, seed: S.seed },
      maxTokens: S.maxTokens,
    });
  }

  function send(text) {
    let t = String(text || "").trim();
    if (!t || generating || !info) return;
    for (const [cmd, on] of [["/think", true], ["/nothink", false], ["/no_think", false]]) {
      if (t === cmd || t.startsWith(cmd + " ") || t.startsWith(cmd + "\n")) {
        setThinking(on);
        t = t.slice(cmd.length).trim();
        toast(on ? "Thinking on" : "Thinking off");
        break;
      }
    }
    if (!t) return;
    const m = { role: "user", content: t };
    msgs.push(m);
    addUserTurn(m);
    startGeneration();
  }

  function regenerate() {
    if (generating) return;
    const last = msgs[msgs.length - 1];
    if (!last || last.role !== "assistant") return;
    msgs.pop();
    last.el.remove();
    startGeneration();
  }

  function newChat() {
    if (generating) invoke("stop");
    msgs.length = 0;
    $("log").innerHTML = "";
    $("hero").hidden = false;
    contextUsed = 0;
    invoke("reset");
    renderMachine();
    fieldMode();
    $("input").focus({ preventScroll: true });
  }

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast("Copied");
    } catch (_) {
      toast("Copy failed");
    }
  }

  // ───────────────────────── engine events ─────────────────────────
  listen("engine", (e) => {
    const ev = e.payload;
    switch (ev.type) {
      case "loading":
        loadingName = ev.name;
        renderModels();
        renderStatus();
        break;
      case "loaded":
        info = ev.info;
        loadingName = null;
        S.spec = info.spec || (S.spec && !info.has_mtp);
        contextUsed = 0;
        renderModels();
        renderStatus();
        applySpec();
        toast(`${prettyName(info.name).plain} · ${info.recipe} — loaded in ${fmt(info.load_secs, 1)} s`);
        $("input").focus({ preventScroll: true });
        if (autoQueue === null) {
          invoke("autosend").then((q) => {
            autoQueue = q || [];
            autoNext();
          });
        }
        break;
      case "load_failed":
        loadingName = null;
        renderModels();
        renderStatus();
        toast(`Could not load the model: ${ev.error}`, 8000);
        break;
      case "spec_changed":
        if (info) info.spec = ev.on;
        S.spec = ev.on;
        applySpec();
        renderMachine();
        break;
      case "prefill": {
        live.prefill = ev.tokens;
        const m = msgs[msgs.length - 1];
        if (m && m.el) m.el.querySelector("[data-wl]").textContent = `reading ${thousands(ev.tokens)} tokens`;
        $("progress").classList.add("reading");
        break;
      }
      case "delta": {
        const m = msgs[msgs.length - 1];
        if (!m || m.role !== "assistant") break;
        const now = performance.now();
        if (!live.first) {
          live.first = now;
          $("progress").classList.remove("reading");
          const chip = m.el.querySelector("[data-chip]");
          if (chip) chip.textContent = "writing";
        }
        const fresh = ev.tokens - live.tokens;
        live.tokens = ev.tokens;
        if (ev.reasoning) {
          m.reasoning += ev.text;
          m.reasoningTokens = ev.tokens;
        } else m.content += ev.text;
        // a faint ember ripple every few tokens, out at the right edge (the field
        // recedes under reading text; this only hints that the model is working)
        if (field && fresh > 0 && (live.tokens % 3 === 0 || fresh > 1)) field.pulse(m.content ? 0.5 : 0.35, 0.93, 0.62);
        // live throughput, sampled ~4x a second for the figure
        const secs = (now - live.first) / 1000;
        if (secs > 0.25) {
          const tps = (live.tokens - 1) / secs;
          setBigTps(tps, true);
          if (!live.lastSample || now - live.lastSample > 250) {
            live.samples.push(tps);
            live.lastSample = now;
            drawSpark();
          }
        }
        $("progress").querySelector("span").style.width = `${Math.min(100, (live.tokens / Math.max(1, S.maxTokens)) * 100)}%`;
        queueRender();
        break;
      }
      case "done": {
        const m = msgs[msgs.length - 1];
        contextUsed = ev.stats.context_used;
        if (m && m.role === "assistant") {
          if (!m.content && !m.reasoning) m.error = ev.stats.stop === "stopped" ? "Stopped before any output." : "The model returned an empty answer.";
          finishTurn(m, ev.stats);
        }
        lastSamples = live.samples.length ? live.samples.slice() : [ev.stats.gen_tps];
        setBigTps(ev.stats.gen_tps, false);
        $("accept").textContent = ev.stats.drafted ? `accept ${Math.round(ev.stats.acceptance * 100)}%` : "";
        setGenerating(false);
        renderMachine();
        drawSpark();
        autoNext();
        break;
      }
      case "failed": {
        const m = msgs[msgs.length - 1];
        if (m && m.role === "assistant" && m.streaming) {
          m.error = ev.error;
          finishTurn(m, null);
        } else toast(ev.error, 8000);
        setGenerating(false);
        break;
      }
    }
  });

  // ───────────────────────── wiring ─────────────────────────
  const input = $("input");
  const grow = () => {
    input.style.height = "auto";
    input.style.height = Math.min(220, input.scrollHeight) + "px";
  };
  input.addEventListener("input", grow);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    if (generating) {
      invoke("stop");
      return;
    }
    const text = input.value;
    if (!text.trim()) return;
    input.value = "";
    grow();
    send(text);
  });

  $("t-theme").addEventListener("click", () => {
    S.theme = S.theme === "ink" ? "paper" : "ink";
    applyTheme();
    save();
  });
  $("t-motion").addEventListener("click", () => {
    S.motion = !S.motion;
    applyMotion();
    save();
  });
  $("t-think").addEventListener("click", () => {
    setThinking(!S.think);
    toast(S.think ? "Thinking on — the model reasons before answering" : "Thinking off");
  });
  $("t-spec").addEventListener("click", () => {
    if (generating) return;
    S.spec = !S.spec;
    applySpec();
    save();
    if (info) invoke("set_spec", { on: S.spec });
  });
  $("t-new").addEventListener("click", newChat);
  $("p-think").addEventListener("click", () => applyPreset("think"));
  $("p-speed").addEventListener("click", () => applyPreset("speed"));
  document.addEventListener("click", (e) => {
    if ($("msel").open && !$("msel").contains(e.target)) $("msel").open = false;
  });
  $("rescan").addEventListener("click", refreshModels);
  $("defaults").addEventListener("click", () => {
    S.sampler = { ...(S.think ? QWEN.think : QWEN.chat) };
    renderSampling();
    save();
    toast("Sampling reset to Qwen's recommended settings");
  });
  $("browse").addEventListener("click", async () => {
    if (generating) return;
    const path = await T.dialog.open({ multiple: false, directory: false, filters: [{ name: "ferrite model", extensions: ["ferrite"] }] });
    if (!path) return;
    if (!models.some((m) => m.path.toLowerCase() === String(path).toLowerCase())) {
      const d = await invoke("describe_model", { path });
      if (d) models.push(d);
    }
    loadModel(String(path));
  });
  $("system").value = S.system;
  $("system").addEventListener("input", (e) => {
    S.system = e.target.value;
    save();
  });

  // links open in the default browser; code plates copy their code
  $("log").addEventListener("click", (e) => {
    const a = e.target.closest("a[data-href]");
    if (a) {
      e.preventDefault();
      T.opener.openUrl(a.dataset.href).catch(() => toast("Could not open the link"));
      return;
    }
    const c = e.target.closest("[data-copy]");
    if (c) {
      const code = c.closest(".code").querySelector("pre").textContent;
      copy(code);
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && (e.key === "n" || e.key === "N")) {
      e.preventDefault();
      newChat();
    } else if (e.key === "Escape" && $("msel").open) {
      $("msel").open = false;
      $("mcur").focus();
    } else if (e.key === "Escape" && generating) {
      invoke("stop");
    } else if (e.ctrlKey && (e.key === "l" || e.key === "L")) {
      e.preventDefault();
      input.focus({ preventScroll: true });
    }
  });

  // ───────────────────────── boot ─────────────────────────
  (async function boot() {
    applyTheme();
    applyMotion();
    applyThink();
    applySpec();
    renderSampling();
    renderStatus();
    fieldMode();
    drawSpark();
    try {
      sys = await invoke("system_info");
    } catch (_) {}
    renderMachine();
    await refreshModels();
    // reopen the last model, else prefer the balanced 2B q6
    let pick = S.modelPath && models.find((m) => m.path.toLowerCase() === S.modelPath.toLowerCase());
    if (!pick) pick = models.find((m) => /2b-q6/i.test(m.name)) || models[0];
    if (pick) loadModel(pick.path);
  })();
})();
