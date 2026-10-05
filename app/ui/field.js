// The dither field: a loss landscape drawn as contour lines, rendered 1-bit with an
// 8×8 Bayer ordered dither (the portfolio's field shader), chunky on purpose.
// The cursor dents the surface; every generated token sends an ember ripple through it.
// It is loud on the empty state and recedes to a faint stipple behind the conversation.
// Rendering is throttled and stops entirely when nothing moves, so it never competes
// with inference for the CPU.
(function () {
  "use strict";

  const VERT = "attribute vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }";

  const FRAG = `
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform float uAmp;
uniform float uFocus;
uniform float uLine;      // contour half-width (fraction of a band)
uniform float uFill;      // how much the high ground is shaded
uniform vec3 uInk;
uniform vec3 uAcc;
uniform vec3 uMouse;      // x, y (canvas px, GL orientation), strength
uniform vec4 uShock[8];   // x, y (canvas px), age (s), strength
float Bayer2(vec2 a){ a = floor(a); return fract(a.x*0.5 + a.y*a.y*0.75); }
#define Bayer4(a) (Bayer2(0.5*(a))*0.25 + Bayer2(a))
#define Bayer8(a) (Bayer4(0.5*(a))*0.25 + Bayer2(a))
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*(3.0 - 2.0*f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ s += a*noise(p); p = p*2.03 + 17.1; a *= 0.5; } return s; }
void main(){
  vec2 frag = gl_FragCoord.xy;
  vec2 uv = frag / uRes;
  vec2 p = (frag - 0.5*uRes) / uRes.y;
  float t = uTime;
  vec2 q = p*1.9 + vec2(t*0.010, t*0.006);
  float h = fbm(q + 0.85*fbm(q*0.65 - t*0.012));
  vec2 m = (uMouse.xy - 0.5*uRes) / uRes.y;
  h -= uMouse.z * 0.20 * exp(-dot(p - m, p - m) * 16.0);
  float ring = 0.0;
  for (int i = 0; i < 8; i++){
    vec4 s = uShock[i];
    if (s.w <= 0.0) continue;
    vec2 c = (s.xy - 0.5*uRes) / uRes.y;
    float r = s.z * 0.30;
    float d = length(p - c);
    float w = exp(-pow((d - r) * 70.0, 2.0)) * s.w * exp(-s.z * 2.4);
    ring = max(ring, w);
    h += w * 0.02;
  }
  float f = fract(h * 11.0);
  float line = 1.0 - smoothstep(uLine * 0.35, uLine, min(f, 1.0 - f));
  float cov = line*0.85 + smoothstep(0.45, 0.95, h)*uFill;
  // hero: the whole field; conversation: recede toward the upper right, away from the text
  float mask = mix(1.0, smoothstep(1.05, 0.05, length((uv - vec2(0.86, 0.80)) * vec2(1.25, 1.0))), uFocus);
  cov *= uAmp * mask;
  float th = Bayer8(frag);
  float ink = step(th, cov) * step(0.015, cov);
  // (the Bayer threshold is exactly 0 in one cell of 64: guard so empty areas stay empty)
  float acc = step(th, min(0.55, ring * 0.7)) * step(0.002, ring);
  float a = max(ink * 0.62, acc);
  vec3 col = mix(uInk, uAcc, acc);
  gl_FragColor = vec4(col * a, a);
}`;

  function rgb(css) {
    const s = css.trim();
    if (s.startsWith("#")) {
      const n = parseInt(s.length === 4 ? s.slice(1).split("").map((c) => c + c).join("") : s.slice(1, 7), 16);
      return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
    }
    const m = s.match(/[\d.]+/g) || [0, 0, 0];
    return [m[0] / 255, m[1] / 255, m[2] / 255];
  }

  // opts: { line, fill, amp, local } — `local` maps the pointer into the canvas's own
  // rectangle (a framed figure) instead of the whole window (the background)
  function init(canvas, opts = {}) {
    const gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false, powerPreference: "low-power" });
    if (!gl) return null;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.warn(gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const U = {};
    for (const n of ["uRes", "uTime", "uAmp", "uFocus", "uLine", "uFill", "uInk", "uAcc", "uMouse", "uShock"]) U[n] = gl.getUniformLocation(prog, n);
    gl.uniform1f(U.uLine, opts.line ?? 0.06);
    gl.uniform1f(U.uFill, opts.fill ?? 0.0);

    const st = {
      amp: opts.amp ?? 1, ampTarget: opts.amp ?? 1, focus: 0, focusTarget: 0,
      mouse: [0, 0, 0], mouseTarget: 0, lastMove: 0,
      shocks: new Float32Array(32), shockHead: 0,
      motion: true, activity: 0, time: Math.random() * 100,
      dirty: true, raf: 0, last: 0, wander: Math.random() * 6.28, visible: true,
    };

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      // one canvas pixel = 2 device pixels: chunky dither, a quarter of the fill cost
      const w = Math.max(1, Math.floor((canvas.clientWidth * dpr) / 2));
      const h = Math.max(1, Math.floor((canvas.clientHeight * dpr) / 2));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        gl.viewport(0, 0, w, h);
      }
      st.dirty = true;
      kick();
    }

    function colors() {
      const cs = getComputedStyle(document.documentElement);
      gl.uniform3fv(U.uInk, rgb(cs.getPropertyValue("--fg")));
      gl.uniform3fv(U.uAcc, rgb(cs.getPropertyValue("--accent")));
      st.dirty = true;
      kick();
    }

    function animating() {
      let shocks = false;
      for (let i = 0; i < 8; i++) if (st.shocks[i * 4 + 3] > 0) shocks = true;
      return (
        st.motion &&
        (shocks || Math.abs(st.amp - st.ampTarget) > 0.004 || Math.abs(st.focus - st.focusTarget) > 0.004 ||
          Math.abs(st.mouse[2] - st.mouseTarget) > 0.01 || performance.now() - st.lastMove < 600 || st.activity > 0)
      );
    }

    function frame(now) {
      st.raf = 0;
      // ~30 fps for pointer interaction, ~15 fps while the model generates (the CPU
      // belongs to inference then), ~12 fps for the slow drift otherwise
      const busy = animating();
      const minDt = st.activity > 0 ? 66 : busy ? 33 : 80;
      if (now - st.last < minDt && !st.dirty) {
        schedule();
        return;
      }
      const dt = Math.min(0.1, (now - (st.last || now)) / 1000);
      st.last = now;
      if (st.motion) st.time += dt;
      const k = st.motion ? 1 - Math.pow(0.004, dt) : 1;
      st.amp += (st.ampTarget - st.amp) * k;
      st.focus += (st.focusTarget - st.focus) * k;
      st.mouse[2] += (st.mouseTarget - st.mouse[2]) * (st.motion ? 1 - Math.pow(0.02, dt) : 1);
      for (let i = 0; i < 8; i++) {
        const o = i * 4;
        if (st.shocks[o + 3] > 0) {
          st.shocks[o + 2] += dt;
          if (st.shocks[o + 2] > 2.2) st.shocks[o + 3] = 0;
        }
      }
      gl.uniform2f(U.uRes, canvas.width, canvas.height);
      gl.uniform1f(U.uTime, st.time);
      gl.uniform1f(U.uAmp, st.amp);
      gl.uniform1f(U.uFocus, st.focus);
      gl.uniform3f(U.uMouse, st.mouse[0], st.mouse[1], st.mouse[2]);
      gl.uniform4fv(U.uShock, st.shocks);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      st.dirty = false;
      if (st.motion || animating()) schedule();
    }

    function schedule() {
      if (!st.raf && !document.hidden && st.visible) st.raf = requestAnimationFrame(frame);
    }
    function kick() {
      schedule();
    }

    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", () => !document.hidden && kick());
    window.addEventListener("pointermove", (e) => {
      const dpr = window.devicePixelRatio || 1;
      const r = canvas.getBoundingClientRect();
      const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
      if (opts.local && !inside) {
        if (st.mouseTarget !== 0) {
          st.mouseTarget = 0;
          kick();
        }
        return;
      }
      st.mouse[0] = ((e.clientX - r.left) * dpr) / 2;
      st.mouse[1] = canvas.height - ((e.clientY - r.top) * dpr) / 2;
      st.mouseTarget = 1;
      st.lastMove = performance.now();
      kick();
    });
    document.addEventListener("pointerleave", () => {
      st.mouseTarget = 0;
      kick();
    });

    resize();
    colors();

    return {
      colors,
      setMotion(on) {
        st.motion = on;
        st.dirty = true;
        kick();
      },
      // background stipple: faint everywhere, a touch livelier while generating
      setMode(mode) {
        st.focusTarget = mode === "hero" ? 0 : 1;
        st.ampTarget = mode === "hero" ? 0.16 : mode === "busy" ? 0.11 : 0.08;
        if (!st.motion) {
          st.focus = st.focusTarget;
          st.amp = st.ampTarget;
        }
        st.dirty = true;
        kick();
      },
      // pause entirely while the canvas is not on screen
      setVisible(v) {
        st.visible = v;
        if (v) resize();
      },
      setAmp(a) {
        st.ampTarget = a;
        if (!st.motion) st.amp = a;
        kick();
      },
      resize,
      setActivity(level) {
        st.activity = level;
        kick();
      },
      // a token: an ember ripple from a slowly wandering point in the upper right
      pulse(strength = 0.6, cx = 0.8, cy = 0.74) {
        if (!st.motion) return;
        st.wander += 0.37;
        const w = canvas.width, h = canvas.height;
        const x = w * (cx + 0.12 * Math.sin(st.wander * 0.9));
        const y = h * (cy + 0.14 * Math.cos(st.wander * 1.3));
        const o = st.shockHead * 4;
        st.shocks[o] = x;
        st.shocks[o + 1] = y;
        st.shocks[o + 2] = 0;
        st.shocks[o + 3] = strength;
        st.shockHead = (st.shockHead + 1) % 8;
        kick();
      },
    };
  }

  window.DitherField = { init };
})();
