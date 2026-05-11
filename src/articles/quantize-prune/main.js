// ============================================================
// Quantization + Pruning — 2-hidden-layer MLP trained on 3-class 2D,
// with live quantization (bit-width slider) and magnitude pruning.
// ============================================================

const C = 3;
const PLANE = { xMin: -3, xMax: 3, yMin: -2.2, yMax: 2.2 };
const STATE = {
  W: null,             // trained FP32 weights: [W1 (2x8), b1 (8), W2 (8x8), b2 (8), W3 (8xC), b3 (C)]
  data: null,
  bits: 8,
  prune: 0,
  recipe: 'ptq',
  qatW: null,          // QAT-finetuned weights (re-finetuned under simulated quantization)
  curveHistory: []
};

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
function randn() { const u = Math.random() || 1e-12, v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

function makeData(n) {
  const out = [];
  const clusters = [
    { mu: [-1.4, 0.2], cov: 0.7, label: 0 },
    { mu: [1.3, 1.2], cov: 0.65, label: 1 },
    { mu: [0.0, -1.5], cov: 0.8, label: 2 }
  ];
  for (let i = 0; i < n; i++) {
    const c = clusters[i % C];
    out.push({ x: c.mu[0] + randn() * c.cov, y: c.mu[1] + randn() * c.cov, label: c.label });
  }
  return out;
}

// Tiny MLP: 2 -> 8 -> 8 -> 3
const H = 8;
function makeMLP() {
  return {
    W1: Array.from({ length: H }, () => Array.from({ length: 2 }, () => randn() * 0.4)),
    b1: new Array(H).fill(0),
    W2: Array.from({ length: H }, () => Array.from({ length: H }, () => randn() * 0.4)),
    b2: new Array(H).fill(0),
    W3: Array.from({ length: C }, () => Array.from({ length: H }, () => randn() * 0.4)),
    b3: new Array(C).fill(0)
  };
}

function relu(z) { return Math.max(0, z); }
function softmax(arr) {
  let m = -Infinity; for (const v of arr) if (v > m) m = v;
  const e = arr.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((x) => x / s);
}

function forward(W, x, y) {
  // First hidden
  const h1 = new Array(H);
  for (let i = 0; i < H; i++) h1[i] = relu(W.W1[i][0] * x + W.W1[i][1] * y + W.b1[i]);
  const h2 = new Array(H);
  for (let i = 0; i < H; i++) {
    let s = W.b2[i];
    for (let j = 0; j < H; j++) s += W.W2[i][j] * h1[j];
    h2[i] = relu(s);
  }
  const logits = new Array(C);
  for (let c = 0; c < C; c++) {
    let s = W.b3[c];
    for (let j = 0; j < H; j++) s += W.W3[c][j] * h2[j];
    logits[c] = s;
  }
  return { h1, h2, logits, probs: softmax(logits) };
}

function trainMLP(data, epochs = 400, lr = 0.05) {
  const W = makeMLP();
  for (let e = 0; e < epochs; e++) {
    // Mini-batch SGD on all points
    for (const ex of data) {
      const f = forward(W, ex.x, ex.y);
      // Backward
      const dlogits = f.probs.slice();
      dlogits[ex.label] -= 1;
      // dW3
      const dW3 = Array.from({ length: C }, () => new Array(H).fill(0));
      const db3 = new Array(C).fill(0);
      const dh2 = new Array(H).fill(0);
      for (let c = 0; c < C; c++) {
        for (let j = 0; j < H; j++) {
          dW3[c][j] = dlogits[c] * f.h2[j];
          dh2[j] += dlogits[c] * W.W3[c][j];
        }
        db3[c] = dlogits[c];
      }
      const dpre2 = dh2.map((v, i) => v * (f.h2[i] > 0 ? 1 : 0));
      const dW2 = Array.from({ length: H }, () => new Array(H).fill(0));
      const db2 = new Array(H).fill(0);
      const dh1 = new Array(H).fill(0);
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < H; j++) {
          dW2[i][j] = dpre2[i] * f.h1[j];
          dh1[j] += dpre2[i] * W.W2[i][j];
        }
        db2[i] = dpre2[i];
      }
      const dpre1 = dh1.map((v, i) => v * (f.h1[i] > 0 ? 1 : 0));
      const dW1 = Array.from({ length: H }, () => new Array(2).fill(0));
      const db1 = new Array(H).fill(0);
      for (let i = 0; i < H; i++) {
        dW1[i][0] = dpre1[i] * ex.x;
        dW1[i][1] = dpre1[i] * ex.y;
        db1[i] = dpre1[i];
      }
      // SGD step
      for (let i = 0; i < H; i++) {
        W.W1[i][0] -= lr * dW1[i][0]; W.W1[i][1] -= lr * dW1[i][1];
        W.b1[i] -= lr * db1[i];
        for (let j = 0; j < H; j++) W.W2[i][j] -= lr * dW2[i][j];
        W.b2[i] -= lr * db2[i];
      }
      for (let c = 0; c < C; c++) {
        for (let j = 0; j < H; j++) W.W3[c][j] -= lr * dW3[c][j];
        W.b3[c] -= lr * db3[c];
      }
    }
  }
  return W;
}

// ---------- Quantize + prune ----------
function quantizeArr(arr, bits) {
  if (bits >= 16) return arr.slice();
  const max = Math.max(...arr.map(Math.abs)) || 1e-6;
  const levels = Math.pow(2, bits - 1) - 1;
  const step = max / levels;
  return arr.map((v) => Math.round(v / step) * step);
}
function flattenW(W) {
  return [].concat(
    ...W.W1.map((r) => r.slice()),
    W.b1.slice(),
    ...W.W2.map((r) => r.slice()),
    W.b2.slice(),
    ...W.W3.map((r) => r.slice()),
    W.b3.slice()
  );
}
function applyCompressedWeights(W, bits, pruneRatio) {
  const flat = flattenW(W);
  // Magnitude prune threshold
  const sorted = flat.map(Math.abs).slice().sort((a, b) => a - b);
  const idx = Math.floor(pruneRatio * sorted.length);
  const thresh = sorted[Math.min(sorted.length - 1, idx)];
  // Apply to each layer
  function process(arr) {
    let processed = arr.map((v) => Math.abs(v) < thresh ? 0 : v);
    processed = quantizeArr(processed, bits);
    return processed;
  }
  const compressed = {
    W1: W.W1.map((r) => process(r)),
    b1: process(W.b1),
    W2: W.W2.map((r) => process(r)),
    b2: process(W.b2),
    W3: W.W3.map((r) => process(r)),
    b3: process(W.b3)
  };
  return compressed;
}

function acc(W, data) {
  let correct = 0;
  for (const ex of data) {
    const f = forward(W, ex.x, ex.y);
    let best = 0; for (let c = 1; c < C; c++) if (f.logits[c] > f.logits[best]) best = c;
    if (best === ex.label) correct++;
  }
  return correct / data.length;
}

function modelSize(W, bits, pruneRatio) {
  const flat = flattenW(W);
  const nonzero = flat.filter((v) => Math.abs(v) > 1e-9).length;
  // estimated effective bits per nonzero
  return (nonzero * bits) / 8; // bytes
}

// ---------- Render ----------
function hex2rgba(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}
const COLORS = ['#2c6fb7', '#d9622b', '#1e7770'];

function renderDecision(canvasId, W) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const Wd = 380, Hd = 380;
  const ctx = setupCanvas(canvas, Wd, Hd);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, Wd, Hd);
  const step = 4;
  for (let py = 0; py < Hd; py += step) {
    for (let px = 0; px < Wd; px += step) {
      const x = PLANE.xMin + (PLANE.xMax - PLANE.xMin) * (px / Wd);
      const y = PLANE.yMax - (PLANE.yMax - PLANE.yMin) * (py / Hd);
      const f = forward(W, x, y);
      let best = 0; for (let c = 1; c < C; c++) if (f.probs[c] > f.probs[best]) best = c;
      ctx.fillStyle = hex2rgba(COLORS[best], 0.18 + 0.45 * f.probs[best]);
      ctx.fillRect(px, py, step, step);
    }
  }
  STATE.data.slice(0, 200).forEach((p) => {
    const px = (p.x - PLANE.xMin) / (PLANE.xMax - PLANE.xMin) * Wd;
    const py = (PLANE.yMax - p.y) / (PLANE.yMax - PLANE.yMin) * Hd;
    ctx.beginPath();
    ctx.arc(px, py, 2.2, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[p.label];
    ctx.fill();
  });
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, Wd, Hd);
}

function renderCurve() {
  const canvas = document.getElementById('qp-curve');
  if (!canvas) return;
  const Wd = 380, Hd = 380;
  const ctx = setupCanvas(canvas, Wd, Hd);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, Wd, Hd);
  const m = { l: 50, r: 14, t: 18, b: 30 };
  const px = Wd - m.l - m.r, py = Hd - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = i / 4;
    const y = m.t + (1 - v) * py;
    ctx.fillText(v.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  // Sweep bits + prune and plot the trade
  const pts = [];
  const baseSize = modelSize(STATE.W, 32, 0);
  for (let b = 1; b <= 16; b++) {
    const comp = applyCompressedWeights(STATE.W, b, 0);
    pts.push({ size: modelSize(comp, b, 0) / baseSize, acc: acc(comp, STATE.data), label: `${b} bits` });
  }
  for (let p = 0.1; p < 1; p += 0.1) {
    const comp = applyCompressedWeights(STATE.W, 32, p);
    pts.push({ size: modelSize(comp, 32, p) / baseSize, acc: acc(comp, STATE.data), label: `${(p * 100).toFixed(0)}% prune` });
  }
  // Plot
  ctx.fillStyle = '#1a1815';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('size (fraction of FP32) →', m.l + 6, m.t + 14);
  pts.forEach((pt) => {
    const x = m.l + pt.size * px;
    const y = m.t + (1 - pt.acc) * py;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = pt.label.includes('bits') ? '#2c6fb7' : '#d9622b';
    ctx.fill();
  });
  // Highlight current
  const cur = applyCompressedWeights(STATE.W, STATE.bits, STATE.prune);
  const cx = m.l + (modelSize(cur, STATE.bits, STATE.prune) / baseSize) * px;
  const cy = m.t + (1 - acc(cur, STATE.data)) * py;
  ctx.strokeStyle = '#1a1815'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.stroke();
}

function renderHist() {
  const canvas = document.getElementById('qp-hist');
  if (!canvas) return;
  const Wd = 880, Hd = 200;
  const ctx = setupCanvas(canvas, Wd, Hd);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, Wd, Hd);
  const m = { l: 50, r: 14, t: 18, b: 30 };
  const px = Wd - m.l - m.r, py = Hd - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const W = applyCompressedWeights(STATE.W, STATE.bits, STATE.prune);
  const flatOrig = flattenW(STATE.W);
  const flatComp = flattenW(W);
  const lo = Math.min(...flatOrig), hi = Math.max(...flatOrig);
  const range = hi - lo;
  const bins = 50;
  const histOrig = new Array(bins).fill(0);
  const histComp = new Array(bins).fill(0);
  flatOrig.forEach((v) => { const b = Math.min(bins - 1, Math.floor((v - lo) / range * bins)); histOrig[b]++; });
  flatComp.forEach((v) => { const b = Math.min(bins - 1, Math.floor((v - lo) / range * bins)); histComp[b]++; });
  const maxC = Math.max(...histOrig, ...histComp, 1);
  const bw = px / bins;
  histOrig.forEach((c, i) => {
    const h = (c / maxC) * py;
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fillRect(m.l + i * bw, m.t + py - h, bw - 1, h);
  });
  histComp.forEach((c, i) => {
    const h = (c / maxC) * py;
    ctx.fillStyle = 'rgba(217, 98, 43, 0.55)';
    ctx.fillRect(m.l + i * bw, m.t + py - h, bw - 1, h);
  });
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('grey = FP32  |  orange = compressed (quantized + pruned)', m.l + 8, m.t + 14);
}

function refreshAll() {
  renderDecision('qp-orig', STATE.W);
  const comp = applyCompressedWeights(STATE.W, STATE.bits, STATE.prune);
  renderDecision('qp-comp', comp);
  renderCurve();
  renderHist();
  const baseSize = modelSize(STATE.W, 32, 0);
  const compSize = modelSize(comp, STATE.bits, STATE.prune);
  document.getElementById('qp-size').textContent = `${(compSize / baseSize * 100).toFixed(1)}%`;
  document.getElementById('qp-acc').textContent = `${(acc(comp, STATE.data) * 100).toFixed(1)}%`;
}

function wire() {
  STATE.data = makeData(300);
  STATE.W = trainMLP(STATE.data);
  document.getElementById('qp-bits').addEventListener('input', (e) => {
    STATE.bits = parseInt(e.target.value, 10);
    document.getElementById('qp-bits-val').textContent = STATE.bits;
    refreshAll();
  });
  document.getElementById('qp-prune').addEventListener('input', (e) => {
    STATE.prune = parseFloat(e.target.value);
    document.getElementById('qp-prune-val').textContent = STATE.prune.toFixed(2);
    refreshAll();
  });
  document.getElementById('qp-recipe').addEventListener('change', (e) => { STATE.recipe = e.target.value; });
  document.getElementById('qp-newdata').addEventListener('click', () => {
    STATE.data = makeData(300);
    STATE.W = trainMLP(STATE.data);
    refreshAll();
  });
  refreshAll();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-quant':
      'q(w) = \\mathrm{round}\\!\\left(\\frac{w}{\\Delta}\\right)\\Delta,\\qquad \\Delta = \\frac{|w|_{\\max}}{2^{b-1} - 1}'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}
function boot() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wire();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
