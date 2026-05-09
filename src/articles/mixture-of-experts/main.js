// ============================================================
// Mixture of Experts — live JS training of 4 linear experts + gate.
// Toy task: 2D regression where the truth is piecewise linear over
// four quadrants. A single linear model can't fit it; four experts
// + a gate can.
// ============================================================

const E = 4;
const COLORS = ['#2c6fb7', '#d9622b', '#1e7770', '#9b59b6'];
const PLANE_MIN = -2.5, PLANE_MAX = 2.5;

const MOE = {
  // Experts: each has w (R^2) and b
  experts: null,
  // Gate: G (E x 2), g_b (E)
  G: null, gb: null,
  data: null,
  step: 0,
  losses: [],
  utilisation: [], // last 4-step utilisation per expert
  routing: 'soft',
  balance: true,
  lam: 0.1,
  lr: 0.05,
  running: false,
  raf: null
};

function randn() {
  const u1 = Math.random() || 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function makeData(n) {
  const out = [];
  // Truth: piecewise linear over four quadrants
  const truths = [
    { w: [ 1.5,  0.5], b:  0.4 }, // ++
    { w: [-0.8,  1.6], b: -0.5 }, // -+
    { w: [-1.2, -0.7], b:  0.7 }, // --
    { w: [ 0.6, -1.3], b: -0.6 }  // +-
  ];
  function quad(x, y) {
    if (x >= 0 && y >= 0) return 0;
    if (x <  0 && y >= 0) return 1;
    if (x <  0 && y <  0) return 2;
    return 3;
  }
  for (let i = 0; i < n; i++) {
    const x = randn() * 1.0;
    const y = randn() * 1.0;
    const q = quad(x, y);
    const t = truths[q].w[0] * x + truths[q].w[1] * y + truths[q].b + 0.05 * randn();
    out.push({ x, y, t, q });
  }
  return out;
}

function resetMoE() {
  MOE.experts = [];
  for (let e = 0; e < E; e++) {
    MOE.experts.push({ w: [randn() * 0.3, randn() * 0.3], b: randn() * 0.1 });
  }
  MOE.G = [];
  for (let e = 0; e < E; e++) MOE.G.push([randn() * 0.2, randn() * 0.2]);
  MOE.gb = new Array(E).fill(0).map(() => randn() * 0.05);
  MOE.step = 0;
  MOE.losses = [];
  MOE.utilisation = [];
  if (!MOE.data) MOE.data = makeData(200);
}

function softmaxArr(arr) {
  let m = -Infinity;
  for (const v of arr) if (v > m) m = v;
  const exps = arr.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((v) => v / s);
}

function gateProbs(x, y) {
  const logits = new Array(E);
  for (let e = 0; e < E; e++) {
    logits[e] = MOE.G[e][0] * x + MOE.G[e][1] * y + MOE.gb[e];
  }
  return softmaxArr(logits);
}

function expertOutput(e, x, y) {
  return MOE.experts[e].w[0] * x + MOE.experts[e].w[1] * y + MOE.experts[e].b;
}

function predict(x, y) {
  const p = gateProbs(x, y);
  if (MOE.routing === 'top1') {
    let best = 0;
    for (let e = 1; e < E; e++) if (p[e] > p[best]) best = e;
    const eOut = expertOutput(best, x, y);
    return { pred: eOut, probs: p, top: best };
  }
  let s = 0;
  for (let e = 0; e < E; e++) s += p[e] * expertOutput(e, x, y);
  return { pred: s, probs: p, top: -1 };
}

function trainStep() {
  // Sample a mini-batch and apply SGD
  const B = 32;
  const lr = MOE.lr;
  let totalLoss = 0;
  // Aggregate gradients
  const dW = new Array(E).fill(0).map(() => [0, 0]);
  const db = new Array(E).fill(0);
  const dG = new Array(E).fill(0).map(() => [0, 0]);
  const dgb = new Array(E).fill(0);
  // Track utilisation
  const counts = new Array(E).fill(0);
  const probSum = new Array(E).fill(0);
  for (let i = 0; i < B; i++) {
    const ex = MOE.data[Math.floor(Math.random() * MOE.data.length)];
    const probs = gateProbs(ex.x, ex.y);
    const eOut = new Array(E);
    for (let e = 0; e < E; e++) eOut[e] = expertOutput(e, ex.x, ex.y);
    let pred, hardE = -1;
    if (MOE.routing === 'top1') {
      let best = 0;
      for (let e = 1; e < E; e++) if (probs[e] > probs[best]) best = e;
      hardE = best;
      pred = eOut[best];
    } else {
      pred = 0;
      for (let e = 0; e < E; e++) pred += probs[e] * eOut[e];
    }
    const err = pred - ex.t;
    totalLoss += 0.5 * err * err;
    // Track utilisation by hard or soft
    if (MOE.routing === 'top1') counts[hardE]++;
    else for (let e = 0; e < E; e++) counts[e] += probs[e];
    for (let e = 0; e < E; e++) probSum[e] += probs[e];
    // Gradients
    if (MOE.routing === 'top1') {
      // Only the chosen expert gets the gradient (straight-through into gate)
      const g = err;
      dW[hardE][0] += g * ex.x;
      dW[hardE][1] += g * ex.y;
      db[hardE] += g;
      // Straight-through gate: treat hard pick as if we used softmax
      // dL/dlogit_e = err * eOut[e] * probs[e] minus mean adjustment (standard softmax xent-like)
      const sumP = probs.reduce((a, b) => a + b * eOut[b], 0); // weighted output
      for (let e = 0; e < E; e++) {
        const dLogit = err * (eOut[e] - sumP) * probs[e] * 0.5; // smaller signal in top1
        dG[e][0] += dLogit * ex.x;
        dG[e][1] += dLogit * ex.y;
        dgb[e] += dLogit;
      }
    } else {
      // Soft mixture
      // y = sum_e p_e * f_e(x);  err = y - t
      // dL/dw_e = err * p_e * x, dL/db_e = err * p_e
      for (let e = 0; e < E; e++) {
        const g = err * probs[e];
        dW[e][0] += g * ex.x;
        dW[e][1] += g * ex.y;
        db[e] += g;
      }
      // dL/dlogit_e = err * (f_e - sum_e' p_e' f_e') * p_e
      let weightedF = 0;
      for (let e = 0; e < E; e++) weightedF += probs[e] * eOut[e];
      for (let e = 0; e < E; e++) {
        const dLogit = err * (eOut[e] - weightedF) * probs[e];
        dG[e][0] += dLogit * ex.x;
        dG[e][1] += dLogit * ex.y;
        dgb[e] += dLogit;
      }
    }
  }
  // Load-balance auxiliary loss (Switch Transformer style)
  if (MOE.balance) {
    const lam = MOE.lam;
    const f = counts.map((c) => c / B);             // fraction of tokens
    const P = probSum.map((s) => s / B);            // mean gate prob
    // L_aux = E * sum_e f_e * P_e (we minimise it)
    // dL_aux/dP_e = E * f_e (treat f as detached)
    // dP_e / dlogit_j ∝ p_e (delta_ej - p_j) averaged over batch — approximate with gradient toward uniform
    // For simplicity we add a gradient that pushes logits of over-utilised experts down.
    for (let e = 0; e < E; e++) {
      const over = (P[e] - 1 / E) + 0.5 * (f[e] - 1 / E);
      // push gate logits via mean x to penalise over-utilised
      // Approximate by adding a constant negative bias gradient proportional to over
      dgb[e] += lam * E * over * B;
    }
  }
  // Apply
  for (let e = 0; e < E; e++) {
    MOE.experts[e].w[0] -= lr * dW[e][0] / B;
    MOE.experts[e].w[1] -= lr * dW[e][1] / B;
    MOE.experts[e].b -= lr * db[e] / B;
    MOE.G[e][0] -= lr * dG[e][0] / B;
    MOE.G[e][1] -= lr * dG[e][1] / B;
    MOE.gb[e] -= lr * dgb[e] / B;
  }
  MOE.step++;
  MOE.losses.push(totalLoss / B);
  if (MOE.losses.length > 1500) MOE.losses = MOE.losses.slice(-1500);
  // Track utilisation timeseries
  MOE.utilisation.push(counts.map((c) => c / B));
  if (MOE.utilisation.length > 600) MOE.utilisation = MOE.utilisation.slice(-600);
}

// ---------- Render ----------
function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function renderGate() {
  const canvas = document.getElementById('moe-gate');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const step = 4;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = PLANE_MIN + (PLANE_MAX - PLANE_MIN) * (px / W);
      const y = PLANE_MAX - (PLANE_MAX - PLANE_MIN) * (py / H);
      const probs = gateProbs(x, y);
      // Mix colors weighted by probs
      let r = 0, g = 0, b = 0;
      for (let e = 0; e < E; e++) {
        const c = COLORS[e];
        const cr = parseInt(c.slice(1, 3), 16);
        const cg = parseInt(c.slice(3, 5), 16);
        const cb = parseInt(c.slice(5, 7), 16);
        r += probs[e] * cr;
        g += probs[e] * cg;
        b += probs[e] * cb;
      }
      ctx.fillStyle = `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
      ctx.globalAlpha = 0.55;
      ctx.fillRect(px, py, step, step);
      ctx.globalAlpha = 1;
    }
  }
  // Plot training data
  if (MOE.data) {
    MOE.data.forEach((ex) => {
      const px = (ex.x - PLANE_MIN) / (PLANE_MAX - PLANE_MIN) * W;
      const py = (PLANE_MAX - ex.y) / (PLANE_MAX - PLANE_MIN) * H;
      ctx.beginPath();
      ctx.arc(px, py, 1.8, 0, Math.PI * 2);
      ctx.fillStyle = '#1a1815';
      ctx.fill();
    });
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderError() {
  const canvas = document.getElementById('moe-error');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const step = 4;
  const truths = [
    { w: [ 1.5,  0.5], b:  0.4 },
    { w: [-0.8,  1.6], b: -0.5 },
    { w: [-1.2, -0.7], b:  0.7 },
    { w: [ 0.6, -1.3], b: -0.6 }
  ];
  function quad(x, y) {
    if (x >= 0 && y >= 0) return 0;
    if (x <  0 && y >= 0) return 1;
    if (x <  0 && y <  0) return 2;
    return 3;
  }
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = PLANE_MIN + (PLANE_MAX - PLANE_MIN) * (px / W);
      const y = PLANE_MAX - (PLANE_MAX - PLANE_MIN) * (py / H);
      const r = predict(x, y);
      const q = quad(x, y);
      const truth = truths[q].w[0] * x + truths[q].w[1] * y + truths[q].b;
      const err = Math.abs(r.pred - truth);
      const t = Math.min(1, err / 1.5);
      ctx.fillStyle = `rgba(217, 98, 43, ${0.05 + 0.85 * t})`;
      ctx.fillRect(px, py, step, step);
    }
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderLossCanvas() {
  const canvas = document.getElementById('moe-loss-canvas');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 12, t: 14, b: 26 };
  const px = W - m.l - m.r, py = (H - m.t - m.b) / 2 - 8;
  // Top: loss curve
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  if (MOE.losses.length > 1) {
    const lo = Math.min(...MOE.losses);
    const hi = Math.max(...MOE.losses);
    const range = Math.max(0.01, hi - lo);
    ctx.strokeStyle = '#2c6fb7';
    ctx.lineWidth = 2;
    ctx.beginPath();
    MOE.losses.forEach((v, i) => {
      const x = m.l + (i / Math.max(1, MOE.losses.length - 1)) * px;
      const y = m.t + (1 - (v - lo) / range) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('MSE loss', m.l + 4, m.t + 14);
  // Bottom: utilisation stacked
  const by = m.t + py + 18;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, by, px, py);
  if (MOE.utilisation.length > 0) {
    const N = MOE.utilisation.length;
    const cw = px / N;
    for (let i = 0; i < N; i++) {
      const u = MOE.utilisation[i];
      let acc = 0;
      for (let e = 0; e < E; e++) {
        const h = u[e] * py;
        ctx.fillStyle = hexToRgba(COLORS[e], 0.85);
        ctx.fillRect(m.l + i * cw, by + py - acc - h, Math.max(1, cw), h);
        acc += h;
      }
    }
  }
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px Manrope';
  ctx.fillText('expert utilisation (uniform = 25% each)', m.l + 4, by + 14);
}

function refreshStats() {
  document.getElementById('moe-step').textContent = MOE.step;
  document.getElementById('moe-loss').textContent = MOE.losses.length ? MOE.losses[MOE.losses.length - 1].toFixed(4) : '—';
}

function loop() {
  if (!MOE.running) return;
  for (let i = 0; i < 5; i++) trainStep();
  renderGate();
  renderError();
  renderLossCanvas();
  refreshStats();
  MOE.raf = requestAnimationFrame(loop);
}

function wireMoE() {
  resetMoE();
  document.getElementById('moe-routing').addEventListener('change', (e) => {
    MOE.routing = e.target.value;
  });
  document.getElementById('moe-balance').addEventListener('change', (e) => {
    MOE.balance = e.target.checked;
  });
  document.getElementById('moe-lambda').addEventListener('input', (e) => {
    MOE.lam = parseFloat(e.target.value);
    document.getElementById('moe-lambda-val').textContent = MOE.lam.toFixed(2);
  });
  const tog = document.getElementById('moe-toggle');
  tog.addEventListener('click', () => {
    MOE.running = !MOE.running;
    tog.textContent = MOE.running ? 'Pause' : 'Start training';
    if (MOE.running) loop();
    else if (MOE.raf) cancelAnimationFrame(MOE.raf);
  });
  document.getElementById('moe-reset').addEventListener('click', () => {
    if (MOE.raf) cancelAnimationFrame(MOE.raf);
    MOE.running = false;
    tog.textContent = 'Start training';
    MOE.data = makeData(200);
    resetMoE();
    renderGate();
    renderError();
    renderLossCanvas();
    refreshStats();
  });
  renderGate();
  renderError();
  renderLossCanvas();
  refreshStats();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-moe':
      'y = \\sum_{e=1}^{E}\\, g_e(x)\\, f_e(x), \\qquad g(x) = \\mathrm{softmax}(W_g x + b_g)',
    'math-aux':
      '\\mathcal{L}_{\\text{bal}} = \\lambda \\cdot E \\,\\sum_{e=1}^{E} f_e \\cdot P_e'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

function init() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wireMoE();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
