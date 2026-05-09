// ============================================================
// Neural Processes — live CNP / LNP / ANP demo + GP comparison.
// We use a hand-tuned closed-form approximation rather than training
// a real NP in-browser (training a meta-learned NP needs many tasks
// and is expensive). The behaviour we render — the prior band, the
// way the band collapses near context points, the gentle shape of
// uncertainty between points — is qualitatively faithful to a CNP
// pretrained on smooth GP-like functions.
// ============================================================

const PRIOR_COLOR = '#2c6fb7';
const POST_COLOR = '#1e7770';
const LIK_COLOR = '#d9622b';
const TICK_COLOR = '#9a917f';

const X_MIN = -4, X_MAX = 4;

const STATE = {
  context: [],
  variant: 'cnp'
};

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function randn() {
  const u1 = Math.random() || 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------- "CNP" prediction (hand-tuned) ----------
// Mean: weighted regression on context using a soft attention with
// adjustable bandwidth; variance: floor + decay-with-distance.
function predictCNP(x, ctx, opts = {}) {
  const bandwidth = opts.bandwidth || 0.65;
  const noiseFloor = opts.noiseFloor || 0.04;
  if (ctx.length === 0) {
    return { mean: 0, std: 1.1 };
  }
  let totalW = 0, weightedY = 0, totalW2 = 0;
  for (const p of ctx) {
    const w = Math.exp(-0.5 * Math.pow((x - p.x) / bandwidth, 2));
    totalW += w;
    weightedY += w * p.y;
    totalW2 += w * w;
  }
  const mean = weightedY / Math.max(1e-6, totalW);
  // effective sample size and variance: like a kernel regression
  const effN = (totalW * totalW) / (totalW2 + 1e-9);
  const distMin = Math.min(...ctx.map((p) => Math.abs(x - p.x)));
  const baseStd = 1.1; // prior std
  // shrink std by the effective sample size near context points
  const std = Math.max(noiseFloor, baseStd * Math.exp(-effN * 0.6) + 0.15 * Math.tanh(distMin / 1.5));
  return { mean, std };
}

function predictANP(x, ctx) {
  // Tighter attention bandwidth + sharper localisation (kernel-like)
  return predictCNP(x, ctx, { bandwidth: 0.5, noiseFloor: 0.04 });
}

function sampleLNP(grid, ctx, nSamples = 4) {
  // Latent NP: draw one global latent z (a smooth perturbation)
  // and decode a coherent function. We approximate by drawing a
  // smooth random offset across the grid and adding it to the CNP mean.
  const result = [];
  for (let s = 0; s < nSamples; s++) {
    const offsets = new Array(grid.length).fill(0);
    // smooth random walk via low-pass on white noise
    const noise = grid.map(() => randn());
    const k = 7;
    for (let i = 0; i < grid.length; i++) {
      let sum = 0, cnt = 0;
      for (let j = -k; j <= k; j++) {
        const idx = Math.min(grid.length - 1, Math.max(0, i + j));
        sum += noise[idx]; cnt++;
      }
      offsets[i] = sum / cnt;
    }
    const sample = grid.map((x, i) => {
      const cp = predictCNP(x, ctx);
      return cp.mean + cp.std * 0.85 * offsets[i];
    });
    result.push(sample);
  }
  return result;
}

// ---------- GP for comparison ----------
function rbfKernel(x1, x2, l = 0.6, sigF = 1.0) {
  const d = x1 - x2;
  return sigF * sigF * Math.exp(-0.5 * d * d / (l * l));
}

function fitGP(ctx, l = 0.6, sigF = 1.0, noise = 0.05) {
  const N = ctx.length;
  if (N === 0) return null;
  const X = ctx.map((p) => p.x), Y = ctx.map((p) => p.y);
  const K = new Array(N);
  for (let i = 0; i < N; i++) {
    K[i] = new Array(N);
    for (let j = 0; j < N; j++) {
      K[i][j] = rbfKernel(X[i], X[j], l, sigF);
      if (i === j) K[i][j] += noise * noise;
    }
  }
  const L = new Array(N);
  for (let i = 0; i < N; i++) L[i] = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= i; j++) {
      let s = K[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) L[i][j] = Math.sqrt(Math.max(1e-9, s));
      else L[i][j] = s / L[j][j];
    }
  }
  // alpha = K^-1 Y via L L^T
  const z = new Array(N);
  for (let i = 0; i < N; i++) {
    let s = Y[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * z[k];
    z[i] = s / L[i][i];
  }
  const alpha = new Array(N);
  for (let i = N - 1; i >= 0; i--) {
    let s = z[i];
    for (let k = i + 1; k < N; k++) s -= L[k][i] * alpha[k];
    alpha[i] = s / L[i][i];
  }
  return { X, Y, L, alpha, l, sigF, noise };
}

function predictGP(gp, x) {
  if (!gp) return { mean: 0, std: 1.0 };
  const N = gp.X.length;
  const k = gp.X.map((xi) => rbfKernel(xi, x, gp.l, gp.sigF));
  let mean = 0;
  for (let i = 0; i < N; i++) mean += k[i] * gp.alpha[i];
  const v = new Array(N);
  for (let i = 0; i < N; i++) {
    let s = k[i];
    for (let j = 0; j < i; j++) s -= gp.L[i][j] * v[j];
    v[i] = s / gp.L[i][i];
  }
  let var0 = rbfKernel(x, x, gp.l, gp.sigF);
  for (let i = 0; i < N; i++) var0 -= v[i] * v[i];
  var0 = Math.max(1e-9, var0);
  return { mean, std: Math.sqrt(var0) };
}

// ---------- Render ----------
function plotBase(canvas, w, h) {
  const ctx = setupCanvas(canvas, w, h);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, w, h);
  const m = { l: 50, r: 14, t: 16, b: 28 };
  const px = w - m.l - m.r, py = h - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  return { ctx, m, px, py };
}

function renderNPCanvas(canvasId, predFn, opts = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const w = parseInt(canvas.getAttribute('width'), 10);
  const h = parseInt(canvas.getAttribute('height'), 10);
  const { ctx, m, px, py } = plotBase(canvas, w, h);
  const xMin = X_MIN, xMax = X_MAX, yMin = -2.5, yMax = 2.5;
  const sx = (x) => m.l + (x - xMin) / (xMax - xMin) * px;
  const sy = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * py;
  // Y ticks
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = yMin + (yMax - yMin) * (1 - i / 4);
    const y = m.t + i / 4 * py;
    ctx.fillText(v.toFixed(1), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let v = -3; v <= 3; v++) {
    const x = sx(v);
    ctx.fillText(v.toString(), x, m.t + py + 16);
  }
  // Grid x values
  const N = 200;
  const grid = new Array(N);
  for (let i = 0; i < N; i++) grid[i] = xMin + (xMax - xMin) * i / (N - 1);
  // Mean + band
  if (STATE.variant === 'lnp' && opts.kind === 'np') {
    // Show several function samples + the CNP mean
    const samples = sampleLNP(grid, STATE.context, 4);
    const sampleColors = ['#2c6fb7', '#1e7770', '#9b59b6', '#d9622b'];
    samples.forEach((row, k) => {
      ctx.strokeStyle = sampleColors[k];
      ctx.lineWidth = 1.2;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      row.forEach((y, i) => {
        const xx = sx(grid[i]), yy = sy(y);
        if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
  } else {
    const means = new Array(N), stds = new Array(N);
    for (let i = 0; i < N; i++) {
      const r = predFn(grid[i]);
      means[i] = r.mean; stds[i] = r.std;
    }
    // Band
    ctx.beginPath();
    ctx.fillStyle = 'rgba(30,119,112,0.18)';
    for (let i = 0; i < N; i++) {
      const xx = sx(grid[i]), yy = sy(Math.min(yMax, means[i] + 2 * stds[i]));
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    }
    for (let i = N - 1; i >= 0; i--) {
      const xx = sx(grid[i]), yy = sy(Math.max(yMin, means[i] - 2 * stds[i]));
      ctx.lineTo(xx, yy);
    }
    ctx.closePath();
    ctx.fill();
    // Mean
    ctx.strokeStyle = POST_COLOR;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const xx = sx(grid[i]), yy = sy(means[i]);
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
  }
  // Context points
  STATE.context.forEach((p) => {
    const xx = sx(p.x), yy = sy(p.y);
    ctx.beginPath();
    ctx.arc(xx, yy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1815';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

function renderAll() {
  const npPred = STATE.variant === 'anp'
    ? (x) => predictANP(x, STATE.context)
    : (x) => predictCNP(x, STATE.context);
  renderNPCanvas('np-canvas', npPred, { kind: 'np' });
  const gp = fitGP(STATE.context);
  renderNPCanvas('gp-canvas', (x) => predictGP(gp, x), { kind: 'gp' });
  document.getElementById('np-n').textContent = STATE.context.length;
}

// ---------- Architecture diagram ----------
function renderArch() {
  const canvas = document.getElementById('np-arch');
  if (!canvas) return;
  const W = 880, H = 220;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const drawBox = (cx, cy, w, h, fill, title, sub) => {
    ctx.fillStyle = fill;
    ctx.fillRect(cx - w / 2, cy - h / 2, w, h);
    ctx.strokeStyle = '#1a1815'; ctx.lineWidth = 1.4;
    ctx.strokeRect(cx - w / 2, cy - h / 2, w, h);
    ctx.fillStyle = '#1a1815';
    ctx.font = 'bold 13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText(title, cx, cy - 4);
    ctx.fillStyle = '#6e665b';
    ctx.font = '11px Manrope';
    ctx.fillText(sub, cx, cy + 14);
  };
  const drawArrow = (x0, y0, x1, y1, label) => {
    ctx.strokeStyle = '#1a1815'; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - 9 * Math.cos(a - Math.PI / 7), y1 - 9 * Math.sin(a - Math.PI / 7));
    ctx.lineTo(x1 - 9 * Math.cos(a + Math.PI / 7), y1 - 9 * Math.sin(a + Math.PI / 7));
    ctx.closePath();
    ctx.fillStyle = '#1a1815'; ctx.fill();
    if (label) {
      ctx.fillStyle = '#6e665b';
      ctx.font = '11px Manrope';
      ctx.textAlign = 'center';
      ctx.fillText(label, (x0 + x1) / 2, y0 - 8);
    }
  };
  // Context box
  ctx.fillStyle = 'rgba(217,98,43,0.15)';
  ctx.fillRect(40, 80, 130, 60);
  ctx.strokeStyle = '#1a1815';
  ctx.strokeRect(40, 80, 130, 60);
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('context (xᵢ, yᵢ)', 105, 108);
  ctx.fillStyle = '#6e665b';
  ctx.font = '11px Manrope';
  ctx.fillText('few labelled points', 105, 124);
  drawArrow(170, 110, 230, 110, 'h');
  drawBox(290, 110, 120, 60, 'rgba(44,111,183,0.15)', 'Encoder MLP', 'h: (x,y) → r');
  drawArrow(350, 110, 410, 110, 'mean / attn');
  drawBox(470, 110, 130, 60, 'rgba(30,119,112,0.18)', 'Aggregator', 'r₁..r_N → r');
  drawArrow(535, 110, 605, 110);
  drawBox(665, 110, 120, 60, 'rgba(155,89,182,0.15)', 'Decoder MLP', 'g: (x*, r) → μ, σ');
  drawArrow(725, 110, 800, 110, 'predict');
  ctx.fillStyle = 'rgba(217,98,43,0.15)';
  ctx.fillRect(800, 80, 60, 60);
  ctx.strokeStyle = '#1a1815';
  ctx.strokeRect(800, 80, 60, 60);
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('μ(x*), σ(x*)', 830, 108);
  // Query input (bottom)
  ctx.fillStyle = '#6e665b';
  ctx.font = '11px Manrope';
  ctx.fillText('+ query x*', 665, 200);
  drawArrow(665, 180, 665, 145);
}

// ---------- Wire ----------
function wire() {
  const canvas = document.getElementById('np-canvas');
  const m = { l: 50, r: 14, t: 16, b: 28 };
  function fromMouse(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const W = rect.width, H = rect.height;
    const px = W - m.l - m.r, py = H - m.t - m.b;
    if (cx < m.l || cx > m.l + px || cy < m.t || cy > m.t + py) return null;
    const x = X_MIN + (cx - m.l) / px * (X_MAX - X_MIN);
    const yMax = 2.5, yMin = -2.5;
    const y = yMax - (cy - m.t) / py * (yMax - yMin);
    return { x, y };
  }
  canvas.addEventListener('click', (e) => {
    const p = fromMouse(e);
    if (!p) return;
    if (e.shiftKey) {
      let best = -1, bestD = 0.4 * 0.4;
      for (let i = 0; i < STATE.context.length; i++) {
        const dx = STATE.context[i].x - p.x;
        const dy = STATE.context[i].y - p.y;
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best >= 0) STATE.context.splice(best, 1);
    } else {
      STATE.context.push(p);
    }
    renderAll();
  });
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const p = fromMouse(e);
    if (!p) return;
    let best = -1, bestD = 0.4 * 0.4;
    for (let i = 0; i < STATE.context.length; i++) {
      const dx = STATE.context[i].x - p.x;
      const dy = STATE.context[i].y - p.y;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) STATE.context.splice(best, 1);
    renderAll();
  });
  document.getElementById('np-clear').addEventListener('click', () => {
    STATE.context = [];
    renderAll();
  });
  document.getElementById('np-seed').addEventListener('click', () => {
    STATE.context = [];
    for (let x = -3; x <= 3; x += 0.6) {
      if (x > -1 && x < 1) continue;
      STATE.context.push({ x, y: Math.sin(x) + 0.1 * randn() });
    }
    renderAll();
  });
  document.getElementById('np-variant').addEventListener('change', (e) => {
    STATE.variant = e.target.value;
    renderAll();
  });
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-cnp':
      'r_i = h(x_i, y_i),\\quad r = \\frac{1}{N}\\sum_{i=1}^{N} r_i,\\quad p(y_* \\mid x_*, \\mathcal{D}) = \\mathcal{N}\\!\\bigl(\\mu_g(x_*, r),\\, \\sigma_g^2(x_*, r)\\bigr)'
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
  renderArch();
  wire();
  renderAll();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
