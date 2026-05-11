// ============================================================
// Conformal Prediction — split + CQR variants live in JS.
// Heteroscedastic 1-D regression: noise grows with |x|. Polynomial
// regression for the point predictor; quantile regression
// approximated with two ±k·σ(x) curves where σ(x) is fit empirically
// for the CQR variant.
// ============================================================

const POST_COLOR = '#1e7770';
const PRIOR_COLOR = '#2c6fb7';
const LIK_COLOR = '#d9622b';
const TICK_COLOR = '#9a917f';

const STATE = {
  data: null,
  alpha: 0.10,
  nCalib: 100,
  variant: 'split',
  history: []
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

function makeData(N, train, calib, test) {
  // x ~ U(-3, 3); y = sin(x) * 0.6 + 0.2 * x + noise; noise ~ N(0, sigma(x)^2);
  // sigma(x) = 0.10 + 0.30 * |x|
  function gen(n) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const x = -3 + 6 * Math.random();
      const sig = 0.10 + 0.30 * Math.abs(x);
      const y = Math.sin(x) * 0.6 + 0.2 * x + sig * randn();
      out.push({ x, y });
    }
    return out;
  }
  return { train: gen(train), calib: gen(calib), test: gen(test) };
}

function fitPoly(data, degree = 4) {
  // Build Vandermonde, solve normal equations with Cholesky
  const N = data.length;
  const D = degree + 1;
  const X = new Array(N);
  for (let i = 0; i < N; i++) {
    X[i] = new Array(D);
    let p = 1;
    for (let d = 0; d < D; d++) { X[i][d] = p; p *= data[i].x; }
  }
  const y = data.map((p) => p.y);
  // X^T X (with ridge)
  const A = new Array(D); for (let i = 0; i < D; i++) A[i] = new Array(D).fill(0);
  for (let r = 0; r < N; r++) for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) A[i][j] += X[r][i] * X[r][j];
  for (let i = 0; i < D; i++) A[i][i] += 0.001;
  const b = new Array(D).fill(0);
  for (let r = 0; r < N; r++) for (let i = 0; i < D; i++) b[i] += X[r][i] * y[r];
  // Cholesky
  const L = new Array(D); for (let i = 0; i < D; i++) L[i] = new Array(D).fill(0);
  for (let i = 0; i < D; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(1e-9, s)) : s / L[j][j];
    }
  }
  const z = new Array(D);
  for (let i = 0; i < D; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * z[k];
    z[i] = s / L[i][i];
  }
  const w = new Array(D);
  for (let i = D - 1; i >= 0; i--) {
    let s = z[i];
    for (let k = i + 1; k < D; k++) s -= L[k][i] * w[k];
    w[i] = s / L[i][i];
  }
  return w;
}

function evalPoly(w, x) {
  let p = 1, y = 0;
  for (let d = 0; d < w.length; d++) { y += w[d] * p; p *= x; }
  return y;
}

function fitSigma(data, w, kSpan = 3) {
  // Fit a quadratic model for residual std as a function of x
  const N = data.length;
  const X = new Array(N), Y = new Array(N);
  for (let i = 0; i < N; i++) {
    const r = data[i].y - evalPoly(w, data[i].x);
    X[i] = data[i].x;
    Y[i] = Math.log(Math.abs(r) + 0.05); // log-residual
  }
  // Quadratic regression on (1, x, x^2)
  const A = [[0,0,0],[0,0,0],[0,0,0]];
  const b = [0,0,0];
  for (let i = 0; i < N; i++) {
    const xi = X[i];
    const phi = [1, xi, xi * xi];
    for (let p = 0; p < 3; p++) {
      for (let q = 0; q < 3; q++) A[p][q] += phi[p] * phi[q];
      b[p] += phi[p] * Y[i];
    }
  }
  for (let i = 0; i < 3; i++) A[i][i] += 0.01;
  // Solve 3x3
  function solve3(A, b) {
    // Cramer's
    function det3(m) { return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]); }
    const D = det3(A);
    const out = new Array(3);
    for (let c = 0; c < 3; c++) {
      const M = A.map((r) => r.slice());
      for (let i = 0; i < 3; i++) M[i][c] = b[i];
      out[c] = det3(M) / D;
    }
    return out;
  }
  const wSig = solve3(A, b);
  return (x) => Math.exp(wSig[0] + wSig[1] * x + wSig[2] * x * x);
}

function quantile(arr, q) {
  const sorted = arr.slice().sort((a, b) => a - b);
  const N = sorted.length;
  // Conformal correction: ceil((1-alpha)(N+1)) / N -> use position
  const idx = Math.min(N - 1, Math.ceil(q * (N + 1)) - 1);
  return sorted[Math.max(0, idx)];
}

function calibrate(data, w, sigmaFn, variant, alpha) {
  if (variant === 'split') {
    const scores = data.map((p) => Math.abs(p.y - evalPoly(w, p.x)));
    return quantile(scores, 1 - alpha);
  }
  // CQR: predict q_lo(x), q_hi(x) as eval ± kσ(x); k chosen via training std
  const k = 1.0; // crude, refined by quantile
  const scores = data.map((p) => {
    const mu = evalPoly(w, p.x);
    const s = sigmaFn(p.x);
    const lo = mu - k * s;
    const hi = mu + k * s;
    return Math.max(lo - p.y, p.y - hi);
  });
  return quantile(scores, 1 - alpha);
}

function intervalAt(x, w, sigmaFn, variant, qHat) {
  const mu = evalPoly(w, x);
  if (variant === 'split') return [mu - qHat, mu + qHat];
  const k = 1.0;
  const s = sigmaFn(x);
  return [mu - k * s - qHat, mu + k * s + qHat];
}

// ---------- Render ----------
function renderCF() {
  const canvas = document.getElementById('cf-canvas');
  if (!canvas) return;
  const W = 880, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 16, b: 32 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const xMin = -3.5, xMax = 3.5;
  const allY = STATE.data.train.concat(STATE.data.calib, STATE.data.test).map((p) => p.y);
  let yMin = Math.min(...allY) - 0.4, yMax = Math.max(...allY) + 0.4;
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
  // Fit
  const w = fitPoly(STATE.data.train.concat(STATE.data.calib.slice(0, Math.floor(STATE.data.calib.length / 2))));
  const sigmaFn = fitSigma(STATE.data.train, w);
  // Use only nCalib calibration points
  const calib = STATE.data.calib.slice(0, STATE.nCalib);
  const qHat = calibrate(calib, w, sigmaFn, STATE.variant, STATE.alpha);
  // Band
  const NN = 240;
  ctx.beginPath();
  ctx.fillStyle = 'rgba(30,119,112,0.18)';
  for (let i = 0; i <= NN; i++) {
    const x = xMin + (xMax - xMin) * (i / NN);
    const [lo, hi] = intervalAt(x, w, sigmaFn, STATE.variant, qHat);
    const xx = sx(x), yy = sy(Math.min(yMax, hi));
    if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
  }
  for (let i = NN; i >= 0; i--) {
    const x = xMin + (xMax - xMin) * (i / NN);
    const [lo, hi] = intervalAt(x, w, sigmaFn, STATE.variant, qHat);
    const xx = sx(x), yy = sy(Math.max(yMin, lo));
    ctx.lineTo(xx, yy);
  }
  ctx.closePath();
  ctx.fill();
  // Mean
  ctx.strokeStyle = PRIOR_COLOR;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= NN; i++) {
    const x = xMin + (xMax - xMin) * (i / NN);
    const xx = sx(x), yy = sy(evalPoly(w, x));
    if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
  }
  ctx.stroke();
  // Train (small grey)
  STATE.data.train.forEach((p) => {
    const xx = sx(p.x), yy = sy(p.y);
    ctx.beginPath();
    ctx.arc(xx, yy, 1.6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.22)';
    ctx.fill();
  });
  // Test points; mark uncovered in red
  let covered = 0; let total = 0;
  STATE.data.test.forEach((p) => {
    const xx = sx(p.x), yy = sy(p.y);
    const [lo, hi] = intervalAt(p.x, w, sigmaFn, STATE.variant, qHat);
    const inside = (p.y >= lo && p.y <= hi);
    total++;
    if (inside) covered++;
    ctx.beginPath();
    ctx.arc(xx, yy, 2.2, 0, Math.PI * 2);
    ctx.fillStyle = inside ? '#1a1815' : '#c03030';
    ctx.fill();
  });
  // Stats
  document.getElementById('cf-target').textContent = `${(100 * (1 - STATE.alpha)).toFixed(1)}%`;
  document.getElementById('cf-emp').textContent = `${(100 * covered / total).toFixed(1)}%`;
  STATE.history.push(covered / total);
  if (STATE.history.length > 80) STATE.history = STATE.history.slice(-80);
  renderCoverage();
}

function renderCoverage() {
  const canvas = document.getElementById('cov-canvas');
  if (!canvas) return;
  const W = 880, H = 120;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 14, b: 22 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = 0.5; v <= 1.0; v += 0.1) {
    const y = m.t + (1 - (v - 0.5) / 0.5) * py;
    ctx.fillText(v.toFixed(1), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  // target line
  const target = 1 - STATE.alpha;
  const yT = m.t + (1 - (target - 0.5) / 0.5) * py;
  ctx.strokeStyle = LIK_COLOR;
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(m.l, yT); ctx.lineTo(m.l + px, yT); ctx.stroke();
  ctx.setLineDash([]);
  // history bars
  const N = STATE.history.length;
  const bw = px / Math.max(N, 1);
  STATE.history.forEach((v, i) => {
    const h = ((v - 0.5) / 0.5) * py;
    ctx.fillStyle = 'rgba(30,119,112,0.55)';
    ctx.fillRect(m.l + i * bw, m.t + py - Math.max(0, h), Math.max(1, bw - 1), Math.max(0, h));
  });
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('coverage', m.l + 4, m.t + 12);
  ctx.fillStyle = LIK_COLOR;
  ctx.fillText(`target = ${target.toFixed(2)}`, m.l + 80, m.t + 12);
}

// ---------- Wire ----------
function refresh() {
  STATE.alpha = parseFloat(document.getElementById('cf-alpha').value);
  STATE.nCalib = parseInt(document.getElementById('cf-n').value, 10);
  STATE.variant = document.getElementById('cf-variant').value;
  document.getElementById('cf-alpha-val').textContent = STATE.alpha.toFixed(2);
  document.getElementById('cf-n-val').textContent = STATE.nCalib;
  if (!STATE.data) STATE.data = makeData(0, 200, 400, 1500);
  // ensure calib is at least nCalib
  if (STATE.data.calib.length < STATE.nCalib) {
    STATE.data = makeData(0, 200, Math.max(400, STATE.nCalib), 1500);
  }
  renderCF();
}

function wire() {
  document.getElementById('cf-resample').addEventListener('click', () => {
    STATE.data = makeData(0, 200, 400, 1500);
    STATE.history = [];
    refresh();
  });
  document.getElementById('cf-alpha').addEventListener('input', refresh);
  document.getElementById('cf-n').addEventListener('input', refresh);
  document.getElementById('cf-variant').addEventListener('change', refresh);
  refresh();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-conf':
      '\\hat q = \\mathrm{Quantile}_{(1-\\alpha)(N+1)/N}\\!\\bigl(\\{\\,|y_i - \\hat f(x_i)|\\,\\}_{i=1}^{N}\\bigr),\\quad C(x) = [\\hat f(x) - \\hat q,\\, \\hat f(x) + \\hat q]',
    'math-aps':
      's_i = \\sum_{c \\in \\pi(x_i,\\,\\le y_i)} \\hat p_c(x_i),\\quad \\hat q = \\mathrm{Quantile}_{1-\\alpha}\\{s_i\\},\\quad C(x) = \\bigl\\{\\,c : \\sum_{c\' \\in \\pi(x,\\,\\le c)} \\hat p_{c\'}(x) \\le \\hat q\\,\\bigr\\}',
    'math-cov-thm':
      '1 - \\alpha \\;\\le\\; \\Pr\\!\\bigl(Y_{\\text{test}} \\in \\hat C(X_{\\text{test}})\\bigr) \\;\\le\\; 1 - \\alpha + \\tfrac{1}{n+1}'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

// ============================================================
// APS classification — 3-class 2D problem; classifier from RBF logreg.
// ============================================================
const APS = { alpha: 0.10, train: null, calib: null, test: null, W: null };

const APS_RBF_CENTERS = [];
for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
  APS_RBF_CENTERS.push([-2.5 + 5 * (c + 0.5) / 4, -2.5 + 5 * (r + 0.5) / 4]);
}
const APS_RBF_GAMMA = 0.7;
function apsPhi(x, y) {
  const out = new Array(APS_RBF_CENTERS.length + 1);
  out[0] = 1;
  for (let i = 0; i < APS_RBF_CENTERS.length; i++) {
    const dx = x - APS_RBF_CENTERS[i][0];
    const dy = y - APS_RBF_CENTERS[i][1];
    out[i + 1] = Math.exp(-APS_RBF_GAMMA * (dx * dx + dy * dy));
  }
  return out;
}
function apsSoftmax(arr) {
  let m = -Infinity;
  for (const v of arr) if (v > m) m = v;
  const exps = arr.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / s);
}
function apsLogits(W, x, y) {
  const f = apsPhi(x, y);
  const C = W.length;
  const out = new Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let d = 0; d < f.length; d++) s += W[c][d] * f[d];
    out[c] = s;
  }
  return out;
}
function apsMakeData(N) {
  const out = [];
  const clusters = [
    { mu: [-1.4, 0.2], cov: 0.7, label: 0 },
    { mu: [1.3, 1.2], cov: 0.65, label: 1 },
    { mu: [0.0, -1.5], cov: 0.8, label: 2 }
  ];
  for (let i = 0; i < N; i++) {
    const c = clusters[i % 3];
    out.push({ x: c.mu[0] + randn() * c.cov, y: c.mu[1] + randn() * c.cov, label: c.label });
  }
  return out;
}
function apsTrain() {
  const C = 3;
  const D = APS_RBF_CENTERS.length + 1;
  let W = [];
  for (let c = 0; c < C; c++) W.push(new Array(D).fill(0).map(() => randn() * 0.05));
  const lr = 0.4, l2 = 0.05;
  for (let it = 0; it < 250; it++) {
    const dW = new Array(C).fill(0).map(() => new Array(D).fill(0));
    for (const ex of APS.train) {
      const f = apsPhi(ex.x, ex.y);
      const probs = apsSoftmax(apsLogits(W, ex.x, ex.y));
      for (let c = 0; c < C; c++) {
        const t = (c === ex.label) ? 1 : 0;
        const g = probs[c] - t;
        for (let d = 0; d < D; d++) dW[c][d] += g * f[d];
      }
    }
    for (let c = 0; c < C; c++) for (let d = 0; d < D; d++) W[c][d] -= lr * (dW[c][d] / APS.train.length + l2 * W[c][d]);
  }
  return W;
}

function apsScore(probs, label) {
  // Score = sum of probabilities of all classes ranked at or above the true class
  const ranked = probs.map((p, c) => ({ p, c })).sort((a, b) => b.p - a.p);
  let s = 0;
  for (const item of ranked) {
    s += item.p;
    if (item.c === label) break;
  }
  return s;
}
function apsCalibrate() {
  const scores = APS.calib.map((ex) => apsScore(apsSoftmax(apsLogits(APS.W, ex.x, ex.y)), ex.label));
  const sorted = scores.slice().sort((a, b) => a - b);
  const N = sorted.length;
  const idx = Math.min(N - 1, Math.ceil((1 - APS.alpha) * (N + 1)) - 1);
  return sorted[Math.max(0, idx)];
}
function apsPredictionSet(W, x, y, qhat) {
  const probs = apsSoftmax(apsLogits(W, x, y));
  const ranked = probs.map((p, c) => ({ p, c })).sort((a, b) => b.p - a.p);
  const set = [];
  let cum = 0;
  for (const item of ranked) {
    set.push(item.c);
    cum += item.p;
    if (cum >= qhat) break;
  }
  return { set, probs };
}

function apsRender() {
  const canvas = document.getElementById('aps-canvas');
  if (!canvas) return;
  const Wd = 880, Hd = 320;
  const ctx = setupCanvas(canvas, Wd, Hd);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, Wd, Hd);
  // Plot test points coloured by set size
  const xMin = -3.2, xMax = 3.2, yMin = -2.6, yMax = 2.6;
  const m = { l: 50, r: 14, t: 18, b: 36 };
  const px = Wd - m.l - m.r, py = Hd - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const sx = (x) => m.l + (x - xMin) / (xMax - xMin) * px;
  const sy = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * py;
  const qhat = apsCalibrate();
  let covered = 0, totalSet = 0;
  // Background heat showing set size
  const step = 4;
  for (let pyy = 0; pyy < py; pyy += step) {
    for (let pxx = 0; pxx < px; pxx += step) {
      const x = xMin + (xMax - xMin) * (pxx / px);
      const yp = yMax - (yMax - yMin) * (pyy / py);
      const r = apsPredictionSet(APS.W, x, yp, qhat);
      const sz = r.set.length;
      const t = (sz - 1) / 2; // 0..1 for sizes 1..3
      ctx.fillStyle = `rgba(217, 98, 43, ${0.05 + 0.45 * t})`;
      ctx.fillRect(m.l + pxx, m.t + pyy, step, step);
    }
  }
  // Test points
  APS.test.forEach((ex) => {
    const r = apsPredictionSet(APS.W, ex.x, ex.y, qhat);
    if (r.set.includes(ex.label)) covered++;
    totalSet += r.set.length;
    const xx = sx(ex.x), yy = sy(ex.y);
    const colors = ['#2c6fb7', '#d9622b', '#1e7770'];
    ctx.beginPath();
    ctx.arc(xx, yy, 2.4, 0, Math.PI * 2);
    ctx.fillStyle = colors[ex.label];
    ctx.fill();
  });
  document.getElementById('aps-target').textContent = `${(100 * (1 - APS.alpha)).toFixed(1)}%`;
  document.getElementById('aps-emp').textContent = `${(100 * covered / APS.test.length).toFixed(1)}%`;
  document.getElementById('aps-size').textContent = (totalSet / APS.test.length).toFixed(2);
}

function apsRefresh() {
  if (!APS.train) {
    APS.train = apsMakeData(200);
    APS.calib = apsMakeData(400);
    APS.test = apsMakeData(800);
    APS.W = apsTrain();
  }
  APS.alpha = parseFloat(document.getElementById('aps-alpha').value);
  document.getElementById('aps-alpha-val').textContent = APS.alpha.toFixed(2);
  apsRender();
}

function wireAPS() {
  const a = document.getElementById('aps-alpha');
  if (!a) return;
  a.addEventListener('input', apsRefresh);
  document.getElementById('aps-resample').addEventListener('click', () => {
    APS.train = apsMakeData(200);
    APS.calib = apsMakeData(400);
    APS.test = apsMakeData(800);
    APS.W = apsTrain();
    apsRefresh();
  });
  apsRefresh();
}

function boot() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wire();
  wireAPS();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
