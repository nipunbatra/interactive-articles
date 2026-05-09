// ============================================================
// Bayesian Optimization, Step by Step
// 1D GP surrogate + EI / UCB / PI / Thompson acquisition.
// All four strategies run in parallel on the same hidden function.
// ============================================================

const PRIOR_COLOR = '#2c6fb7';
const LIK_COLOR = '#d9622b';
const POST_COLOR = '#1e7770';
const TICK_COLOR = '#9a917f';

const X_MIN = 0, X_MAX = 6;

const HIDDEN = {
  sine: (x) => 0.6 * Math.sin(1.6 * x) + 0.5 * Math.exp(-0.5 * (x - 4) * (x - 4)),
  bimodal: (x) => 0.9 * Math.exp(-1.5 * (x - 1.4) * (x - 1.4))
                + 1.1 * Math.exp(-2.0 * (x - 4.3) * (x - 4.3))
                - 0.4,
  sawtooth: (x) => Math.abs(((x * 1.5) % 2) - 1) - 0.4
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

// ---------- GP (RBF kernel, fixed lengthscale, fixed noise) ----------
function rbfKernel(x1, x2, l = 0.6, sigF = 1.0) {
  const d = x1 - x2;
  return sigF * sigF * Math.exp(-0.5 * d * d / (l * l));
}

function fitGP(X, Y, l = 0.6, sigF = 1.0, noise = 0.05) {
  const N = X.length;
  if (N === 0) return null;
  const K = new Array(N);
  for (let i = 0; i < N; i++) {
    K[i] = new Array(N);
    for (let j = 0; j < N; j++) {
      K[i][j] = rbfKernel(X[i], X[j], l, sigF);
      if (i === j) K[i][j] += noise * noise;
    }
  }
  // Cholesky
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
  // Solve K alpha = Y -> alpha = L^-T L^-1 Y
  function solveLT(L, b) {
    const x = new Array(N);
    for (let i = 0; i < N; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }
  function solveLT_T(L, b) {
    const x = new Array(N);
    for (let i = N - 1; i >= 0; i--) {
      let s = b[i];
      for (let k = i + 1; k < N; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }
  const a = solveLT(L, Y);
  const alpha = solveLT_T(L, a);
  return { X, Y, L, alpha, l, sigF, noise };
}

function gpPredict(gp, x) {
  if (!gp) return { mean: 0, std: 1.0 };
  const N = gp.X.length;
  const k = new Array(N);
  for (let i = 0; i < N; i++) k[i] = rbfKernel(gp.X[i], x, gp.l, gp.sigF);
  let mean = 0;
  for (let i = 0; i < N; i++) mean += k[i] * gp.alpha[i];
  // v = L^-1 k
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

// ---------- Acquisition functions ----------
function normPdf(z) { return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); }
function normCdf(z) {
  // Abramowitz & Stegun approx
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = normPdf(z);
  const p = d * (0.319381530 * t - 0.356563782 * t * t + 1.781477937 * t * t * t
                  - 1.821255978 * Math.pow(t, 4) + 1.330274429 * Math.pow(t, 5));
  return z >= 0 ? 1 - p : p;
}

function ei(mu, sig, fStar, xi = 0.01) {
  if (sig < 1e-9) return 0;
  const z = (mu - fStar - xi) / sig;
  return (mu - fStar - xi) * normCdf(z) + sig * normPdf(z);
}
function ucb(mu, sig, kappa = 2.0) { return mu + kappa * sig; }
function pi(mu, sig, fStar, xi = 0.01) {
  if (sig < 1e-9) return 0;
  return normCdf((mu - fStar - xi) / sig);
}

// Thompson sampling: draw a sample from the GP posterior on a fine grid; pick its argmax.
function thompsonScores(gp, grid) {
  const N = grid.length;
  const means = new Array(N);
  const stds = new Array(N);
  for (let i = 0; i < N; i++) {
    const r = gpPredict(gp, grid[i]);
    means[i] = r.mean; stds[i] = r.std;
  }
  // Sample independently per grid point (approximation; ignores covariance)
  const out = new Array(N);
  for (let i = 0; i < N; i++) out[i] = means[i] + stds[i] * randn();
  return out;
}

// ---------- State ----------
const STRATS = ['ei', 'ucb', 'pi', 'ts'];
const STRAT_COLORS = {
  ei: '#2c6fb7', ucb: '#d9622b', pi: '#1e7770', ts: '#9b59b6'
};
const STATE = {
  fnKey: 'sine',
  active: 'ei',
  strats: {}
};

function makeStrat() {
  return { X: [], Y: [], gp: null, best: -Infinity, history: [] };
}

function trueFn(x) { return HIDDEN[STATE.fnKey](x); }

function reset() {
  STRATS.forEach((k) => { STATE.strats[k] = makeStrat(); });
  // Initial seed evaluations: 3 points spread across the domain
  const seeds = [1.2, 3.0, 4.8];
  seeds.forEach((x) => {
    const y = trueFn(x) + 0.02 * randn();
    STRATS.forEach((k) => {
      STATE.strats[k].X.push(x);
      STATE.strats[k].Y.push(y);
      STATE.strats[k].best = Math.max(STATE.strats[k].best, y);
      STATE.strats[k].history.push(STATE.strats[k].best);
    });
  });
  STRATS.forEach((k) => {
    STATE.strats[k].gp = fitGP(STATE.strats[k].X, STATE.strats[k].Y);
  });
}

function pickNext(strat, gp, fStar) {
  const N = 200;
  const grid = new Array(N);
  for (let i = 0; i < N; i++) grid[i] = X_MIN + (X_MAX - X_MIN) * i / (N - 1);
  let bestIdx = 0, bestVal = -Infinity;
  if (strat === 'ts') {
    const samples = thompsonScores(gp, grid);
    for (let i = 0; i < N; i++) {
      if (samples[i] > bestVal) { bestVal = samples[i]; bestIdx = i; }
    }
  } else {
    for (let i = 0; i < N; i++) {
      const r = gpPredict(gp, grid[i]);
      let val;
      if (strat === 'ei') val = ei(r.mean, r.std, fStar);
      else if (strat === 'ucb') val = ucb(r.mean, r.std);
      else if (strat === 'pi') val = pi(r.mean, r.std, fStar);
      // tie-break with tiny noise
      val += 1e-9 * Math.random();
      if (val > bestVal) { bestVal = val; bestIdx = i; }
    }
  }
  return grid[bestIdx];
}

function acquireOne() {
  STRATS.forEach((k) => {
    const s = STATE.strats[k];
    if (!s.gp) s.gp = fitGP(s.X, s.Y);
    const xNext = pickNext(k, s.gp, s.best);
    const y = trueFn(xNext) + 0.02 * randn();
    s.X.push(xNext); s.Y.push(y);
    s.best = Math.max(s.best, y);
    s.history.push(s.best);
    s.gp = fitGP(s.X, s.Y);
  });
}

// ---------- Render ----------
function plot(canvas, opts) {
  const w = opts.w, h = opts.h;
  const ctx = setupCanvas(canvas, w, h);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, w, h);
  const m = Object.assign({ l: 50, r: 14, t: 14, b: 28 }, opts.margin || {});
  const px = w - m.l - m.r, py = h - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  return { ctx, m, px, py };
}

function renderSurrogate() {
  const canvas = document.getElementById('bo-surrogate');
  if (!canvas) return;
  const { ctx, m, px, py } = plot(canvas, { w: 880, h: 280 });
  const s = STATE.strats[STATE.active];
  const N = 200;
  const xs = new Array(N), means = new Array(N), stds = new Array(N), trueY = new Array(N);
  for (let i = 0; i < N; i++) {
    xs[i] = X_MIN + (X_MAX - X_MIN) * i / (N - 1);
    const r = gpPredict(s.gp, xs[i]);
    means[i] = r.mean; stds[i] = r.std;
    trueY[i] = trueFn(xs[i]);
  }
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < N; i++) {
    lo = Math.min(lo, means[i] - 2 * stds[i], trueY[i]);
    hi = Math.max(hi, means[i] + 2 * stds[i], trueY[i]);
  }
  for (const y of s.Y) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
  lo -= 0.2; hi += 0.2;
  const sx = (x) => m.l + (x - X_MIN) / (X_MAX - X_MIN) * px;
  const sy = (y) => m.t + (1 - (y - lo) / (hi - lo)) * py;
  // Y ticks
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * (1 - i / 4);
    const y = m.t + i / 4 * py;
    ctx.fillText(v.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  // True function (faint)
  ctx.strokeStyle = 'rgba(0,0,0,0.35)';
  ctx.lineWidth = 1.2;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = sx(xs[i]), y = sy(trueY[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // Confidence band
  ctx.beginPath();
  ctx.fillStyle = 'rgba(30,119,112,0.18)';
  for (let i = 0; i < N; i++) {
    const x = sx(xs[i]), y = sy(means[i] + 2 * stds[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  for (let i = N - 1; i >= 0; i--) {
    const x = sx(xs[i]), y = sy(means[i] - 2 * stds[i]);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  // Mean
  ctx.strokeStyle = POST_COLOR;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = sx(xs[i]), y = sy(means[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // Observations
  for (let i = 0; i < s.X.length; i++) {
    const x = sx(s.X[i]), y = sy(s.Y[i]);
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1815';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  // Best so far marker (horizontal line)
  ctx.strokeStyle = 'rgba(217,98,43,0.6)';
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(m.l, sy(s.best)); ctx.lineTo(m.l + px, sy(s.best));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#9c3f15';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('best so far', m.l + 4, sy(s.best) - 4);

  ctx.fillStyle = '#3b342b';
  ctx.font = '13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('x', m.l + px / 2, m.t + py + 22);
  ctx.save();
  ctx.translate(14, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('f(x)', 0, 0);
  ctx.restore();
}

function renderAcquisition() {
  const canvas = document.getElementById('bo-acquisition');
  if (!canvas) return;
  const { ctx, m, px, py } = plot(canvas, { w: 880, h: 180, margin: { l: 50, r: 14, t: 12, b: 24 } });
  const s = STATE.strats[STATE.active];
  const N = 200;
  const xs = new Array(N), avals = new Array(N);
  for (let i = 0; i < N; i++) {
    xs[i] = X_MIN + (X_MAX - X_MIN) * i / (N - 1);
    const r = gpPredict(s.gp, xs[i]);
    let v;
    if (STATE.active === 'ei') v = ei(r.mean, r.std, s.best);
    else if (STATE.active === 'ucb') v = ucb(r.mean, r.std);
    else if (STATE.active === 'pi') v = pi(r.mean, r.std, s.best);
    else v = r.mean + r.std * 0.5; // approximate
    avals[i] = v;
  }
  let lo = Math.min(...avals);
  let hi = Math.max(...avals);
  if (hi - lo < 1e-6) { hi = lo + 1; }
  const sx = (x) => m.l + (x - X_MIN) / (X_MAX - X_MIN) * px;
  const sy = (y) => m.t + (1 - (y - lo) / (hi - lo)) * py;
  // Fill
  ctx.beginPath();
  ctx.fillStyle = 'rgba(217,98,43,0.18)';
  ctx.moveTo(sx(xs[0]), sy(lo));
  for (let i = 0; i < N; i++) ctx.lineTo(sx(xs[i]), sy(avals[i]));
  ctx.lineTo(sx(xs[N - 1]), sy(lo));
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = LIK_COLOR;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = sx(xs[i]), y = sy(avals[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // Mark argmax
  let bestIdx = 0;
  for (let i = 1; i < N; i++) if (avals[i] > avals[bestIdx]) bestIdx = i;
  ctx.strokeStyle = '#1a1815';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(sx(xs[bestIdx]), m.t);
  ctx.lineTo(sx(xs[bestIdx]), m.t + py);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#3b342b';
  ctx.font = '13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('x', m.l + px / 2, m.t + py + 18);
  ctx.save();
  ctx.translate(14, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('a(x)', 0, 0);
  ctx.restore();
}

function renderConvergence() {
  const canvas = document.getElementById('bo-convergence');
  if (!canvas) return;
  const { ctx, m, px, py } = plot(canvas, { w: 880, h: 200, margin: { l: 50, r: 14, t: 12, b: 28 } });
  const allHist = STRATS.flatMap((k) => STATE.strats[k].history);
  const N = STATE.strats[STRATS[0]].history.length;
  let lo = Math.min(...allHist) - 0.05, hi = Math.max(...allHist) + 0.05;
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * (1 - i / 4);
    const y = m.t + i / 4 * py;
    ctx.fillText(v.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const x = m.l + (i / 5) * px;
    ctx.fillText(Math.round((N - 1) * i / 5), x, m.t + py + 16);
  }
  STRATS.forEach((k) => {
    const data = STATE.strats[k].history;
    ctx.strokeStyle = STRAT_COLORS[k];
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = m.l + (i / Math.max(1, N - 1)) * px;
      const y = m.t + (1 - (v - lo) / (hi - lo)) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  // Legend
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  let lx = m.l + 6, ly = m.t + 14;
  STRATS.forEach((k) => {
    ctx.strokeStyle = STRAT_COLORS[k]; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke();
    ctx.fillStyle = '#3b342b';
    ctx.fillText(k.toUpperCase(), lx + 18, ly + 3);
    lx += 18 + ctx.measureText(k.toUpperCase()).width + 14;
  });
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('queries', m.l + px / 2, m.t + py + 22);
}

function refreshStats() {
  const s = STATE.strats[STATE.active];
  document.getElementById('bo-n').textContent = s.X.length;
  document.getElementById('bo-best').textContent = s.best.toFixed(3);
}

function renderAll() {
  renderSurrogate();
  renderAcquisition();
  renderConvergence();
  refreshStats();
}

function wireBO() {
  reset();
  document.getElementById('bo-acq').addEventListener('change', (e) => {
    STATE.active = e.target.value;
    renderAll();
  });
  document.getElementById('bo-fn').addEventListener('change', (e) => {
    STATE.fnKey = e.target.value;
    reset();
    renderAll();
  });
  document.getElementById('bo-step').addEventListener('click', () => { acquireOne(); renderAll(); });
  document.getElementById('bo-step5').addEventListener('click', () => { for (let i = 0; i < 5; i++) acquireOne(); renderAll(); });
  document.getElementById('bo-reset').addEventListener('click', () => { reset(); renderAll(); });
  renderAll();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-bo':
      '\\begin{aligned} \\text{EI:}\\quad & a(x) = (\\mu(x) - f^* - \\xi)\\,\\Phi(z) + \\sigma(x)\\,\\phi(z) \\\\ \\text{UCB:}\\quad & a(x) = \\mu(x) + \\kappa\\,\\sigma(x) \\\\ \\text{PI:}\\quad & a(x) = \\Phi\\!\\left(\\tfrac{\\mu(x) - f^* - \\xi}{\\sigma(x)}\\right) \\\\ \\text{Thompson:}\\quad & a(x) = \\tilde f(x)\\sim \\mathcal{GP}\\,\\text{posterior} \\end{aligned}'
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
  wireBO();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
