// ============================================================
// Double Descent — random-feature ridge regression on a 1D curve.
// Train error and test error vs feature count P; show second descent.
// ============================================================

const STATE = {
  N: 40, sigma: 0.3, lam: 1e-4,
  Xtrain: null, Ytrain: null, Xtest: null, Ytest: null
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

function trueFn(x) { return Math.sin(2 * x) + 0.5 * Math.cos(0.7 * x) + 0.4 * x * 0.3; }

function regen() {
  const N = STATE.N;
  STATE.Xtrain = []; STATE.Ytrain = [];
  for (let i = 0; i < N; i++) {
    const x = -3 + 6 * Math.random();
    STATE.Xtrain.push(x);
    STATE.Ytrain.push(trueFn(x) + STATE.sigma * randn());
  }
  STATE.Xtest = []; STATE.Ytest = [];
  for (let i = 0; i < 400; i++) {
    const x = -3 + 6 * (i / 399);
    STATE.Xtest.push(x);
    STATE.Ytest.push(trueFn(x));
  }
}

// Random feature: phi_p(x) = sin(w_p * x + b_p)
function makeFeatures(P) {
  const W = new Array(P), B = new Array(P);
  for (let p = 0; p < P; p++) {
    W[p] = 1.0 * randn();
    B[p] = 2 * Math.PI * Math.random();
  }
  return { W, B };
}
function phi(x, F) {
  const out = new Array(F.W.length);
  for (let p = 0; p < F.W.length; p++) out[p] = Math.sin(F.W[p] * x + F.B[p]);
  return out;
}

// Solve (X^T X + lambda I) w = X^T y via Gauss elimination (small P only)
function ridgeFit(X, Y, lambda) {
  const N = X.length, P = X[0].length;
  // Use min(P, N) approach: when P > N, use kernel form (fits exactly with min-norm)
  if (P <= N + 50) {
    // Normal equations P x P
    const A = []; for (let i = 0; i < P; i++) { const r = new Array(P).fill(0); A.push(r); }
    const b = new Array(P).fill(0);
    for (let n = 0; n < N; n++) {
      const phi = X[n];
      for (let i = 0; i < P; i++) {
        for (let j = 0; j < P; j++) A[i][j] += phi[i] * phi[j];
        b[i] += phi[i] * Y[n];
      }
    }
    for (let i = 0; i < P; i++) A[i][i] += lambda;
    return solveLinear(A, b);
  } else {
    // Kernel form: w = X^T (XX^T + lambda I)^-1 y
    const K = []; for (let i = 0; i < N; i++) K.push(new Array(N).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let p = 0; p < P; p++) s += X[i][p] * X[j][p];
        K[i][j] = s;
      }
      K[i][i] += lambda;
    }
    const alpha = solveLinear(K, Y);
    const w = new Array(P).fill(0);
    for (let p = 0; p < P; p++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += X[n][p] * alpha[n];
      w[p] = s;
    }
    return w;
  }
}
function solveLinear(A, b) {
  const N = A.length;
  // Gaussian elimination with partial pivot
  const M = []; for (let i = 0; i < N; i++) M.push(A[i].concat([b[i]]));
  for (let i = 0; i < N; i++) {
    let max = Math.abs(M[i][i]), pivot = i;
    for (let r = i + 1; r < N; r++) if (Math.abs(M[r][i]) > max) { max = Math.abs(M[r][i]); pivot = r; }
    if (pivot !== i) [M[i], M[pivot]] = [M[pivot], M[i]];
    if (Math.abs(M[i][i]) < 1e-12) M[i][i] = 1e-12;
    for (let r = 0; r < N; r++) if (r !== i) {
      const f = M[r][i] / M[i][i];
      for (let c = i; c <= N; c++) M[r][c] -= f * M[i][c];
    }
  }
  const x = new Array(N);
  for (let i = 0; i < N; i++) x[i] = M[i][N] / M[i][i];
  return x;
}

function trainTestError(P) {
  const F = makeFeatures(P);
  const Xtr = STATE.Xtrain.map((x) => phi(x, F));
  const w = ridgeFit(Xtr, STATE.Ytrain, STATE.lam);
  // Train error
  let tr = 0;
  for (let n = 0; n < STATE.Xtrain.length; n++) {
    let pred = 0;
    for (let p = 0; p < P; p++) pred += Xtr[n][p] * w[p];
    tr += (pred - STATE.Ytrain[n]) ** 2;
  }
  tr = Math.sqrt(tr / STATE.Xtrain.length);
  // Test error
  const Xte = STATE.Xtest.map((x) => phi(x, F));
  let te = 0;
  for (let n = 0; n < STATE.Xtest.length; n++) {
    let pred = 0;
    for (let p = 0; p < P; p++) pred += Xte[n][p] * w[p];
    te += (pred - STATE.Ytest[n]) ** 2;
  }
  te = Math.sqrt(te / STATE.Xtest.length);
  return { tr, te, w, F };
}

// ---------- Render ----------
function renderCurve() {
  const canvas = document.getElementById('dd-curve');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 14, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);

  const Pmax = 4 * STATE.N;
  const Ps = [];
  const trainErrs = [];
  const testErrs = [];
  // Sample 30 P values from 1..Pmax
  for (let i = 1; i <= 30; i++) {
    const P = Math.max(2, Math.round(Pmax * i / 30));
    const r = trainTestError(P);
    Ps.push(P);
    trainErrs.push(r.tr);
    testErrs.push(r.te);
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of trainErrs.concat(testErrs)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  hi = Math.min(hi, lo + 5);
  // ticks
  ctx.fillStyle = '#9a917f';
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
    const v = Math.round(Pmax * i / 5);
    const x = m.l + (i / 5) * px;
    ctx.fillText(v.toString(), x, m.t + py + 16);
  }
  // Threshold
  const Nx = m.l + (STATE.N / Pmax) * px;
  ctx.strokeStyle = 'rgba(217,98,43,0.6)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(Nx, m.t); ctx.lineTo(Nx, m.t + py); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#9c3f15';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('P = N', Nx + 4, m.t + 14);

  function plot(arr, color, dashed) {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [4, 3] : []);
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = m.l + (Ps[i] / Pmax) * px;
      const y = m.t + (1 - (Math.min(hi, v) - lo) / (hi - lo)) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  plot(testErrs, '#2c6fb7', false);
  plot(trainErrs, '#1e7770', true);
  // Legend
  ctx.font = '11px Manrope'; ctx.fillStyle = '#3b342b';
  ctx.textAlign = 'left';
  ctx.fillText('test RMSE', m.l + 8, m.t + 16);
  ctx.fillStyle = '#1e7770';
  ctx.fillText('train RMSE (dashed)', m.l + 90, m.t + 16);
}

function renderFit() {
  const canvas = document.getElementById('dd-fit');
  if (!canvas) return;
  const W = 880, H = 280;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  // Three model sizes
  const sizes = [Math.max(2, Math.round(STATE.N * 0.3)), STATE.N, STATE.N * 4];
  const colors = ['#1e7770', '#d9622b', '#2c6fb7'];
  const labels = ['small', 'P=N', 'very large'];
  const xMin = -3, xMax = 3, yMin = -2, yMax = 2;
  // True fn
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (xMax - xMin) * (i / 200);
    const y = trueFn(x);
    const sx = m.l + (x - xMin) / (xMax - xMin) * px;
    const sy = m.t + (1 - (y - yMin) / (yMax - yMin)) * py;
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // Training points
  STATE.Xtrain.forEach((x, i) => {
    const sx = m.l + (x - xMin) / (xMax - xMin) * px;
    const sy = m.t + (1 - (STATE.Ytrain[i] - yMin) / (yMax - yMin)) * py;
    ctx.beginPath(); ctx.arc(sx, sy, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1815'; ctx.fill();
  });
  // Three fits
  sizes.forEach((P, idx) => {
    const r = trainTestError(P);
    ctx.strokeStyle = colors[idx]; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (xMax - xMin) * (i / 200);
      const f = phi(x, r.F);
      let pred = 0;
      for (let p = 0; p < P; p++) pred += f[p] * r.w[p];
      const sx = m.l + (x - xMin) / (xMax - xMin) * px;
      const sy = m.t + (1 - (Math.max(yMin, Math.min(yMax, pred)) - yMin) / (yMax - yMin)) * py;
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    ctx.fillStyle = colors[idx];
    ctx.font = '11px Manrope';
    ctx.textAlign = 'left';
    ctx.fillText(`${labels[idx]} (P=${P})`, m.l + 6 + idx * 130, m.t + 14);
  });
}

function refresh() {
  STATE.N = parseInt(document.getElementById('dd-N').value, 10);
  STATE.sigma = parseFloat(document.getElementById('dd-sig').value);
  const lamLog = parseFloat(document.getElementById('dd-lam').value);
  STATE.lam = Math.pow(10, lamLog);
  document.getElementById('dd-N-val').textContent = STATE.N;
  document.getElementById('dd-sig-val').textContent = STATE.sigma.toFixed(2);
  document.getElementById('dd-lam-val').textContent = lamLog.toFixed(1);
  if (!STATE.Xtrain || STATE.Xtrain.length !== STATE.N) regen();
  renderCurve();
  renderFit();
}

// ---------- Sample-wise double descent ----------
const SW = { P: 120, sigma: 0.3, lam: 1e-4, F: null, Xfull: null, Yfull: null };
function swRegen() {
  // Make a generous training pool; subsample for each N.
  const Nmax = 300;
  SW.Xfull = []; SW.Yfull = [];
  for (let i = 0; i < Nmax; i++) {
    const x = -3 + 6 * Math.random();
    SW.Xfull.push(x);
    SW.Yfull.push(trueFn(x) + SW.sigma * randn());
  }
  // Fix features so the curve is comparable across N.
  SW.F = makeFeatures(SW.P);
}
function swTestError(N) {
  const Xtr = []; const Ytr = [];
  for (let i = 0; i < N; i++) {
    Xtr.push(phi(SW.Xfull[i], SW.F));
    Ytr.push(SW.Yfull[i]);
  }
  const w = ridgeFit(Xtr, Ytr, SW.lam);
  // Test on 400 clean points.
  let te = 0; const Nt = 400;
  for (let i = 0; i < Nt; i++) {
    const x = -3 + 6 * (i / (Nt - 1));
    const ph = phi(x, SW.F);
    let pred = 0;
    for (let p = 0; p < SW.P; p++) pred += ph[p] * w[p];
    te += (pred - trueFn(x)) ** 2;
  }
  return Math.sqrt(te / Nt);
}
function renderSampleWise() {
  const canvas = document.getElementById('sw-curve');
  if (!canvas) return;
  const W = 880, H = 300;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 14, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  // Sweep N from 4 to 2*P (so we go across the threshold P=N from both sides).
  const Nmax = Math.min(300, 2 * SW.P + 20);
  const Ns = [], errs = [];
  for (let i = 0; i < 40; i++) {
    const N = Math.max(4, Math.round(Nmax * (i + 1) / 40));
    Ns.push(N); errs.push(swTestError(N));
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of errs) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  hi = Math.min(hi, lo + 6);
  ctx.fillStyle = '#9a917f'; ctx.font = '11px IBM Plex Mono'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * (1 - i / 4);
    const y = m.t + i / 4 * py;
    ctx.fillText(v.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const v = Math.round(Nmax * i / 5);
    const x = m.l + (i / 5) * px;
    ctx.fillText(v.toString(), x, m.t + py + 16);
  }
  // Threshold line at N = P
  const Nx = m.l + (SW.P / Nmax) * px;
  ctx.strokeStyle = 'rgba(217,98,43,0.6)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(Nx, m.t); ctx.lineTo(Nx, m.t + py); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#9c3f15'; ctx.font = '11px Manrope'; ctx.textAlign = 'left';
  ctx.fillText('N = P', Nx + 4, m.t + 14);
  // Plot
  ctx.strokeStyle = '#2c6fb7'; ctx.lineWidth = 2;
  ctx.beginPath();
  errs.forEach((v, i) => {
    const x = m.l + (Ns[i] / Nmax) * px;
    const y = m.t + (1 - (Math.min(hi, v) - lo) / (hi - lo)) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // Markers
  errs.forEach((v, i) => {
    const x = m.l + (Ns[i] / Nmax) * px;
    const y = m.t + (1 - (Math.min(hi, v) - lo) / (hi - lo)) * py;
    ctx.fillStyle = '#2c6fb7';
    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  });
  ctx.fillStyle = '#3b342b'; ctx.font = '11px Manrope'; ctx.textAlign = 'left';
  ctx.fillText('test RMSE', m.l + 8, m.t + 16);
  ctx.fillStyle = '#9a917f';
  ctx.fillText('training set size N →', m.l + px - 130, m.t + py + 16);
}
function refreshSW() {
  SW.P = parseInt(document.getElementById('sw-P').value, 10);
  SW.sigma = parseFloat(document.getElementById('sw-sig').value);
  document.getElementById('sw-P-val').textContent = SW.P;
  document.getElementById('sw-sig-val').textContent = SW.sigma.toFixed(2);
  swRegen();
  renderSampleWise();
}

function wire() {
  ['dd-N', 'dd-sig', 'dd-lam'].forEach((id) => {
    document.getElementById(id).addEventListener('input', refresh);
  });
  document.getElementById('dd-resample').addEventListener('click', () => {
    regen();
    refresh();
  });
  refresh();
  // Sample-wise demo
  const swP = document.getElementById('sw-P');
  if (swP) {
    ['sw-P', 'sw-sig'].forEach((id) => {
      document.getElementById(id).addEventListener('input', refreshSW);
    });
    refreshSW();
  }
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-dd':
      '\\hat w = \\arg\\min_w \\,\\|\\Phi w - y\\|^2 + \\lambda\\|w\\|^2,\\qquad \\Phi \\in \\mathbb{R}^{N \\times P}',
    'math-bv':
      '\\mathbb{E}\\,[(\\hat f(x) - f(x))^2] \\;=\\; \\underbrace{(\\mathbb{E}\\hat f - f)^2}_{\\text{bias}^2} \\;+\\; \\underbrace{\\operatorname{Var}(\\hat f)}_{\\text{variance}} \\;+\\; \\sigma^2',
    'math-spike':
      '\\hat w = (\\Phi^\\top\\Phi + \\lambda I)^{-1}\\Phi^\\top y = \\sum_i \\frac{\\sigma_i}{\\sigma_i^2 + \\lambda}\\, v_i\\, (u_i^\\top y)'
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
