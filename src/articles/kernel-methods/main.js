// ============================================================
// Kernel methods — kernel-ridge classifier on a 2D plane,
// + NTK panel comparing finite-width NN vs analytic NTK regressor.
// ============================================================

const STATE = {
  kernel: 'rbf', h: 0.5, d: 3,
  points: [],
  // NTK
  width: 128
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

function kernel(p, q) {
  const dx = p[0] - q[0], dy = p[1] - q[1];
  const d2 = dx * dx + dy * dy;
  const h = STATE.h;
  if (STATE.kernel === 'rbf') return Math.exp(-d2 / (2 * h * h));
  if (STATE.kernel === 'laplace') return Math.exp(-Math.sqrt(d2) / h);
  if (STATE.kernel === 'poly') return Math.pow(p[0] * q[0] + p[1] * q[1] + 1, STATE.d) / Math.pow(STATE.d * 2, STATE.d);
  return p[0] * q[0] + p[1] * q[1]; // linear
}

function fitClassifier() {
  const N = STATE.points.length;
  if (N === 0) return null;
  const K = []; for (let i = 0; i < N; i++) K.push(new Array(N).fill(0));
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) K[i][j] = kernel([STATE.points[i].x, STATE.points[i].y], [STATE.points[j].x, STATE.points[j].y]);
  const lambda = 0.05;
  for (let i = 0; i < N; i++) K[i][i] += lambda;
  const y = STATE.points.map((p) => p.label === 0 ? -1 : 1);
  return { alpha: solveLinear(K, y), N };
}

function predict(model, p) {
  if (!model) return 0;
  let s = 0;
  for (let i = 0; i < model.N; i++) s += model.alpha[i] * kernel([STATE.points[i].x, STATE.points[i].y], p);
  return s;
}

function solveLinear(A, b) {
  const N = A.length;
  const M = []; for (let i = 0; i < N; i++) M.push(A[i].concat([b[i]]));
  for (let i = 0; i < N; i++) {
    let mx = Math.abs(M[i][i]), pivot = i;
    for (let r = i + 1; r < N; r++) if (Math.abs(M[r][i]) > mx) { mx = Math.abs(M[r][i]); pivot = r; }
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

// ---------- Render kernel panel ----------
function renderKernel() {
  const canvas = document.getElementById('km-canvas');
  if (!canvas) return;
  const W = 640, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const xMin = -3, xMax = 3, yMin = -2, yMax = 2;
  const model = fitClassifier();
  const step = 4;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = xMin + (xMax - xMin) * (px / W);
      const y = yMax - (yMax - yMin) * (py / H);
      const v = predict(model, [x, y]);
      const t = Math.tanh(v);
      const r = t > 0 ? Math.round(217 + (44 - 217) * (t)) : Math.round(217 + (-t) * 0);
      // simpler: red->white->blue
      let col;
      if (t > 0) col = `rgba(217, 98, 43, ${0.15 + 0.45 * t})`;
      else col = `rgba(44, 111, 183, ${0.15 + 0.45 * (-t)})`;
      ctx.fillStyle = col;
      ctx.fillRect(px, py, step, step);
    }
  }
  // Points
  STATE.points.forEach((p) => {
    const px = (p.x - xMin) / (xMax - xMin) * W;
    const py = (yMax - p.y) / (yMax - yMin) * H;
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 0 ? '#2c6fb7' : '#d9622b';
    ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.stroke();
  });
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

// ---------- NTK panel ----------
// 1-hidden-layer ReLU NN with width W:
//   f(x) = sum_i a_i * relu(w_i * x + b_i),  a_i ~ N(0, 1/W), w_i, b_i ~ N(0, 1)
// NTK_relu(x, x') analytically (Cho & Saul 2009 / Arora 2019):
//   K(x, x') = (1/π) * (||x|| * ||x'||) * (sin θ + (π - θ) cos θ),
// where cos θ = <x, x'> / (||x|| ||x'||). For 1D inputs this gives a
// scalar NTK we use to fit a regression with kernel ridge.
function ntkKernel(x, xp) {
  const ax = Math.abs(x) + 1e-9;
  const axp = Math.abs(xp) + 1e-9;
  const ipt = x * xp;
  const cos = Math.max(-1, Math.min(1, ipt / (ax * axp)));
  const theta = Math.acos(cos);
  return (1 / Math.PI) * ax * axp * (Math.sin(theta) + (Math.PI - theta) * Math.cos(theta));
}

function fitNTKRegressor(xs, ys) {
  const N = xs.length;
  const K = []; for (let i = 0; i < N; i++) K.push(new Array(N).fill(0));
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) K[i][j] = ntkKernel(xs[i], xs[j]);
  for (let i = 0; i < N; i++) K[i][i] += 0.01;
  const alpha = solveLinear(K, ys);
  return (x) => {
    let s = 0;
    for (let i = 0; i < N; i++) s += alpha[i] * ntkKernel(xs[i], x);
    return s;
  };
}

function makeFiniteNet(width) {
  const ws = new Array(width), bs = new Array(width), as = new Array(width);
  for (let i = 0; i < width; i++) {
    ws[i] = randn(); bs[i] = randn();
    as[i] = randn() / Math.sqrt(width);
  }
  return { ws, bs, as };
}

function netForward(net, x) {
  let s = 0;
  for (let i = 0; i < net.ws.length; i++) s += net.as[i] * Math.max(0, net.ws[i] * x + net.bs[i]);
  return s;
}

function fitFiniteNet(net, xs, ys, lam = 0.01, steps = 800, lr = 0.05) {
  // Gradient descent on a's only (NTK regime essentially), with quadratic loss
  for (let it = 0; it < steps; it++) {
    const da = new Array(net.ws.length).fill(0);
    for (let n = 0; n < xs.length; n++) {
      const phi = new Array(net.ws.length);
      for (let i = 0; i < net.ws.length; i++) phi[i] = Math.max(0, net.ws[i] * xs[n] + net.bs[i]);
      let pred = 0;
      for (let i = 0; i < net.ws.length; i++) pred += net.as[i] * phi[i];
      const e = pred - ys[n];
      for (let i = 0; i < net.ws.length; i++) da[i] += e * phi[i];
    }
    for (let i = 0; i < net.ws.length; i++) net.as[i] -= lr * (da[i] / xs.length + lam * net.as[i]);
  }
}

function renderNTK() {
  const canvas = document.getElementById('ntk-canvas');
  if (!canvas) return;
  const W = 880, H = 280;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 32 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  // Training data: 12 points on sin(2x)
  const xs = [], ys = [];
  for (let i = 0; i < 12; i++) {
    const x = -2 + 4 * (i / 11);
    xs.push(x); ys.push(Math.sin(2 * x) + 0.05 * randn());
  }
  // Fit NTK regressor (analytic)
  const ntk = fitNTKRegressor(xs, ys);
  // Fit finite-width net
  const net = makeFiniteNet(STATE.width);
  fitFiniteNet(net, xs, ys);
  // Plot
  const xMin = -2.2, xMax = 2.2, yMin = -1.5, yMax = 1.5;
  const sx = (x) => m.l + (x - xMin) / (xMax - xMin) * px;
  const sy = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * py;
  // Truth
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (xMax - xMin) * (i / 200);
    const y = Math.sin(2 * x);
    if (i === 0) ctx.moveTo(sx(x), sy(y)); else ctx.lineTo(sx(x), sy(y));
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // Finite net
  ctx.strokeStyle = '#1e7770'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (xMax - xMin) * (i / 200);
    const y = netForward(net, x);
    if (i === 0) ctx.moveTo(sx(x), sy(Math.max(yMin, Math.min(yMax, y)))); else ctx.lineTo(sx(x), sy(Math.max(yMin, Math.min(yMax, y))));
  }
  ctx.stroke();
  // NTK regressor
  ctx.strokeStyle = '#d9622b'; ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (xMax - xMin) * (i / 200);
    const y = ntk(x);
    if (i === 0) ctx.moveTo(sx(x), sy(Math.max(yMin, Math.min(yMax, y)))); else ctx.lineTo(sx(x), sy(Math.max(yMin, Math.min(yMax, y))));
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // Points
  xs.forEach((x, i) => {
    ctx.beginPath(); ctx.arc(sx(x), sy(ys[i]), 3, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1815'; ctx.fill();
  });
  // Legend
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText(`finite NN (width ${STATE.width})`, m.l + 8, m.t + 16);
  ctx.fillText('analytic NTK regressor (dashed)', m.l + 200, m.t + 16);
}

// ---------- Wire ----------
function wire() {
  const canvas = document.getElementById('km-canvas');
  function add(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const xMin = -3, xMax = 3, yMin = -2, yMax = 2;
    const x = xMin + (cx / rect.width) * (xMax - xMin);
    const y = yMax - (cy / rect.height) * (yMax - yMin);
    const label = (e.shiftKey || e.button === 2) ? 1 : 0;
    STATE.points.push({ x, y, label });
    renderKernel();
  }
  canvas.addEventListener('click', (e) => add(e));
  canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); add(e); });
  document.getElementById('km-kernel').addEventListener('change', (e) => {
    STATE.kernel = e.target.value; renderKernel();
  });
  document.getElementById('km-h').addEventListener('input', (e) => {
    STATE.h = parseFloat(e.target.value);
    document.getElementById('km-h-val').textContent = STATE.h.toFixed(2);
    renderKernel();
  });
  document.getElementById('km-d').addEventListener('input', (e) => {
    STATE.d = parseInt(e.target.value, 10);
    document.getElementById('km-d-val').textContent = STATE.d;
    renderKernel();
  });
  document.getElementById('km-clear').addEventListener('click', () => {
    STATE.points = []; renderKernel();
  });
  document.getElementById('km-seed').addEventListener('click', () => {
    STATE.points = [
      { x: -1.2, y: 0.6, label: 0 }, { x: 1.2, y: 0.6, label: 0 },
      { x: -1.2, y: -0.6, label: 1 }, { x: 1.2, y: -0.6, label: 1 },
      { x: -0.5, y: 0, label: 0 }, { x: 0.5, y: 0, label: 1 }
    ];
    renderKernel();
  });
  document.getElementById('ntk-w').addEventListener('input', (e) => {
    STATE.width = parseInt(e.target.value, 10);
    document.getElementById('ntk-w-val').textContent = STATE.width;
    renderNTK();
  });
  document.getElementById('ntk-resample').addEventListener('click', renderNTK);
  renderKernel();
  renderNTK();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-kernel':
      'k(x, x\') = \\phi(x)^\\top \\phi(x\'),\\qquad \\hat f(x) = \\sum_{i} \\alpha_i\\, k(x_i, x)',
    'math-ntk':
      'K_{\\text{NTK}}(x, x\') = \\bigl\\langle \\nabla_\\theta f_\\theta(x),\\; \\nabla_\\theta f_\\theta(x\') \\bigr\\rangle\\Big|_{\\theta = \\theta_0}'
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
