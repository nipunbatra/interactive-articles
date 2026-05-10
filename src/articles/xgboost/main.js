// ============================================================
// XGBoost / Gradient Boosting on a 1D regression toy.
// We do simple GBM with squared loss; depth-limited stumps fit residuals.
// ============================================================

const STATE = {
  data: null, trees: [], lr: 0.3, depth: 3, lossHistory: []
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

function trueFn(x) { return Math.sin(2 * x) + 0.4 * Math.cos(0.6 * x); }

function newData(n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const x = -3 + 6 * Math.random();
    out.push({ x, y: trueFn(x) + 0.2 * randn() });
  }
  return out;
}

// 1D regression tree (squared loss, depth-limited)
function buildTree1D(points, depth, maxDepth) {
  const n = points.length;
  if (n === 0) return { leaf: true, val: 0, n };
  const ymean = points.reduce((s, p) => s + p.y, 0) / n;
  if (depth >= maxDepth || n < 2) return { leaf: true, val: ymean, n };
  // Find best split on x
  const sorted = points.slice().sort((a, b) => a.x - b.x);
  let bestGain = -Infinity, bestSplit = null;
  const totalSum = sorted.reduce((s, p) => s + p.y, 0);
  let leftSum = 0;
  for (let i = 1; i < sorted.length; i++) {
    leftSum += sorted[i - 1].y;
    if (sorted[i].x === sorted[i - 1].x) continue;
    const nL = i, nR = sorted.length - i;
    const meanL = leftSum / nL;
    const meanR = (totalSum - leftSum) / nR;
    const gain = nL * meanL * meanL + nR * meanR * meanR;
    if (gain > bestGain) {
      bestGain = gain;
      bestSplit = { thresh: 0.5 * (sorted[i - 1].x + sorted[i].x), nL, nR };
    }
  }
  if (!bestSplit) return { leaf: true, val: ymean, n };
  const left = points.filter((p) => p.x <= bestSplit.thresh);
  const right = points.filter((p) => p.x > bestSplit.thresh);
  return {
    leaf: false, thresh: bestSplit.thresh, n,
    left: buildTree1D(left, depth + 1, maxDepth),
    right: buildTree1D(right, depth + 1, maxDepth)
  };
}
function tree1DPredict(t, x) {
  if (t.leaf) return t.val;
  return x <= t.thresh ? tree1DPredict(t.left, x) : tree1DPredict(t.right, x);
}

function ensemblePredict(trees, x) {
  let s = 0;
  for (const t of trees) s += STATE.lr * tree1DPredict(t, x);
  return s;
}

function rmse(data, predFn) {
  let s = 0;
  for (const p of data) s += (p.y - predFn(p.x)) ** 2;
  return Math.sqrt(s / data.length);
}

function addTree() {
  // Compute residuals
  const residuals = STATE.data.map((p) => ({ x: p.x, y: p.y - ensemblePredict(STATE.trees, p.x) }));
  const tree = buildTree1D(residuals, 0, STATE.depth);
  STATE.trees.push(tree);
  STATE.lossHistory.push(rmse(STATE.data, (x) => ensemblePredict(STATE.trees, x)));
}

// ---------- Comparators ----------
let CMP = { tree: null, forest: null };
function buildComparators() {
  CMP.tree = buildTree1D(STATE.data, 0, 8);
  CMP.forest = [];
  for (let i = 0; i < 25; i++) {
    const sample = [];
    for (let j = 0; j < STATE.data.length; j++) sample.push(STATE.data[Math.floor(Math.random() * STATE.data.length)]);
    CMP.forest.push(buildTree1D(sample, 0, 6));
  }
}
function rfPredict(x) {
  let s = 0;
  for (const t of CMP.forest) s += tree1DPredict(t, x);
  return s / CMP.forest.length;
}

// ---------- Render ----------
function renderFit() {
  const canvas = document.getElementById('xg-fit');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const xMin = -3.2, xMax = 3.2, yMin = -2, yMax = 2;
  const sx = (x) => m.l + (x - xMin) / (xMax - xMin) * px;
  const sy = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * py;
  // Y ticks
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = -2; v <= 2; v++) {
    const y = sy(v);
    ctx.fillText(v.toString(), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  // True curve
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (xMax - xMin) * (i / 200);
    if (i === 0) ctx.moveTo(sx(x), sy(trueFn(x))); else ctx.lineTo(sx(x), sy(trueFn(x)));
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // Latest tree (orange dashed) — show its standalone contribution scaled by lr
  if (STATE.trees.length > 0) {
    ctx.strokeStyle = 'rgba(217, 98, 43, 0.7)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    const lastTree = STATE.trees[STATE.trees.length - 1];
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (xMax - xMin) * (i / 200);
      const y = STATE.lr * tree1DPredict(lastTree, x);
      if (i === 0) ctx.moveTo(sx(x), sy(y)); else ctx.lineTo(sx(x), sy(y));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  // Ensemble (teal solid)
  ctx.strokeStyle = '#1e7770'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (xMax - xMin) * (i / 200);
    const y = ensemblePredict(STATE.trees, x);
    if (i === 0) ctx.moveTo(sx(x), sy(y)); else ctx.lineTo(sx(x), sy(y));
  }
  ctx.stroke();
  // Data
  STATE.data.forEach((p) => {
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(p.y), 2.4, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1815';
    ctx.fill();
  });
  // Legend
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('truth (dashed) · ensemble (teal) · latest tree contribution (orange dashed)', m.l + 8, m.t + 14);
}

function renderResidual() {
  const canvas = document.getElementById('xg-residual');
  if (!canvas) return;
  const W = 880, H = 200;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const xMin = -3.2, xMax = 3.2;
  const yMin = -1, yMax = 1;
  const sx = (x) => m.l + (x - xMin) / (xMax - xMin) * px;
  const sy = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * py;
  // 0 line
  ctx.strokeStyle = '#9a917f';
  ctx.beginPath(); ctx.moveTo(m.l, sy(0)); ctx.lineTo(m.l + px, sy(0)); ctx.stroke();
  // residuals
  STATE.data.forEach((p) => {
    const r = p.y - ensemblePredict(STATE.trees, p.x);
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(Math.max(yMin, Math.min(yMax, r))), 2, 0, Math.PI * 2);
    ctx.fillStyle = '#d9622b'; ctx.fill();
  });
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('residual = y - ensemble(x)', m.l + 8, m.t + 14);
}

function renderLossCurve() {
  const canvas = document.getElementById('xg-loss');
  if (!canvas) return;
  const W = 880, H = 200;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  if (STATE.lossHistory.length === 0) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText('Press Add tree to start.', m.l + px / 2, m.t + py / 2);
    return;
  }
  const lo = 0;
  const hi = Math.max(STATE.lossHistory[0], 0.01) * 1.05;
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
  ctx.strokeStyle = '#2c6fb7'; ctx.lineWidth = 2;
  ctx.beginPath();
  STATE.lossHistory.forEach((v, i) => {
    const x = m.l + (i / Math.max(1, STATE.lossHistory.length - 1)) * px;
    const y = m.t + (1 - (v - lo) / (hi - lo)) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderCmp(canvasId, predFn) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = 380, H = 240;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 36, r: 12, t: 14, b: 26 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const xMin = -3.2, xMax = 3.2, yMin = -2, yMax = 2;
  const sx = (x) => m.l + (x - xMin) / (xMax - xMin) * px;
  const sy = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * py;
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  for (let i = 0; i <= 100; i++) {
    const x = xMin + (xMax - xMin) * (i / 100);
    if (i === 0) ctx.moveTo(sx(x), sy(trueFn(x))); else ctx.lineTo(sx(x), sy(trueFn(x)));
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = '#1e7770'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 100; i++) {
    const x = xMin + (xMax - xMin) * (i / 100);
    const y = predFn(x);
    if (i === 0) ctx.moveTo(sx(x), sy(y)); else ctx.lineTo(sx(x), sy(y));
  }
  ctx.stroke();
  STATE.data.forEach((p) => {
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(p.y), 1.6, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1815'; ctx.fill();
  });
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'left';
  ctx.fillText(`RMSE ${rmse(STATE.data, predFn).toFixed(3)}`, m.l + 6, m.t + 14);
}

function renderAll() {
  renderFit();
  renderResidual();
  renderLossCurve();
  if (CMP.tree) renderCmp('xg-cmp-tree', (x) => tree1DPredict(CMP.tree, x));
  if (CMP.forest) renderCmp('xg-cmp-rf', rfPredict);
  renderCmp('xg-cmp-boost', (x) => ensemblePredict(STATE.trees, x));
  document.getElementById('xg-n').textContent = STATE.trees.length;
  document.getElementById('xg-rmse').textContent = STATE.lossHistory.length ? STATE.lossHistory[STATE.lossHistory.length - 1].toFixed(3) : '—';
}

function reset() {
  STATE.trees = [];
  STATE.lossHistory = [];
  STATE.lossHistory.push(rmse(STATE.data, () => 0));
}

function wire() {
  STATE.data = newData(60);
  buildComparators();
  reset();
  document.getElementById('xg-add').addEventListener('click', () => { addTree(); renderAll(); });
  document.getElementById('xg-add10').addEventListener('click', () => { for (let i = 0; i < 10; i++) addTree(); renderAll(); });
  document.getElementById('xg-reset').addEventListener('click', () => { reset(); renderAll(); });
  document.getElementById('xg-newdata').addEventListener('click', () => {
    STATE.data = newData(60); buildComparators(); reset(); renderAll();
  });
  document.getElementById('xg-lr').addEventListener('input', (e) => {
    STATE.lr = parseFloat(e.target.value);
    document.getElementById('xg-lr-val').textContent = STATE.lr.toFixed(2);
    renderAll();
  });
  document.getElementById('xg-depth').addEventListener('input', (e) => {
    STATE.depth = parseInt(e.target.value, 10);
    document.getElementById('xg-depth-val').textContent = STATE.depth;
  });
  renderAll();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-boost':
      '\\hat y_t(x) = \\hat y_{t-1}(x) + \\eta\\, f_t(x),\\qquad f_t \\approx \\arg\\min_f \\sum_i L\\bigl(y_i,\\; \\hat y_{t-1}(x_i) + f(x_i)\\bigr)',
    'math-xgb':
      '\\mathrm{Gain} = \\tfrac{1}{2}\\!\\left[\\frac{(\\sum g_L)^2}{\\sum h_L + \\lambda} + \\frac{(\\sum g_R)^2}{\\sum h_R + \\lambda} - \\frac{(\\sum g)^2}{\\sum h + \\lambda}\\right] - \\gamma,\\qquad w_j = -\\frac{\\sum_{i \\in j} g_i}{\\sum_{i \\in j} h_i + \\lambda}'
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
