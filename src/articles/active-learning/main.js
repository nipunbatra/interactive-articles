// ============================================================
// Active Learning, Picked Live
// 3-class 2D classification with a small RBF-features logistic regression
// trained from scratch each acquisition step. Four acquisition strategies
// run in parallel on the same starting set so the user can race them.
// ============================================================

const C = 3;
const COLORS = ['#2c6fb7', '#d9622b', '#1e7770'];
const PLANE_MIN = -3, PLANE_MAX = 3;
const POOL_SIZE = 200;
const SEED_SIZE = 5;
const RBF_CENTERS = []; // 16 RBF centers on a grid
for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
  RBF_CENTERS.push([PLANE_MIN + (PLANE_MAX - PLANE_MIN) * (c + 0.5) / 4,
                    PLANE_MIN + (PLANE_MAX - PLANE_MIN) * (r + 0.5) / 4]);
}
const RBF_GAMMA = 0.6;

function randn() {
  const u1 = Math.random() || 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function phi(x, y) {
  const v = new Array(RBF_CENTERS.length + 1);
  v[0] = 1; // bias
  for (let i = 0; i < RBF_CENTERS.length; i++) {
    const dx = x - RBF_CENTERS[i][0];
    const dy = y - RBF_CENTERS[i][1];
    v[i + 1] = Math.exp(-RBF_GAMMA * (dx * dx + dy * dy));
  }
  return v;
}

function softmax(arr) {
  let m = -Infinity;
  for (const v of arr) if (v > m) m = v;
  const exps = arr.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / s);
}

function predictProbs(W, x, y) {
  const f = phi(x, y);
  const D = f.length;
  const logits = new Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let d = 0; d < D; d++) s += W[c][d] * f[d];
    logits[c] = s;
  }
  return softmax(logits);
}

function trainModel(labels, points, epochs = 80, lr = 0.5, l2 = 0.05, init = null) {
  const D = RBF_CENTERS.length + 1;
  let W;
  if (init) W = init.map((row) => row.slice());
  else {
    W = new Array(C);
    for (let c = 0; c < C; c++) W[c] = new Array(D).fill(0).map(() => randn() * 0.05);
  }
  if (labels.length === 0) return W;
  for (let e = 0; e < epochs; e++) {
    const dW = new Array(C).fill(0).map(() => new Array(D).fill(0));
    for (const li of labels) {
      const p = points[li];
      const f = phi(p.x, p.y);
      const probs = predictProbs(W, p.x, p.y);
      for (let c = 0; c < C; c++) {
        const t = (c === p.label) ? 1 : 0;
        const g = probs[c] - t;
        for (let d = 0; d < D; d++) dW[c][d] += g * f[d];
      }
    }
    for (let c = 0; c < C; c++)
      for (let d = 0; d < D; d++)
        W[c][d] -= lr * (dW[c][d] / labels.length + l2 * W[c][d]);
  }
  return W;
}

// ---------- Pool generation ----------
function generatePool(seed) {
  const arr = [];
  // 3 clusters with overlap
  const clusters = [
    { mu: [-1.6,  0.5], cov: 0.9, label: 0 },
    { mu: [ 1.4,  1.2], cov: 0.7, label: 1 },
    { mu: [ 0.0, -1.4], cov: 0.85, label: 2 }
  ];
  for (let i = 0; i < POOL_SIZE; i++) {
    const c = clusters[i % C];
    arr.push({
      x: c.mu[0] + randn() * c.cov,
      y: c.mu[1] + randn() * c.cov,
      label: c.label
    });
  }
  return arr;
}

const STATE = {
  pool: null,
  test: null,
  initLabels: null,    // shared starting indices
  strategies: {}       // strategy -> { labels: [...indices], W, accs: [...] }
};

function makeStrategyState() {
  return { labels: STATE.initLabels.slice(), W: null, accs: [] };
}

function init() {
  STATE.pool = generatePool();
  STATE.test = generatePool();
  STATE.initLabels = [];
  // Pick 1-2 random per class so we have at least one of each
  const used = new Set();
  for (let c = 0; c < C; c++) {
    let attempts = 0;
    while (attempts < 200) {
      const idx = Math.floor(Math.random() * STATE.pool.length);
      if (!used.has(idx) && STATE.pool[idx].label === c) {
        STATE.initLabels.push(idx); used.add(idx); break;
      }
      attempts++;
    }
  }
  // Pad with random points to SEED_SIZE
  while (STATE.initLabels.length < SEED_SIZE) {
    const idx = Math.floor(Math.random() * STATE.pool.length);
    if (!used.has(idx)) { STATE.initLabels.push(idx); used.add(idx); }
  }
  ['random', 'uncertainty', 'margin', 'committee'].forEach((s) => {
    STATE.strategies[s] = makeStrategyState();
  });
  retrainAll();
}

function retrainAll() {
  Object.keys(STATE.strategies).forEach((k) => {
    const s = STATE.strategies[k];
    s.W = trainModel(s.labels, STATE.pool);
    s.accs.push({ n: s.labels.length, acc: testAcc(s.W) });
  });
}

function testAcc(W) {
  let correct = 0;
  for (const p of STATE.test) {
    const probs = predictProbs(W, p.x, p.y);
    let best = 0;
    for (let c = 1; c < C; c++) if (probs[c] > probs[best]) best = c;
    if (best === p.label) correct++;
  }
  return correct / STATE.test.length;
}

// ---------- Acquisition ----------
function uncertaintyScore(W, x, y) {
  const probs = predictProbs(W, x, y);
  return 1 - Math.max(...probs);
}
function marginScore(W, x, y) {
  const probs = predictProbs(W, x, y).slice().sort((a, b) => b - a);
  return -(probs[0] - probs[1]);
}

function trainEnsemble(labels) {
  const M = 5;
  const out = [];
  for (let m = 0; m < M; m++) {
    const sample = [];
    for (let i = 0; i < labels.length; i++) {
      sample.push(labels[Math.floor(Math.random() * labels.length)]);
    }
    out.push(trainModel(sample, STATE.pool, 60, 0.4, 0.04));
  }
  return out;
}

function committeeScore(committee, x, y) {
  // Vote entropy across the committee
  const counts = new Array(C).fill(0);
  for (const W of committee) {
    const p = predictProbs(W, x, y);
    let best = 0;
    for (let c = 1; c < C; c++) if (p[c] > p[best]) best = c;
    counts[best]++;
  }
  const T = committee.length;
  let H = 0;
  for (let c = 0; c < C; c++) {
    const p = counts[c] / T;
    if (p > 0) H -= p * Math.log2(p);
  }
  return H;
}

function pickAcquisition(strategy) {
  const s = STATE.strategies[strategy];
  const labelled = new Set(s.labels);
  let bestIdx = -1, bestScore = -Infinity;
  if (strategy === 'random') {
    const remaining = [];
    for (let i = 0; i < STATE.pool.length; i++) if (!labelled.has(i)) remaining.push(i);
    bestIdx = remaining[Math.floor(Math.random() * remaining.length)];
  } else if (strategy === 'committee') {
    const committee = trainEnsemble(s.labels);
    for (let i = 0; i < STATE.pool.length; i++) {
      if (labelled.has(i)) continue;
      const p = STATE.pool[i];
      const sc = committeeScore(committee, p.x, p.y) + 1e-6 * Math.random();
      if (sc > bestScore) { bestScore = sc; bestIdx = i; }
    }
  } else {
    const fn = strategy === 'uncertainty' ? uncertaintyScore : marginScore;
    for (let i = 0; i < STATE.pool.length; i++) {
      if (labelled.has(i)) continue;
      const p = STATE.pool[i];
      const sc = fn(s.W, p.x, p.y) + 1e-6 * Math.random();
      if (sc > bestScore) { bestScore = sc; bestIdx = i; }
    }
  }
  return bestIdx;
}

function acquireOne() {
  Object.keys(STATE.strategies).forEach((k) => {
    const s = STATE.strategies[k];
    if (s.labels.length >= STATE.pool.length) return;
    const idx = pickAcquisition(k);
    if (idx >= 0) s.labels.push(idx);
  });
  retrainAll();
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

function renderBoard() {
  const canvas = document.getElementById('al-board');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const strat = document.getElementById('al-strategy').value;
  const s = STATE.strategies[strat];
  // Decision regions (low resolution)
  const step = 4;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = PLANE_MIN + (PLANE_MAX - PLANE_MIN) * (px / W);
      const y = PLANE_MAX - (PLANE_MAX - PLANE_MIN) * (py / H);
      const probs = predictProbs(s.W, x, y);
      let best = 0;
      for (let c = 1; c < C; c++) if (probs[c] > probs[best]) best = c;
      ctx.fillStyle = hexToRgba(COLORS[best], 0.18 + 0.45 * probs[best]);
      ctx.fillRect(px, py, step, step);
    }
  }
  // Pool (small) and labelled (large)
  const labelled = new Set(s.labels);
  STATE.pool.forEach((p, i) => {
    const px = (p.x - PLANE_MIN) / (PLANE_MAX - PLANE_MIN) * W;
    const py = (PLANE_MAX - p.y) / (PLANE_MAX - PLANE_MIN) * H;
    if (labelled.has(i)) {
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[p.label];
      ctx.fill();
      ctx.strokeStyle = '#1a1815';
      ctx.lineWidth = 1.6;
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(px, py, 1.6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.45)';
      ctx.fill();
    }
  });
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderAcq() {
  const canvas = document.getElementById('al-acq');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const strat = document.getElementById('al-strategy').value;
  const s = STATE.strategies[strat];
  // Heatmap of acquisition score on a grid
  const step = 4;
  let lo = Infinity, hi = -Infinity;
  const scores = [];
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = PLANE_MIN + (PLANE_MAX - PLANE_MIN) * (px / W);
      const y = PLANE_MAX - (PLANE_MAX - PLANE_MIN) * (py / H);
      let sc;
      if (strat === 'random') sc = 0.5;
      else if (strat === 'uncertainty') sc = uncertaintyScore(s.W, x, y);
      else if (strat === 'margin') sc = marginScore(s.W, x, y);
      else sc = committeeScore(trainEnsembleFromW(s.W), x, y);
      scores.push({ px, py, sc });
      if (sc < lo) lo = sc;
      if (sc > hi) hi = sc;
    }
  }
  const range = Math.max(1e-6, hi - lo);
  scores.forEach(({ px, py, sc }) => {
    const t = (sc - lo) / range;
    const r = Math.round(253 - 50 * t);
    const g = Math.round(252 - 130 * t);
    const b = Math.round(249 - 150 * t);
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.fillRect(px, py, step, step);
  });
  // Mark the just-acquired point if any
  const lastIdx = s.labels[s.labels.length - 1];
  if (lastIdx != null) {
    const p = STATE.pool[lastIdx];
    const px = (p.x - PLANE_MIN) / (PLANE_MAX - PLANE_MIN) * W;
    const py = (PLANE_MAX - p.y) / (PLANE_MAX - PLANE_MIN) * H;
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.strokeStyle = '#1a1815';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

// trainEnsembleFromW is a cheap proxy: perturb W to get a small committee
function trainEnsembleFromW(W) {
  const M = 4;
  const c = new Array(M);
  for (let i = 0; i < M; i++) {
    c[i] = W.map((row) => row.map((v) => v + randn() * 0.15));
  }
  return c;
}

function renderCurve() {
  const canvas = document.getElementById('al-curve');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 16, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  // Y axis: 0..1 accuracy
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = 0; v <= 4; v++) {
    const acc = v / 4;
    const y = m.t + (1 - acc) * py;
    ctx.fillText(acc.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  // X axis: labels
  const allN = Object.values(STATE.strategies).flatMap((s) => s.accs.map((a) => a.n));
  const maxN = Math.max(SEED_SIZE + 1, ...allN);
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const v = SEED_SIZE + (maxN - SEED_SIZE) * (i / 5);
    const x = m.l + (i / 5) * px;
    ctx.fillText(Math.round(v), x, m.t + py + 16);
  }
  const stratColors = {
    random: '#6e665b',
    uncertainty: '#2c6fb7',
    margin: '#d9622b',
    committee: '#1e7770'
  };
  Object.keys(STATE.strategies).forEach((k) => {
    const data = STATE.strategies[k].accs;
    if (data.length < 1) return;
    ctx.strokeStyle = stratColors[k];
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = m.l + ((d.n - SEED_SIZE) / Math.max(1, maxN - SEED_SIZE)) * px;
      const y = m.t + (1 - d.acc) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  // Legend
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  let lx = m.l + 6, ly = m.t + 14;
  Object.keys(stratColors).forEach((k) => {
    ctx.strokeStyle = stratColors[k]; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke();
    ctx.fillStyle = '#3b342b';
    ctx.fillText(k, lx + 18, ly + 3);
    lx += 18 + ctx.measureText(k).width + 12;
  });
}

function refreshStats() {
  const strat = document.getElementById('al-strategy').value;
  const s = STATE.strategies[strat];
  document.getElementById('al-budget').textContent = s.labels.length;
  const lastAcc = s.accs.length ? s.accs[s.accs.length - 1].acc : 0;
  document.getElementById('al-acc').textContent = (lastAcc * 100).toFixed(1) + '%';
}

function renderAll() {
  renderBoard();
  renderAcq();
  renderCurve();
  refreshStats();
}

function wireAL() {
  init();
  document.getElementById('al-strategy').addEventListener('change', renderAll);
  document.getElementById('al-step').addEventListener('click', () => { acquireOne(); renderAll(); });
  document.getElementById('al-step5').addEventListener('click', () => { for (let i = 0; i < 5; i++) acquireOne(); renderAll(); });
  document.getElementById('al-reset').addEventListener('click', () => { init(); renderAll(); });
  renderAll();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-aq':
      '\\begin{aligned} \\text{Uncertainty:}\\quad & a(x) = 1 - \\max_c p(c\\mid x) \\\\ \\text{Margin:}\\quad & a(x) = -\\bigl(p_{(1)}(x) - p_{(2)}(x)\\bigr) \\\\ \\text{BALD:}\\quad & a(x) = H[\\bar p(\\cdot\\mid x)] - \\mathbb{E}_W H[p(\\cdot\\mid x, W)] \\end{aligned}'
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
  wireAL();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
