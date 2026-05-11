// ============================================================
// Decision Trees — CART for binary classification on a 2D plane.
// Live: tree, random forest of 25, logistic regression side-by-side.
// ============================================================

const STATE = {
  points: [],
  criterion: 'gini',
  maxDepth: 5,
  minLeaf: 1,
  forestTrees: null
};
const PLANE = { xMin: -3, xMax: 3, yMin: -2.2, yMax: 2.2 };

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

// ---------- CART ----------
function impurity(labels, criterion) {
  if (labels.length === 0) return 0;
  let n0 = 0;
  for (const l of labels) if (l === 0) n0++;
  const p0 = n0 / labels.length;
  const p1 = 1 - p0;
  if (criterion === 'gini') return 1 - p0 * p0 - p1 * p1;
  // entropy
  let h = 0;
  if (p0 > 0) h -= p0 * Math.log2(p0);
  if (p1 > 0) h -= p1 * Math.log2(p1);
  return h;
}

function bestSplit(points, criterion, minLeaf) {
  if (points.length < 2) return null;
  const labels = points.map((p) => p.label);
  const baseI = impurity(labels, criterion);
  let best = null;
  for (let f = 0; f < 2; f++) {
    const sorted = points.slice().sort((a, b) => a[f === 0 ? 'x' : 'y'] - b[f === 0 ? 'x' : 'y']);
    for (let i = 1; i < sorted.length; i++) {
      const v0 = sorted[i - 1][f === 0 ? 'x' : 'y'];
      const v1 = sorted[i][f === 0 ? 'x' : 'y'];
      if (v0 === v1) continue;
      const thresh = 0.5 * (v0 + v1);
      const left = [], right = [];
      for (const p of points) {
        if (p[f === 0 ? 'x' : 'y'] <= thresh) left.push(p); else right.push(p);
      }
      if (left.length < minLeaf || right.length < minLeaf) continue;
      const iL = impurity(left.map((p) => p.label), criterion);
      const iR = impurity(right.map((p) => p.label), criterion);
      const drop = baseI - (left.length / points.length) * iL - (right.length / points.length) * iR;
      if (!best || drop > best.drop) {
        best = { feature: f, thresh, drop, left, right };
      }
    }
  }
  return best;
}

function buildTree(points, depth, maxDepth, criterion, minLeaf) {
  const labels = points.map((p) => p.label);
  let n0 = 0;
  for (const l of labels) if (l === 0) n0++;
  const majority = n0 >= points.length - n0 ? 0 : 1;
  if (depth >= maxDepth || points.length <= 2 * minLeaf || impurity(labels, criterion) < 1e-6) {
    return { leaf: true, label: majority, n: points.length };
  }
  const split = bestSplit(points, criterion, minLeaf);
  if (!split || split.drop <= 0) {
    return { leaf: true, label: majority, n: points.length };
  }
  return {
    leaf: false,
    feature: split.feature, thresh: split.thresh, n: points.length,
    left: buildTree(split.left, depth + 1, maxDepth, criterion, minLeaf),
    right: buildTree(split.right, depth + 1, maxDepth, criterion, minLeaf)
  };
}
function treePredict(tree, x, y) {
  if (tree.leaf) return tree.label;
  const v = tree.feature === 0 ? x : y;
  return v <= tree.thresh ? treePredict(tree.left, x, y) : treePredict(tree.right, x, y);
}

// Random forest
function bootstrap(arr) {
  const out = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[Math.floor(Math.random() * arr.length)];
  return out;
}
function buildForest(points, n, maxDepth, criterion, minLeaf) {
  const trees = [];
  for (let i = 0; i < n; i++) {
    const sample = bootstrap(points);
    trees.push(buildTree(sample, 0, maxDepth, criterion, minLeaf));
  }
  return trees;
}
function forestPredict(forest, x, y) {
  let v = 0;
  for (const t of forest) v += treePredict(t, x, y);
  return v >= forest.length / 2 ? 1 : 0;
}

// Logistic regression
function fitLogistic(points) {
  let w = [randn() * 0.1, randn() * 0.1, 0];
  const lr = 0.3;
  for (let it = 0; it < 200; it++) {
    let dw = [0, 0, 0];
    for (const p of points) {
      const z = w[0] * p.x + w[1] * p.y + w[2];
      const pr = 1 / (1 + Math.exp(-z));
      const e = pr - p.label;
      dw[0] += e * p.x; dw[1] += e * p.y; dw[2] += e;
    }
    if (points.length > 0) for (let i = 0; i < 3; i++) w[i] -= lr * dw[i] / points.length;
  }
  return w;
}
function logisticPredict(w, x, y) {
  return (w[0] * x + w[1] * y + w[2]) > 0 ? 1 : 0;
}

// ---------- Render ----------
function renderPanel(canvasId, predictFn) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const step = 4;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = PLANE.xMin + (PLANE.xMax - PLANE.xMin) * (px / W);
      const y = PLANE.yMax - (PLANE.yMax - PLANE.yMin) * (py / H);
      const cls = predictFn(x, y);
      ctx.fillStyle = cls === 0 ? 'rgba(44, 111, 183, 0.22)' : 'rgba(217, 98, 43, 0.22)';
      ctx.fillRect(px, py, step, step);
    }
  }
  STATE.points.forEach((p) => {
    const px = (p.x - PLANE.xMin) / (PLANE.xMax - PLANE.xMin) * W;
    const py = (PLANE.yMax - p.y) / (PLANE.yMax - PLANE.yMin) * H;
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 0 ? '#2c6fb7' : '#d9622b';
    ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.stroke();
  });
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderTreeShape(tree) {
  const canvas = document.getElementById('dt-shape');
  if (!canvas) return;
  const W = 880, H = 240;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  // BFS layout: assign x by index, y by depth
  const levels = [];
  function walk(node, depth) {
    if (!levels[depth]) levels[depth] = [];
    levels[depth].push(node);
    node._depth = depth;
    if (!node.leaf) { walk(node.left, depth + 1); walk(node.right, depth + 1); }
  }
  walk(tree, 0);
  const D = levels.length;
  const margin = 30;
  const rowH = (H - 2 * margin) / Math.max(1, D - 1);
  function renderNode(node, parentX, parentY, x0, x1) {
    const cx = (x0 + x1) / 2;
    const cy = margin + node._depth * rowH;
    if (parentX != null) {
      ctx.strokeStyle = 'rgba(0,0,0,0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(parentX, parentY); ctx.lineTo(cx, cy); ctx.stroke();
    }
    if (node.leaf) {
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = node.label === 0 ? '#2c6fb7' : '#d9622b';
      ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.stroke();
      ctx.fillStyle = '#1a1815';
      ctx.font = '10px IBM Plex Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`n=${node.n}`, cx, cy + 18);
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy, 9, 0, Math.PI * 2);
      ctx.fillStyle = '#fdfcf9';
      ctx.fill();
      ctx.strokeStyle = '#1a1815'; ctx.lineWidth = 1.4; ctx.stroke();
      ctx.fillStyle = '#1a1815';
      ctx.font = '9px IBM Plex Mono';
      ctx.textAlign = 'center';
      ctx.fillText(node.feature === 0 ? 'x' : 'y', cx, cy + 3);
      ctx.fillStyle = '#6e665b';
      ctx.font = '9px IBM Plex Mono';
      ctx.fillText(`≤ ${node.thresh.toFixed(2)}`, cx, cy + 18);
      const mid = (x0 + x1) / 2;
      renderNode(node.left, cx, cy, x0, mid);
      renderNode(node.right, cx, cy, mid, x1);
    }
  }
  renderNode(tree, null, null, margin, W - margin);
}

function renderAll() {
  const tree = buildTree(STATE.points, 0, STATE.maxDepth, STATE.criterion, STATE.minLeaf);
  STATE.forestTrees = STATE.points.length > 0 ? buildForest(STATE.points, 25, STATE.maxDepth, STATE.criterion, STATE.minLeaf) : [];
  const lr = STATE.points.length > 0 ? fitLogistic(STATE.points) : [0, 0, 0];
  renderPanel('dt-tree', (x, y) => treePredict(tree, x, y));
  renderPanel('dt-rf', (x, y) => STATE.forestTrees.length ? forestPredict(STATE.forestTrees, x, y) : 0);
  renderPanel('dt-lr', (x, y) => logisticPredict(lr, x, y));
  renderTreeShape(tree);
}

function wire() {
  const canvas = document.getElementById('dt-tree');
  function add(e, label) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const x = PLANE.xMin + (cx / rect.width) * (PLANE.xMax - PLANE.xMin);
    const y = PLANE.yMax - (cy / rect.height) * (PLANE.yMax - PLANE.yMin);
    STATE.points.push({ x, y, label });
    renderAll();
  }
  canvas.addEventListener('click', (e) => add(e, e.shiftKey ? 1 : 0));
  canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); add(e, 1); });
  document.getElementById('dt-crit').addEventListener('change', (e) => {
    STATE.criterion = e.target.value; renderAll();
  });
  document.getElementById('dt-depth').addEventListener('input', (e) => {
    STATE.maxDepth = parseInt(e.target.value, 10);
    document.getElementById('dt-depth-val').textContent = STATE.maxDepth;
    renderAll();
  });
  document.getElementById('dt-leaf').addEventListener('input', (e) => {
    STATE.minLeaf = parseInt(e.target.value, 10);
    document.getElementById('dt-leaf-val').textContent = STATE.minLeaf;
    renderAll();
  });
  document.getElementById('dt-clear').addEventListener('click', () => {
    STATE.points = []; renderAll();
  });
  document.getElementById('dt-seed').addEventListener('click', () => {
    STATE.points = [];
    // XOR-ish + spirals mix
    for (let i = 0; i < 80; i++) {
      const t = Math.random() * Math.PI * 2;
      const r = 0.5 + Math.random() * 1.4;
      const cls = Math.floor(t / Math.PI) % 2;
      const x = r * Math.cos(t) + 0.2 * randn();
      const y = r * Math.sin(t) + 0.2 * randn();
      STATE.points.push({ x, y, label: cls });
    }
    renderAll();
  });
  renderAll();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-impurity':
      '\\mathrm{Gini}(p) = 1 - p^2 - (1-p)^2,\\qquad H(p) = -p\\log_2 p - (1-p)\\log_2(1-p)',
    'math-mse':
      'I_{\\text{reg}}(\\text{node}) = \\frac{1}{N}\\sum_{i \\in \\text{node}} (y_i - \\bar y_{\\text{node}})^2,\\qquad \\hat y(\\text{leaf}) = \\bar y_{\\text{leaf}}',
    'math-ccp':
      'C_\\alpha(T) = \\sum_{\\ell \\in \\text{leaves}(T)} N_\\ell \\cdot I(\\ell) \\;+\\; \\alpha \\cdot |\\text{leaves}(T)|'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}
// ---------- Information-gain demo (1D) ----------
const IG = { t: 0.5, criterion: 'gini', data: null };
function makeIGDataset() {
  // 60 points along x ∈ [0,1] with two slightly-overlapping clusters
  const pts = [];
  for (let i = 0; i < 30; i++) {
    pts.push({ x: 0.25 + 0.13 * randn(), label: 0 });
  }
  for (let i = 0; i < 30; i++) {
    pts.push({ x: 0.7 + 0.13 * randn(), label: 1 });
  }
  return pts.map((p) => ({ x: Math.max(0, Math.min(1, p.x)), label: p.label }));
}
function impurity1D(labels, criterion) {
  if (labels.length === 0) return 0;
  let n0 = 0; for (const l of labels) if (l === 0) n0++;
  const p0 = n0 / labels.length, p1 = 1 - p0;
  if (criterion === 'gini') return 1 - p0 * p0 - p1 * p1;
  let h = 0;
  if (p0 > 0) h -= p0 * Math.log2(p0);
  if (p1 > 0) h -= p1 * Math.log2(p1);
  return h;
}
function igCurve(data, criterion) {
  const sorted = data.slice().sort((a, b) => a.x - b.x);
  const N = sorted.length;
  const all = sorted.map((p) => p.label);
  const baseI = impurity1D(all, criterion);
  // 200 thresholds spanning [0,1]
  const ts = [], deltas = [];
  for (let k = 0; k <= 200; k++) {
    const t = k / 200;
    const left = [], right = [];
    for (const p of sorted) {
      if (p.x <= t) left.push(p.label); else right.push(p.label);
    }
    if (left.length === 0 || right.length === 0) {
      ts.push(t); deltas.push(0); continue;
    }
    const Il = impurity1D(left, criterion);
    const Ir = impurity1D(right, criterion);
    const d = baseI - (left.length / N) * Il - (right.length / N) * Ir;
    ts.push(t); deltas.push(d);
  }
  return { ts, deltas, baseI };
}
function renderIG() {
  const canvas = document.getElementById('ig-canvas');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 20, t: 20, b: 30 };
  const px = W - m.l - m.r;
  const top = { y: m.t, h: 100 };
  const bot = { y: m.t + 120, h: H - m.t - 120 - m.b };
  // Top panel: data
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, top.y, px, top.h);
  // axis label
  for (let i = 0; i <= 5; i++) {
    const x = m.l + (i / 5) * px;
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(x, top.y); ctx.lineTo(x, top.y + top.h); ctx.stroke();
    ctx.fillStyle = '#9a917f'; ctx.font = '11px IBM Plex Mono'; ctx.textAlign = 'center';
    ctx.fillText((i / 5).toFixed(1), x, top.y + top.h + 14);
  }
  // points
  for (const p of IG.data) {
    const x = m.l + p.x * px;
    const y = top.y + top.h / 2 + (p.label === 0 ? -20 : 20) + 8 * (Math.random() - 0.5);
    ctx.fillStyle = p.label === 0 ? '#2c6fb7' : '#d9622b';
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
  }
  // threshold line on top
  const tx = m.l + IG.t * px;
  ctx.strokeStyle = '#1a1815'; ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(tx, top.y); ctx.lineTo(tx, top.y + top.h); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#1a1815'; ctx.font = '11px Manrope'; ctx.textAlign = 'center';
  ctx.fillText(`t = ${IG.t.toFixed(2)}`, tx, top.y - 4);
  // Bottom panel: IG curve
  const { ts, deltas, baseI } = igCurve(IG.data, IG.criterion);
  let hi = 0; for (const d of deltas) hi = Math.max(hi, d);
  hi = Math.max(0.05, hi);
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, bot.y, px, bot.h);
  ctx.fillStyle = '#9a917f'; ctx.font = '11px IBM Plex Mono'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = hi * (1 - i / 4);
    const y = bot.y + i / 4 * bot.h;
    ctx.fillText(v.toFixed(3), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.strokeStyle = '#1e7770'; ctx.lineWidth = 2;
  ctx.beginPath();
  ts.forEach((t, i) => {
    const x = m.l + t * px;
    const y = bot.y + (1 - deltas[i] / hi) * bot.h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // Current threshold marker
  ctx.strokeStyle = '#1a1815';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(tx, bot.y); ctx.lineTo(tx, bot.y + bot.h); ctx.stroke();
  ctx.setLineDash([]);
  // Current delta value
  const idx = Math.round(IG.t * 200);
  const dCur = deltas[Math.min(200, Math.max(0, idx))];
  ctx.fillStyle = '#1e7770'; ctx.font = '11px Manrope'; ctx.textAlign = 'left';
  ctx.fillText(`Δ = ${dCur.toFixed(4)}`, m.l + 8, bot.y + 16);
  ctx.fillStyle = '#9a917f';
  ctx.fillText(`I(parent) = ${baseI.toFixed(3)}`, m.l + 120, bot.y + 16);
  // Best threshold
  let bestT = 0, bestD = -Infinity;
  for (let i = 0; i < ts.length; i++) if (deltas[i] > bestD) { bestD = deltas[i]; bestT = ts[i]; }
  const bx = m.l + bestT * px;
  ctx.strokeStyle = '#d9622b'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(bx, bot.y); ctx.lineTo(bx, bot.y + bot.h); ctx.stroke();
  ctx.fillStyle = '#d9622b'; ctx.textAlign = 'center';
  ctx.fillText(`argmax t* = ${bestT.toFixed(3)}`, bx, bot.y + bot.h - 6);
}
function refreshIG() {
  IG.t = parseFloat(document.getElementById('ig-t').value);
  IG.criterion = document.getElementById('ig-crit').value;
  document.getElementById('ig-t-val').textContent = IG.t.toFixed(2);
  renderIG();
}
function wireIG() {
  IG.data = makeIGDataset();
  ['ig-t', 'ig-crit'].forEach((id) => {
    document.getElementById(id).addEventListener('input', refreshIG);
  });
  document.getElementById('ig-reseed').addEventListener('click', () => {
    IG.data = makeIGDataset();
    refreshIG();
  });
  refreshIG();
}

function boot() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wire();
  if (document.getElementById('ig-canvas')) wireIG();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
