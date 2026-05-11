// ============================================================
// Random Forest — bagging only / RF (m features per split) / Extra-Trees
// on a 2D classification problem, with live tree-by-tree OOB.
// ============================================================

const STATE = {
  data: null,
  forests: { bag: [], rf: [], extra: [] },
  oob: { bag: [], rf: [], extra: [] }, // OOB-vote table per point per forest
  oobHistory: [],     // {n, errBag, errRf, errExtra}
  fiHistory: [],      // {n, fi: [f0, f1] for RF}
  maxDepth: 6, mtry: 1
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

function newData(n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = Math.random() * Math.PI * 2;
    const r = 0.5 + Math.random() * 1.4;
    const cls = Math.floor(t / Math.PI) % 2;
    out.push({ x: r * Math.cos(t) + 0.18 * randn(), y: r * Math.sin(t) + 0.18 * randn(), label: cls });
  }
  return out;
}

function bootstrapIdx(N) {
  const idx = new Array(N);
  for (let i = 0; i < N; i++) idx[i] = Math.floor(Math.random() * N);
  return idx;
}

function impurity(labels) {
  if (labels.length === 0) return 0;
  let n0 = 0;
  for (const l of labels) if (l === 0) n0++;
  const p0 = n0 / labels.length;
  return 1 - p0 * p0 - (1 - p0) * (1 - p0);
}

function bestSplit(points, mtry, randomSplit) {
  if (points.length < 2) return null;
  const baseI = impurity(points.map((p) => p.label));
  // Pick mtry features at random
  const feats = [0, 1];
  for (let i = feats.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [feats[i], feats[j]] = [feats[j], feats[i]];
  }
  const chosen = feats.slice(0, mtry);
  let best = null;
  let totalFI = [0, 0];
  for (const f of chosen) {
    const key = f === 0 ? 'x' : 'y';
    const vals = points.map((p) => p[key]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    if (lo === hi) continue;
    let cand;
    if (randomSplit) {
      const thresh = lo + (hi - lo) * Math.random();
      const left = [], right = [];
      for (const p of points) { (p[key] <= thresh ? left : right).push(p); }
      if (left.length === 0 || right.length === 0) continue;
      const iL = impurity(left.map((p) => p.label));
      const iR = impurity(right.map((p) => p.label));
      const drop = baseI - (left.length / points.length) * iL - (right.length / points.length) * iR;
      cand = { feature: f, thresh, drop, left, right };
    } else {
      const sorted = points.slice().sort((a, b) => a[key] - b[key]);
      for (let i = 1; i < sorted.length; i++) {
        const v0 = sorted[i - 1][key], v1 = sorted[i][key];
        if (v0 === v1) continue;
        const thresh = 0.5 * (v0 + v1);
        const left = [], right = [];
        for (const p of points) { (p[key] <= thresh ? left : right).push(p); }
        if (left.length === 0 || right.length === 0) continue;
        const iL = impurity(left.map((p) => p.label));
        const iR = impurity(right.map((p) => p.label));
        const drop = baseI - (left.length / points.length) * iL - (right.length / points.length) * iR;
        if (!cand || drop > cand.drop) cand = { feature: f, thresh, drop, left, right };
      }
    }
    if (cand && (!best || cand.drop > best.drop)) best = cand;
    if (cand) totalFI[f] += Math.max(0, cand.drop) * points.length;
  }
  return best ? { ...best, fiContrib: totalFI } : null;
}

function buildTree(points, depth, maxDepth, mtry, randomSplit, fi) {
  const labels = points.map((p) => p.label);
  let n0 = 0; for (const l of labels) if (l === 0) n0++;
  const majority = n0 >= labels.length - n0 ? 0 : 1;
  if (depth >= maxDepth || labels.length <= 2 || impurity(labels) < 1e-6) {
    return { leaf: true, label: majority };
  }
  const split = bestSplit(points, mtry, randomSplit);
  if (!split || split.drop <= 0) return { leaf: true, label: majority };
  fi[0] += split.fiContrib[0]; fi[1] += split.fiContrib[1];
  return {
    leaf: false, feature: split.feature, thresh: split.thresh,
    left: buildTree(split.left, depth + 1, maxDepth, mtry, randomSplit, fi),
    right: buildTree(split.right, depth + 1, maxDepth, mtry, randomSplit, fi)
  };
}
function treePredict(t, x, y) {
  if (t.leaf) return t.label;
  const v = t.feature === 0 ? x : y;
  return v <= t.thresh ? treePredict(t.left, x, y) : treePredict(t.right, x, y);
}

function ensembleVote(forest, x, y) {
  if (forest.length === 0) return 0;
  let s = 0;
  for (const tr of forest) s += treePredict(tr.tree, x, y);
  return s >= forest.length / 2 ? 1 : 0;
}

function addTreeAll() {
  const N = STATE.data.length;
  const variants = [
    { name: 'bag', mtry: 2, randomSplit: false },
    { name: 'rf', mtry: STATE.mtry, randomSplit: false },
    { name: 'extra', mtry: 2, randomSplit: true }
  ];
  for (const v of variants) {
    const bIdx = bootstrapIdx(N);
    const used = new Set(bIdx);
    const sample = bIdx.map((i) => STATE.data[i]);
    const fi = [0, 0];
    const tree = buildTree(sample, 0, STATE.maxDepth, v.mtry, v.randomSplit, fi);
    STATE.forests[v.name].push({ tree, oobIdx: [], fi });
    // OOB: for each point not in used, store the prediction (running vote)
    if (!STATE.oob[v.name]) STATE.oob[v.name] = [];
    if (STATE.oob[v.name].length !== N) {
      STATE.oob[v.name] = new Array(N).fill(null).map(() => ({ pos: 0, neg: 0 }));
    }
    for (let i = 0; i < N; i++) {
      if (used.has(i)) continue;
      const p = STATE.data[i];
      const pred = treePredict(tree, p.x, p.y);
      if (pred === 1) STATE.oob[v.name][i].pos++; else STATE.oob[v.name][i].neg++;
    }
  }
  // Compute OOB errors
  function oobErr(name) {
    let err = 0, c = 0;
    for (let i = 0; i < N; i++) {
      const r = STATE.oob[name][i];
      if (r.pos + r.neg === 0) continue;
      const pred = r.pos > r.neg ? 1 : 0;
      if (pred !== STATE.data[i].label) err++;
      c++;
    }
    return c > 0 ? err / c : null;
  }
  STATE.oobHistory.push({
    n: STATE.forests.bag.length,
    bag: oobErr('bag'), rf: oobErr('rf'), extra: oobErr('extra')
  });
  // RF cumulative feature importance
  let fi = [0, 0];
  STATE.forests.rf.forEach((t) => { fi[0] += t.fi[0]; fi[1] += t.fi[1]; });
  const s = fi[0] + fi[1] || 1;
  STATE.fiHistory.push({ n: STATE.forests.rf.length, fi: [fi[0] / s, fi[1] / s] });
}

function renderForestPanel(canvasId, forest) {
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
      const cls = ensembleVote(forest, x, y);
      ctx.fillStyle = cls === 0 ? 'rgba(44,111,183,0.22)' : 'rgba(217,98,43,0.22)';
      ctx.fillRect(px, py, step, step);
    }
  }
  STATE.data.forEach((p) => {
    const px = (p.x - PLANE.xMin) / (PLANE.xMax - PLANE.xMin) * W;
    const py = (PLANE.yMax - p.y) / (PLANE.yMax - PLANE.yMin) * H;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 0 ? '#2c6fb7' : '#d9622b';
    ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.stroke();
  });
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderStatsCanvas() {
  const canvas = document.getElementById('rf-stats');
  if (!canvas) return;
  const W = 880, H = 260;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  // Left side: OOB curves; right side: FI bars
  const m = { l: 50, r: 280, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
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
  const series = [
    { key: 'bag', color: '#9b59b6', label: 'bagging' },
    { key: 'rf', color: '#2c6fb7', label: 'RF' },
    { key: 'extra', color: '#1e7770', label: 'Extra-Trees' }
  ];
  const N = STATE.oobHistory.length;
  series.forEach((s) => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath();
    STATE.oobHistory.forEach((h, i) => {
      if (h[s.key] == null) return;
      const x = m.l + (i / Math.max(1, N - 1)) * px;
      const y = m.t + (1 - Math.min(1, h[s.key])) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('OOB error vs # trees', m.l + 8, m.t + 14);
  let lx = m.l + 130, ly = m.t + 14;
  series.forEach((s) => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke();
    ctx.fillStyle = '#3b342b';
    ctx.fillText(s.label, lx + 18, ly + 4);
    lx += 18 + ctx.measureText(s.label).width + 18;
  });
  // Right: feature importance for RF
  const fi = STATE.fiHistory.length ? STATE.fiHistory[STATE.fiHistory.length - 1].fi : [0.5, 0.5];
  const rx = m.l + px + 30, ry = m.t;
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 12px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('RF feature importance (MDI)', rx, ry + 12);
  const barW = 210;
  ['x', 'y'].forEach((name, i) => {
    const y = ry + 40 + i * 50;
    ctx.fillStyle = '#3b342b';
    ctx.font = '12px Manrope';
    ctx.fillText(`feature ${name}`, rx, y - 6);
    ctx.strokeStyle = '#e2d8c6';
    ctx.strokeRect(rx, y, barW, 20);
    ctx.fillStyle = '#2c6fb7';
    ctx.fillRect(rx, y, fi[i] * barW, 20);
    ctx.fillStyle = '#1a1815';
    ctx.font = '11px IBM Plex Mono';
    ctx.fillText(`${(fi[i] * 100).toFixed(0)}%`, rx + barW + 8, y + 14);
  });
}

function renderAll() {
  renderForestPanel('rf-bag', STATE.forests.bag);
  renderForestPanel('rf-rf', STATE.forests.rf);
  renderForestPanel('rf-extra', STATE.forests.extra);
  renderStatsCanvas();
  document.getElementById('rf-n').textContent = STATE.forests.bag.length;
  const lastOOB = STATE.oobHistory.length ? STATE.oobHistory[STATE.oobHistory.length - 1].rf : null;
  document.getElementById('rf-oob').textContent = lastOOB == null ? '—' : (lastOOB * 100).toFixed(1) + '%';
}

function reset() {
  STATE.forests = { bag: [], rf: [], extra: [] };
  STATE.oob = { bag: [], rf: [], extra: [] };
  STATE.oobHistory = [];
  STATE.fiHistory = [];
}

function wire() {
  STATE.data = newData(100);
  reset();
  document.getElementById('rf-add').addEventListener('click', () => { addTreeAll(); renderAll(); });
  document.getElementById('rf-add10').addEventListener('click', () => { for (let i = 0; i < 10; i++) addTreeAll(); renderAll(); });
  document.getElementById('rf-reset').addEventListener('click', () => { reset(); renderAll(); });
  document.getElementById('rf-newdata').addEventListener('click', () => { STATE.data = newData(100); reset(); renderAll(); });
  document.getElementById('rf-depth').addEventListener('input', (e) => {
    STATE.maxDepth = parseInt(e.target.value, 10);
    document.getElementById('rf-depth-val').textContent = STATE.maxDepth;
  });
  document.getElementById('rf-mtry').addEventListener('input', (e) => {
    STATE.mtry = parseInt(e.target.value, 10);
    document.getElementById('rf-mtry-val').textContent = STATE.mtry;
  });
  renderAll();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-rf':
      '\\hat y(x) = \\mathrm{mode}\\bigl\\{T_b(x)\\bigr\\}_{b=1}^{B},\\qquad T_b\\text{ on bootstrap sample}\\;\\mathcal{B}_b,\\;\\text{splits on a random subset of }m\\le d\\text{ features}',
    'math-oob':
      '\\mathrm{OOB\\;error} = \\frac{1}{N}\\sum_{i=1}^{N} \\mathbb{1}\\!\\left[\\hat y_i^{\\text{OOB}} \\neq y_i\\right],\\quad \\hat y_i^{\\text{OOB}} = \\mathrm{mode}\\bigl\\{T_b(x_i) : i \\notin \\mathcal{B}_b\\bigr\\}'
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
