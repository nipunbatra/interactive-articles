// ============================================================
// Graph Neural Networks — message passing on a 14-node graph.
// ============================================================

const N = 14;
const STATE = {
  nodes: null,        // [{x, y, feat, label?}]
  edges: null,
  K: 2,
  agg: 'mean',
  classifier: null
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

function makeGraph() {
  const W = 880, H = 440;
  const nodes = [];
  // Place nodes via deterministic-ish layout
  for (let i = 0; i < N; i++) {
    const angle = 2 * Math.PI * i / N + 0.2 * Math.sin(i);
    const r = 150 + 50 * Math.sin(i * 1.3);
    nodes.push({
      x: W / 2 + r * Math.cos(angle),
      y: H / 2 + r * Math.sin(angle),
      feat: 0,
      label: -1
    });
  }
  // Random close-neighbour edges (kNN, k=3)
  const edges = [];
  for (let i = 0; i < N; i++) {
    const dists = [];
    for (let j = 0; j < N; j++) if (i !== j) {
      const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
      dists.push({ j, d: Math.hypot(dx, dy) });
    }
    dists.sort((a, b) => a.d - b.d);
    for (let k = 0; k < 3; k++) {
      const j = dists[k].j;
      const key1 = i + '-' + j, key2 = j + '-' + i;
      if (!edges.some((e) => e.key === key1 || e.key === key2)) {
        edges.push({ a: i, b: j, key: key1 });
      }
    }
  }
  // Assign 4 labels: 2 of class 0, 2 of class 1, by spatial cluster
  const labelled = [0, 5, 7, 12];
  labelled.forEach((idx, k) => { nodes[idx].label = k % 2; });
  return { nodes, edges };
}

function neighbours(i) {
  const out = [];
  STATE.edges.forEach((e) => {
    if (e.a === i) out.push(e.b);
    if (e.b === i) out.push(e.a);
  });
  return out;
}

function messagePass(featuresIn, agg) {
  const out = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    const neigh = neighbours(i);
    if (neigh.length === 0) { out[i] = featuresIn[i]; continue; }
    let val;
    if (agg === 'mean') {
      let s = featuresIn[i];
      for (const j of neigh) s += featuresIn[j];
      val = s / (neigh.length + 1);
    } else if (agg === 'max') {
      val = featuresIn[i];
      for (const j of neigh) val = Math.max(val, featuresIn[j]);
    } else { // attn (softmax over scaled feature similarity)
      // weights = softmax over (feat_i + feat_j) — toy "self-attentional"
      const logits = neigh.map((j) => featuresIn[i] + featuresIn[j]);
      // include self
      logits.push(featuresIn[i] * 2);
      const m = Math.max(...logits);
      const exps = logits.map((v) => Math.exp(v - m));
      const Z = exps.reduce((a, b) => a + b, 0);
      let s = 0;
      neigh.forEach((j, k) => s += exps[k] / Z * featuresIn[j]);
      s += exps[exps.length - 1] / Z * featuresIn[i];
      val = s;
    }
    out[i] = val;
  }
  return out;
}

function propagate() {
  let f = STATE.nodes.map((n) => n.feat);
  for (let k = 0; k < STATE.K; k++) f = messagePass(f, STATE.agg);
  return f;
}

// ---------- Simple node classifier (logistic on propagated feature) ----------
function trainClassifier(propagated) {
  // 1-d feature + intercept; binary
  let w = 0, b = 0;
  const lr = 0.5;
  const labelled = STATE.nodes.map((n, i) => ({ i, label: n.label })).filter((x) => x.label >= 0);
  for (let it = 0; it < 200; it++) {
    let dw = 0, db = 0;
    for (const ex of labelled) {
      const z = w * propagated[ex.i] + b;
      const p = 1 / (1 + Math.exp(-z));
      const g = p - ex.label;
      dw += g * propagated[ex.i]; db += g;
    }
    w -= lr * dw / labelled.length;
    b -= lr * db / labelled.length;
  }
  return { w, b };
}

// ---------- Render ----------
function render() {
  const canvas = document.getElementById('gnn-canvas');
  if (!canvas) return;
  const W = 880, H = 440;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  // Edges
  ctx.strokeStyle = 'rgba(0,0,0,0.18)';
  ctx.lineWidth = 1.2;
  STATE.edges.forEach((e) => {
    const a = STATE.nodes[e.a], b = STATE.nodes[e.b];
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  });
  // Propagate
  const f = propagate();
  // Train classifier
  const cls = trainClassifier(f);
  // Nodes
  STATE.nodes.forEach((node, i) => {
    const v = f[i];
    const t = Math.max(0, Math.min(1, v));
    const r = Math.round(253 - 50 * t);
    const g = Math.round(252 - 130 * t);
    const b = Math.round(249 - 150 * t);
    ctx.beginPath();
    ctx.arc(node.x, node.y, 18, 0, Math.PI * 2);
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.fill();
    // Classifier prediction ring
    const z = cls.w * f[i] + cls.b;
    const p = 1 / (1 + Math.exp(-z));
    const ringColor = p > 0.5 ? '#d9622b' : '#2c6fb7';
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = node.label >= 0 ? 4 : 2;
    ctx.stroke();
    // If labelled, draw a dashed black inner ring
    if (node.label >= 0) {
      ctx.strokeStyle = '#1a1815';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(node.x, node.y, 12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // Label text
    ctx.fillStyle = '#1a1815';
    ctx.font = '11px IBM Plex Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`${i}`, node.x, node.y - 24);
    ctx.fillStyle = '#3b342b';
    ctx.font = '10px IBM Plex Mono';
    ctx.fillText(v.toFixed(2), node.x, node.y + 5);
  });
  // Legend
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText(`K = ${STATE.K} round(s) of ${STATE.agg} aggregation`, 16, 22);
  ctx.fillText('Click a node to spike its feature.', 16, 40);
  ctx.fillStyle = '#2c6fb7';
  ctx.fillText('blue ring = pred class 0', 16, 60);
  ctx.fillStyle = '#d9622b';
  ctx.fillText('orange ring = pred class 1', 180, 60);
  ctx.fillStyle = '#1a1815';
  ctx.fillText('dashed inner = labelled seed', 380, 60);
}

function wire() {
  const g = makeGraph();
  STATE.nodes = g.nodes; STATE.edges = g.edges;
  document.getElementById('gnn-agg').addEventListener('change', (e) => {
    STATE.agg = e.target.value; render();
  });
  document.getElementById('gnn-k').addEventListener('input', (e) => {
    STATE.K = parseInt(e.target.value, 10);
    document.getElementById('gnn-k-val').textContent = STATE.K;
    render();
  });
  document.getElementById('gnn-clear').addEventListener('click', () => {
    STATE.nodes.forEach((n) => { n.feat = 0; });
    render();
  });
  document.getElementById('gnn-reroll').addEventListener('click', () => {
    const g = makeGraph();
    STATE.nodes = g.nodes; STATE.edges = g.edges;
    render();
  });
  const canvas = document.getElementById('gnn-canvas');
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    let best = -1, bestD = Infinity;
    STATE.nodes.forEach((n, i) => {
      const dx = (cx / rect.width) * 880 - n.x;
      const dy = (cy / rect.height) * 440 - n.y;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    });
    if (best >= 0 && bestD < 30 * 30) {
      STATE.nodes.forEach((n) => { n.feat = 0; });
      STATE.nodes[best].feat = 1;
      render();
    }
  });
  render();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-mp':
      'h_i^{(\\ell+1)} = \\phi\\!\\left(h_i^{(\\ell)},\\;\\bigoplus_{j \\in \\mathcal{N}(i)} \\psi(h_i^{(\\ell)},\\,h_j^{(\\ell)})\\right)'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

// ============================================================
// Oversmoothing curve — accuracy vs K for three aggregators
// ============================================================
function computeOversmoothCurve() {
  // For each aggregator and each K in 0..8, run K message passes from
  // a random initial feature set, compute (1) classifier accuracy on
  // unlabelled nodes (using a logistic head trained at this K), (2)
  // cross-node feature variance.
  const Ks = [];
  for (let k = 0; k <= 8; k++) Ks.push(k);
  const aggs = ['mean', 'max', 'attn'];
  const trials = 4;
  const results = {};
  aggs.forEach((agg) => {
    results[agg] = { accs: [], vars: [] };
    Ks.forEach((K) => {
      let accSum = 0, varSum = 0;
      for (let t = 0; t < trials; t++) {
        // Initialise random features
        const feats0 = STATE.nodes.map(() => Math.random());
        // Save and restore state
        const saved = STATE.nodes.map((n) => n.feat);
        STATE.nodes.forEach((n, i) => { n.feat = feats0[i]; });
        const savedAgg = STATE.agg, savedK = STATE.K;
        STATE.agg = agg; STATE.K = K;
        const f = propagate();
        const cls = trainClassifier(f);
        // Test accuracy: all nodes (use ground-truth labels: alternate by spatial cluster)
        // Use existing labelled seeds + assume unseen nodes' labels = nearest seed's label.
        // Simplification: test on labelled nodes themselves, since we lack ground truth
        // for non-seeds. Better: synth ground truth from a 2-cluster spatial split.
        const W = 880;
        const split = W / 2;
        let correct = 0, total = 0;
        STATE.nodes.forEach((n, i) => {
          const trueClass = n.x < split ? 0 : 1;
          const z = cls.w * f[i] + cls.b;
          const p = 1 / (1 + Math.exp(-z));
          const pred = p > 0.5 ? 1 : 0;
          if (pred === trueClass) correct++;
          total++;
        });
        accSum += correct / total;
        // Variance across nodes
        const mean = f.reduce((a, b) => a + b, 0) / f.length;
        const v = f.reduce((s, x) => s + (x - mean) * (x - mean), 0) / f.length;
        varSum += v;
        // Restore
        STATE.nodes.forEach((n, i) => { n.feat = saved[i]; });
        STATE.agg = savedAgg; STATE.K = savedK;
      }
      results[agg].accs.push(accSum / trials);
      results[agg].vars.push(Math.max(1e-6, varSum / trials));
    });
  });
  return { Ks, results };
}

function renderOversmoothCurve() {
  const canvas = document.getElementById('os-curve');
  if (!canvas) return;
  const W = 880, H = 320;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 60, t: 18, b: 32 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const { Ks, results } = computeOversmoothCurve();
  // Left axis: accuracy 0..1
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
  // Right axis: log variance, range 1e-4..1
  ctx.textAlign = 'left';
  for (let i = 0; i <= 4; i++) {
    const v = -4 + i;
    const y = m.t + (1 - i / 4) * py;
    ctx.fillStyle = '#d9622b';
    ctx.fillText(`10^${v}`, m.l + px + 4, y + 3);
  }
  ctx.textAlign = 'center';
  ctx.fillStyle = '#9a917f';
  for (let k = 0; k <= 8; k++) {
    const x = m.l + (k / 8) * px;
    ctx.fillText(k.toString(), x, m.t + py + 16);
  }
  const colors = { mean: '#2c6fb7', max: '#1e7770', attn: '#9b59b6' };
  // Plot accuracy (solid)
  Object.keys(results).forEach((agg) => {
    ctx.strokeStyle = colors[agg]; ctx.lineWidth = 2;
    ctx.beginPath();
    results[agg].accs.forEach((a, i) => {
      const x = m.l + (i / 8) * px;
      const y = m.t + (1 - a) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  // Plot variance (dashed) on log scale -4..0
  Object.keys(results).forEach((agg) => {
    ctx.strokeStyle = colors[agg]; ctx.lineWidth = 1.4;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    results[agg].vars.forEach((v, i) => {
      const x = m.l + (i / 8) * px;
      const lv = Math.max(-4, Math.min(0, Math.log10(v)));
      const y = m.t + (1 - (lv + 4) / 4) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  });
  // Legend
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  let lx = m.l + 8, ly = m.t + 14;
  Object.keys(colors).forEach((agg) => {
    ctx.strokeStyle = colors[agg]; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke();
    ctx.fillStyle = '#3b342b';
    ctx.fillText(agg, lx + 18, ly + 4);
    lx += 18 + ctx.measureText(agg).width + 14;
  });
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('rounds K', m.l + px / 2, m.t + py + 28);
  ctx.save();
  ctx.translate(14, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('accuracy', 0, 0);
  ctx.restore();
  ctx.save();
  ctx.translate(W - 14, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#9c3f15';
  ctx.fillText('variance (log, dashed)', 0, 0);
  ctx.restore();
}

function boot() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wire();
  setTimeout(renderOversmoothCurve, 100);
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
