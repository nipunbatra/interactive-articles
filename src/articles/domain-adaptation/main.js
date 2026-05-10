// ============================================================
// Domain Adaptation — vanilla classifier vs CORAL-aligned + energy OOD.
// ============================================================

const COLORS = ['#2c6fb7', '#d9622b'];
const PLANE_MIN = -4, PLANE_MAX = 4;
const C = 2;

const STATE = {
  source: null, target: null, ood: null,
  drift: 1.5, useCoral: false,
  W: null
};

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
function randn() { const u1 = Math.random() || 1e-12, u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function makeDomain(centerOff, rotation, n) {
  const out = [];
  const cos = Math.cos(rotation), sin = Math.sin(rotation);
  const centers = [
    [-1.0, -0.4], [1.0, 0.4]
  ];
  for (let i = 0; i < n; i++) {
    const lab = i % C;
    const cx = centers[lab][0], cy = centers[lab][1];
    const xr = cx + 0.6 * randn();
    const yr = cy + 0.6 * randn();
    // rotate + translate
    const x = cos * xr - sin * yr + centerOff[0];
    const y = sin * xr + cos * yr + centerOff[1];
    out.push({ x, y, label: lab });
  }
  return out;
}

function regenDomains() {
  STATE.source = makeDomain([0, 0], 0, 200);
  const drift = STATE.drift;
  STATE.target = makeDomain([drift * 0.8, drift * 0.5], drift * 0.4, 200);
  // OOD: distant Gaussian
  STATE.ood = [];
  for (let i = 0; i < 150; i++) {
    STATE.ood.push({ x: 4 * (Math.random() * 2 - 1) + (Math.random() < 0.5 ? -3.5 : 3.5),
                     y: 4 * (Math.random() * 2 - 1) + (Math.random() < 0.5 ? -3.5 : 3.5) });
  }
}

// ---------- Train logistic regression on source ----------
function trainModel(data) {
  const D = 3;
  let W = [[randn() * 0.1, randn() * 0.1, 0], [randn() * 0.1, randn() * 0.1, 0]];
  const lr = 0.3;
  for (let it = 0; it < 200; it++) {
    const dW = [[0,0,0],[0,0,0]];
    for (const ex of data) {
      const phi = [ex.x, ex.y, 1];
      const logits = [0,0];
      for (let c = 0; c < C; c++) for (let d = 0; d < D; d++) logits[c] += W[c][d] * phi[d];
      const m = Math.max(...logits);
      const expsArr = logits.map((v) => Math.exp(v - m));
      const Z = expsArr.reduce((a, b) => a + b, 0) || 1;
      const probs = expsArr.map((e) => e / Z);
      for (let c = 0; c < C; c++) {
        const t = (c === ex.label) ? 1 : 0;
        const g = probs[c] - t;
        for (let d = 0; d < D; d++) dW[c][d] += g * phi[d];
      }
    }
    for (let c = 0; c < C; c++) for (let d = 0; d < D; d++) W[c][d] -= lr * dW[c][d] / data.length;
  }
  return W;
}

// ---------- CORAL alignment ----------
function meanCov(data) {
  const N = data.length;
  let mx = 0, my = 0;
  for (const p of data) { mx += p.x; my += p.y; }
  mx /= N; my /= N;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of data) {
    sxx += (p.x - mx) * (p.x - mx);
    sxy += (p.x - mx) * (p.y - my);
    syy += (p.y - my) * (p.y - my);
  }
  const denom = Math.max(1, N - 1);
  return { mx, my, cov: [[sxx / denom, sxy / denom], [sxy / denom, syy / denom]] };
}

function matSqrt2x2(M) {
  // Closed-form symmetric 2x2 sqrt via eigen
  const a = M[0][0], b = M[0][1], d = M[1][1];
  const tr = a + d, det = a * d - b * b;
  const tmp = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + tmp, l2 = tr / 2 - tmp;
  const sl1 = Math.sqrt(Math.max(0, l1)), sl2 = Math.sqrt(Math.max(0, l2));
  // eigenvector for l1
  let v1, v2;
  if (Math.abs(b) > 1e-9) {
    v1 = [b, l1 - a]; v2 = [b, l2 - a];
  } else {
    v1 = [1, 0]; v2 = [0, 1];
  }
  function n(v) { const r = Math.hypot(v[0], v[1]) || 1; return [v[0]/r, v[1]/r]; }
  v1 = n(v1); v2 = n(v2);
  // M^{1/2} = V Λ^{1/2} V^T
  const D = [[sl1, 0], [0, sl2]];
  const V = [[v1[0], v2[0]], [v1[1], v2[1]]];
  function mm(A, B) {
    return [
      [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
      [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
    ];
  }
  const Vt = [[V[0][0], V[1][0]], [V[0][1], V[1][1]]];
  return mm(mm(V, D), Vt);
}
function inv2x2(M) {
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  return [
    [M[1][1] / det, -M[0][1] / det],
    [-M[1][0] / det, M[0][0] / det]
  ];
}
function applyAffine(p, A, t) {
  return {
    x: A[0][0] * p.x + A[0][1] * p.y + t[0],
    y: A[1][0] * p.x + A[1][1] * p.y + t[1],
    label: p.label
  };
}

function coralTransformedSource() {
  // Whiten source by C_S^{-1/2}, then color by C_T^{1/2}; centre the source on target mean.
  const { mx: mxS, my: myS, cov: CS } = meanCov(STATE.source);
  const { mx: mxT, my: myT, cov: CT } = meanCov(STATE.target);
  // Add tiny ridge for stability
  CS[0][0] += 0.05; CS[1][1] += 0.05;
  CT[0][0] += 0.05; CT[1][1] += 0.05;
  const CSroot = matSqrt2x2(CS);
  const CTroot = matSqrt2x2(CT);
  const CSinvRoot = inv2x2(CSroot);
  // First center source at origin
  const centred = STATE.source.map((p) => ({ x: p.x - mxS, y: p.y - myS, label: p.label }));
  // Apply CSinvRoot then CTroot
  const A = [
    [CTroot[0][0] * CSinvRoot[0][0] + CTroot[0][1] * CSinvRoot[1][0],
     CTroot[0][0] * CSinvRoot[0][1] + CTroot[0][1] * CSinvRoot[1][1]],
    [CTroot[1][0] * CSinvRoot[0][0] + CTroot[1][1] * CSinvRoot[1][0],
     CTroot[1][0] * CSinvRoot[0][1] + CTroot[1][1] * CSinvRoot[1][1]]
  ];
  return centred.map((p) => applyAffine(p, A, [mxT, myT]));
}

// ---------- Predict ----------
function predict(W, x, y) {
  const phi = [x, y, 1];
  const logits = [0, 0];
  for (let c = 0; c < C; c++) for (let d = 0; d < 3; d++) logits[c] += W[c][d] * phi[d];
  return logits;
}
function softmaxArgmax(logits) {
  let best = 0;
  for (let c = 1; c < logits.length; c++) if (logits[c] > logits[best]) best = c;
  return best;
}

// ---------- Energy score ----------
function energyScore(W, x, y) {
  const logits = predict(W, x, y);
  const m = Math.max(...logits);
  const lse = m + Math.log(logits.reduce((a, l) => a + Math.exp(l - m), 0));
  return -lse; // higher = more OOD
}

// ---------- Render ----------
function renderBoard() {
  const canvas = document.getElementById('da-board');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const trainData = STATE.useCoral ? coralTransformedSource() : STATE.source;
  STATE.W = trainModel(trainData);
  // decision regions
  const step = 4;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = PLANE_MIN + (PLANE_MAX - PLANE_MIN) * (px / W);
      const y = PLANE_MAX - (PLANE_MAX - PLANE_MIN) * (py / H);
      const logits = predict(STATE.W, x, y);
      const m = Math.max(...logits);
      const exps = logits.map((v) => Math.exp(v - m));
      const Z = exps.reduce((a, b) => a + b, 0) || 1;
      const probs = exps.map((e) => e / Z);
      const best = softmaxArgmax(logits);
      ctx.fillStyle = hexToRgba(COLORS[best], 0.18 + 0.45 * probs[best]);
      ctx.fillRect(px, py, step, step);
    }
  }
  // Source filled, target as ×
  trainData.forEach((p) => {
    const px = (p.x - PLANE_MIN) / (PLANE_MAX - PLANE_MIN) * W;
    const py = (PLANE_MAX - p.y) / (PLANE_MAX - PLANE_MIN) * H;
    ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[p.label]; ctx.fill();
  });
  STATE.target.forEach((p) => {
    const px = (p.x - PLANE_MIN) / (PLANE_MAX - PLANE_MIN) * W;
    const py = (PLANE_MAX - p.y) / (PLANE_MAX - PLANE_MIN) * H;
    ctx.strokeStyle = COLORS[p.label];
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(px - 3, py - 3); ctx.lineTo(px + 3, py + 3);
    ctx.moveTo(px + 3, py - 3); ctx.lineTo(px - 3, py + 3);
    ctx.stroke();
  });
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderEnergy() {
  const canvas = document.getElementById('da-energy');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  // Energy heatmap
  const step = 4;
  let lo = Infinity, hi = -Infinity;
  const grid = [];
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = PLANE_MIN + (PLANE_MAX - PLANE_MIN) * (px / W);
      const y = PLANE_MAX - (PLANE_MAX - PLANE_MIN) * (py / H);
      const e = energyScore(STATE.W, x, y);
      grid.push({ px, py, e });
      if (e < lo) lo = e; if (e > hi) hi = e;
    }
  }
  const range = Math.max(1e-6, hi - lo);
  grid.forEach(({ px, py, e }) => {
    const t = (e - lo) / range;
    ctx.fillStyle = `rgba(217, 98, 43, ${0.05 + 0.7 * t})`;
    ctx.fillRect(px, py, step, step);
  });
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderROC() {
  const canvas = document.getElementById('da-roc');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(m.l, m.t + py); ctx.lineTo(m.l + px, m.t); ctx.stroke();
  ctx.setLineDash([]);
  // Energy on target (in-domain) vs ood
  const targetScores = STATE.target.map((p) => energyScore(STATE.W, p.x, p.y));
  const oodScores = STATE.ood.map((p) => energyScore(STATE.W, p.x, p.y));
  const all = targetScores.concat(oodScores).slice().sort((a, b) => a - b);
  // ROC: at each threshold, TPR (OOD with score > threshold) vs FPR (target with score > threshold)
  let auc = 0;
  let prevFPR = 0, prevTPR = 0;
  ctx.strokeStyle = '#2c6fb7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < all.length; i++) {
    const t = all[i];
    let tp = 0, fp = 0;
    for (const s of oodScores) if (s > t) tp++;
    for (const s of targetScores) if (s > t) fp++;
    const tpr = tp / oodScores.length;
    const fpr = fp / targetScores.length;
    const x = m.l + fpr * px, y = m.t + (1 - tpr) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    auc += (prevFPR - fpr) * (tpr + prevTPR) / 2;
    prevFPR = fpr; prevTPR = tpr;
  }
  ctx.stroke();
  auc = Math.abs(auc);
  // Update AUC stat
  document.getElementById('da-auc').textContent = auc.toFixed(2);
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = i / 4;
    const y = m.t + (1 - v) * py;
    ctx.fillText(v.toFixed(1), m.l - 4, y + 3);
  }
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const v = i / 4;
    const x = m.l + v * px;
    ctx.fillText(v.toFixed(1), x, m.t + py + 16);
  }
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('FPR', m.l + px / 2, m.t + py + 28);
  ctx.save();
  ctx.translate(14, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('TPR', 0, 0);
  ctx.restore();
}

function refreshStats() {
  let srcCorrect = 0, tgtCorrect = 0;
  STATE.source.forEach((p) => {
    if (softmaxArgmax(predict(STATE.W, p.x, p.y)) === p.label) srcCorrect++;
  });
  STATE.target.forEach((p) => {
    if (softmaxArgmax(predict(STATE.W, p.x, p.y)) === p.label) tgtCorrect++;
  });
  document.getElementById('da-src-acc').textContent = `${(100 * srcCorrect / STATE.source.length).toFixed(1)}%`;
  document.getElementById('da-tgt-acc').textContent = `${(100 * tgtCorrect / STATE.target.length).toFixed(1)}%`;
}

function renderAll() { renderBoard(); renderEnergy(); renderROC(); refreshStats(); }

function wire() {
  regenDomains();
  document.getElementById('da-drift').addEventListener('input', (e) => {
    STATE.drift = parseFloat(e.target.value);
    document.getElementById('da-drift-val').textContent = STATE.drift.toFixed(1);
    regenDomains();
    renderAll();
  });
  document.getElementById('da-coral').addEventListener('change', (e) => {
    STATE.useCoral = e.target.checked;
    renderAll();
  });
  document.getElementById('da-resample').addEventListener('click', () => {
    regenDomains();
    renderAll();
  });
  renderAll();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-coral':
      'X_S \\leftarrow (X_S - \\mu_S)\\,C_S^{-1/2}\\,C_T^{1/2} + \\mu_T',
    'math-energy':
      '\\mathrm{Energy}(x) = -\\log\\!\\sum_c \\exp(z_c(x))'
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
