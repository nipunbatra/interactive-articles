// ============================================================
// Calibration & reliability diagrams
// 3-class RBF logistic regression trained from scratch in browser;
// reliability diagram + ECE/MCE/Brier/NLL all live; temperature
// scaling slider applied at inference.
// ============================================================

const C = 3;
const COLORS = ['#2c6fb7', '#d9622b', '#1e7770'];

const RBF_CENTERS = [];
for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
  RBF_CENTERS.push([-2.5 + 5 * (c + 0.5) / 4, -2.5 + 5 * (r + 0.5) / 4]);
}
const RBF_GAMMA = 0.7;

const STATE = {
  W: null,
  data: null,
  test: null,
  step: 0,
  T: 1.0,
  history: [], // {step, acc, ece, brier, nll}
  running: false,
  raf: null
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

function phi(x, y) {
  const out = new Array(RBF_CENTERS.length + 1);
  out[0] = 1;
  for (let i = 0; i < RBF_CENTERS.length; i++) {
    const dx = x - RBF_CENTERS[i][0];
    const dy = y - RBF_CENTERS[i][1];
    out[i + 1] = Math.exp(-RBF_GAMMA * (dx * dx + dy * dy));
  }
  return out;
}

function softmax(arr, T = 1) {
  const scaled = arr.map((v) => v / T);
  let m = -Infinity;
  for (const v of scaled) if (v > m) m = v;
  const exps = scaled.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / s);
}

function predictLogits(W, x, y) {
  const f = phi(x, y);
  const D = f.length;
  const logits = new Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let d = 0; d < D; d++) s += W[c][d] * f[d];
    logits[c] = s;
  }
  return logits;
}

function makeData(N) {
  const out = [];
  const clusters = [
    { mu: [-1.4,  0.2], cov: 0.8, label: 0 },
    { mu: [ 1.3,  1.2], cov: 0.7, label: 1 },
    { mu: [ 0.0, -1.5], cov: 0.85, label: 2 }
  ];
  for (let i = 0; i < N; i++) {
    const c = clusters[i % C];
    out.push({
      x: c.mu[0] + randn() * c.cov,
      y: c.mu[1] + randn() * c.cov,
      label: c.label
    });
  }
  return out;
}

function reset() {
  STATE.data = makeData(150);
  STATE.test = makeData(800);
  const D = RBF_CENTERS.length + 1;
  STATE.W = [];
  for (let c = 0; c < C; c++) {
    STATE.W.push(new Array(D).fill(0).map(() => randn() * 0.05));
  }
  STATE.step = 0;
  STATE.history = [];
}

function trainStep() {
  const D = STATE.W[0].length;
  const dW = new Array(C).fill(0).map(() => new Array(D).fill(0));
  const lr = 0.4;
  // Push past calibration sweet spot to induce over-confidence.
  // Use NO L2 regularisation here so logits grow.
  for (const ex of STATE.data) {
    const f = phi(ex.x, ex.y);
    const logits = predictLogits(STATE.W, ex.x, ex.y);
    const probs = softmax(logits);
    for (let c = 0; c < C; c++) {
      const t = (c === ex.label) ? 1 : 0;
      const g = probs[c] - t;
      for (let d = 0; d < D; d++) dW[c][d] += g * f[d];
    }
  }
  for (let c = 0; c < C; c++) for (let d = 0; d < D; d++) STATE.W[c][d] -= lr * dW[c][d] / STATE.data.length;
  STATE.step++;
  // Compute metrics on test set
  const stats = computeStats(STATE.test, STATE.W, STATE.T);
  STATE.history.push({ step: STATE.step, ...stats });
  if (STATE.history.length > 600) STATE.history = STATE.history.slice(-600);
}

function computeStats(test, W, T) {
  let correct = 0, brier = 0, nll = 0;
  const probsList = [];
  for (const ex of test) {
    const logits = predictLogits(W, ex.x, ex.y);
    const probs = softmax(logits, T);
    let best = 0;
    for (let c = 1; c < C; c++) if (probs[c] > probs[best]) best = c;
    if (best === ex.label) correct++;
    nll -= Math.log(Math.max(probs[ex.label], 1e-12));
    for (let c = 0; c < C; c++) {
      const t = (c === ex.label) ? 1 : 0;
      brier += (probs[c] - t) * (probs[c] - t);
    }
    probsList.push({ probs, label: ex.label, top: best, conf: probs[best] });
  }
  // ECE / MCE: 10 bins on confidence
  const nBins = 10;
  const bins = new Array(nBins).fill(0).map(() => ({ count: 0, accSum: 0, confSum: 0 }));
  probsList.forEach((p) => {
    const b = Math.min(nBins - 1, Math.floor(p.conf * nBins));
    bins[b].count++;
    bins[b].confSum += p.conf;
    if (p.top === p.label) bins[b].accSum += 1;
  });
  let ece = 0, mce = 0;
  const N = test.length;
  const binData = bins.map((b) => {
    if (b.count === 0) return null;
    const acc = b.accSum / b.count;
    const conf = b.confSum / b.count;
    const w = b.count / N;
    const gap = Math.abs(acc - conf);
    ece += w * gap;
    mce = Math.max(mce, gap);
    return { acc, conf, w, count: b.count };
  });
  return { acc: correct / N, ece, mce, brier: brier / N, nll: nll / N, bins: binData };
}

// ---------- Render ----------
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function renderBoard() {
  const canvas = document.getElementById('cal-board');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const xMin = -3, xMax = 3, yMin = -3, yMax = 3;
  const step = 4;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = xMin + (xMax - xMin) * (px / W);
      const y = yMax - (yMax - yMin) * (py / H);
      const probs = softmax(predictLogits(STATE.W, x, y), STATE.T);
      let best = 0;
      for (let c = 1; c < C; c++) if (probs[c] > probs[best]) best = c;
      ctx.fillStyle = hexToRgba(COLORS[best], 0.15 + 0.55 * probs[best]);
      ctx.fillRect(px, py, step, step);
    }
  }
  STATE.test.slice(0, 200).forEach((p) => {
    const px = (p.x - xMin) / (xMax - xMin) * W;
    const py = (yMax - p.y) / (yMax - yMin) * H;
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[p.label];
    ctx.fill();
  });
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderReliability() {
  const canvas = document.getElementById('cal-reliab');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  // Diagonal
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(m.l, m.t + py); ctx.lineTo(m.l + px, m.t); ctx.stroke();
  ctx.setLineDash([]);
  // Bins
  const stats = computeStats(STATE.test, STATE.W, STATE.T);
  const nBins = 10;
  const bw = px / nBins;
  stats.bins.forEach((b, i) => {
    if (!b) return;
    const x0 = m.l + i * bw + 1;
    const h = b.acc * py;
    ctx.fillStyle = 'rgba(44,111,183,0.55)';
    ctx.fillRect(x0, m.t + py - h, bw - 2, h);
    // Mark mean confidence as a tick on that bin
    const cx = x0 + bw / 2 - 1;
    const cy = m.t + py - b.conf * py;
    ctx.fillStyle = '#d9622b';
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fill();
  });
  // Axis labels
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
    const x = m.l + (i / 4) * px;
    ctx.fillText((i / 4).toFixed(1), x, m.t + py + 16);
  }
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('confidence', m.l + px / 2, m.t + py + 28);
  ctx.save();
  ctx.translate(14, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('accuracy', 0, 0);
  ctx.restore();
  // Legend
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgba(44,111,183,0.85)';
  ctx.fillRect(m.l + 6, m.t + 6, 12, 12);
  ctx.fillStyle = '#3b342b';
  ctx.fillText('bin accuracy', m.l + 22, m.t + 16);
  ctx.fillStyle = '#d9622b';
  ctx.beginPath();
  ctx.arc(m.l + 110, m.t + 12, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#3b342b';
  ctx.fillText('mean confidence', m.l + 120, m.t + 16);
}

function renderMetrics() {
  const canvas = document.getElementById('cal-metrics');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  if (STATE.history.length === 0) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText('Train to populate metrics', m.l + px / 2, m.t + py / 2);
    return;
  }
  const series = [
    { key: 'acc', color: '#1e7770', label: 'accuracy', max: 1 },
    { key: 'ece', color: '#d9622b', label: 'ECE', max: 0.5 },
    { key: 'brier', color: '#9b59b6', label: 'Brier', max: 1 },
    { key: 'nll', color: '#2c6fb7', label: 'NLL/log_e', max: 2 }
  ];
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = m.t + i / 4 * py;
    ctx.fillText((1 - i / 4).toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  const N = STATE.history.length;
  series.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    STATE.history.forEach((h, i) => {
      const x = m.l + (i / Math.max(1, N - 1)) * px;
      const v = Math.min(1, h[s.key] / s.max);
      const y = m.t + (1 - v) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  let lx = m.l + 8, ly = m.t + 14;
  series.forEach((s) => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke();
    ctx.fillStyle = '#3b342b';
    ctx.fillText(s.label, lx + 18, ly + 4);
    lx += 18 + ctx.measureText(s.label).width + 14;
  });
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('step', m.l + px / 2, m.t + py + 22);
}

function refreshStats() {
  const stats = computeStats(STATE.test, STATE.W, STATE.T);
  document.getElementById('cal-step').textContent = STATE.step;
  document.getElementById('cal-acc').textContent = (stats.acc * 100).toFixed(1) + '%';
  document.getElementById('cal-ece').textContent = stats.ece.toFixed(3);
}

function renderAll() {
  renderBoard();
  renderReliability();
  renderMetrics();
  refreshStats();
}

function loop() {
  if (!STATE.running) return;
  for (let i = 0; i < 4; i++) trainStep();
  renderAll();
  STATE.raf = requestAnimationFrame(loop);
}

function wire() {
  reset();
  document.getElementById('cal-T').addEventListener('input', (e) => {
    STATE.T = parseFloat(e.target.value);
    document.getElementById('cal-T-val').textContent = STATE.T.toFixed(2);
    renderAll();
  });
  const tog = document.getElementById('cal-toggle');
  tog.addEventListener('click', () => {
    STATE.running = !STATE.running;
    tog.textContent = STATE.running ? 'Pause' : 'Start training';
    if (STATE.running) loop();
    else if (STATE.raf) cancelAnimationFrame(STATE.raf);
  });
  document.getElementById('cal-reset').addEventListener('click', () => {
    if (STATE.raf) cancelAnimationFrame(STATE.raf);
    STATE.running = false;
    tog.textContent = 'Start training';
    reset();
    renderAll();
  });
  renderAll();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-cal':
      '\\mathrm{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{N} \\, \\bigl|\\; \\mathrm{acc}(B_m) - \\mathrm{conf}(B_m) \\;\\bigr|',
    'math-temp':
      'p^{(T)}_c = \\frac{e^{z_c / T}}{\\sum_{c\'} e^{z_{c\'} / T}}, \\qquad T^* = \\arg\\min_T \\;\\mathrm{NLL}(T)\\text{ on val}'
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
