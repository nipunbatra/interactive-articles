// ============================================================
// Knowledge Distillation, Live
// Teacher: smooth Gaussian-mixture classifier (4 classes on a 2-D plane)
// Student: a single linear layer (D=2 → K=4)
// All training runs live in the browser; no autograd library.
// ============================================================

const CLASSES = 4;
const TEACHER_CENTERS = [
  { x: -1.4, y: -1.2 },
  { x:  1.5, y: -1.0 },
  { x: -0.8, y:  1.6 },
  { x:  1.7, y:  1.5 }
];
const TEACHER_WIDTH_SQ = 1.2; // controls smoothness
const CLASS_COLORS = ['#2c6fb7', '#d9622b', '#1e7770', '#9b59b6'];

const KD = {
  W: [[0, 0, 0, 0], [0, 0, 0, 0]], // 2 x 4
  b: [0, 0, 0, 0],
  step: 0,
  losses: [],
  studentAcc: 0,
  teacherAcc: 0,
  recipe: 'hybrid',
  T: 4.0,
  alpha: 0.30,
  lr: 0.10,
  running: false,
  raf: null,
  trainData: null
};

function teacherLogits(x, y) {
  // Smooth, "pretrained-looking" logits: distance-based.
  const out = new Array(CLASSES);
  for (let c = 0; c < CLASSES; c++) {
    const dx = x - TEACHER_CENTERS[c].x;
    const dy = y - TEACHER_CENTERS[c].y;
    const d2 = dx * dx + dy * dy;
    out[c] = -d2 / TEACHER_WIDTH_SQ;
  }
  return out;
}

function softmaxArr(logits, T = 1) {
  const scaled = logits.map((v) => v / T);
  let m = -Infinity;
  for (const v of scaled) if (v > m) m = v;
  const exps = scaled.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / s);
}

function argMax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function studentLogits(x, y, W = KD.W, b = KD.b) {
  // 2 x 4
  const out = new Array(CLASSES);
  for (let c = 0; c < CLASSES; c++) {
    out[c] = x * W[0][c] + y * W[1][c] + b[c];
  }
  return out;
}

function makeTrainData(n) {
  const data = [];
  for (let i = 0; i < n; i++) {
    const c = i % CLASSES;
    const cx = TEACHER_CENTERS[c].x;
    const cy = TEACHER_CENTERS[c].y;
    const x = cx + 0.7 * (Math.random() * 2 - 1);
    const y = cy + 0.7 * (Math.random() * 2 - 1);
    data.push({ x, y, label: c });
  }
  return data;
}

function resetStudent() {
  KD.W = [
    [randn() * 0.1, randn() * 0.1, randn() * 0.1, randn() * 0.1],
    [randn() * 0.1, randn() * 0.1, randn() * 0.1, randn() * 0.1]
  ];
  KD.b = [0, 0, 0, 0];
  KD.step = 0;
  KD.losses = [];
}

function randn() {
  const u1 = Math.random() || 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------- Training step ----------
function trainStep(batch) {
  // Accumulate gradients over batch
  let dW = [[0, 0, 0, 0], [0, 0, 0, 0]];
  let dB = [0, 0, 0, 0];
  let totalLoss = 0;
  for (const ex of batch) {
    const tLogits = teacherLogits(ex.x, ex.y);
    const sLogits = studentLogits(ex.x, ex.y);
    let grad = new Array(CLASSES).fill(0);
    let loss = 0;

    if (KD.recipe === 'hard' || KD.recipe === 'hybrid') {
      // Hard CE: dL/dz_c = q_c - one_hot
      const sProb = softmaxArr(sLogits, 1);
      for (let c = 0; c < CLASSES; c++) {
        const t = (c === ex.label) ? 1 : 0;
        grad[c] += (KD.recipe === 'hybrid' ? KD.alpha : 1.0) * (sProb[c] - t);
        if (c === ex.label) loss -= Math.log(Math.max(sProb[c], 1e-12));
      }
      if (KD.recipe === 'hybrid') loss *= KD.alpha;
    }
    if (KD.recipe === 'soft' || KD.recipe === 'hybrid') {
      // KD KL(p_T || q_T) where q_T = softmax(z/T)
      const T = KD.T;
      const pT = softmaxArr(tLogits, T);
      const qT = softmaxArr(sLogits, T);
      const w = (KD.recipe === 'hybrid') ? (1 - KD.alpha) : 1.0;
      // d(KL)/dz_c = T^2 * (q_T - p_T) / T = T * (q_T - p_T)... actually:
      // L = T^2 * sum_c p_T_c * log(p_T_c / q_T_c)
      // dL/dz_c = T^2 * (q_T_c - p_T_c) / T = T * (q_T_c - p_T_c)
      // (the T^2 prefactor times the chain factor 1/T → T)
      for (let c = 0; c < CLASSES; c++) {
        grad[c] += w * T * (qT[c] - pT[c]);
        loss += w * T * T * pT[c] * Math.log(Math.max(pT[c], 1e-12) / Math.max(qT[c], 1e-12));
      }
    }
    // Add gradient contribution
    for (let c = 0; c < CLASSES; c++) {
      dW[0][c] += grad[c] * ex.x;
      dW[1][c] += grad[c] * ex.y;
      dB[c] += grad[c];
    }
    totalLoss += loss;
  }
  const N = batch.length;
  for (let c = 0; c < CLASSES; c++) {
    KD.W[0][c] -= KD.lr * dW[0][c] / N;
    KD.W[1][c] -= KD.lr * dW[1][c] / N;
    KD.b[c] -= KD.lr * dB[c] / N;
  }
  KD.step++;
  KD.losses.push(totalLoss / N);
  if (KD.losses.length > 1500) KD.losses = KD.losses.slice(-1500);
}

// ---------- Decision-boundary canvas ----------
function drawDecision(canvasId, logitFn) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = 380, H = 380;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const xMin = -3, xMax = 3, yMin = -3, yMax = 3;
  // Render decision regions as a grid
  const step = 4;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = xMin + (xMax - xMin) * (px / W);
      const y = yMax - (yMax - yMin) * (py / H);
      const logits = logitFn(x, y);
      const probs = softmaxArr(logits, 1);
      const bestC = argMax(probs);
      const conf = probs[bestC];
      const col = CLASS_COLORS[bestC];
      const alpha = 0.20 + 0.45 * conf;
      ctx.fillStyle = hexToRgba(col, alpha);
      ctx.fillRect(px, py, step, step);
    }
  }
  // Draw class centers as crosses
  TEACHER_CENTERS.forEach((c, i) => {
    const px = (c.x - xMin) / (xMax - xMin) * W;
    const py = (yMax - c.y) / (yMax - yMin) * H;
    ctx.strokeStyle = '#1a1815';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px - 6, py); ctx.lineTo(px + 6, py);
    ctx.moveTo(px, py - 6); ctx.lineTo(px, py + 6);
    ctx.stroke();
  });
  // Draw training points
  if (KD.trainData) {
    KD.trainData.forEach((ex) => {
      const px = (ex.x - xMin) / (xMax - xMin) * W;
      const py = (yMax - ex.y) / (yMax - yMin) * H;
      ctx.beginPath();
      ctx.arc(px, py, 2.6, 0, Math.PI * 2);
      ctx.fillStyle = CLASS_COLORS[ex.label];
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 0.8;
      ctx.stroke();
    });
  }
  // Border
  ctx.strokeStyle = '#e2d8c6';
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, W, H);
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ---------- Loss curve ----------
function drawLossCurve() {
  const canvas = document.getElementById('kd-loss-canvas');
  if (!canvas) return;
  const W = 380, H = 380;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#fdfcf9';
  ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 12, t: 14, b: 30 };
  const px = W - m.l - m.r;
  const py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  if (KD.losses.length === 0) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText('Loss curve appears here', m.l + px / 2, m.t + py / 2);
    return;
  }
  const minL = Math.min(...KD.losses, 0);
  const maxL = Math.max(...KD.losses, 1);
  const range = Math.max(0.01, maxL - minL);
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = minL + range * (1 - i / 4);
    const y = m.t + (i / 4) * py;
    ctx.fillText(v.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath();
    ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y);
    ctx.stroke();
  }
  ctx.strokeStyle = '#2c6fb7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  KD.losses.forEach((v, i) => {
    const x = m.l + (i / Math.max(1, KD.losses.length - 1)) * px;
    const y = m.t + (1 - (v - minL) / range) * py;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText(`step ${KD.step} · loss ${KD.losses.length ? KD.losses[KD.losses.length - 1].toFixed(3) : '—'}`, m.l + px / 2, m.t + py + 18);
}

// ---------- Step 1 (temperature softmax demo) ----------
function renderTempCanvas() {
  const canvas = document.getElementById('t-canvas');
  if (!canvas) return;
  const W = 880, H = 240;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#fdfcf9';
  ctx.fillRect(0, 0, W, H);
  const T = parseFloat(document.getElementById('t-slider').value);
  document.getElementById('t-val').textContent = T.toFixed(1);
  // Hand-picked logits
  const logits = [4.0, 1.6, 0.8, -0.3, -1.2, -2.5];
  const Ts = [1, 2, T, 16];
  const labels = ['T = 1', 'T = 2', `T = ${T.toFixed(1)} (slider)`, 'T = 16'];
  const colors = ['#2c6fb7', '#1e7770', '#d9622b', '#9b59b6'];
  const m = { l: 50, r: 16, t: 22, b: 28 };
  const px = (W - m.l - m.r) / 4;
  const py = H - m.t - m.b;
  Ts.forEach((temp, i) => {
    const probs = softmaxArr(logits, temp);
    const x0 = m.l + i * px;
    const bw = (px - 16) / probs.length;
    ctx.fillStyle = '#1a1815';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'left';
    ctx.fillText(labels[i], x0, m.t - 4);
    probs.forEach((p, j) => {
      const h = p * py;
      ctx.fillStyle = hexToRgba(colors[i], 0.25 + 0.55 * p);
      ctx.fillRect(x0 + j * bw + 6, m.t + py - h, bw - 4, h);
    });
    ctx.strokeStyle = '#e2d8c6';
    ctx.strokeRect(x0 + 4, m.t, px - 14, py);
    ctx.fillStyle = '#9a917f';
    ctx.font = '10px IBM Plex Mono';
    ctx.textAlign = 'left';
    ctx.fillText(probs.map((p) => p.toFixed(2)).join(' / '), x0 + 6, m.t + py + 14);
  });
}

// ---------- Step 4 (gradient richness) ----------
function renderRichness() {
  const canvas = document.getElementById('kd-richness-canvas');
  if (!canvas) return;
  const W = 880, H = 280;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#fdfcf9';
  ctx.fillRect(0, 0, W, H);
  // Take a sample point near class 0
  const x = -1.3, y = -1.4;
  const tLogits = teacherLogits(x, y);
  const sLogits = [1.5, -0.5, 0.2, 0.0]; // arbitrary student logits before training
  const T = 4.0;
  const sProb1 = softmaxArr(sLogits, 1);
  const oneHot = [1, 0, 0, 0];
  const hardGrad = sProb1.map((q, c) => q - oneHot[c]);
  const pT = softmaxArr(tLogits, T);
  const qT = softmaxArr(sLogits, T);
  const softGrad = qT.map((q, c) => T * (q - pT[c]));
  const labels = ['cls 0 (true)', 'cls 1', 'cls 2', 'cls 3'];
  const colors = CLASS_COLORS;
  const m = { l: 90, r: 16, t: 26, b: 36 };
  const px = (W - m.l - m.r) / 2;
  const py = H - m.t - m.b;
  const drawBars = (offset, grads, title) => {
    ctx.fillStyle = '#1a1815';
    ctx.font = '14px Manrope';
    ctx.textAlign = 'left';
    ctx.fillText(title, m.l + offset, m.t - 8);
    const maxAbs = Math.max(...grads.map(Math.abs), 0.01);
    const baseY = m.t + py / 2;
    grads.forEach((g, c) => {
      const bw = (px - 24) / grads.length;
      const x0 = m.l + offset + c * bw + 6;
      const h = (g / maxAbs) * (py / 2 - 14);
      ctx.fillStyle = hexToRgba(colors[c], 0.85);
      ctx.fillRect(x0, baseY - (h > 0 ? h : 0), bw - 6, Math.abs(h));
      ctx.fillStyle = '#3b342b';
      ctx.font = '11px IBM Plex Mono';
      ctx.textAlign = 'center';
      ctx.fillText(g.toFixed(2), x0 + bw / 2 - 3, baseY + (h > 0 ? -h - 4 : -h + 14));
      ctx.fillStyle = '#9a917f';
      ctx.font = '10px Manrope';
      ctx.fillText(labels[c], x0 + bw / 2 - 3, m.t + py + 16);
    });
    // baseline
    ctx.strokeStyle = '#e2d8c6';
    ctx.beginPath();
    ctx.moveTo(m.l + offset, baseY);
    ctx.lineTo(m.l + offset + px, baseY);
    ctx.stroke();
  };
  drawBars(0, hardGrad, 'Hard target gradient ∂L/∂z');
  drawBars(px + 16, softGrad, 'Soft target gradient (T=4)');
}

// ---------- Loop / boot ----------
function evalAccuracy() {
  if (!KD.trainData) return { s: 0, t: 0 };
  let sCorrect = 0, tCorrect = 0;
  for (const ex of KD.trainData) {
    const sLogits = studentLogits(ex.x, ex.y);
    const tLogits = teacherLogits(ex.x, ex.y);
    if (argMax(sLogits) === ex.label) sCorrect++;
    if (argMax(tLogits) === ex.label) tCorrect++;
  }
  return { s: sCorrect / KD.trainData.length, t: tCorrect / KD.trainData.length };
}

function refreshStats() {
  const { s, t } = evalAccuracy();
  KD.studentAcc = s;
  KD.teacherAcc = t;
  document.getElementById('kd-step').textContent = KD.step;
  document.getElementById('kd-acc').textContent = (s * 100).toFixed(1) + '%';
  document.getElementById('kd-tacc').textContent = (t * 100).toFixed(1) + '%';
}

function trainLoop() {
  if (!KD.running) return;
  for (let i = 0; i < 4; i++) {
    // Randomly sample a batch from trainData
    const batch = [];
    for (let j = 0; j < 16; j++) {
      batch.push(KD.trainData[Math.floor(Math.random() * KD.trainData.length)]);
    }
    trainStep(batch);
  }
  drawDecision('kd-student-canvas', (x, y) => studentLogits(x, y));
  drawLossCurve();
  refreshStats();
  KD.raf = requestAnimationFrame(trainLoop);
}

function wireKD() {
  KD.trainData = makeTrainData(80);
  resetStudent();

  document.getElementById('kd-recipe').addEventListener('change', (e) => {
    KD.recipe = e.target.value;
  });
  document.getElementById('kd-T').addEventListener('input', (e) => {
    KD.T = parseFloat(e.target.value);
    document.getElementById('kd-T-val').textContent = KD.T.toFixed(1);
  });
  document.getElementById('kd-alpha').addEventListener('input', (e) => {
    KD.alpha = parseFloat(e.target.value);
    document.getElementById('kd-alpha-val').textContent = KD.alpha.toFixed(2);
  });
  const tog = document.getElementById('kd-toggle');
  tog.addEventListener('click', () => {
    KD.running = !KD.running;
    tog.textContent = KD.running ? 'Pause training' : 'Start training';
    if (KD.running) trainLoop();
    else if (KD.raf) cancelAnimationFrame(KD.raf);
  });
  document.getElementById('kd-reset').addEventListener('click', () => {
    if (KD.raf) cancelAnimationFrame(KD.raf);
    KD.running = false;
    tog.textContent = 'Start training';
    KD.trainData = makeTrainData(80);
    resetStudent();
    drawDecision('kd-teacher-canvas', teacherLogits);
    drawDecision('kd-student-canvas', (x, y) => studentLogits(x, y));
    drawLossCurve();
    refreshStats();
  });
  drawDecision('kd-teacher-canvas', teacherLogits);
  drawDecision('kd-student-canvas', (x, y) => studentLogits(x, y));
  drawLossCurve();
  refreshStats();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-soft':
      'p_c^{(T)} = \\frac{\\exp(z_c / T)}{\\sum_{c\'} \\exp(z_{c\'} / T)}',
    'math-loss':
      '\\mathcal{L}_{\\text{KD}} = \\alpha\\,\\mathrm{CE}(y, q^{(1)}) + (1-\\alpha)\\,T^2 \\,\\mathrm{KL}\\!\\left(p^{(T)} \\,\\|\\, q^{(T)}\\right)'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

function init() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  document.getElementById('t-slider').addEventListener('input', renderTempCanvas);
  renderTempCanvas();
  renderRichness();
  wireKD();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
