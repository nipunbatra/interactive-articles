// ============================================================
// Residual Connections — interactive
// All three live demos in pure JS: forward activation magnitude,
// backward gradient magnitude, and a side-by-side training race
// of plain vs residual stacks on a 2-D regression toy.
// ============================================================

const PRIOR_COLOR = '#2c6fb7';
const LIK_COLOR = '#d9622b';
const POST_COLOR = '#1e7770';
const TICK_COLOR = '#9a917f';

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

// 8-dim hidden state, single weight matrix per layer
const HIDDEN = 8;

// ---------- Step 1: Forward activation magnitudes ----------
function forwardActs(L, scale, residual, weights) {
  // weights: array of W (HIDDEN x HIDDEN), b (HIDDEN), one per layer
  const norms = new Array(L + 1);
  let h = new Array(HIDDEN).fill(0);
  for (let i = 0; i < HIDDEN; i++) h[i] = randn() * 0.3;
  norms[0] = Math.hypot(...h);
  for (let l = 0; l < L; l++) {
    const W = weights[l].W;
    const b = weights[l].b;
    const z = new Array(HIDDEN).fill(0);
    for (let i = 0; i < HIDDEN; i++) {
      let s = b[i];
      for (let j = 0; j < HIDDEN; j++) s += W[i][j] * h[j];
      z[i] = Math.tanh(s);
    }
    const next = new Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) next[i] = residual ? h[i] + z[i] : z[i];
    h = next;
    norms[l + 1] = Math.hypot(...h);
  }
  return norms;
}

function makeRandomWeights(L, scale) {
  const w = new Array(L);
  for (let l = 0; l < L; l++) {
    const W = new Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      W[i] = new Array(HIDDEN);
      for (let j = 0; j < HIDDEN; j++) W[i][j] = randn() * scale / Math.sqrt(HIDDEN);
    }
    const b = new Array(HIDDEN).fill(0);
    w[l] = { W, b };
  }
  return w;
}

function renderActCanvas() {
  const canvas = document.getElementById('act-canvas');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 18, t: 24, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  const L = parseInt(document.getElementById('act-L').value, 10);
  const scale = parseFloat(document.getElementById('act-scale').value);
  document.getElementById('act-L-val').textContent = L;
  document.getElementById('act-scale-val').textContent = scale.toFixed(2);
  // Average over a few seeds to smooth
  const trials = 20;
  let plainAcc = new Array(L + 1).fill(0);
  let residAcc = new Array(L + 1).fill(0);
  for (let t = 0; t < trials; t++) {
    const w = makeRandomWeights(L, scale);
    const np = forwardActs(L, scale, false, w);
    const nr = forwardActs(L, scale, true, w);
    for (let i = 0; i <= L; i++) { plainAcc[i] += np[i]; residAcc[i] += nr[i]; }
  }
  for (let i = 0; i <= L; i++) { plainAcc[i] /= trials; residAcc[i] /= trials; }
  let yMax = 0;
  for (let i = 0; i <= L; i++) yMax = Math.max(yMax, plainAcc[i], residAcc[i]);
  yMax = Math.max(yMax, 1) * 1.15;
  // axes
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = 0; v <= 4; v++) {
    const val = (v / 4) * yMax;
    const y = m.t + (1 - val / yMax) * py;
    ctx.fillText(val.toFixed(1), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const x = m.l + (i / 5) * px;
    ctx.fillText(Math.round(L * i / 5), x, m.t + py + 16);
  }
  // plain
  ctx.strokeStyle = LIK_COLOR;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  for (let i = 0; i <= L; i++) {
    const x = m.l + (i / L) * px;
    const y = m.t + (1 - plainAcc[i] / yMax) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // residual
  ctx.strokeStyle = POST_COLOR;
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  for (let i = 0; i <= L; i++) {
    const x = m.l + (i / L) * px;
    const y = m.t + (1 - residAcc[i] / yMax) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = '#3b342b';
  ctx.font = '13px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('layer index l', m.l + px / 2 - 50, H - 4);
  ctx.fillText('‖h_l‖₂', m.l + 8, m.t + 14);
  // legend
  ctx.strokeStyle = LIK_COLOR; ctx.setLineDash([6,4]); ctx.beginPath();
  ctx.moveTo(m.l + px - 200, m.t + 14); ctx.lineTo(m.l + px - 180, m.t + 14); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillText('plain', m.l + px - 175, m.t + 18);
  ctx.strokeStyle = POST_COLOR; ctx.beginPath();
  ctx.moveTo(m.l + px - 130, m.t + 14); ctx.lineTo(m.l + px - 110, m.t + 14); ctx.stroke();
  ctx.fillText('residual', m.l + px - 105, m.t + 18);
}

// ---------- Step 2: Backward gradient magnitudes (random Jacobian) ----------
function renderGradCanvas() {
  const canvas = document.getElementById('grad-canvas');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 18, t: 24, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  const L = parseInt(document.getElementById('act-L').value, 10);
  const scale = parseFloat(document.getElementById('act-scale').value);
  // Use random per-layer Jacobian J_l = sigma' * W_l with sigma' = 1 - h^2 (treat ~constant 0.7 average)
  const trials = 30;
  const plain = new Array(L + 1).fill(0);
  const resid = new Array(L + 1).fill(0);
  for (let t = 0; t < trials; t++) {
    let gP = new Array(HIDDEN).fill(0);
    let gR = new Array(HIDDEN).fill(0);
    for (let i = 0; i < HIDDEN; i++) gP[i] = gR[i] = randn();
    plain[L] += Math.hypot(...gP);
    resid[L] += Math.hypot(...gR);
    for (let l = L - 1; l >= 0; l--) {
      const w = makeRandomWeights(1, scale)[0];
      // For tanh, derivative ~ 0.7 average (depends on activation magnitude); use 0.7
      const factor = 0.7;
      // Plain: g <- W^T (factor * g)
      const newP = new Array(HIDDEN).fill(0);
      const newR = new Array(HIDDEN).fill(0);
      for (let i = 0; i < HIDDEN; i++) {
        let sp = 0, sr = 0;
        for (let j = 0; j < HIDDEN; j++) {
          sp += w.W[j][i] * factor * gP[j];
          sr += w.W[j][i] * factor * gR[j];
        }
        newP[i] = sp;
        newR[i] = sr + gR[i]; // + identity contribution
      }
      gP = newP; gR = newR;
      plain[l] += Math.hypot(...gP);
      resid[l] += Math.hypot(...gR);
    }
  }
  for (let i = 0; i <= L; i++) { plain[i] /= trials; resid[i] /= trials; }
  // log scale plot
  const yMin = -5;
  const yMax = 2;
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = yMin; v <= yMax; v++) {
    const y = m.t + (1 - (v - yMin) / (yMax - yMin)) * py;
    ctx.fillText(`10^${v}`, m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const x = m.l + (i / 5) * px;
    ctx.fillText(Math.round(L * i / 5), x, m.t + py + 16);
  }
  function plotLine(arr, color, dashed) {
    ctx.strokeStyle = color; ctx.lineWidth = 2.2;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    for (let i = 0; i <= L; i++) {
      const v = Math.log10(Math.max(1e-7, arr[i]));
      const x = m.l + (i / L) * px;
      const y = m.t + (1 - (Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  plotLine(plain, LIK_COLOR, true);
  plotLine(resid, POST_COLOR, false);

  ctx.fillStyle = '#3b342b';
  ctx.font = '13px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('layer index l (loss is at l = L)', m.l + px / 2 - 90, H - 4);
  ctx.fillText('‖∂ℒ/∂h_l‖₂  (log)', m.l + 8, m.t + 14);
}

// ---------- Step 3: Train both side-by-side on a regression toy ----------
const TRAIN = {
  L: 12,
  lr: 0.05,
  step: 0,
  losses: { plain: [], resid: [] },
  netP: null,
  netR: null,
  data: null,
  running: false,
  raf: null
};

function makeNet(L, scale) {
  return makeRandomWeights(L, scale);
}

function makeRegressionData(n) {
  const data = [];
  for (let i = 0; i < n; i++) {
    const x = new Array(HIDDEN);
    for (let j = 0; j < HIDDEN; j++) x[j] = randn() * 0.5;
    // Target: tanh of weighted sum
    let target = 0;
    for (let j = 0; j < HIDDEN; j++) target += x[j] * (j % 2 === 0 ? 1 : -1) * 0.4;
    target = Math.tanh(target);
    data.push({ x, y: target });
  }
  return data;
}

function forwardWithIntermediates(net, x, residual) {
  // Returns sequence of h's; last h's first component is the prediction.
  const L = net.length;
  let h = x.slice();
  const hs = [h.slice()];
  const zs = [];
  for (let l = 0; l < L; l++) {
    const z = new Array(HIDDEN).fill(0);
    for (let i = 0; i < HIDDEN; i++) {
      let s = net[l].b[i];
      for (let j = 0; j < HIDDEN; j++) s += net[l].W[i][j] * h[j];
      z[i] = Math.tanh(s);
    }
    const next = new Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) next[i] = residual ? h[i] + z[i] : z[i];
    zs.push(z);
    h = next;
    hs.push(h.slice());
  }
  return { hs, zs, pred: h[0] };
}

function backprop(net, hs, zs, residual, dPred) {
  // Output is hs[L][0]; loss = 0.5 * (pred - target)^2 → dh_L = [dPred, 0, ...0]
  const L = net.length;
  const grads = new Array(L);
  let dh = new Array(HIDDEN).fill(0);
  dh[0] = dPred;
  for (let l = L - 1; l >= 0; l--) {
    // h_{l+1} = (residual ? h_l : 0) + tanh(W_l h_l + b_l)
    // dh_l_local = dh_{l+1} (residual contribution)
    // dz = dh_{l+1} (the tanh output goes into next directly)
    const dz = new Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      // ∂tanh/∂(W h + b) = (1 - z^2)
      dz[i] = dh[i] * (1 - zs[l][i] * zs[l][i]);
    }
    // dW_l = dz · h_l^T ;  db_l = dz
    const dW = new Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      dW[i] = new Array(HIDDEN);
      for (let j = 0; j < HIDDEN; j++) dW[i][j] = dz[i] * hs[l][j];
    }
    grads[l] = { dW, db: dz };
    // dh_l = (residual ? dh : 0) + W^T dz
    const dhNext = residual ? dh.slice() : new Array(HIDDEN).fill(0);
    for (let j = 0; j < HIDDEN; j++) {
      let s = 0;
      for (let i = 0; i < HIDDEN; i++) s += net[l].W[i][j] * dz[i];
      dhNext[j] += s;
    }
    dh = dhNext;
  }
  return grads;
}

function applyGrads(net, grads, lr) {
  const L = net.length;
  for (let l = 0; l < L; l++) {
    for (let i = 0; i < HIDDEN; i++) {
      for (let j = 0; j < HIDDEN; j++) net[l].W[i][j] -= lr * grads[l].dW[i][j];
      net[l].b[i] -= lr * grads[l].db[i];
    }
  }
}

function trainOnce(net, residual, batch, lr) {
  let loss = 0;
  // Accumulate grads
  let totalGrads = null;
  batch.forEach((ex) => {
    const fwd = forwardWithIntermediates(net, ex.x, residual);
    const e = fwd.pred - ex.y;
    loss += 0.5 * e * e;
    const grads = backprop(net, fwd.hs, fwd.zs, residual, e);
    if (!totalGrads) totalGrads = grads.map((g) => ({
      dW: g.dW.map((r) => r.slice()),
      db: g.db.slice()
    }));
    else {
      for (let l = 0; l < grads.length; l++) {
        for (let i = 0; i < HIDDEN; i++) {
          for (let j = 0; j < HIDDEN; j++) totalGrads[l].dW[i][j] += grads[l].dW[i][j];
          totalGrads[l].db[i] += grads[l].db[i];
        }
      }
    }
  });
  // Average
  const N = batch.length;
  for (let l = 0; l < totalGrads.length; l++) {
    for (let i = 0; i < HIDDEN; i++) {
      for (let j = 0; j < HIDDEN; j++) totalGrads[l].dW[i][j] /= N;
      totalGrads[l].db[i] /= N;
    }
  }
  // Gradient clipping
  let totalNorm = 0;
  for (let l = 0; l < totalGrads.length; l++) {
    for (let i = 0; i < HIDDEN; i++) {
      for (let j = 0; j < HIDDEN; j++) totalNorm += totalGrads[l].dW[i][j] ** 2;
      totalNorm += totalGrads[l].db[i] ** 2;
    }
  }
  totalNorm = Math.sqrt(totalNorm);
  const clipAt = 5.0;
  if (totalNorm > clipAt) {
    const scale = clipAt / totalNorm;
    for (let l = 0; l < totalGrads.length; l++) {
      for (let i = 0; i < HIDDEN; i++) {
        for (let j = 0; j < HIDDEN; j++) totalGrads[l].dW[i][j] *= scale;
        totalGrads[l].db[i] *= scale;
      }
    }
  }
  applyGrads(net, totalGrads, lr);
  return loss / N;
}

function renderLossCanvas() {
  const canvas = document.getElementById('loss-canvas');
  if (!canvas) return;
  const W = 880, H = 280;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 18, t: 24, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  const N = Math.max(TRAIN.losses.plain.length, TRAIN.losses.resid.length);
  if (N === 0) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText('Press Start training to begin.', m.l + px / 2, m.t + py / 2);
    return;
  }
  // log10 of loss
  const allValues = TRAIN.losses.plain.concat(TRAIN.losses.resid);
  const minV = Math.min(...allValues, 0.001);
  const maxV = Math.max(...allValues, 1);
  const minLog = Math.floor(Math.log10(Math.max(minV, 1e-6)));
  const maxLog = Math.ceil(Math.log10(Math.max(maxV, 0.01)));
  const range = Math.max(1, maxLog - minLog);
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = minLog; v <= maxLog; v++) {
    const y = m.t + (1 - (v - minLog) / range) * py;
    ctx.fillText(`10^${v}`, m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const x = m.l + (i / 5) * px;
    ctx.fillText(Math.round(N * i / 5), x, m.t + py + 16);
  }
  function plotSeries(arr, color, dashed) {
    if (arr.length < 1) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2.2;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = m.l + (i / Math.max(1, N - 1)) * px;
      const lv = Math.log10(Math.max(v, 1e-7));
      const y = m.t + (1 - (lv - minLog) / range) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  plotSeries(TRAIN.losses.plain, LIK_COLOR, true);
  plotSeries(TRAIN.losses.resid, POST_COLOR, false);

  ctx.fillStyle = '#3b342b';
  ctx.font = '13px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('plain', m.l + px - 130, m.t + 18);
  ctx.fillText('residual', m.l + px - 70, m.t + 18);
  ctx.strokeStyle = LIK_COLOR; ctx.setLineDash([6,4]); ctx.beginPath();
  ctx.moveTo(m.l + px - 145, m.t + 14); ctx.lineTo(m.l + px - 130, m.t + 14); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = POST_COLOR; ctx.beginPath();
  ctx.moveTo(m.l + px - 90, m.t + 14); ctx.lineTo(m.l + px - 70, m.t + 14); ctx.stroke();
}

function refreshTrainStats() {
  document.getElementById('train-step').textContent = TRAIN.step;
  const lp = TRAIN.losses.plain;
  const lr = TRAIN.losses.resid;
  document.getElementById('plain-loss').textContent = lp.length ? lp[lp.length - 1].toFixed(4) : '—';
  document.getElementById('resid-loss').textContent = lr.length ? lr[lr.length - 1].toFixed(4) : '—';
}

function trainLoop() {
  if (!TRAIN.running) return;
  for (let i = 0; i < 4; i++) {
    const batch = [];
    for (let j = 0; j < 16; j++) batch.push(TRAIN.data[Math.floor(Math.random() * TRAIN.data.length)]);
    const lp = trainOnce(TRAIN.netP, false, batch, TRAIN.lr);
    const lr = trainOnce(TRAIN.netR, true, batch, TRAIN.lr);
    TRAIN.losses.plain.push(lp);
    TRAIN.losses.resid.push(lr);
    TRAIN.step++;
  }
  if (TRAIN.losses.plain.length > 800) {
    TRAIN.losses.plain = TRAIN.losses.plain.slice(-800);
    TRAIN.losses.resid = TRAIN.losses.resid.slice(-800);
  }
  renderLossCanvas();
  refreshTrainStats();
  TRAIN.raf = requestAnimationFrame(trainLoop);
}

function resetTrain() {
  TRAIN.netP = makeNet(TRAIN.L, 0.6);
  // Use the *same* initial weights for residual net for fairness
  TRAIN.netR = TRAIN.netP.map((l) => ({
    W: l.W.map((r) => r.slice()),
    b: l.b.slice()
  }));
  TRAIN.step = 0;
  TRAIN.losses = { plain: [], resid: [] };
  if (!TRAIN.data) TRAIN.data = makeRegressionData(120);
  renderLossCanvas();
  refreshTrainStats();
}

function wireTrain() {
  const Lel = document.getElementById('train-L');
  const lrEl = document.getElementById('train-lr');
  Lel.addEventListener('input', () => {
    TRAIN.L = parseInt(Lel.value, 10);
    document.getElementById('train-L-val').textContent = TRAIN.L;
    if (TRAIN.raf) cancelAnimationFrame(TRAIN.raf);
    TRAIN.running = false;
    document.getElementById('train-toggle').textContent = 'Start training';
    resetTrain();
    renderActCanvas();
    renderGradCanvas();
  });
  lrEl.addEventListener('input', () => {
    TRAIN.lr = parseFloat(lrEl.value);
    document.getElementById('train-lr-val').textContent = TRAIN.lr.toFixed(3);
  });
  const tog = document.getElementById('train-toggle');
  tog.addEventListener('click', () => {
    TRAIN.running = !TRAIN.running;
    tog.textContent = TRAIN.running ? 'Pause' : 'Start training';
    if (TRAIN.running) trainLoop();
    else if (TRAIN.raf) cancelAnimationFrame(TRAIN.raf);
  });
  document.getElementById('train-reset').addEventListener('click', () => {
    if (TRAIN.raf) cancelAnimationFrame(TRAIN.raf);
    TRAIN.running = false;
    tog.textContent = 'Start training';
    resetTrain();
  });
  resetTrain();
}

// ---------- Step 4: Loss surface 2D slice ----------
function lossOnRegression(net, residual, data) {
  let s = 0;
  for (const ex of data) {
    const fwd = forwardWithIntermediates(net, ex.x, residual);
    const e = fwd.pred - ex.y;
    s += 0.5 * e * e;
  }
  return s / data.length;
}

function makeDir(net) {
  return net.map((l) => ({
    W: l.W.map((r) => r.map(() => randn())),
    b: l.b.map(() => randn())
  }));
}

function perturbedNet(net, dir1, dir2, a, b) {
  return net.map((l, i) => ({
    W: l.W.map((r, ii) => r.map((v, j) => v + a * dir1[i].W[ii][j] + b * dir2[i].W[ii][j])),
    b: l.b.map((v, ii) => v + a * dir1[i].b[ii] + b * dir2[i].b[ii])
  }));
}

function renderSurface(canvasId, residual) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = 380, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 36, r: 16, t: 16, b: 26 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  // Build a small net at L=24
  const L = 24;
  const net = makeNet(L, 0.6);
  if (!TRAIN.data) TRAIN.data = makeRegressionData(120);
  // Train it briefly to get a meaningful center
  for (let s = 0; s < 60; s++) {
    const batch = [];
    for (let j = 0; j < 16; j++) batch.push(TRAIN.data[Math.floor(Math.random() * TRAIN.data.length)]);
    trainOnce(net, residual, batch, 0.05);
  }
  const dir1 = makeDir(net);
  const dir2 = makeDir(net);
  // Filter normalisation (Li et al.) — scale dirs to match net's per-layer norms
  for (let l = 0; l < L; l++) {
    let nW = 0;
    for (let i = 0; i < HIDDEN; i++) for (let j = 0; j < HIDDEN; j++) nW += net[l].W[i][j] * net[l].W[i][j];
    nW = Math.sqrt(nW) || 1;
    let n1 = 0, n2 = 0;
    for (let i = 0; i < HIDDEN; i++) for (let j = 0; j < HIDDEN; j++) {
      n1 += dir1[l].W[i][j] * dir1[l].W[i][j];
      n2 += dir2[l].W[i][j] * dir2[l].W[i][j];
    }
    n1 = Math.sqrt(n1) || 1; n2 = Math.sqrt(n2) || 1;
    for (let i = 0; i < HIDDEN; i++) for (let j = 0; j < HIDDEN; j++) {
      dir1[l].W[i][j] *= nW / n1;
      dir2[l].W[i][j] *= nW / n2;
    }
  }
  // Sample a small grid
  const grid = 18;
  const losses = new Array(grid);
  let lossMin = Infinity, lossMax = -Infinity;
  for (let gx = 0; gx < grid; gx++) {
    losses[gx] = new Array(grid);
    for (let gy = 0; gy < grid; gy++) {
      const a = (gx / (grid - 1) - 0.5) * 0.6;
      const b = (gy / (grid - 1) - 0.5) * 0.6;
      const newNet = perturbedNet(net, dir1, dir2, a, b);
      let l = lossOnRegression(newNet, residual, TRAIN.data.slice(0, 30));
      losses[gx][gy] = l;
      if (l < lossMin) lossMin = l;
      if (l > lossMax) lossMax = l;
    }
  }
  // Render heatmap
  const cellW = px / grid, cellH = py / grid;
  for (let gx = 0; gx < grid; gx++) {
    for (let gy = 0; gy < grid; gy++) {
      const t = (losses[gx][gy] - lossMin) / Math.max(1e-6, lossMax - lossMin);
      const r = Math.round(253 - 70 * t);
      const g = Math.round(252 - 100 * t);
      const b = Math.round(249 - 130 * t);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(m.l + gx * cellW, m.t + gy * cellH, cellW + 1, cellH + 1);
    }
  }
  // Contours — simple: draw level sets via threshold
  const levels = 7;
  for (let lvl = 1; lvl <= levels; lvl++) {
    const thresh = lossMin + (lossMax - lossMin) * (lvl / (levels + 1));
    ctx.strokeStyle = `rgba(30,119,112,${0.15 + 0.06 * lvl})`;
    ctx.lineWidth = 1;
    for (let gx = 0; gx < grid - 1; gx++) {
      for (let gy = 0; gy < grid - 1; gy++) {
        const a = losses[gx][gy], b = losses[gx + 1][gy], c = losses[gx][gy + 1], d = losses[gx + 1][gy + 1];
        const pass = (a < thresh) + (b < thresh) + (c < thresh) + (d < thresh);
        if (pass > 0 && pass < 4) {
          ctx.strokeRect(m.l + (gx + 0.5) * cellW, m.t + (gy + 0.5) * cellH, cellW, cellH);
        }
      }
    }
  }
  // Center marker
  ctx.fillStyle = '#1a1815';
  ctx.beginPath();
  ctx.arc(m.l + px / 2, m.t + py / 2, 3.5, 0, Math.PI * 2);
  ctx.fill();
}

// ---------- Math + boot ----------
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-jacobian':
      '\\frac{\\partial h_{l+1}}{\\partial h_l} = I + \\frac{\\partial F(h_l)}{\\partial h_l}',
    'math-stream':
      'h_{l+1} = h_l + F_l(h_l) \\;\\;\\Longrightarrow\\;\\; h_L = h_0 + \\sum_{l=0}^{L-1} F_l(h_l)'
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
  document.getElementById('act-L').addEventListener('input', () => { renderActCanvas(); renderGradCanvas(); });
  document.getElementById('act-scale').addEventListener('input', () => { renderActCanvas(); renderGradCanvas(); });
  renderActCanvas();
  renderGradCanvas();
  wireTrain();
  // Surface (single render — slow)
  setTimeout(() => {
    renderSurface('surface-plain', false);
    renderSurface('surface-resid', true);
  }, 50);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
