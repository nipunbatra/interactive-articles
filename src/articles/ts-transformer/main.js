// ============================================================
// Time-series Transformers — synthetic AQ-flavour signal + four
// forecasters: seasonal-naive, ARIMA-flavour (exp smoothing + AR),
// LSTM-flavour (decaying-trend + recent average), and a
// PatchTST-flavour (linear regression on patch-mean features +
// seasonal carryover). All four forecast deterministically; this
// is for intuition, not accuracy claims.
// ============================================================

const TICK = '#9a917f';
const STATE = {
  series: null,
  horizon: 96,
  patch: 16,
  seasonalDay: 24,
  seasonalWeek: 24 * 7
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

function makeSeries(n) {
  const out = new Array(n);
  for (let t = 0; t < n; t++) {
    const daily = 0.8 * Math.sin(2 * Math.PI * t / 24);
    const weekly = 0.4 * Math.sin(2 * Math.PI * t / (24 * 7));
    const drift = 0.0008 * t;
    const noise = 0.15 * randn();
    out[t] = 1.5 + daily + weekly + drift + noise;
  }
  return out;
}

// ---------- Forecasters ----------
function seasonalNaive(context, h, S) {
  const out = new Array(h);
  for (let i = 0; i < h; i++) {
    const idx = context.length - S + (i % S);
    out[i] = context[Math.max(0, idx)];
  }
  return out;
}

function arimaFlavour(context, h) {
  // Exp smoothing + AR(1) residual on top of seasonal-naive
  const S = 24;
  const seasonal = seasonalNaive(context, h, S);
  // Exp smoothing of residual
  const alpha = 0.3;
  let level = context[context.length - 1] - context[Math.max(0, context.length - 1 - S)];
  for (let t = context.length - 24; t < context.length; t++) {
    if (t - S < 0) continue;
    const resid = context[t] - context[t - S];
    level = alpha * resid + (1 - alpha) * level;
  }
  return seasonal.map((y, i) => y + level * Math.exp(-i * 0.02));
}

function lstmFlavour(context, h) {
  // Decaying trend + recent mean + soft seasonal carryover
  const tail = context.slice(-12);
  const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
  const grad = (tail[tail.length - 1] - tail[0]) / tail.length;
  const seasonal = seasonalNaive(context, h, 24);
  const out = new Array(h);
  for (let i = 0; i < h; i++) {
    const decay = Math.exp(-i * 0.04);
    out[i] = (1 - 0.7 * decay) * seasonal[i] + 0.7 * decay * (mean + grad * i);
  }
  return out;
}

function patchTSTFlavour(context, h, P) {
  // Linear regression on patch-mean features + seasonal carryover.
  // Build patches, take patch means; learn a linear map from past
  // patch means to next-h windows via least squares.
  if (context.length < P * 4) return seasonalNaive(context, h, 24);
  const Np = Math.floor(context.length / P);
  const means = new Array(Np);
  for (let i = 0; i < Np; i++) {
    let s = 0;
    for (let j = 0; j < P; j++) s += context[i * P + j];
    means[i] = s / P;
  }
  // Use last K patches' means as features; predict mean of each future patch
  const K = Math.min(8, Np - 1);
  const Hp = Math.ceil(h / P);
  // Train via shifting window over historical patches:
  // X_i = means[i:i+K], target = means[i+K]
  const X = [], y = [];
  for (let i = 0; i + K < Np; i++) {
    X.push(means.slice(i, i + K));
    y.push(means[i + K]);
  }
  // Solve linear regression: w = (X^T X)^-1 X^T y, with regularisation
  const D = K;
  const XtX = new Array(D);
  for (let i = 0; i < D; i++) {
    XtX[i] = new Array(D).fill(0);
    for (let r = 0; r < X.length; r++) {
      for (let j = 0; j < D; j++) XtX[i][j] += X[r][i] * X[r][j];
    }
    XtX[i][i] += 0.01;
  }
  const Xty = new Array(D).fill(0);
  for (let r = 0; r < X.length; r++) for (let i = 0; i < D; i++) Xty[i] += X[r][i] * y[r];
  // Cholesky solve
  const L = new Array(D); for (let i = 0; i < D; i++) L[i] = new Array(D).fill(0);
  for (let i = 0; i < D; i++) {
    for (let j = 0; j <= i; j++) {
      let s = XtX[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(1e-9, s)) : s / L[j][j];
    }
  }
  const z = new Array(D);
  for (let i = 0; i < D; i++) {
    let s = Xty[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * z[k];
    z[i] = s / L[i][i];
  }
  const w = new Array(D);
  for (let i = D - 1; i >= 0; i--) {
    let s = z[i];
    for (let k = i + 1; k < D; k++) s -= L[k][i] * w[k];
    w[i] = s / L[i][i];
  }
  // Forecast patch means iteratively
  const tail = means.slice(-K);
  const futureMeans = new Array(Hp);
  let cur = tail.slice();
  for (let i = 0; i < Hp; i++) {
    let pred = 0;
    for (let d = 0; d < D; d++) pred += w[d] * cur[d];
    futureMeans[i] = pred;
    cur.shift(); cur.push(pred);
  }
  // Expand patch means into a horizon-length series + add seasonal residual
  const seasonal = seasonalNaive(context, h, 24);
  const seasonalMean = new Array(Hp).fill(0);
  for (let i = 0; i < Hp; i++) {
    for (let j = 0; j < P; j++) seasonalMean[i] += seasonal[i * P + j] || 0;
    seasonalMean[i] /= P;
  }
  const out = new Array(h);
  for (let i = 0; i < h; i++) {
    const pi = Math.floor(i / P);
    const offset = (futureMeans[pi] || 0) - seasonalMean[pi];
    out[i] = seasonal[i] + 0.7 * offset;
  }
  return out;
}

function rmse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
  return Math.sqrt(s / a.length);
}

// ---------- Render ----------
function renderTS() {
  const canvas = document.getElementById('ts-canvas');
  if (!canvas) return;
  const W = 880, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 16, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);

  const N = STATE.series.length;
  const h = STATE.horizon;
  const cutoff = N - h;
  const context = STATE.series.slice(0, cutoff);
  const truth = STATE.series.slice(cutoff);
  const fc = {
    naive: seasonalNaive(context, h, 24),
    arima: arimaFlavour(context, h),
    lstm: lstmFlavour(context, h),
    patch: patchTSTFlavour(context, h, STATE.patch)
  };
  const all = [...STATE.series, ...fc.naive, ...fc.arima, ...fc.lstm, ...fc.patch];
  let lo = Math.min(...all), hi = Math.max(...all);
  lo -= 0.1; hi += 0.1;

  const sx = (i) => m.l + (i / (N - 1)) * px;
  const sy = (y) => m.t + (1 - (y - lo) / (hi - lo)) * py;
  // Y ticks
  ctx.fillStyle = TICK;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * (1 - i / 4);
    const y = m.t + i / 4 * py;
    ctx.fillText(v.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  // Plot context
  ctx.strokeStyle = '#1a1815';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let i = 0; i < context.length; i++) {
    const x = sx(i), y = sy(context[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // Plot truth (faint)
  ctx.strokeStyle = 'rgba(0,0,0,0.35)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  for (let i = 0; i < truth.length; i++) {
    const x = sx(cutoff + i), y = sy(truth[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // Plot forecasts
  const colors = {
    naive: '#9a917f', arima: '#2c6fb7', lstm: '#d9622b', patch: '#1e7770'
  };
  const labels = {
    naive: 'seasonal-naive', arima: 'ARIMA-ish',
    lstm: 'LSTM-ish', patch: 'PatchTST-ish'
  };
  Object.keys(fc).forEach((k) => {
    ctx.strokeStyle = colors[k];
    ctx.lineWidth = 2;
    ctx.beginPath();
    fc[k].forEach((v, i) => {
      const x = sx(cutoff + i), y = sy(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  // Cutoff line
  ctx.strokeStyle = 'rgba(217,98,43,0.55)';
  ctx.lineWidth = 1.6;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(sx(cutoff), m.t);
  ctx.lineTo(sx(cutoff), m.t + py);
  ctx.stroke();
  ctx.setLineDash([]);
  // Legend
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  let lx = m.l + 8, ly = m.t + 14;
  Object.keys(fc).forEach((k) => {
    ctx.strokeStyle = colors[k]; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke();
    ctx.fillStyle = '#3b342b';
    ctx.fillText(labels[k], lx + 18, ly + 3);
    lx += 18 + ctx.measureText(labels[k]).width + 16;
  });
  // Axis labels
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('time', m.l + px / 2, m.t + py + 20);

  // Stats
  const stats = document.getElementById('ts-stats');
  if (stats) {
    stats.innerHTML = `
      <span>RMSE on horizon (lower is better):</span>
      <span style="color:${colors.naive}">naive ${rmse(fc.naive, truth).toFixed(3)}</span>
      <span style="color:${colors.arima}">ARIMA-ish ${rmse(fc.arima, truth).toFixed(3)}</span>
      <span style="color:${colors.lstm}">LSTM-ish ${rmse(fc.lstm, truth).toFixed(3)}</span>
      <span style="color:${colors.patch}">PatchTST-ish ${rmse(fc.patch, truth).toFixed(3)}</span>
    `;
  }
}

// ---------- Architecture diagram ----------
function renderPatchArch() {
  const canvas = document.getElementById('patch-arch');
  if (!canvas) return;
  const W = 880, H = 240;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  // Sequence at top, patches below it, transformer block, forecast head
  const seqY = 50, patchY = 105;
  const seqLen = 32;
  const cellW = 18;
  const startX = (W - seqLen * cellW) / 2;
  // Sequence values
  ctx.fillStyle = '#1a1815';
  ctx.font = '13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('input sequence (length L)', W / 2, 22);
  for (let i = 0; i < seqLen; i++) {
    const v = 0.5 + 0.4 * Math.sin(i * 0.5) + 0.05 * Math.sin(i * 1.7);
    ctx.fillStyle = `rgba(44,111,183,${0.4 + 0.5 * v})`;
    ctx.fillRect(startX + i * cellW, seqY - 12, cellW - 1, 24);
  }
  // Patches (group every 4)
  ctx.font = '11px Manrope';
  ctx.fillStyle = '#1a1815';
  ctx.fillText('patches of length P (here P=4)', W / 2, patchY - 20);
  for (let p = 0; p < seqLen / 4; p++) {
    const x0 = startX + p * 4 * cellW;
    ctx.fillStyle = 'rgba(30,119,112,0.12)';
    ctx.fillRect(x0 + 2, patchY - 10, 4 * cellW - 5, 24);
    ctx.strokeStyle = 'rgba(30,119,112,0.6)';
    ctx.strokeRect(x0 + 2, patchY - 10, 4 * cellW - 5, 24);
  }
  // Transformer block
  const txY = 175;
  ctx.fillStyle = 'rgba(155,89,182,0.16)';
  ctx.fillRect(W / 2 - 240, txY - 16, 480, 32);
  ctx.strokeStyle = '#1a1815'; ctx.strokeRect(W / 2 - 240, txY - 16, 480, 32);
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('Transformer encoder over patch tokens (+ positional encoding)', W / 2, txY + 4);
  // Forecast arrow
  ctx.strokeStyle = '#1a1815';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(W / 2, txY + 16);
  ctx.lineTo(W / 2, 215);
  ctx.stroke();
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('linear forecast head → horizon H', W / 2, 230);
  // Connector lines from sequence to patches
  ctx.strokeStyle = '#9a917f'; ctx.lineWidth = 0.6;
  for (let p = 0; p < seqLen / 4; p++) {
    const cx = startX + p * 4 * cellW + 2 * cellW;
    ctx.beginPath();
    ctx.moveTo(cx, seqY + 14);
    ctx.lineTo(cx, patchY - 12);
    ctx.stroke();
  }
}

// ---------- Wire ----------
function refresh() {
  if (!STATE.series) STATE.series = makeSeries(24 * 21);
  STATE.horizon = parseInt(document.getElementById('ts-h').value, 10);
  STATE.patch = parseInt(document.getElementById('ts-p').value, 10);
  document.getElementById('ts-h-val').textContent = STATE.horizon;
  document.getElementById('ts-p-val').textContent = STATE.patch;
  renderTS();
}

function wire() {
  document.getElementById('ts-h').addEventListener('input', refresh);
  document.getElementById('ts-p').addEventListener('input', refresh);
  document.getElementById('ts-newseries').addEventListener('click', () => {
    STATE.series = makeSeries(24 * 21);
    refresh();
  });
  refresh();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-patch':
      'X \\in \\mathbb{R}^L \\;\\to\\; \\bigl[\\,X[1{:}P],\\, X[P{+}1{:}2P],\\,\\dots\\,\\bigr] \\;\\to\\; \\text{Embed} \\;\\to\\; \\text{Transformer} \\;\\to\\; \\text{Linear head} \\;\\to\\; \\hat X \\in \\mathbb{R}^H'
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
  renderPatchArch();
  wire();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
