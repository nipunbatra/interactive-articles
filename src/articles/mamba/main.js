// ============================================================
// Mamba/SSM — discrete-time scalar SSM forward pass.
// ============================================================

const L = 256;
const STATE = { sig: 'impulse', delta: 0.3, selective: false };

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function makeSignal(kind) {
  const x = new Array(L).fill(0);
  if (kind === 'impulse') {
    [40, 120, 200].forEach((t) => { x[t] = 1; });
  } else if (kind === 'square') {
    for (let t = 80; t < 180; t++) x[t] = 0.6;
  } else if (kind === 'needle') {
    for (let t = 0; t < L; t++) x[t] = 0.05 * (Math.random() * 2 - 1);
    x[170] = 1.4;
  }
  return x;
}

function ssmForward(x, deltaFixed, selective) {
  const A = -0.2;
  const B = 1.0;
  const C = 1.0;
  const h = new Array(L).fill(0);
  const out = new Array(L).fill(0);
  let state = 0;
  for (let t = 0; t < L; t++) {
    const xt = x[t];
    const dt = selective ? deltaFixed * (1 + 8 * Math.abs(xt)) : deltaFixed;
    // Discretise with bilinear (zoh-ish): A_bar = exp(dt * A), B_bar = (A_bar - 1)/A * B
    const Abar = Math.exp(dt * A);
    const Bbar = (Abar - 1) / A * B;
    state = Abar * state + Bbar * xt;
    h[t] = state;
    out[t] = C * state;
  }
  return { h, out };
}

function renderForward() {
  const canvas = document.getElementById('mb-canvas');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 14, b: 28 };
  const px = W - m.l - m.r;
  const py = (H - m.t - m.b) / 2 - 8;
  const x = makeSignal(STATE.sig);
  const { h } = ssmForward(x, STATE.delta, STATE.selective);
  const lo = Math.min(...x.concat(h.map((v) => Math.abs(v))), 0);
  let topMax = Math.max(0.05, ...x.map(Math.abs));
  let botMax = Math.max(0.05, ...h.map(Math.abs));
  // Top: input
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  ctx.strokeStyle = '#2c6fb7'; ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let t = 0; t < L; t++) {
    const xx = m.l + (t / (L - 1)) * px;
    const yy = m.t + py - (x[t] / topMax) * (py * 0.9) - py * 0.05;
    if (t === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
  }
  ctx.stroke();
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('input x_t', m.l + 6, m.t + 14);
  // Bottom: |h_t|
  const by = m.t + py + 16;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, by, px, py);
  ctx.strokeStyle = '#1e7770'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let t = 0; t < L; t++) {
    const xx = m.l + (t / (L - 1)) * px;
    const yy = by + py - (Math.abs(h[t]) / botMax) * (py * 0.92) - py * 0.04;
    if (t === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
  }
  ctx.stroke();
  ctx.fillStyle = '#3b342b';
  ctx.fillText('|h_t| (state magnitude)', m.l + 6, by + 14);
}

function renderCost() {
  const canvas = document.getElementById('mb-cost');
  if (!canvas) return;
  const W = 880, H = 240;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 14, t: 18, b: 32 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  // X: log N from 32 to 1e6; Y: log ops
  const Nmin = 32, Nmax = 1e6;
  const dState = 16; // SSM state dim used for cost
  function ssmOps(n) { return n * dState * 2; }
  function attnOps(n) { return n * n; }
  const allLogs = [];
  for (let i = 0; i <= 60; i++) {
    const N = Math.pow(10, Math.log10(Nmin) + (Math.log10(Nmax) - Math.log10(Nmin)) * (i / 60));
    allLogs.push(Math.log10(ssmOps(N)));
    allLogs.push(Math.log10(attnOps(N)));
  }
  const yLo = Math.floor(Math.min(...allLogs));
  const yHi = Math.ceil(Math.max(...allLogs));
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = yLo; v <= yHi; v++) {
    const y = m.t + (1 - (v - yLo) / (yHi - yLo)) * py;
    ctx.fillText(`10^${v}`, m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let v = Math.log10(Nmin); v <= Math.log10(Nmax) + 0.001; v++) {
    const x = m.l + (v - Math.log10(Nmin)) / (Math.log10(Nmax) - Math.log10(Nmin)) * px;
    ctx.fillText(`10^${Math.round(v)}`, x, m.t + py + 16);
  }
  function plot(fn, color, dashed, label, lx) {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    for (let i = 0; i <= 60; i++) {
      const N = Math.pow(10, Math.log10(Nmin) + (Math.log10(Nmax) - Math.log10(Nmin)) * (i / 60));
      const x = m.l + (Math.log10(N) - Math.log10(Nmin)) / (Math.log10(Nmax) - Math.log10(Nmin)) * px;
      const y = m.t + (1 - (Math.log10(fn(N)) - yLo) / (yHi - yLo)) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = '11px Manrope';
    ctx.textAlign = 'left';
    ctx.fillText(label, lx, m.t + 14);
  }
  plot(ssmOps, '#1e7770', false, 'Mamba O(N · d_state)', m.l + 8);
  plot(attnOps, '#d9622b', true, 'Attention O(N²)', m.l + 200);
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('sequence length N (log)', m.l + px / 2, m.t + py + 28);
}

function wire() {
  document.getElementById('mb-sig').addEventListener('change', (e) => {
    STATE.sig = e.target.value; renderForward();
  });
  document.getElementById('mb-d').addEventListener('input', (e) => {
    STATE.delta = parseFloat(e.target.value);
    document.getElementById('mb-d-val').textContent = STATE.delta.toFixed(2);
    renderForward();
  });
  document.getElementById('mb-sel').addEventListener('change', (e) => {
    STATE.selective = e.target.checked;
    renderForward();
  });
  renderForward();
  renderCost();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-ssm':
      'h_t = \\bar A\\,h_{t-1} + \\bar B\\, x_t,\\qquad y_t = C\\, h_t',
    'math-mamba':
      '\\bar A_t = \\exp(\\Delta_t A),\\quad \\bar B_t = \\frac{\\bar A_t - I}{A} B_t,\\qquad \\Delta_t = \\mathrm{softplus}(\\mathrm{Linear}(x_t))',
    'math-scan':
      '\\text{Define the operator}\\;(c_1, M_1) \\oplus (c_2, M_2) = (M_2\\, c_1 + c_2,\\;\\; M_2 M_1).\\;\\;\\text{It is associative; tree-reduce in } O(\\log N) \\text{ depth.}'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}
// ============================================================
// Step 3½ — Parallel-scan tree viz
// ============================================================
const SCAN = { N: 8, rnnTime: 0, animating: false, raf: null };

function renderScanTree() {
  const canvas = document.getElementById('scan-canvas');
  if (!canvas) return;
  const N = SCAN.N;
  const W = 880, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 30, r: 30, t: 30, b: 30 };
  const px = W - m.l - m.r;
  const levels = Math.ceil(Math.log2(N)) + 1;
  const py = H - m.t - m.b;
  const rowH = py / levels;
  const cellW = px / N;
  // Bottom row: input tokens 1..N
  function nodeXY(level, idx) {
    return [m.l + (idx + 0.5) * cellW, m.t + py - level * rowH];
  }
  // Edges first
  ctx.strokeStyle = 'rgba(0,0,0,0.25)';
  ctx.lineWidth = 1;
  for (let lvl = 0; lvl < levels - 1; lvl++) {
    const stride = 1 << lvl; // 1, 2, 4, ...
    for (let j = 0; j < N; j++) {
      const partner = j ^ stride;
      if (partner < j) continue;
      const a = nodeXY(lvl, j);
      const b = nodeXY(lvl, partner);
      const c = nodeXY(lvl + 1, Math.max(j, partner));
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]); ctx.lineTo(c[0], c[1]);
      ctx.moveTo(b[0], b[1]); ctx.lineTo(c[0], c[1]);
      ctx.stroke();
    }
  }
  // Nodes
  for (let lvl = 0; lvl < levels; lvl++) {
    for (let j = 0; j < N; j++) {
      const [x, y] = nodeXY(lvl, j);
      const isPrefix = lvl > 0;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = isPrefix ? '#1e7770' : '#2c6fb7';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = '10px IBM Plex Mono';
      ctx.textAlign = 'center';
      ctx.fillText(`${j + 1}`, x, y + 3);
    }
  }
  // Level labels
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'left';
  for (let lvl = 0; lvl < levels; lvl++) {
    const [x, y] = nodeXY(lvl, 0);
    const txt = lvl === 0 ? 'inputs x_t' : `level ${lvl} (stride 2^${lvl - 1})`;
    ctx.fillText(txt, m.l - 8, y - 14);
  }
}

function renderScanCost() {
  const canvas = document.getElementById('scan-cost-canvas');
  if (!canvas) return;
  const W = 880, H = 220;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 14, t: 16, b: 32 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  // X: log N from 8 to 1e6; Y: depth (steps)
  const Nmin = 8, Nmax = 1e6;
  const yMax = 25;
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = 0; v <= 25; v += 5) {
    const y = m.t + (1 - v / yMax) * py;
    ctx.fillText(v.toString(), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let v = Math.log10(Nmin); v <= Math.log10(Nmax) + 0.001; v++) {
    const x = m.l + (v - Math.log10(Nmin)) / (Math.log10(Nmax) - Math.log10(Nmin)) * px;
    ctx.fillText(`10^${Math.round(v)}`, x, m.t + py + 16);
  }
  // RNN: O(N), but we cap viz at 25
  ctx.strokeStyle = '#d9622b'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
  ctx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const N = Math.pow(10, Math.log10(Nmin) + (Math.log10(Nmax) - Math.log10(Nmin)) * (i / 60));
    const depth = Math.min(yMax, N / 8000); // visual scaling
    const x = m.l + (Math.log10(N) - Math.log10(Nmin)) / (Math.log10(Nmax) - Math.log10(Nmin)) * px;
    const y = m.t + (1 - depth / yMax) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // Parallel scan: O(log N)
  ctx.strokeStyle = '#1e7770'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const N = Math.pow(10, Math.log10(Nmin) + (Math.log10(Nmax) - Math.log10(Nmin)) * (i / 60));
    const depth = Math.log2(N);
    const x = m.l + (Math.log10(N) - Math.log10(Nmin)) / (Math.log10(Nmax) - Math.log10(Nmin)) * px;
    const y = m.t + (1 - depth / yMax) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // Marker for current N
  const xc = m.l + (Math.log10(SCAN.N) - Math.log10(Nmin)) / (Math.log10(Nmax) - Math.log10(Nmin)) * px;
  ctx.strokeStyle = 'rgba(0,0,0,0.5)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(xc, m.t); ctx.lineTo(xc, m.t + py); ctx.stroke();
  ctx.setLineDash([]);
  // Legend
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('parallel scan O(log N)', m.l + 8, m.t + 14);
  ctx.fillStyle = '#9c3f15';
  ctx.fillText('sequential RNN O(N) (clipped at 25)', m.l + 200, m.t + 14);
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('sequence length N (log)', m.l + px / 2, m.t + py + 28);
  ctx.save();
  ctx.translate(14, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('depth (sequential steps)', 0, 0);
  ctx.restore();
}

function wireScan() {
  const Nslider = document.getElementById('scan-N');
  if (!Nslider) return;
  Nslider.addEventListener('input', () => {
    SCAN.N = parseInt(Nslider.value, 10);
    document.getElementById('scan-N-val').textContent = SCAN.N;
    renderScanTree();
    renderScanCost();
  });
  renderScanTree();
  renderScanCost();
}

function boot() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wire();
  wireScan();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
