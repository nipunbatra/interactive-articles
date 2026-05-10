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
      '\\bar A_t = \\exp(\\Delta_t A),\\quad \\bar B_t = \\frac{\\bar A_t - I}{A} B_t,\\qquad \\Delta_t = \\mathrm{softplus}(\\mathrm{Linear}(x_t))'
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
