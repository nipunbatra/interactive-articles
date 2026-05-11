// ============================================================
// Federated Learning — 6 clients, 3-class 2D classification.
// FedAvg vs FedProx; non-IID slider; communication-rounds-vs-acc curve.
// Each client trains a small logistic-regression-on-2D model.
// ============================================================

const C = 3;
const K = 6;
const COLORS = ['#2c6fb7', '#d9622b', '#1e7770'];
const PLANE = { xMin: -3, xMax: 3, yMin: -2.2, yMax: 2.2 };

const STATE = {
  globalW: null,     // C x 3 (x, y, 1)
  clients: null,     // array of { data, classBias }
  test: null,
  history: [],
  niid: 0.5, E: 3, algo: 'fedavg'
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

function softmax(arr) {
  let m = -Infinity;
  for (const v of arr) if (v > m) m = v;
  const e = arr.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((x) => x / s);
}
function logits(W, x, y) {
  const out = new Array(C);
  for (let c = 0; c < C; c++) out[c] = W[c][0] * x + W[c][1] * y + W[c][2];
  return out;
}

function makeClassData(label, n) {
  const centers = [[-1.4, 0.0], [1.3, 0.4], [0.0, -1.5]];
  const out = [];
  for (let i = 0; i < n; i++) {
    const cx = centers[label][0], cy = centers[label][1];
    out.push({ x: cx + 0.6 * randn(), y: cy + 0.6 * randn(), label });
  }
  return out;
}

function buildClients() {
  // non-IID: each client samples a class distribution skewed toward one class
  STATE.clients = [];
  for (let k = 0; k < K; k++) {
    const preferred = k % C;
    const data = [];
    // Mixture: with prob α pick preferred class, with (1-α) uniform
    const nPerClient = 50;
    for (let i = 0; i < nPerClient; i++) {
      const useSkew = Math.random() < STATE.niid;
      const cls = useSkew ? preferred : Math.floor(Math.random() * C);
      data.push(makeClassData(cls, 1)[0]);
    }
    STATE.clients.push({ data, preferred });
  }
  STATE.test = [];
  for (let i = 0; i < 600; i++) STATE.test.push(makeClassData(i % C, 1)[0]);
}

function localTrain(globalW, clientData, E, algo) {
  // Local SGD with optional proximal term
  let W = globalW.map((r) => r.slice());
  const lr = 0.1;
  const mu = algo === 'fedprox' ? 0.1 : 0;
  for (let e = 0; e < E; e++) {
    const dW = new Array(C).fill(0).map(() => [0, 0, 0]);
    for (const ex of clientData) {
      const phi = [ex.x, ex.y, 1];
      const probs = softmax(logits(W, ex.x, ex.y));
      for (let c = 0; c < C; c++) {
        const t = (c === ex.label) ? 1 : 0;
        const g = probs[c] - t;
        for (let d = 0; d < 3; d++) dW[c][d] += g * phi[d];
      }
    }
    for (let c = 0; c < C; c++) {
      for (let d = 0; d < 3; d++) {
        const prox = mu * (W[c][d] - globalW[c][d]);
        W[c][d] -= lr * (dW[c][d] / clientData.length + prox);
      }
    }
  }
  return W;
}

function fedRound() {
  // Each client trains locally, server averages
  const clientWs = STATE.clients.map((cl) => localTrain(STATE.globalW, cl.data, STATE.E, STATE.algo));
  // Weighted average by n_k (here equal)
  const next = new Array(C).fill(0).map(() => [0, 0, 0]);
  for (const W of clientWs) {
    for (let c = 0; c < C; c++) for (let d = 0; d < 3; d++) next[c][d] += W[c][d] / K;
  }
  STATE.globalW = next;
  // Test accuracy
  let correct = 0;
  STATE.test.forEach((ex) => {
    const z = logits(STATE.globalW, ex.x, ex.y);
    let best = 0; for (let c = 1; c < C; c++) if (z[c] > z[best]) best = c;
    if (best === ex.label) correct++;
  });
  STATE.history.push({ round: STATE.history.length + 1, acc: correct / STATE.test.length });
}

function reset() {
  STATE.globalW = [];
  for (let c = 0; c < C; c++) STATE.globalW.push([randn() * 0.1, randn() * 0.1, 0]);
  STATE.history = [];
  buildClients();
}

function hex2rgba(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function renderGlobalPanel() {
  const canvas = document.getElementById('fl-global');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const step = 4;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const x = PLANE.xMin + (PLANE.xMax - PLANE.xMin) * (px / W);
      const y = PLANE.yMax - (PLANE.yMax - PLANE.yMin) * (py / H);
      const probs = softmax(logits(STATE.globalW, x, y));
      let best = 0; for (let c = 1; c < C; c++) if (probs[c] > probs[best]) best = c;
      ctx.fillStyle = hex2rgba(COLORS[best], 0.18 + 0.45 * probs[best]);
      ctx.fillRect(px, py, step, step);
    }
  }
  STATE.test.slice(0, 150).forEach((ex) => {
    const px = (ex.x - PLANE.xMin) / (PLANE.xMax - PLANE.xMin) * W;
    const py = (PLANE.yMax - ex.y) / (PLANE.yMax - PLANE.yMin) * H;
    ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[ex.label]; ctx.fill();
  });
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderClientsPanel() {
  const canvas = document.getElementById('fl-clients');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 28 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  // Stacked bars: one row per client, 3-class fractions
  const rowH = py / K;
  for (let k = 0; k < K; k++) {
    const counts = [0, 0, 0];
    STATE.clients[k].data.forEach((ex) => counts[ex.label]++);
    const total = counts.reduce((a, b) => a + b, 0);
    let x = m.l;
    for (let c = 0; c < C; c++) {
      const w = (counts[c] / total) * px;
      ctx.fillStyle = COLORS[c];
      ctx.fillRect(x, m.t + k * rowH + 2, w, rowH - 4);
      x += w;
    }
    ctx.fillStyle = '#1a1815';
    ctx.font = '11px IBM Plex Mono';
    ctx.textAlign = 'right';
    ctx.fillText(`c${k + 1}`, m.l - 6, m.t + k * rowH + rowH / 2 + 4);
  }
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('per-client class mix (skewed by α)', m.l + 6, m.t + 14);
}

function renderLossCanvas() {
  const canvas = document.getElementById('fl-loss');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 28 };
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
  if (STATE.history.length === 0) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText('Press Run 1 round to start.', m.l + px / 2, m.t + py / 2);
    return;
  }
  const N = STATE.history.length;
  ctx.strokeStyle = STATE.algo === 'fedavg' ? '#d9622b' : '#1e7770';
  ctx.lineWidth = 2;
  ctx.beginPath();
  STATE.history.forEach((h, i) => {
    const x = m.l + (i / Math.max(1, N - 1)) * px;
    const y = m.t + (1 - h.acc) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderAll() {
  renderGlobalPanel();
  renderClientsPanel();
  renderLossCanvas();
  document.getElementById('fl-round-count').textContent = STATE.history.length;
  const last = STATE.history.length ? STATE.history[STATE.history.length - 1] : null;
  document.getElementById('fl-acc').textContent = last ? `${(last.acc * 100).toFixed(1)}%` : '—';
}

function wire() {
  reset();
  document.getElementById('fl-round').addEventListener('click', () => { fedRound(); renderAll(); });
  document.getElementById('fl-round10').addEventListener('click', () => {
    for (let i = 0; i < 10; i++) fedRound(); renderAll();
  });
  document.getElementById('fl-reset').addEventListener('click', () => { reset(); renderAll(); });
  document.getElementById('fl-algo').addEventListener('change', (e) => { STATE.algo = e.target.value; });
  document.getElementById('fl-niid').addEventListener('input', (e) => {
    STATE.niid = parseFloat(e.target.value);
    document.getElementById('fl-niid-val').textContent = STATE.niid.toFixed(2);
    buildClients();
    renderAll();
  });
  document.getElementById('fl-E').addEventListener('input', (e) => {
    STATE.E = parseInt(e.target.value, 10);
    document.getElementById('fl-E-val').textContent = STATE.E;
  });
  renderAll();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-fedavg':
      '\\theta^{(t+1)} = \\sum_{k=1}^{K} \\frac{n_k}{n}\\,\\theta_k^{(t+1)}\\;\\;\\text{where}\\;\\;\\theta_k^{(t+1)} = \\theta^{(t)} - \\eta\\,\\nabla \\mathcal{L}_k(\\theta^{(t)})',
    'math-fedprox':
      '\\theta_k^{(t+1)} = \\arg\\min_w\\;\\mathcal{L}_k(w) + \\frac{\\mu}{2}\\,\\bigl\\lVert w - \\theta^{(t)} \\bigr\\rVert^2',
    'math-dp':
      '\\tilde \\theta_k = \\mathrm{clip}_C(\\theta_k - \\theta) + \\mathcal{N}(0, \\sigma^2 C^2 I),\\quad \\text{server sees}\\;\\sum_k \\tilde \\theta_k\\;\\;\\Rightarrow\\;(\\varepsilon, \\delta)\\text{-DP}'
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
