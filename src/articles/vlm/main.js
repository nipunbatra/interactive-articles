// ============================================================
// Vision-Language Models, End to End.
// Mini-CLIP: 4 toy "image" types × 4 textual "captions"; tiny image
// encoder (3-d input → 8-d embedding) and tiny text encoder (4-d
// one-hot → 8-d embedding) trained jointly with the symmetric
// InfoNCE loss. Cosine similarity matrix updates live.
// ============================================================

const POST_COLOR = '#1e7770';
const PRIOR_COLOR = '#2c6fb7';
const LIK_COLOR = '#d9622b';

const N = 4; // 4 image types, 4 captions
const D = 8; // shared embedding dim

// "Image" features: 3-d (mean RGB after a tiny pretend backbone)
const IMAGE_FEATS = [
  [0.85, 0.30, 0.20], // red square
  [0.20, 0.85, 0.35], // green disc
  [0.30, 0.45, 0.95], // blue triangle
  [0.95, 0.85, 0.30]  // yellow pentagon
];
const IMAGE_LABELS = ['red square', 'green disc', 'blue triangle', 'yellow pentagon'];
const TEXT_LABELS = ['"a red shape"', '"a green dot"', '"a blue triangle"', '"a yellow pentagon"'];

const STATE = {
  Wi: null, Bi: null, // image projector: 3 -> 8
  Wt: null, Bt: null, // text projector: 4 (one-hot) -> 8
  tau: 0.25,
  step: 0,
  losses: [],
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

function reset() {
  STATE.Wi = []; for (let i = 0; i < 3; i++) STATE.Wi.push(new Array(D).fill(0).map(() => randn() * 0.3));
  STATE.Bi = new Array(D).fill(0);
  STATE.Wt = []; for (let i = 0; i < 4; i++) STATE.Wt.push(new Array(D).fill(0).map(() => randn() * 0.3));
  STATE.Bt = new Array(D).fill(0);
  STATE.step = 0;
  STATE.losses = [];
}

function imageEmbedding(idx) {
  const f = IMAGE_FEATS[idx];
  const out = new Array(D).fill(0);
  for (let d = 0; d < D; d++) {
    let s = STATE.Bi[d];
    for (let c = 0; c < 3; c++) s += STATE.Wi[c][d] * f[c];
    out[d] = Math.tanh(s);
  }
  return out;
}
function textEmbedding(idx) {
  const out = new Array(D).fill(0);
  for (let d = 0; d < D; d++) {
    let s = STATE.Bt[d];
    s += STATE.Wt[idx][d];
    out[d] = Math.tanh(s);
  }
  return out;
}

function l2Norm(v) {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s) || 1;
}
function normalize(v) {
  const n = l2Norm(v);
  return v.map((x) => x / n);
}

function similarityMatrix() {
  const I = []; const T = [];
  for (let i = 0; i < N; i++) I.push(normalize(imageEmbedding(i)));
  for (let j = 0; j < N; j++) T.push(normalize(textEmbedding(j)));
  const S = new Array(N);
  for (let i = 0; i < N; i++) {
    S[i] = new Array(N);
    for (let j = 0; j < N; j++) {
      let dot = 0;
      for (let d = 0; d < D; d++) dot += I[i][d] * T[j][d];
      S[i][j] = dot;
    }
  }
  return { S, I, T };
}

function softmaxRow(row) {
  let m = -Infinity;
  for (const v of row) if (v > m) m = v;
  const exps = row.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / s);
}

function trainStep() {
  // Compute embeddings and gradient analytically for the symmetric InfoNCE loss
  // Loss = 0.5 * (CE_i2t + CE_t2i)
  // where CE_i2t[i] = -log softmax(S[i, :] / tau)[i]  (target = i)
  const { S, I, T } = similarityMatrix();
  const tau = STATE.tau;
  // Forward
  const rowsI = S.map((row) => row.map((v) => v / tau));
  const probsI = rowsI.map(softmaxRow);
  // Image-to-text gradient: d(CE)/dS[i,j] = (probsI[i,j] - 1[j==i]) / tau
  const colsT = []; // column-stack per text
  for (let j = 0; j < N; j++) {
    colsT.push(new Array(N));
    for (let i = 0; i < N; i++) colsT[j][i] = S[i][j] / tau;
  }
  const probsT = colsT.map(softmaxRow);

  // Loss
  let loss = 0;
  for (let i = 0; i < N; i++) loss -= 0.5 * Math.log(Math.max(probsI[i][i], 1e-12));
  for (let j = 0; j < N; j++) loss -= 0.5 * Math.log(Math.max(probsT[j][j], 1e-12));

  // dS[i, j] = 0.5 * ( (probsI[i,j] - 1[i==j]) / tau + (probsT[j,i] - 1[i==j]) / tau )
  const dS = new Array(N);
  for (let i = 0; i < N; i++) {
    dS[i] = new Array(N);
    for (let j = 0; j < N; j++) {
      const t1 = probsI[i][j] - (i === j ? 1 : 0);
      const t2 = probsT[j][i] - (i === j ? 1 : 0);
      dS[i][j] = 0.5 * (t1 + t2) / tau;
    }
  }
  // S[i, j] = (I[i] · T[j]) where I, T are unit-normalised.
  // For training simplicity, treat I, T as the post-tanh embeddings divided by their norm.
  // We'll backprop through the dot product but ignore the unit-norm derivative (it's small).
  // dI[i] += sum_j dS[i,j] * T[j] / |I[i]|  (we approximate with raw)
  const dI = new Array(N);
  const dT = new Array(N);
  for (let i = 0; i < N; i++) {
    dI[i] = new Array(D).fill(0);
    for (let j = 0; j < N; j++) {
      for (let d = 0; d < D; d++) dI[i][d] += dS[i][j] * T[j][d];
    }
  }
  for (let j = 0; j < N; j++) {
    dT[j] = new Array(D).fill(0);
    for (let i = 0; i < N; i++) {
      for (let d = 0; d < D; d++) dT[j][d] += dS[i][j] * I[i][d];
    }
  }
  // Backprop through tanh and through the projection
  // For image i: pre-tanh z_d = sum_c W_i[c][d] * f_c[i] + B_i[d]; emb = tanh(z); dz = dI * (1 - tanh^2(z)) (we approximate with dI directly since values are close)
  const lr = 0.05;
  // Update Wi, Bi
  for (let c = 0; c < 3; c++) for (let d = 0; d < D; d++) {
    let g = 0;
    for (let i = 0; i < N; i++) {
      const val = imageEmbedding(i)[d];
      const sigmaPrime = 1 - val * val;
      g += dI[i][d] * sigmaPrime * IMAGE_FEATS[i][c];
    }
    STATE.Wi[c][d] -= lr * g / N;
  }
  for (let d = 0; d < D; d++) {
    let g = 0;
    for (let i = 0; i < N; i++) {
      const val = imageEmbedding(i)[d];
      const sigmaPrime = 1 - val * val;
      g += dI[i][d] * sigmaPrime;
    }
    STATE.Bi[d] -= lr * g / N;
  }
  // Update Wt, Bt
  for (let j = 0; j < N; j++) for (let d = 0; d < D; d++) {
    const val = textEmbedding(j)[d];
    const sigmaPrime = 1 - val * val;
    STATE.Wt[j][d] -= lr * dT[j][d] * sigmaPrime;
  }
  for (let d = 0; d < D; d++) {
    let g = 0;
    for (let j = 0; j < N; j++) {
      const val = textEmbedding(j)[d];
      const sigmaPrime = 1 - val * val;
      g += dT[j][d] * sigmaPrime;
    }
    STATE.Bt[d] -= lr * g / N;
  }
  STATE.step++;
  STATE.losses.push(loss);
  if (STATE.losses.length > 1500) STATE.losses = STATE.losses.slice(-1500);
}

// ---------- Render: data panel ----------
function renderData() {
  const canvas = document.getElementById('clip-data');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const cell = W / N;
  for (let i = 0; i < N; i++) {
    // Draw image type
    const f = IMAGE_FEATS[i];
    const r = Math.round(255 * f[0]);
    const g = Math.round(255 * f[1]);
    const b = Math.round(255 * f[2]);
    const cy = i * cell + cell / 2;
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    if (i === 0) ctx.fillRect(20, cy - 18, 36, 36);
    else if (i === 1) {
      ctx.beginPath(); ctx.arc(38, cy, 18, 0, Math.PI * 2); ctx.fill();
    } else if (i === 2) {
      ctx.beginPath();
      ctx.moveTo(38, cy - 20); ctx.lineTo(20, cy + 16); ctx.lineTo(56, cy + 16); ctx.closePath(); ctx.fill();
    } else {
      ctx.beginPath();
      for (let k = 0; k < 5; k++) {
        const a = -Math.PI / 2 + 2 * Math.PI * k / 5;
        const x = 38 + 18 * Math.cos(a);
        const y = cy + 18 * Math.sin(a);
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath(); ctx.fill();
    }
    ctx.fillStyle = '#3b342b';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'left';
    ctx.fillText(IMAGE_LABELS[i], 70, cy + 4);
    ctx.fillStyle = '#6e665b';
    ctx.font = 'italic 13px Manrope';
    ctx.fillText(TEXT_LABELS[i], 200, cy + 4);
  }
  // Title
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 13px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('image     →     caption', 20, 18);
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

// ---------- Render: similarity matrix ----------
function renderSim() {
  const canvas = document.getElementById('clip-sim');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const { S } = similarityMatrix();
  const m = { l: 70, r: 14, t: 32, b: 14 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  const cw = px / N, ch = py / N;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v = S[i][j]; // cosine in [-1, 1]
      const t = (v + 1) / 2;
      const r = Math.round(217 + (44 - 217) * t);
      const g = Math.round(98 + (111 - 98) * t);
      const b = Math.round(43 + (183 - 43) * t);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(m.l + j * cw, m.t + i * ch, cw + 0.5, ch + 0.5);
      // value
      ctx.fillStyle = '#fff';
      ctx.font = '12px IBM Plex Mono';
      ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(2), m.l + j * cw + cw / 2, m.t + i * ch + ch / 2 + 4);
    }
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  // Axis labels
  ctx.fillStyle = '#1a1815';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  for (let j = 0; j < N; j++) {
    ctx.fillText(`t${j+1}`, m.l + j * cw + cw / 2, m.t - 6);
  }
  ctx.textAlign = 'right';
  for (let i = 0; i < N; i++) {
    ctx.fillText(`i${i+1}`, m.l - 6, m.t + i * ch + ch / 2 + 4);
  }
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('text →', m.l, 18);
}

// ---------- Render: loss curve ----------
function renderLoss() {
  const canvas = document.getElementById('clip-loss-canvas');
  if (!canvas) return;
  const W = 380, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 12, t: 14, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  if (STATE.losses.length === 0) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText('Press Start training', m.l + px / 2, m.t + py / 2);
    return;
  }
  const lo = Math.min(...STATE.losses);
  const hi = Math.max(...STATE.losses);
  const range = Math.max(0.01, hi - lo);
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * (1 - i / 4);
    const y = m.t + i / 4 * py;
    ctx.fillText(v.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.strokeStyle = PRIOR_COLOR;
  ctx.lineWidth = 2;
  ctx.beginPath();
  STATE.losses.forEach((v, i) => {
    const x = m.l + (i / Math.max(1, STATE.losses.length - 1)) * px;
    const y = m.t + (1 - (v - lo) / range) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ---------- Render: architecture diagram ----------
function renderArch() {
  const canvas = document.getElementById('arch-canvas');
  if (!canvas) return;
  const W = 880, H = 240;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const drawBox = (cx, cy, w, h, fill, title, sub) => {
    ctx.fillStyle = fill;
    ctx.fillRect(cx - w / 2, cy - h / 2, w, h);
    ctx.strokeStyle = '#1a1815';
    ctx.lineWidth = 1.4;
    ctx.strokeRect(cx - w / 2, cy - h / 2, w, h);
    ctx.fillStyle = '#1a1815';
    ctx.font = 'bold 13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText(title, cx, cy - 4);
    ctx.fillStyle = '#6e665b';
    ctx.font = '11px Manrope';
    ctx.fillText(sub, cx, cy + 14);
  };
  const drawArrow = (x0, y0, x1, y1, label) => {
    ctx.strokeStyle = '#1a1815';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    // arrowhead
    const a = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - 9 * Math.cos(a - Math.PI / 7), y1 - 9 * Math.sin(a - Math.PI / 7));
    ctx.lineTo(x1 - 9 * Math.cos(a + Math.PI / 7), y1 - 9 * Math.sin(a + Math.PI / 7));
    ctx.closePath();
    ctx.fillStyle = '#1a1815'; ctx.fill();
    if (label) {
      ctx.fillStyle = '#6e665b';
      ctx.font = '11px Manrope';
      ctx.textAlign = 'center';
      ctx.fillText(label, (x0 + x1) / 2, y0 - 8);
    }
  };
  // Image input
  ctx.fillStyle = 'rgba(217,98,43,0.15)';
  ctx.fillRect(40, 80, 100, 80);
  ctx.strokeStyle = '#1a1815';
  ctx.strokeRect(40, 80, 100, 80);
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('image', 90, 116);
  ctx.fillStyle = '#6e665b';
  ctx.font = '11px Manrope';
  ctx.fillText('224×224×3', 90, 134);
  drawArrow(140, 120, 200, 120);
  drawBox(260, 120, 120, 80, 'rgba(44,111,183,0.18)', 'Vision encoder', 'CLIP / DINOv2 ViT');
  drawArrow(320, 120, 380, 120, 'patch tokens');
  drawBox(440, 120, 120, 80, 'rgba(30,119,112,0.18)', 'Projector', 'MLP / Q-Former');
  drawArrow(500, 120, 560, 120, 'image tokens');
  drawBox(640, 120, 140, 80, 'rgba(155,89,182,0.18)', 'Language model', 'Llama / Qwen / GPT');
  drawArrow(710, 120, 800, 120);
  // Text outputs
  ctx.fillStyle = 'rgba(217,98,43,0.15)';
  ctx.fillRect(800, 80, 60, 80);
  ctx.strokeStyle = '#1a1815';
  ctx.strokeRect(800, 80, 60, 80);
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('text', 830, 116);
  ctx.fillStyle = '#6e665b';
  ctx.font = '11px Manrope';
  ctx.fillText('autoregressive', 830, 134);
  // Text input under projector / LLM
  ctx.fillStyle = '#6e665b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('+ instruction text tokens', 640, 220);
  drawArrow(640, 200, 640, 165);
}

// ---------- Wiring ----------
function wire() {
  reset();
  document.getElementById('clip-tau').addEventListener('input', (e) => {
    STATE.tau = parseFloat(e.target.value);
    document.getElementById('clip-tau-val').textContent = STATE.tau.toFixed(2);
  });
  const tog = document.getElementById('clip-toggle');
  tog.addEventListener('click', () => {
    STATE.running = !STATE.running;
    tog.textContent = STATE.running ? 'Pause' : 'Start training';
    if (STATE.running) loop();
    else if (STATE.raf) cancelAnimationFrame(STATE.raf);
  });
  document.getElementById('clip-reset').addEventListener('click', () => {
    if (STATE.raf) cancelAnimationFrame(STATE.raf);
    STATE.running = false;
    tog.textContent = 'Start training';
    reset();
    renderAll();
  });
  renderAll();
}

function refreshStats() {
  document.getElementById('clip-step').textContent = STATE.step;
  document.getElementById('clip-loss').textContent = STATE.losses.length ? STATE.losses[STATE.losses.length - 1].toFixed(3) : '—';
}

function renderAll() {
  renderData();
  renderSim();
  renderLoss();
  refreshStats();
}

function loop() {
  if (!STATE.running) return;
  for (let i = 0; i < 4; i++) trainStep();
  renderAll();
  STATE.raf = requestAnimationFrame(loop);
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-clip':
      '\\mathcal{L}_{\\text{InfoNCE}} = -\\frac{1}{2K}\\sum_{k=1}^{K}\\!\\left[\\log\\frac{\\exp(s_{k,k}/\\tau)}{\\sum_{j} \\exp(s_{k,j}/\\tau)} + \\log\\frac{\\exp(s_{k,k}/\\tau)}{\\sum_{i} \\exp(s_{i,k}/\\tau)}\\right]',
    'math-stage2':
      '\\mathcal{L}_{\\text{cap}} = -\\sum_{t} \\log p_{\\text{LLM}}(y_t \\mid y_{<t}, \\mathrm{Proj}(\\mathrm{Enc}(I)))',
    'math-stage3':
      '\\mathcal{L}_{\\text{IT}} = -\\sum_{t \\in \\text{response}} \\log p_{\\text{LLM}}(y_t \\mid y_{<t}, \\mathrm{Proj}(\\mathrm{Enc}(I)), \\text{instr})'
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
  renderArch();
  wire();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
