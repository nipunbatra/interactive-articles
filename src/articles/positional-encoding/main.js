// ============================================================
// Positional Encodings — Sinusoid, RoPE, ALiBi
// Five interactives, all canvas-based:
//   1) Sinusoidal heatmap (Step 1)
//   2) Sinusoidal cosine-similarity matrix (Step 1)
//   3) RoPE 2-D vector rotation panel (Step 2)
//   4) RoPE dot-product matrix (Step 2)
//   5) ALiBi bias matrix + slope schedule (Step 3)
//   6) Extrapolation curves for all three (Step 4)
// ============================================================

const SIN = { L: 64, D: 32 };
const ROPE = { p1: 5, p2: 15, base: 10000, dim: 64 };
const ALI  = { head: 2, L: 32, nHeads: 8 };

// ---------------- Canvas helper ----------------
function setupCanvas(canvas, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const aspectH = opts.aspect ? Math.round(cssW * opts.aspect) : canvas.clientHeight;
  canvas.style.height = aspectH + 'px';
  canvas.width = Math.max(1, cssW * dpr);
  canvas.height = Math.max(1, aspectH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, aspectH);
  return { ctx, w: cssW, h: aspectH };
}

// ---------------- Color maps ----------------
function divergingColor(v, vMax) {
  // -1 → blue, 0 → off-white, +1 → red. v normalised by vMax.
  const t = Math.max(-1, Math.min(1, v / vMax));
  if (t >= 0) {
    const r = Math.round(253 - 110 * t);
    const g = Math.round(248 - 175 * t);
    const b = Math.round(243 - 200 * t);
    return `rgb(${r},${g},${b})`;
  } else {
    const a = -t;
    const r = Math.round(253 - 200 * a);
    const g = Math.round(248 - 145 * a);
    const b = Math.round(243 - 60 * a);
    return `rgb(${r},${g},${b})`;
  }
}

function sequentialColor(v, vMax) {
  // 0 → off-white, 1 → deep blue. For non-negative magnitudes.
  const t = Math.max(0, Math.min(1, v / vMax));
  const r = Math.round(253 - 213 * t);
  const g = Math.round(252 - 144 * t);
  const b = Math.round(249 - 65 * t);
  return `rgb(${r},${g},${b})`;
}

// =====================================================
// STEP 1: Sinusoidal heatmap
// =====================================================
function sinusoidalPE(L, d) {
  const pe = new Float32Array(L * d);
  for (let pos = 0; pos < L; pos++) {
    for (let i = 0; i < d; i++) {
      const pair = Math.floor(i / 2);
      const angle = pos / Math.pow(10000, (2 * pair) / d);
      pe[pos * d + i] = (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
    }
  }
  return pe;
}

function renderSinHeatmap() {
  const canvas = document.getElementById('sin-heatmap');
  const { ctx, w, h } = setupCanvas(canvas, { aspect: SIN.L / SIN.D * 0.45 + 0.15 });
  const pe = sinusoidalPE(SIN.L, SIN.D);
  const cellW = w / SIN.D, cellH = h / SIN.L;
  for (let pos = 0; pos < SIN.L; pos++) {
    for (let i = 0; i < SIN.D; i++) {
      const v = pe[pos * SIN.D + i];
      ctx.fillStyle = divergingColor(v, 1.0);
      ctx.fillRect(i * cellW, pos * cellH, cellW + 0.5, cellH + 0.5);
    }
  }
}

function renderSinSim() {
  const canvas = document.getElementById('sin-sim');
  const { ctx, w, h } = setupCanvas(canvas, { aspect: 1 });
  const L = SIN.L, d = SIN.D;
  const pe = sinusoidalPE(L, d);
  // Cosine similarity. Pre-compute norms.
  const norm = new Float32Array(L);
  for (let i = 0; i < L; i++) {
    let s = 0;
    for (let k = 0; k < d; k++) { const v = pe[i * d + k]; s += v * v; }
    norm[i] = Math.sqrt(s) || 1e-9;
  }
  const cell = Math.min(w, h) / L;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      let dot = 0;
      for (let k = 0; k < d; k++) dot += pe[i * d + k] * pe[j * d + k];
      const sim = dot / (norm[i] * norm[j]);
      ctx.fillStyle = divergingColor(sim, 1.0);
      ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
    }
  }
}

// =====================================================
// STEP 2: RoPE
// =====================================================
function rotate2D(x, y, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return { x: c * x - s * y, y: s * x + c * y };
}

function renderRopeVectors() {
  const canvas = document.getElementById('rope-vectors');
  const { ctx, w, h } = setupCanvas(canvas, { aspect: 0.85 });

  // Two original (unrotated) Q and K vectors.
  const q0 = { x: 0.85, y: 0.25 };
  const k0 = { x: 0.55, y: 0.75 };
  const baseAngle = (pos, dim) => pos * Math.pow(ROPE.base, -2 * dim / ROPE.dim);

  // Show the rotation for the first 2-D pair (dim=0).
  const a1 = baseAngle(ROPE.p1, 0);
  const a2 = baseAngle(ROPE.p2, 0);
  const q = rotate2D(q0.x, q0.y, a1);
  const k = rotate2D(k0.x, k0.y, a2);

  // Frame
  const cx = w / 2, cy = h / 2;
  const scale = Math.min(w, h) * 0.32;
  ctx.strokeStyle = '#bdb29c'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - scale * 1.1, cy); ctx.lineTo(cx + scale * 1.1, cy);
  ctx.moveTo(cx, cy - scale * 1.1); ctx.lineTo(cx, cy + scale * 1.1);
  ctx.stroke();

  // Unit circle
  ctx.strokeStyle = 'rgba(0,0,0,0.07)';
  ctx.beginPath(); ctx.arc(cx, cy, scale, 0, Math.PI * 2); ctx.stroke();

  // Original Q and K (dashed faint)
  function drawArrow(x0, y0, fx, fy, color, label, dashed = false) {
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.setLineDash(dashed ? [4, 4] : []);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(fx, fy); ctx.stroke();
    // arrow head
    const ang = Math.atan2(fy - y0, fx - x0);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(fx, fy);
    ctx.lineTo(fx - 9 * Math.cos(ang - 0.4), fy - 9 * Math.sin(ang - 0.4));
    ctx.lineTo(fx - 9 * Math.cos(ang + 0.4), fy - 9 * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
    ctx.setLineDash([]);
    // Label
    ctx.fillStyle = color; ctx.font = 'bold 12px IBM Plex Mono, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(label, fx + 6, fy - 4);
  }
  drawArrow(cx, cy, cx + q0.x * scale, cy - q0.y * scale, '#9a9084', 'Q₀', true);
  drawArrow(cx, cy, cx + k0.x * scale, cy - k0.y * scale, '#bdb29c', 'K₀', true);
  drawArrow(cx, cy, cx + q.x * scale, cy - q.y * scale, '#2c6fb7', `Q(p=${ROPE.p1})`);
  drawArrow(cx, cy, cx + k.x * scale, cy - k.y * scale, '#7e4ea3', `K(p=${ROPE.p2})`);

  // Dot product between rotated Q and K = original dot rotated by (a2 - a1).
  const dot = q.x * k.x + q.y * k.y;
  const dotOriginal = q0.x * k0.x + q0.y * k0.y;
  ctx.fillStyle = '#2f2a22'; ctx.font = 'bold 12px IBM Plex Mono, monospace';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText(`⟨Q(p₁), K(p₂)⟩ = ${dot.toFixed(3)}`, 12, 12);
  ctx.fillStyle = '#9a9084';
  ctx.fillText(`⟨Q₀, K₀⟩ = ${dotOriginal.toFixed(3)}  (unrotated)`, 12, 28);

  const readout = document.getElementById('rope-readout');
  readout.innerHTML =
    `rotation angles for dim 0: θ(p₁)=<strong>${(a1).toFixed(3)}</strong> rad, ` +
    `θ(p₂)=<strong>${(a2).toFixed(3)}</strong> rad. ` +
    `Offset Δ = p₂ − p₁ = <strong>${ROPE.p2 - ROPE.p1}</strong>. ` +
    `Try p=(5,15) and p=(20,30) — same Δ, same dot product.`;
}

function renderRopeDotmap() {
  const canvas = document.getElementById('rope-dotmap');
  const { ctx, w, h } = setupCanvas(canvas, { aspect: 1 });
  const N = 32;
  // Stable per-pair Q0 and K0 vectors so the map is meaningful.
  const Q0 = { x: 0.85, y: 0.25 };
  const K0 = { x: 0.55, y: 0.75 };
  // Compute dot product for all pairs (i, j) of positions in [0, N).
  const cell = Math.min(w, h) / N;
  let vMax = 0;
  const vals = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    const aQ = i * Math.pow(ROPE.base, 0);  // for dim=0, theta = pos
    const q = rotate2D(Q0.x, Q0.y, aQ);
    for (let j = 0; j < N; j++) {
      const aK = j * Math.pow(ROPE.base, 0);
      const k = rotate2D(K0.x, K0.y, aK);
      const d = q.x * k.x + q.y * k.y;
      vals[i * N + j] = d;
      if (Math.abs(d) > vMax) vMax = Math.abs(d);
    }
  }
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      ctx.fillStyle = divergingColor(vals[i * N + j], vMax);
      ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
    }
  }
  // Highlight current p1, p2
  if (ROPE.p1 < N && ROPE.p2 < N) {
    ctx.strokeStyle = '#1a1815'; ctx.lineWidth = 2;
    ctx.strokeRect(ROPE.p2 * cell, ROPE.p1 * cell, cell, cell);
  }
}

// =====================================================
// STEP 3: ALiBi
// =====================================================
function alibiSlope(headIdx, nHeads) {
  // The canonical ALiBi geometric sequence: m_h = 2^{-8h/H}.
  return Math.pow(2, -8 * (headIdx + 1) / nHeads);
}

function renderAlibiBias() {
  const canvas = document.getElementById('ali-bias');
  const { ctx, w, h } = setupCanvas(canvas, { aspect: 1 });
  const L = ALI.L;
  const m = alibiSlope(ALI.head, ALI.nHeads);
  const cell = Math.min(w, h) / L;
  // Bias = -m * |i - j|. Largest absolute value at the corners (L-1).
  const vMax = m * (L - 1);
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      if (j > i) {
        // Causal: future tokens are masked entirely.
        ctx.fillStyle = '#f3eee3';
        ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
        continue;
      }
      const bias = -m * Math.abs(i - j);
      // Negative-valued; sequentialColor mapping over magnitude.
      ctx.fillStyle = sequentialColor(-bias, vMax || 1);
      ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
    }
  }
  document.getElementById('ali-slope-val').textContent = `m = ${m.toFixed(3)}`;
}

function renderAlibiSlopes() {
  const canvas = document.getElementById('ali-slopes');
  const { ctx, w, h } = setupCanvas(canvas, { aspect: 0.7 });
  const H = ALI.nHeads;
  const slopes = [];
  for (let i = 0; i < H; i++) slopes.push(alibiSlope(i, H));
  const padL = 36, padR = 12, padT = 14, padB = 22;
  const innerW = w - padL - padR, innerH = h - padT - padB;
  const barW = innerW / H;
  const maxS = Math.max(...slopes);
  for (let i = 0; i < H; i++) {
    const bh = (slopes[i] / maxS) * innerH;
    const x = padL + i * barW + 2;
    const y = h - padB - bh;
    ctx.fillStyle = (i === ALI.head) ? '#d9622b' : '#bdb29c';
    ctx.fillRect(x, y, Math.max(2, barW - 4), bh);
    ctx.fillStyle = '#6e665b'; ctx.font = '10px IBM Plex Mono, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`h=${i}`, x + barW / 2 - 2, h - padB + 4);
    ctx.fillText(slopes[i].toFixed(3), x + barW / 2 - 2, y - 14);
  }
  // y-axis label
  ctx.fillStyle = '#9a9084'; ctx.font = '10px Manrope, sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText('per-head slope m', 4, 2);
}

// =====================================================
// STEP 4: Extrapolation curves (synthetic)
// =====================================================
function renderExtrapolation() {
  const trainLen = 2048;
  const maxLen = 32768;
  const N = 80;
  const distances = Array.from({ length: N }, (_, i) => Math.round(64 * Math.pow(maxLen / 64, i / (N - 1))));

  // Synthetic curves that mirror the published qualitative findings:
  //   sinusoidal collapses past trainLen,
  //   RoPE drifts down,
  //   ALiBi stays roughly flat.
  function sinScore(d) {
    if (d <= trainLen) return 1 - 0.05 * (d / trainLen);
    const over = Math.log(d / trainLen);
    return Math.max(0.05, 0.95 - 0.55 * over);  // sharp drop
  }
  function ropeScore(d) {
    if (d <= trainLen) return 1 - 0.04 * (d / trainLen);
    const over = Math.log(d / trainLen);
    return Math.max(0.30, 0.96 - 0.20 * over);  // gradual drop
  }
  function alibiScore(d) {
    if (d <= trainLen) return 1 - 0.03 * (d / trainLen);
    const over = Math.log(d / trainLen);
    return Math.max(0.78, 0.97 - 0.04 * over);  // nearly flat
  }

  function drawCurve(canvasId, scoreFn, color) {
    const canvas = document.getElementById(canvasId);
    const { ctx, w, h } = setupCanvas(canvas, { aspect: 0.8 });
    const padL = 38, padR = 10, padT = 16, padB = 28;
    const innerW = w - padL - padR, innerH = h - padT - padB;
    // log x-axis, linear y in [0, 1]
    const xMin = Math.log10(64), xMax = Math.log10(maxLen);
    const xToPx = (d) => padL + ((Math.log10(d) - xMin) / (xMax - xMin)) * innerW;
    const yToPx = (v) => padT + (1 - v) * innerH;
    // Axis grid
    ctx.strokeStyle = '#e8e2d8'; ctx.lineWidth = 1;
    ctx.fillStyle = '#9a9084'; ctx.font = '9px IBM Plex Mono, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    [64, 256, 1024, 4096, 16384].forEach(d => {
      if (d > maxLen) return;
      const x = xToPx(d);
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, h - padB); ctx.stroke();
      ctx.fillText(d >= 1024 ? `${d/1024}k` : d, x, h - padB + 4);
    });
    // Training-length marker
    ctx.strokeStyle = 'rgba(217,98,43,0.5)';
    const xT = xToPx(trainLen);
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(xT, padT); ctx.lineTo(xT, h - padB); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(217,98,43,0.8)';
    ctx.textAlign = 'left';
    ctx.fillText('trained ≤ 2k', xT + 3, padT + 2);
    // y-axis labels
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillStyle = '#9a9084';
    [0, 0.5, 1].forEach(v => {
      ctx.fillText(v.toFixed(1), padL - 4, yToPx(v));
    });
    // Curve
    ctx.strokeStyle = color; ctx.lineWidth = 2.4;
    ctx.beginPath();
    distances.forEach((d, i) => {
      const x = xToPx(d), y = yToPx(scoreFn(d));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawCurve('ext-sin', sinScore, '#2c6fb7');
  drawCurve('ext-rope', ropeScore, '#7e4ea3');
  drawCurve('ext-alibi', alibiScore, '#d9622b');
}

// =====================================================
// Wiring
// =====================================================
function wireSinusoidal() {
  document.getElementById('sin-L').addEventListener('input', (e) => {
    SIN.L = parseInt(e.target.value, 10);
    document.getElementById('sin-L-val').textContent = SIN.L;
    renderSinHeatmap(); renderSinSim();
  });
  document.getElementById('sin-D').addEventListener('input', (e) => {
    SIN.D = parseInt(e.target.value, 10);
    document.getElementById('sin-D-val').textContent = SIN.D;
    renderSinHeatmap(); renderSinSim();
  });
  renderSinHeatmap();
  renderSinSim();
}

function wireRope() {
  function syncP() {
    document.getElementById('rope-p1-val').textContent = ROPE.p1;
    document.getElementById('rope-p2-val').textContent = ROPE.p2;
    document.getElementById('rope-base-val').textContent = ROPE.base;
  }
  document.getElementById('rope-p1').addEventListener('input', (e) => {
    ROPE.p1 = parseInt(e.target.value, 10); syncP();
    renderRopeVectors(); renderRopeDotmap();
  });
  document.getElementById('rope-p2').addEventListener('input', (e) => {
    ROPE.p2 = parseInt(e.target.value, 10); syncP();
    renderRopeVectors(); renderRopeDotmap();
  });
  document.getElementById('rope-base').addEventListener('input', (e) => {
    ROPE.base = parseInt(e.target.value, 10); syncP();
    renderRopeVectors(); renderRopeDotmap();
  });
  syncP();
  renderRopeVectors();
  renderRopeDotmap();
}

function wireAlibi() {
  function sync() {
    document.getElementById('ali-head-val').textContent = ALI.head;
    document.getElementById('ali-L-val').textContent = ALI.L;
  }
  document.getElementById('ali-head').addEventListener('input', (e) => {
    ALI.head = parseInt(e.target.value, 10); sync();
    renderAlibiBias(); renderAlibiSlopes();
  });
  document.getElementById('ali-L').addEventListener('input', (e) => {
    ALI.L = parseInt(e.target.value, 10); sync();
    renderAlibiBias();
  });
  sync();
  renderAlibiBias();
  renderAlibiSlopes();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-sinusoidal':
      '\\mathrm{PE}_{(\\text{pos},\\, 2i)} = \\sin\\!\\left(\\frac{\\text{pos}}{10000^{2i/d}}\\right), \\quad \\mathrm{PE}_{(\\text{pos},\\, 2i+1)} = \\cos\\!\\left(\\frac{\\text{pos}}{10000^{2i/d}}\\right)',
    'math-rope':
      'R_\\theta(\\text{pos}) \\, \\begin{bmatrix} q_{2i} \\\\ q_{2i+1} \\end{bmatrix} \\;=\\; \\begin{bmatrix} \\cos(\\text{pos} \\cdot \\theta_i) & -\\sin(\\text{pos} \\cdot \\theta_i) \\\\ \\sin(\\text{pos} \\cdot \\theta_i) & \\cos(\\text{pos} \\cdot \\theta_i) \\end{bmatrix} \\begin{bmatrix} q_{2i} \\\\ q_{2i+1} \\end{bmatrix}',
    'math-alibi':
      '\\text{score}(i, j) = \\frac{Q_i K_j^\\top}{\\sqrt d} - m_h \\cdot |i - j|, \\quad m_h = 2^{-8h / H}'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

function boot() {
  wireSinusoidal();
  wireRope();
  wireAlibi();
  renderExtrapolation();
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  let timer = null;
  window.addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      renderSinHeatmap(); renderSinSim();
      renderRopeVectors(); renderRopeDotmap();
      renderAlibiBias(); renderAlibiSlopes();
      renderExtrapolation();
    }, 120);
  });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
