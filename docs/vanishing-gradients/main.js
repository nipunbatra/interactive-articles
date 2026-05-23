// ============================================================
// Vanishing & Exploding Gradients
// Three interactives:
//   1) Single-layer derivative panel (Step 1)
//   2) Main multiplicative chain (Step 2)
//   3) Weight-scale + active-fraction advanced panel (Step 3)
// ============================================================

// ---------------- Activation functions and derivatives ----------------
const ACTS = {
  sigmoid: {
    name: 'sigmoid',
    f:  (z) => 1 / (1 + Math.exp(-z)),
    df: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); },
    yRange: [0, 1],
    dyRange: [0, 0.30],
    maxDeriv: 0.25,
    maxAt: 0.0
  },
  tanh: {
    name: 'tanh',
    f:  (z) => Math.tanh(z),
    df: (z) => 1 - Math.tanh(z) ** 2,
    yRange: [-1.05, 1.05],
    dyRange: [0, 1.05],
    maxDeriv: 1.0,
    maxAt: 0.0
  },
  relu: {
    name: 'ReLU',
    f:  (z) => Math.max(0, z),
    df: (z) => z > 0 ? 1 : 0,
    yRange: [-0.5, 6.5],
    dyRange: [-0.1, 1.2],
    maxDeriv: 1.0,
    maxAt: 1.0
  }
};

// ---------------- Canvas helpers ----------------
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = Math.max(1, w * dpr);
    canvas.height = Math.max(1, h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawCurve(ctx, w, h, fn, zRange, yRange, opts = {}) {
  const padL = 36, padR = 12, padT = 12, padB = 24;
  const innerW = w - padL - padR, innerH = h - padT - padB;
  // Axes
  ctx.strokeStyle = '#bdb29c';
  ctx.lineWidth = 1;
  // y-zero line if in range
  if (yRange[0] < 0 && yRange[1] > 0) {
    const yZero = padT + (yRange[1] / (yRange[1] - yRange[0])) * innerH;
    ctx.beginPath(); ctx.moveTo(padL, yZero); ctx.lineTo(w - padR, yZero); ctx.stroke();
  } else {
    ctx.beginPath(); ctx.moveTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
  }
  // x-zero line
  const xZero = padL + ((-zRange[0]) / (zRange[1] - zRange[0])) * innerW;
  ctx.strokeStyle = '#d8d0bd';
  ctx.beginPath(); ctx.moveTo(xZero, padT); ctx.lineTo(xZero, h - padB); ctx.stroke();

  // y-axis ticks (3 ticks)
  ctx.fillStyle = '#9a9084';
  ctx.font = '10px IBM Plex Mono, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 2; i++) {
    const y = padT + (i / 2) * innerH;
    const yVal = yRange[1] - (i / 2) * (yRange[1] - yRange[0]);
    ctx.fillText(yVal.toFixed(2), padL - 4, y);
  }
  // x-axis ticks
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let i = 0; i <= 4; i++) {
    const x = padL + (i / 4) * innerW;
    const xVal = zRange[0] + (i / 4) * (zRange[1] - zRange[0]);
    ctx.fillText(xVal.toFixed(0), x, h - padB + 4);
  }

  // Curve
  ctx.strokeStyle = opts.color || '#2c6fb7';
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  const N = 240;
  for (let i = 0; i <= N; i++) {
    const z = zRange[0] + (i / N) * (zRange[1] - zRange[0]);
    const v = fn(z);
    const x = padL + (i / N) * innerW;
    const y = padT + ((yRange[1] - v) / (yRange[1] - yRange[0])) * innerH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Dot at current z
  if (opts.dot != null) {
    const z = opts.dot, v = fn(z);
    const x = padL + ((z - zRange[0]) / (zRange[1] - zRange[0])) * innerW;
    const y = padT + ((yRange[1] - v) / (yRange[1] - yRange[0])) * innerH;
    ctx.fillStyle = '#d9622b';
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'white'; ctx.lineWidth = 1.8; ctx.stroke();
    // value label
    ctx.fillStyle = '#2f2a22';
    ctx.font = 'bold 11px IBM Plex Mono, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText(v.toFixed(3), x + 8, y - 4);
  }
}

// =====================================================
// STEP 1: derivative panel
// =====================================================
const DERIV = { act: 'sigmoid', z: 0 };

function renderDeriv() {
  const a = ACTS[DERIV.act];
  const c1 = document.getElementById('deriv-sigma');
  const c2 = document.getElementById('deriv-prime');
  const ctx1 = setupCanvas(c1);
  const ctx2 = setupCanvas(c2);
  drawCurve(ctx1.ctx, ctx1.w, ctx1.h, a.f, [-6, 6], a.yRange, { color: '#2c6fb7', dot: DERIV.z });
  drawCurve(ctx2.ctx, ctx2.w, ctx2.h, a.df, [-6, 6], a.dyRange, { color: '#d9622b', dot: DERIV.z });
  document.getElementById('deriv-z-val').textContent = DERIV.z.toFixed(2);
  document.getElementById('deriv-cur').textContent = a.df(DERIV.z).toFixed(3);
  document.getElementById('deriv-max').textContent = a.maxDeriv.toFixed(3);
}

function wireDeriv() {
  const tabs = document.querySelectorAll('#deriv-tabs .tab-btn');
  tabs.forEach((b) => b.addEventListener('click', () => {
    tabs.forEach((x) => x.classList.remove('is-active'));
    b.classList.add('is-active');
    DERIV.act = b.dataset.act;
    renderDeriv();
  }));
  document.getElementById('deriv-z').addEventListener('input', (e) => {
    DERIV.z = parseFloat(e.target.value);
    renderDeriv();
  });
  renderDeriv();
}

// =====================================================
// STEP 2: main multiplicative chain
// =====================================================
const MAIN = { act: 'tanh', depth: 12 };

function bestCaseFactor(actName) {
  // Use the activation's max derivative as the best-case chain factor.
  return ACTS[actName].maxDeriv;
}

function gradientChain(depth, factor) {
  // Input-layer gradient = factor^depth (best case).
  // Returns array of per-layer gradients: index 0 = output layer (factor^1),
  // index depth-1 = input layer (factor^depth).
  // The output layer's "gradient" passed to it from the loss is just 1; multiplying by σ' gives factor.
  const arr = [];
  for (let l = 1; l <= depth; l++) arr.push(Math.pow(factor, l));
  return arr;
}

function classify(g, exploding = false) {
  if (g > 1e2) return 'is-explode';
  if (g > 1e-3) return 'is-healthy';
  if (g >= 1e-6) return 'is-warn';
  return 'is-dead';
}

function formatGrad(g) {
  if (!isFinite(g)) return '∞';
  if (g === 0) return '0';
  const expo = Math.floor(Math.log10(Math.abs(g)));
  const mant = g / Math.pow(10, expo);
  return `${mant.toFixed(2)}e${expo >= 0 ? '+' : ''}${expo}`;
}

function drawBars(canvas, grads, opts = {}) {
  const { ctx, w, h: rawH } = (() => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    // Set h based on n
    const n = grads.length;
    const h = Math.max(180, Math.min(420, n * 14 + 56));
    canvas.style.height = h + 'px';
    canvas.width = Math.max(1, w * dpr);
    canvas.height = Math.max(1, h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  })();
  const h = rawH;

  const padL = 52, padR = 14, padT = 22, padB = 22;
  const innerW = w - padL - padR, innerH = h - padT - padB;
  const logMin = -20, logMax = 5;

  // Grid lines and labels at every 5 powers of 10
  ctx.strokeStyle = '#e8e2d8'; ctx.lineWidth = 0.5;
  ctx.fillStyle = '#9a9084'; ctx.font = '10px IBM Plex Mono, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let p = logMin; p <= logMax; p += 5) {
    const x = padL + ((p - logMin) / (logMax - logMin)) * innerW;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, h - padB); ctx.stroke();
    ctx.fillText(`10^${p}`, x, h - padB + 4);
  }
  // "1" line
  const xOne = padL + ((0 - logMin) / (logMax - logMin)) * innerW;
  ctx.strokeStyle = '#bdb29c'; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(xOne, padT); ctx.lineTo(xOne, h - padB); ctx.stroke();

  const n = grads.length;
  if (n === 0) return;
  const barGap = 2;
  const barH = Math.max(3, Math.min(16, (innerH - barGap * (n - 1)) / n));
  const totalH = n * barH + (n - 1) * barGap;
  const yOff = padT + (innerH - totalH) / 2;

  // Bars: index 0 (output layer) at top, last index (input layer) at bottom.
  for (let i = 0; i < n; i++) {
    const grad = grads[i];
    const logVal = grad > 0 ? Math.log10(grad) : logMin;
    const clamped = Math.max(logMin, Math.min(logMax, logVal));
    const y = yOff + i * (barH + barGap);

    let color = '#1e7770';
    if (grad > 1e2) color = '#7e4ea3';
    else if (grad > 1e-3) color = '#1e7770';
    else if (grad >= 1e-6) color = '#d9622b';
    else color = '#c53030';

    if (clamped >= 0) {
      // Bar grows right from "1" line for exploding gradients
      const barW = ((clamped - 0) / (logMax - 0)) * (innerW - (xOne - padL));
      ctx.fillStyle = color;
      ctx.fillRect(xOne, y, Math.max(1, barW), barH);
    } else {
      // Bar grows right from logMin to value (the bulk of the chart)
      const barW = ((clamped - logMin) / (logMax - logMin)) * innerW;
      ctx.fillStyle = color;
      ctx.fillRect(padL, y, Math.max(1, barW), barH);
    }

    // Layer label every nth tick
    if (n <= 30 || i % Math.ceil(n / 25) === 0 || i === n - 1 || i === 0) {
      ctx.fillStyle = '#9a9084'; ctx.font = '10px IBM Plex Mono, monospace';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      const label = (i === 0) ? 'L=1 (out)' : (i === n - 1 ? `L=${n} (in)` : `L=${i + 1}`);
      ctx.fillText(label, padL - 6, y + barH / 2);
    }
  }
  // Header
  ctx.fillStyle = '#6e665b'; ctx.font = '11px Manrope, sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText('output (top)  →  input (bottom): each row is one more multiplication into the chain', padL, 4);
}

function renderMain() {
  const factor = bestCaseFactor(MAIN.act);
  const grads = gradientChain(MAIN.depth, factor);
  drawBars(document.getElementById('main-chart'), grads);

  const inputGrad = grads[grads.length - 1];
  const cls = classify(inputGrad);
  const card = document.getElementById('card-grad');
  card.classList.remove('is-healthy', 'is-warn', 'is-dead', 'is-explode');
  card.classList.add(cls);
  document.getElementById('val-grad').textContent = formatGrad(inputGrad);
  document.getElementById('val-factor').textContent = factor.toFixed(3);
  // Depth at which it hits 1e-9
  let dThresh = '—';
  if (factor > 0 && factor < 1) {
    dThresh = Math.ceil(Math.log(1e-9) / Math.log(factor)).toString();
  } else if (factor >= 1) {
    dThresh = '∞ (stays alive)';
  }
  document.getElementById('val-depth-thresh').textContent = dThresh;

  // Chain product equation
  const eq = document.getElementById('chain-eq');
  const factorStr = factor.toFixed(3);
  let chainStr = '';
  if (MAIN.depth <= 6) {
    chainStr = Array(MAIN.depth).fill(`(${factorStr})`).join(' × ');
  } else {
    chainStr = `(${factorStr}) × (${factorStr}) × … × (${factorStr})  [${MAIN.depth} factors]`;
  }
  eq.textContent = chainStr;
  const res = document.getElementById('chain-result');
  res.textContent = `= ${factorStr}^${MAIN.depth} = ${formatGrad(inputGrad)}`;
  res.classList.remove('is-healthy', 'is-warn', 'is-dead', 'is-explode');
  res.classList.add(cls);

  document.getElementById('main-depth-val').textContent = MAIN.depth;
}

function wireMain() {
  const tabs = document.querySelectorAll('#main-tabs .tab-btn');
  tabs.forEach((b) => b.addEventListener('click', () => {
    tabs.forEach((x) => x.classList.remove('is-active'));
    b.classList.add('is-active');
    MAIN.act = b.dataset.act;
    renderMain();
  }));
  document.getElementById('main-depth').addEventListener('input', (e) => {
    MAIN.depth = parseInt(e.target.value, 10);
    renderMain();
  });
  renderMain();
}

// =====================================================
// STEP 3: weight-scale advanced panel
// =====================================================
const ADV = { w: 1.0, act: 'tanh', depth: 20, active: 0.5 };

function renderAdv() {
  const a = ACTS[ADV.act];
  let sigma = a.maxDeriv;
  if (ADV.act === 'relu') sigma = ADV.active;  // expected ReLU derivative under active fraction
  const factor = ADV.w * sigma;
  const grads = [];
  for (let l = 1; l <= ADV.depth; l++) grads.push(Math.pow(factor, l));
  drawBars(document.getElementById('adv-chart'), grads);

  const inputGrad = grads[grads.length - 1];
  const cls = classify(inputGrad);
  const card = document.getElementById('adv-card-grad');
  card.classList.remove('is-healthy', 'is-warn', 'is-dead', 'is-explode');
  card.classList.add(cls);
  document.getElementById('adv-grad').textContent = formatGrad(inputGrad);
  document.getElementById('adv-factor').textContent = factor.toFixed(3);

  let regime;
  if (factor > 1.05) regime = 'exploding';
  else if (factor > 0.95) regime = 'stable';
  else regime = 'vanishing';
  document.getElementById('adv-regime').textContent = regime;

  document.getElementById('adv-weight-val').textContent = ADV.w.toFixed(2);
  document.getElementById('adv-depth-val').textContent = ADV.depth;
  document.getElementById('adv-active-val').textContent = Math.round(ADV.active * 100) + '%';

  // Hide active fraction unless ReLU
  const activeRow = document.getElementById('adv-active').closest('label');
  const activeNum = document.getElementById('adv-active-val');
  if (activeRow) activeRow.style.display = ADV.act === 'relu' ? '' : 'none';
  if (activeNum) activeNum.style.display = ADV.act === 'relu' ? '' : 'none';
}

function wireAdv() {
  document.getElementById('adv-weight').addEventListener('input', (e) => {
    ADV.w = parseFloat(e.target.value); renderAdv();
  });
  document.getElementById('adv-act').addEventListener('change', (e) => {
    ADV.act = e.target.value; renderAdv();
  });
  document.getElementById('adv-depth').addEventListener('input', (e) => {
    ADV.depth = parseInt(e.target.value, 10); renderAdv();
  });
  document.getElementById('adv-active').addEventListener('input', (e) => {
    ADV.active = parseInt(e.target.value, 10) / 100; renderAdv();
  });
  renderAdv();
}

// =====================================================
// KaTeX
// =====================================================
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-prod':
      '\\frac{\\partial \\mathcal{L}}{\\partial W_1} \\;\\propto\\; \\prod_{\\ell=1}^{L} W_\\ell^\\top \\, \\sigma\'(z_\\ell)  \\;\\;\\Longrightarrow\\;\\; |\\nabla| \\sim (\\text{factor})^L'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

function boot() {
  wireDeriv();
  wireMain();
  wireAdv();
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  let resizeT = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => { renderDeriv(); renderMain(); renderAdv(); }, 120);
  });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
