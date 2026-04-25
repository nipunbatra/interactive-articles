// ============================================================
// The Optimizer Race
// Four landscapes, three optimizers, live computation.
// ============================================================

// ---------- Landscapes ----------
// Each landscape has a closed-form loss f(x,y) and its gradient,
// plus a global minimum location, a safe viewport, and a KaTeX formula.
const LANDSCAPES = {
  ravine: {
    label: 'Narrow ravine',
    caption:
      'Elongated bowl: steep in the y-direction, gentle in the x-direction. Classic "direction mismatch" test.',
    formula: 'L(x, y) = \\frac{x^2}{20} + y^2',
    f: (x, y) => (x * x) / 20 + y * y,
    grad: (x, y) => ({ dx: x / 10, dy: 2 * y }),
    min: { x: 0, y: 0 },
    viewport: { xMin: -14, xMax: 14, yMin: -5, yMax: 5 },
    maxLoss: 25
  },
  saddle: {
    label: 'Saddle point',
    caption:
      'Curvature +1 along x and -1 along y: the minimum is at one end of the y-axis, but the origin looks almost flat.',
    formula: 'L(x, y) = x^2 - y^2 + 0.05\\,y^4',
    f: (x, y) => x * x - y * y + 0.05 * y * y * y * y,
    grad: (x, y) => ({ dx: 2 * x, dy: -2 * y + 0.2 * y * y * y }),
    // Real minima along y = ±sqrt(10), set display minimum at y = +sqrt(10)
    min: { x: 0, y: Math.sqrt(10) },
    viewport: { xMin: -6, xMax: 6, yMin: -5, yMax: 5 },
    maxLoss: 25
  },
  plateau: {
    label: 'Plateau + bowl',
    caption:
      'Dead-flat middle, bowl on the right. Gradients are near-zero for a long distance, stalling SGD.',
    // Smooth "soft" bowl: very small gradient in middle, quadratic far right
    f: (x, y) => {
      const base = 0.05 * Math.log1p(Math.exp(6 * (x - 0))) * Math.log1p(Math.exp(6 * (x - 0))) / 8;
      return base + 0.6 * y * y;
    },
    grad: (x, y) => {
      // Use an approximation that stays smooth: d/dx ≈ tiny near middle, grows past 0
      const s = 6 * x;
      const sig = 1 / (1 + Math.exp(-s));
      const softplus = Math.log1p(Math.exp(s));
      const dx = 0.05 * 2 * softplus * sig * 6 / 8;
      return { dx, dy: 1.2 * y };
    },
    min: { x: -6, y: 0 },
    viewport: { xMin: -7, xMax: 6, yMin: -4, yMax: 4 },
    maxLoss: 6
  },
  rosenbrock: {
    label: 'Rosenbrock',
    caption:
      'The textbook hard case: a curved valley that bends around (1, 1). Notorious for tripping up optimizers.',
    formula:
      'L(x, y) = (1 - x)^2 + 50\\,(y - x^2)^2',
    f: (x, y) => (1 - x) ** 2 + 50 * (y - x * x) ** 2,
    grad: (x, y) => ({
      dx: -2 * (1 - x) - 200 * x * (y - x * x),
      dy: 100 * (y - x * x)
    }),
    min: { x: 1, y: 1 },
    viewport: { xMin: -2.2, xMax: 2.2, yMin: -1, yMax: 3 },
    maxLoss: 300
  }
};

// Fix plateau formula rendering (the gradient is simplified; formula shown is symbolic)
LANDSCAPES.plateau.formula =
  'L(x, y) = \\tfrac{1}{8}\\,\\mathrm{softplus}(6x)^2 \\cdot \\tfrac{1}{20} + 0.6\\,y^2';

// Sharp Trap · added per Gemini review
// Wide true minimum at (2,2) but a sharp local minimum at (0,0).
// Adam tends to fall into the sharp trap; SGD+momentum can escape due to inertia.
LANDSCAPES.sharpTrap = {
  label: 'Sharp Trap',
  caption: 'A wide, shallow true minimum at (2, 2) and a much sharper local minimum at (0, 0). Adaptive optimizers can get stuck in the sharp trap.',
  formula: 'L(x, y) = 1 - \\exp(-(100x^2 + y^2)) + 0.1\\,((x-2)^2 + (y-2)^2)',
  f: (x, y) => 1 - Math.exp(-(100 * x * x + y * y)) + 0.1 * ((x - 2) ** 2 + (y - 2) ** 2),
  grad: (x, y) => ({
    dx: 200 * x * Math.exp(-(100 * x * x + y * y)) + 0.2 * (x - 2),
    dy: 2 * y * Math.exp(-(100 * x * x + y * y)) + 0.2 * (y - 2)
  }),
  min: { x: 2, y: 2 },
  viewport: { xMin: -1.5, xMax: 4, yMin: -1.5, yMax: 4 },
  maxLoss: 5
};

// ---------- Optimizers ----------
function makeSGD() { return {}; }
function makeMomentum() { return { vx: 0, vy: 0 }; }
function makeAdam() { return { mx: 0, my: 0, vx: 0, vy: 0, t: 0 }; }

function stepSGD(state, grad, lr) {
  state.x -= lr * grad.dx;
  state.y -= lr * grad.dy;
}
function stepMomentum(state, grad, lr, beta = 0.9) {
  state.vx = beta * state.vx + grad.dx;
  state.vy = beta * state.vy + grad.dy;
  state.x -= lr * state.vx;
  state.y -= lr * state.vy;
}
function stepAdam(state, grad, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
  state.t += 1;
  state.mx = beta1 * state.mx + (1 - beta1) * grad.dx;
  state.my = beta1 * state.my + (1 - beta1) * grad.dy;
  state.vx = beta2 * state.vx + (1 - beta2) * grad.dx * grad.dx;
  state.vy = beta2 * state.vy + (1 - beta2) * grad.dy * grad.dy;
  const mhx = state.mx / (1 - Math.pow(beta1, state.t));
  const mhy = state.my / (1 - Math.pow(beta1, state.t));
  const vhx = state.vx / (1 - Math.pow(beta2, state.t));
  const vhy = state.vy / (1 - Math.pow(beta2, state.t));
  // Adam uses a lr that's typically larger than SGD on the same problem;
  // scale by a factor so the visual comparison is informative.
  state.x -= (lr * 3) * mhx / (Math.sqrt(vhx) + eps);
  state.y -= (lr * 3) * mhy / (Math.sqrt(vhy) + eps);
}

// ---------- State ----------
let raceState = {
  landscapeKey: 'ravine',
  lr: 0.05,
  maxSteps: 300,
  paths: { sgd: [], momentum: [], adam: [] },
  losses: { sgd: [], momentum: [], adam: [] },
  steps: { sgd: 0, momentum: 0, adam: 0 },
  running: false,
  raf: null,
  optStates: null,
  start: null
};

function resetRace(startX, startY) {
  const L = LANDSCAPES[raceState.landscapeKey];
  raceState.start = { x: startX, y: startY };
  raceState.paths = {
    sgd: [{ x: startX, y: startY }],
    momentum: [{ x: startX, y: startY }],
    adam: [{ x: startX, y: startY }]
  };
  raceState.losses = {
    sgd: [L.f(startX, startY)],
    momentum: [L.f(startX, startY)],
    adam: [L.f(startX, startY)]
  };
  raceState.steps = { sgd: 0, momentum: 0, adam: 0 };
  raceState.optStates = {
    sgd: { x: startX, y: startY, ...makeSGD() },
    momentum: { x: startX, y: startY, ...makeMomentum() },
    adam: { x: startX, y: startY, ...makeAdam() }
  };
  raceState.running = true;
  if (raceState.raf) cancelAnimationFrame(raceState.raf);
  tick();
}

function clipState(s, L) {
  const pad = 2;
  if (s.x < L.viewport.xMin - pad) s.x = L.viewport.xMin - pad;
  if (s.x > L.viewport.xMax + pad) s.x = L.viewport.xMax + pad;
  if (s.y < L.viewport.yMin - pad) s.y = L.viewport.yMin - pad;
  if (s.y > L.viewport.yMax + pad) s.y = L.viewport.yMax + pad;
  if (!isFinite(s.x)) s.x = L.viewport.xMax + pad;
  if (!isFinite(s.y)) s.y = L.viewport.yMax + pad;
}

function tick() {
  if (!raceState.running) return;
  const L = LANDSCAPES[raceState.landscapeKey];
  const lr = raceState.lr;
  const stepsPerFrame = 2;

  for (let s = 0; s < stepsPerFrame; s++) {
    let anyActive = false;

    const sgd = raceState.optStates.sgd;
    if (raceState.steps.sgd < raceState.maxSteps) {
      const g = L.grad(sgd.x, sgd.y);
      stepSGD(sgd, g, lr);
      clipState(sgd, L);
      raceState.paths.sgd.push({ x: sgd.x, y: sgd.y });
      raceState.losses.sgd.push(L.f(sgd.x, sgd.y));
      raceState.steps.sgd += 1;
      anyActive = true;
    }

    const mom = raceState.optStates.momentum;
    if (raceState.steps.momentum < raceState.maxSteps) {
      const g = L.grad(mom.x, mom.y);
      stepMomentum(mom, g, lr);
      clipState(mom, L);
      raceState.paths.momentum.push({ x: mom.x, y: mom.y });
      raceState.losses.momentum.push(L.f(mom.x, mom.y));
      raceState.steps.momentum += 1;
      anyActive = true;
    }

    const ad = raceState.optStates.adam;
    if (raceState.steps.adam < raceState.maxSteps) {
      const g = L.grad(ad.x, ad.y);
      stepAdam(ad, g, lr);
      clipState(ad, L);
      raceState.paths.adam.push({ x: ad.x, y: ad.y });
      raceState.losses.adam.push(L.f(ad.x, ad.y));
      raceState.steps.adam += 1;
      anyActive = true;
    }

    if (!anyActive) {
      raceState.running = false;
      break;
    }
  }

  drawTerrain();
  drawLossCurves();
  updateScoreboard();

  if (raceState.running) {
    raceState.raf = requestAnimationFrame(tick);
  }
}

// ---------- Drawing ----------
function setupCanvas(canvas, logicalWidth, logicalHeight) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = logicalWidth * dpr;
  canvas.height = logicalHeight * dpr;
  canvas.style.width = logicalWidth + 'px';
  canvas.style.height = logicalHeight + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: logicalWidth, h: logicalHeight };
}

function drawTerrain() {
  const canvas = document.getElementById('optCanvas');
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas, 800, 500);
  ctx.fillStyle = '#fdfcf9';
  ctx.fillRect(0, 0, w, h);

  const L = LANDSCAPES[raceState.landscapeKey];
  const vp = L.viewport;
  const margin = { top: 20, right: 20, bottom: 20, left: 20 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  function toScreen(x, y) {
    const sx = margin.left + ((x - vp.xMin) / (vp.xMax - vp.xMin)) * plotW;
    const sy = margin.top + plotH - ((y - vp.yMin) / (vp.yMax - vp.yMin)) * plotH;
    return { x: sx, y: sy };
  }

  // Contour plot via marching‐grid
  const cols = 160;
  const rows = 100;
  const dx = (vp.xMax - vp.xMin) / cols;
  const dy = (vp.yMax - vp.yMin) / rows;
  const vals = new Float32Array((cols + 1) * (rows + 1));
  let maxV = -Infinity;
  for (let j = 0; j <= rows; j++) {
    for (let i = 0; i <= cols; i++) {
      const x = vp.xMin + i * dx;
      const y = vp.yMin + j * dy;
      const v = L.f(x, y);
      vals[j * (cols + 1) + i] = v;
      if (isFinite(v) && v > maxV) maxV = v;
    }
  }
  if (!isFinite(maxV)) maxV = L.maxLoss;
  const loss99 = Math.min(maxV, L.maxLoss);

  // Shaded background by height
  const cellW = plotW / cols;
  const cellH = plotH / rows;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const v = vals[j * (cols + 1) + i];
      const t = Math.min(1, Math.max(0, v / loss99));
      // Beige→warm shading
      const r = Math.round(253 - t * 30);
      const g = Math.round(249 - t * 70);
      const b = Math.round(239 - t * 100);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(margin.left + i * cellW, margin.top + plotH - (j + 1) * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  // Contour lines
  const contours = 12;
  ctx.strokeStyle = 'rgba(30, 30, 40, 0.18)';
  ctx.lineWidth = 1;
  for (let c = 1; c <= contours; c++) {
    const level = loss99 * (c / contours);
    ctx.beginPath();
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const a = vals[j * (cols + 1) + i];
        const b = vals[j * (cols + 1) + i + 1];
        const cc = vals[(j + 1) * (cols + 1) + i];
        if ((a - level) * (b - level) < 0) {
          const x1 = vp.xMin + i * dx;
          const y1 = vp.yMin + j * dy;
          const p1 = toScreen(x1, y1);
          const p2 = toScreen(x1 + dx, y1);
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
        }
        if ((a - level) * (cc - level) < 0) {
          const x1 = vp.xMin + i * dx;
          const y1 = vp.yMin + j * dy;
          const p1 = toScreen(x1, y1);
          const p2 = toScreen(x1, y1 + dy);
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
        }
      }
    }
    ctx.stroke();
  }

  // Global min marker
  const m = toScreen(L.min.x, L.min.y);
  ctx.fillStyle = '#1a1815';
  ctx.font = '22px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('★', m.x, m.y);

  // Paths
  const opts = [
    { id: 'sgd', color: '#ca5b2a' },
    { id: 'momentum', color: '#245b8f' },
    { id: 'adam', color: '#1e7770' }
  ];
  opts.forEach((o) => {
    const p = raceState.paths[o.id];
    if (!p || !p.length) return;
    ctx.strokeStyle = o.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < p.length; i++) {
      const sp = toScreen(p[i].x, p[i].y);
      if (i === 0) ctx.moveTo(sp.x, sp.y);
      else ctx.lineTo(sp.x, sp.y);
    }
    ctx.stroke();

    const last = toScreen(p[p.length - 1].x, p[p.length - 1].y);
    ctx.beginPath();
    ctx.arc(last.x, last.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = o.color;
    ctx.fill();
  });

  // Start marker
  if (raceState.start) {
    const sp = toScreen(raceState.start.x, raceState.start.y);
    ctx.strokeStyle = '#1a1815';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(sp.x, sp.y, 6, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function drawLossCurves() {
  const canvas = document.getElementById('lossCanvas');
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas, 800, 220);
  ctx.clearRect(0, 0, w, h);

  const margin = { top: 20, right: 20, bottom: 30, left: 60 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  // Grid
  ctx.strokeStyle = '#f0ebe1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (plotH / 4) * i;
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
  }
  ctx.stroke();

  ctx.strokeStyle = '#c4beb1';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();

  const all = [...raceState.losses.sgd, ...raceState.losses.momentum, ...raceState.losses.adam];
  if (!all.length) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Click anywhere on the terrain to start', margin.left + plotW / 2, margin.top + plotH / 2);
    return;
  }

  const logs = all.map((v) => Math.log10(Math.max(v, 1e-8)));
  const minL = Math.min(...logs) - 0.2;
  const maxL = Math.max(...logs) + 0.2;
  const maxSteps = Math.max(
    raceState.losses.sgd.length,
    raceState.losses.momentum.length,
    raceState.losses.adam.length,
    10
  );

  const curves = [
    { id: 'sgd', color: '#ca5b2a' },
    { id: 'momentum', color: '#245b8f' },
    { id: 'adam', color: '#1e7770' }
  ];
  curves.forEach((c) => {
    const losses = raceState.losses[c.id];
    if (!losses.length) return;
    ctx.strokeStyle = c.color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < losses.length; i++) {
      const t = i / Math.max(1, maxSteps - 1);
      const lg = Math.log10(Math.max(losses[i], 1e-8));
      const px = margin.left + t * plotW;
      const py = margin.top + plotH - ((lg - minL) / (maxL - minL)) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  });

  ctx.fillStyle = '#9a917f';
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('10^' + maxL.toFixed(1), margin.left - 6, margin.top + 4);
  ctx.fillText('10^' + minL.toFixed(1), margin.left - 6, margin.top + plotH + 4);
  ctx.textAlign = 'center';
  ctx.fillText('step', margin.left + plotW / 2, margin.top + plotH + 22);
}

function updateScoreboard() {
  const fmt = (v) => {
    if (!isFinite(v)) return '∞';
    if (v >= 100) return v.toExponential(1);
    if (v >= 1) return v.toFixed(2);
    if (v >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
  };
  const ids = ['sgd', 'mom', 'adam'];
  const keys = ['sgd', 'momentum', 'adam'];
  ids.forEach((id, i) => {
    const key = keys[i];
    const lossEl = document.getElementById(`${id}-loss`);
    const stepsEl = document.getElementById(`${id}-steps`);
    const losses = raceState.losses[key];
    const final = losses.length ? losses[losses.length - 1] : NaN;
    if (lossEl) lossEl.textContent = losses.length ? fmt(final) : '—';
    if (stepsEl) stepsEl.textContent = 'steps: ' + raceState.steps[key];
  });
}

// ---------- Landscape switching ----------
function loadLandscape(key) {
  raceState.landscapeKey = key;
  raceState.running = false;
  if (raceState.raf) cancelAnimationFrame(raceState.raf);
  raceState.paths = { sgd: [], momentum: [], adam: [] };
  raceState.losses = { sgd: [], momentum: [], adam: [] };
  raceState.steps = { sgd: 0, momentum: 0, adam: 0 };
  raceState.start = null;

  document.querySelectorAll('#landscape-buttons [data-landscape]').forEach((b) => {
    b.classList.toggle('is-active', b.dataset.landscape === key);
  });
  const cap = document.getElementById('landscape-caption');
  if (cap) cap.textContent = LANDSCAPES[key].caption;

  renderLandscapeFormula();
  drawTerrain();
  drawLossCurves();
  updateScoreboard();
}

function renderLandscapeFormula() {
  const el = document.getElementById('math-landscape');
  if (!el || !window.katex) return;
  try {
    katex.render(LANDSCAPES[raceState.landscapeKey].formula, el, {
      displayMode: true,
      throwOnError: false
    });
  } catch (_) { /* no-op */ }
}

function renderStaticMath() {
  if (!window.katex) return;
  const blocks = {
    'math-sgd': 'W_{t+1} = W_t - \\alpha\\,\\nabla L(W_t)',
    'math-momentum':
      'V_{t+1} = \\beta V_t + \\nabla L(W_t) \\\\ W_{t+1} = W_t - \\alpha V_{t+1}',
    'math-adam':
      'm_{t+1} = \\beta_1 m_t + (1 - \\beta_1)\\,\\nabla L(W_t) \\\\' +
      'v_{t+1} = \\beta_2 v_t + (1 - \\beta_2)\\,\\nabla L(W_t)^2 \\\\' +
      'W_{t+1} = W_t - \\alpha\\,\\frac{\\hat m_{t+1}}{\\sqrt{\\hat v_{t+1}} + \\epsilon}'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try {
      katex.render(blocks[id], el, { displayMode: true, throwOnError: false });
    } catch (_) { /* no-op */ }
  });
  renderLandscapeFormula();
}

// ---------- Click to drop start ----------
function installCanvasInput() {
  const canvas = document.getElementById('optCanvas');
  if (!canvas) return;
  const logicalWidth = 800;
  const logicalHeight = 500;
  const margin = { top: 20, right: 20, bottom: 20, left: 20 };
  const plotW = logicalWidth - margin.left - margin.right;
  const plotH = logicalHeight - margin.top - margin.bottom;

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const vp = LANDSCAPES[raceState.landscapeKey].viewport;
    const x = vp.xMin + ((mx - margin.left) / plotW) * (vp.xMax - vp.xMin);
    const y = vp.yMin + ((margin.top + plotH - my) / plotH) * (vp.yMax - vp.yMin);
    resetRace(x, y);
  });
}

// ---------- Controls ----------
function installControls() {
  document.querySelectorAll('#landscape-buttons [data-landscape]').forEach((b) => {
    b.addEventListener('click', () => loadLandscape(b.dataset.landscape));
  });

  const lrSlider = document.getElementById('lr-slider');
  const lrVal = document.getElementById('lr-val');
  if (lrSlider && lrVal) {
    lrSlider.addEventListener('input', () => {
      raceState.lr = parseFloat(lrSlider.value);
      lrVal.textContent = raceState.lr.toFixed(3);
    });
  }

  const stepsSlider = document.getElementById('steps-slider');
  const stepsVal = document.getElementById('steps-val');
  if (stepsSlider && stepsVal) {
    stepsSlider.addEventListener('input', () => {
      raceState.maxSteps = parseInt(stepsSlider.value, 10);
      stepsVal.textContent = raceState.maxSteps;
    });
  }
}

// ---------- Boot ----------
function init() {
  if (window.katex) {
    renderStaticMath();
  } else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderStaticMath);
  }
  installControls();
  installCanvasInput();
  loadLandscape('ravine');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
