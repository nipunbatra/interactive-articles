// ============================================================
// A Visual Proof That One Layer Can Compute Anything
// Nielsen's construction, made interactive. All math is real:
// sigmoids, steps, bumps, and towers evaluated live in the browser.
// ============================================================

// ---------- Math primitives ----------
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
function logit(p) {
  const c = Math.min(0.999, Math.max(0.001, p));
  return Math.log(c / (1 - c));
}
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

// ---------- Canvas helpers ----------
function setupCanvas(canvas, logicalWidth, logicalHeight) {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== logicalWidth * dpr) {
    canvas.width = logicalWidth * dpr;
    canvas.height = logicalHeight * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
  return { ctx, w: logicalWidth, h: logicalHeight };
}

function drawAxes(ctx, margin, w, h, xRange, yRange, xLabel = 'x', yLabel = 'y') {
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  ctx.strokeStyle = '#f0ebe1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 5; i++) {
    const x = margin.left + (plotW / 5) * i;
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotH);
  }
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (plotH / 4) * i;
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
  }
  ctx.stroke();

  // Zero line
  const yZero = margin.top + plotH - ((0 - yRange[0]) / (yRange[1] - yRange[0])) * plotH;
  if (yZero > margin.top && yZero < margin.top + plotH) {
    ctx.strokeStyle = '#d6cdb8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(margin.left, yZero);
    ctx.lineTo(margin.left + plotW, yZero);
    ctx.stroke();
  }

  ctx.strokeStyle = '#c4beb1';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();

  ctx.fillStyle = '#9a917f';
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(xRange[0].toFixed(1), margin.left, margin.top + plotH + 16);
  ctx.fillText(xRange[1].toFixed(1), margin.left + plotW, margin.top + plotH + 16);
  ctx.fillText(xLabel, margin.left + plotW / 2, h - 4);

  ctx.textAlign = 'right';
  ctx.fillText(yRange[0].toFixed(1), margin.left - 6, margin.top + plotH + 4);
  ctx.fillText(yRange[1].toFixed(1), margin.left - 6, margin.top + 4);
}

function plotCurve(ctx, xs, ys, margin, w, h, xRange, yRange, color, lineWidth = 2, dash = []) {
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dash);
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const px = margin.left + ((xs[i] - xRange[0]) / (xRange[1] - xRange[0])) * plotW;
    let yVal = clamp(ys[i], yRange[0], yRange[1]);
    const py = margin.top + plotH - ((yVal - yRange[0]) / (yRange[1] - yRange[0])) * plotH;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function dataToPixel(x, y, margin, w, h, xRange, yRange) {
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  return [
    margin.left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * plotW,
    margin.top + plotH - ((clamp(y, yRange[0], yRange[1]) - yRange[0]) / (yRange[1] - yRange[0])) * plotH,
  ];
}

// ---------- Sequential colormap (dark → blue → orange → cream) ----------
const CMAP_STOPS = [
  [0.0, [26, 24, 21]],
  [0.22, [40, 64, 110]],
  [0.45, [44, 111, 183]],
  [0.62, [120, 130, 150]],
  [0.74, [217, 98, 43]],
  [0.88, [240, 165, 78]],
  [1.0, [255, 230, 188]],
];
function colormap(t) {
  t = clamp(t, 0, 1);
  for (let i = 1; i < CMAP_STOPS.length; i++) {
    if (t <= CMAP_STOPS[i][0]) {
      const [t0, c0] = CMAP_STOPS[i - 1];
      const [t1, c1] = CMAP_STOPS[i];
      const f = (t - t0) / (t1 - t0 || 1);
      const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
      const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
      const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
      return `rgb(${r},${g},${b})`;
    }
  }
  return 'rgb(255,230,188)';
}

// Paint a precomputed G×G scalar field as a heatmap. field[gy*G+gx], y up.
function paintField(canvas, field, G, vmin, vmax, side = 440) {
  const { ctx, w, h } = setupCanvas(canvas, side, side);
  ctx.clearRect(0, 0, w, h);
  const cw = w / G;
  const ch = h / G;
  const span = vmax - vmin || 1;
  for (let gy = 0; gy < G; gy++) {
    for (let gx = 0; gx < G; gx++) {
      const t = (field[gy * G + gx] - vmin) / span;
      ctx.fillStyle = colormap(t);
      // y up: row 0 (gy=0) at bottom
      ctx.fillRect(gx * cw, (G - 1 - gy) * ch, cw + 0.6, ch + 0.6);
    }
  }
  // subtle frame
  ctx.strokeStyle = 'rgba(0,0,0,0.12)';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
}

// ============================================================
// STEP 1 — one sigmoid neuron
// ============================================================
function initNeuron() {
  const canvas = document.getElementById('neuronCanvas');
  const wS = document.getElementById('n1-w');
  const bS = document.getElementById('n1-b');
  const wV = document.getElementById('val-n1-w');
  const bV = document.getElementById('val-n1-b');
  const formula = document.getElementById('neuron-formula');
  const margin = { top: 20, right: 30, bottom: 35, left: 45 };
  const xRange = [0, 1];
  const yRange = [-0.08, 1.08];

  function draw() {
    const w = parseFloat(wS.value);
    const b = parseFloat(bS.value);
    wV.textContent = w.toFixed(1);
    bV.textContent = b.toFixed(1);
    if (formula) formula.textContent = `y = σ(${w.toFixed(1)}·x + ${b.toFixed(1)})`;

    const { ctx, w: cw, h: ch } = setupCanvas(canvas, 920, 300);
    ctx.clearRect(0, 0, cw, ch);
    drawAxes(ctx, margin, cw, ch, xRange, yRange);

    const xs = [], ys = [];
    for (let i = 0; i <= 400; i++) {
      const x = i / 400;
      xs.push(x);
      ys.push(sigmoid(w * x + b));
    }
    plotCurve(ctx, xs, ys, margin, cw, ch, xRange, yRange, '#2c6fb7', 3);

    // mark the half-way point x = -b/w
    if (Math.abs(w) > 1e-6) {
      const mid = -b / w;
      if (mid >= 0 && mid <= 1) {
        const [px, py] = dataToPixel(mid, 0.5, margin, cw, ch, xRange, yRange);
        ctx.fillStyle = '#d9622b';
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = '600 12px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`σ = 0.5 at x = ${mid.toFixed(2)}`, px, py - 12);
      }
    }
  }

  wS.addEventListener('input', draw);
  bS.addEventListener('input', draw);
  draw();
}

// ============================================================
// STEP 2 — high weight → step, parametrized by position s
// ============================================================
function initStep() {
  const canvas = document.getElementById('stepCanvas');
  const sS = document.getElementById('step-s');
  const wS = document.getElementById('step-w');
  const sV = document.getElementById('val-step-s');
  const wV = document.getElementById('val-step-w');
  const margin = { top: 20, right: 30, bottom: 35, left: 45 };
  const xRange = [0, 1];
  const yRange = [-0.08, 1.08];

  function draw() {
    const s = parseFloat(sS.value);
    const w = parseFloat(wS.value);
    sV.textContent = s.toFixed(2);
    wV.textContent = w.toFixed(0);

    const { ctx, w: cw, h: ch } = setupCanvas(canvas, 920, 300);
    ctx.clearRect(0, 0, cw, ch);
    drawAxes(ctx, margin, cw, ch, xRange, yRange);

    const xs = [], ys = [];
    for (let i = 0; i <= 600; i++) {
      const x = i / 600;
      xs.push(x);
      ys.push(sigmoid(w * (x - s)));
    }
    plotCurve(ctx, xs, ys, margin, cw, ch, xRange, yRange, '#2c6fb7', 3);

    // cliff marker
    const plotH = ch - margin.top - margin.bottom;
    const [px] = dataToPixel(s, 0, margin, cw, ch, xRange, yRange);
    ctx.strokeStyle = 'rgba(217, 98, 43, 0.55)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(px, margin.top);
    ctx.lineTo(px, margin.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#d9622b';
    ctx.font = '600 13px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`step at s = ${s.toFixed(2)}`, px, margin.top + 14);
  }

  sS.addEventListener('input', draw);
  wS.addEventListener('input', draw);
  draw();
}

// ============================================================
// STEP 3 — two steps make a bump
// ============================================================
function initBump() {
  const canvas = document.getElementById('bumpCanvas');
  const s1S = document.getElementById('bump-s1');
  const s2S = document.getElementById('bump-s2');
  const hS = document.getElementById('bump-h');
  const s1V = document.getElementById('val-bump-s1');
  const s2V = document.getElementById('val-bump-s2');
  const hV = document.getElementById('val-bump-h');
  const margin = { top: 20, right: 30, bottom: 35, left: 45 };
  const xRange = [0, 1];
  const yRange = [-1.6, 1.6];
  const W = 50; // steepness

  function draw() {
    let s1 = parseFloat(s1S.value);
    let s2 = parseFloat(s2S.value);
    const h = parseFloat(hS.value);
    if (s1 > s2) [s1, s2] = [s2, s1];
    s1V.textContent = s1.toFixed(2);
    s2V.textContent = s2.toFixed(2);
    hV.textContent = h.toFixed(2);

    const { ctx, w: cw, h: ch } = setupCanvas(canvas, 920, 320);
    ctx.clearRect(0, 0, cw, ch);
    drawAxes(ctx, margin, cw, ch, xRange, yRange);

    const xs = [], step1 = [], step2 = [], sum = [];
    for (let i = 0; i <= 600; i++) {
      const x = i / 600;
      const a = h * sigmoid(W * (x - s1));
      const b = -h * sigmoid(W * (x - s2));
      xs.push(x);
      step1.push(a);
      step2.push(b);
      sum.push(a + b);
    }
    plotCurve(ctx, xs, step1, margin, cw, ch, xRange, yRange, 'rgba(44,111,183,0.55)', 2, [6, 4]);
    plotCurve(ctx, xs, step2, margin, cw, ch, xRange, yRange, 'rgba(30,119,112,0.55)', 2, [6, 4]);
    plotCurve(ctx, xs, sum, margin, cw, ch, xRange, yRange, '#d9622b', 3.2);

    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(44,111,183,0.85)';
    ctx.fillText('+h · step(s₁)', margin.left + 8, margin.top + 14);
    ctx.fillStyle = 'rgba(30,119,112,0.85)';
    ctx.fillText('−h · step(s₂)', margin.left + 8, margin.top + 30);
    ctx.fillStyle = '#d9622b';
    ctx.fillText('bump = sum', margin.left + 8, margin.top + 46);
  }

  [s1S, s2S, hS].forEach((el) => el.addEventListener('input', draw));
  draw();
}

// ============================================================
// STEP 4 — design a 1D function by hand
// ============================================================
const DESIGN_TARGETS = {
  wiggle: {
    label: "Nielsen's wiggle",
    raw: (x) => 0.22 + 0.36 * x * x + 0.32 * x * Math.sin(15 * x) + 0.05 * Math.cos(40 * x),
  },
  wave: { label: 'Sine wave', raw: (x) => 0.5 + 0.36 * Math.sin(2 * Math.PI * x) },
  hill: { label: 'Single hill', raw: (x) => 0.12 + 0.8 * Math.exp(-Math.pow((x - 0.5) / 0.16, 2)) },
  plateaus: {
    label: 'Plateaus',
    raw: (x) => 0.2 + 0.32 * sigmoid(26 * (x - 0.33)) + 0.32 * sigmoid(26 * (x - 0.66)),
  },
};
function targetFn(key) {
  const raw = DESIGN_TARGETS[key].raw;
  return (x) => clamp(raw(x), 0.04, 0.96);
}

let designState = { target: 'wiggle', N: 6, p: [], view: 'fn' };

function initDesign() {
  const canvas = document.getElementById('designCanvas');
  const nS = document.getElementById('design-n');
  const nV = document.getElementById('val-design-n');
  const heightsBox = document.getElementById('design-heights');
  const btnAuto = document.getElementById('btn-design-autofit');
  const btnMath = document.getElementById('btn-design-math');
  const targetButtons = document.querySelectorAll('#design-target-buttons [data-target]');
  const neuronsStat = document.getElementById('stat-design-neurons');
  const slabStat = document.getElementById('stat-design-slab');
  const errStat = document.getElementById('stat-design-err');
  const caption = document.getElementById('design-caption');

  const margin = { top: 22, right: 30, bottom: 35, left: 50 };
  const xRange = [0, 1];

  function steepness() {
    return 24 * designState.N; // sharper as slabs get narrower
  }
  function bumpVal(x, i) {
    const W = steepness();
    const left = i / designState.N;
    const right = (i + 1) / designState.N;
    return sigmoid(W * (x - left)) - sigmoid(W * (x - right));
  }
  function hiddenOutput(x) {
    let H = 0;
    for (let i = 0; i < designState.N; i++) H += logit(designState.p[i]) * bumpVal(x, i);
    return H;
  }

  function autofit() {
    const f = targetFn(designState.target);
    for (let i = 0; i < designState.N; i++) {
      const mid = (i + 0.5) / designState.N;
      designState.p[i] = clamp(f(mid), 0.02, 0.98);
    }
  }

  function rebuildSliders() {
    heightsBox.innerHTML = '';
    for (let i = 0; i < designState.N; i++) {
      const slot = document.createElement('div');
      slot.className = 'height-slot';
      const input = document.createElement('input');
      input.type = 'range';
      input.min = '0.02';
      input.max = '0.98';
      input.step = '0.02';
      input.value = String(designState.p[i]);
      input.className = 'height-range';
      input.setAttribute('aria-label', `Bump ${i + 1} height`);
      input.addEventListener('input', () => {
        designState.p[i] = parseFloat(input.value);
        draw();
      });
      const tag = document.createElement('span');
      tag.className = 'height-tag';
      tag.textContent = i + 1;
      slot.appendChild(input);
      slot.appendChild(tag);
      heightsBox.appendChild(slot);
    }
  }

  function draw() {
    const f = targetFn(designState.target);
    const logitView = designState.view === 'logit';
    nV.textContent = designState.N;
    neuronsStat.textContent = designState.N * 2;
    slabStat.textContent = (1 / designState.N).toFixed(2);
    btnMath.classList.toggle('is-active', logitView);
    btnMath.textContent = logitView ? 'Show the function view' : 'Show the σ⁻¹ goal';

    const yRange = logitView ? [-6, 6] : [-0.05, 1.05];

    const { ctx, w: cw, h: ch } = setupCanvas(canvas, 920, 360);
    ctx.clearRect(0, 0, cw, ch);
    drawAxes(ctx, margin, cw, ch, xRange, yRange, 'x', logitView ? 'σ⁻¹(f)' : 'y');

    // slab boundaries
    const plotH = ch - margin.top - margin.bottom;
    ctx.strokeStyle = 'rgba(120,110,90,0.18)';
    ctx.lineWidth = 1;
    for (let i = 1; i < designState.N; i++) {
      const [px] = dataToPixel(i / designState.N, 0, margin, cw, ch, xRange, yRange);
      ctx.beginPath();
      ctx.moveTo(px, margin.top);
      ctx.lineTo(px, margin.top + plotH);
      ctx.stroke();
    }

    const xs = [], goal = [], net = [];
    let err = 0;
    const M = 500;
    for (let i = 0; i <= M; i++) {
      const x = i / M;
      const H = hiddenOutput(x);
      const fv = f(x);
      xs.push(x);
      if (logitView) {
        goal.push(logit(fv));
        net.push(H);
      } else {
        goal.push(fv);
        net.push(sigmoid(H));
      }
      err += Math.abs(sigmoid(H) - fv);
    }
    err /= M + 1;

    plotCurve(ctx, xs, goal, margin, cw, ch, xRange, yRange, '#d9622b', 3);
    plotCurve(ctx, xs, net, margin, cw, ch, xRange, yRange, '#2c6fb7', 2.6);

    errStat.textContent = err.toFixed(3);
    if (caption) {
      caption.innerHTML = logitView
        ? '<span style="color:var(--warm);font-weight:600">Orange</span>: the σ⁻¹ goal (logit of the target). <span style="color:var(--accent);font-weight:600">Blue</span>: your weighted hidden output H(x).'
        : '<span style="color:var(--warm);font-weight:600">Orange</span>: target f(x). <span style="color:var(--accent);font-weight:600">Blue</span>: network output σ(H(x)).';
    }
  }

  // wiring
  nS.addEventListener('input', () => {
    designState.N = parseInt(nS.value, 10);
    autofit();
    rebuildSliders();
    draw();
  });
  btnAuto.addEventListener('click', () => {
    autofit();
    rebuildSliders();
    draw();
  });
  btnMath.addEventListener('click', () => {
    designState.view = designState.view === 'logit' ? 'fn' : 'logit';
    draw();
  });
  targetButtons.forEach((b) => {
    b.addEventListener('click', () => {
      targetButtons.forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      designState.target = b.dataset.target;
      autofit();
      rebuildSliders();
      draw();
    });
  });

  designState.N = parseInt(nS.value, 10);
  autofit();
  rebuildSliders();
  draw();
}

// ============================================================
// STEP 5 — two inputs: build a tower
// ============================================================
function initTower() {
  const sumCanvas = document.getElementById('towerSumCanvas');
  const outCanvas = document.getElementById('towerOutCanvas');
  const cxS = document.getElementById('tower-cx');
  const cyS = document.getElementById('tower-cy');
  const sizeS = document.getElementById('tower-size');
  const thrS = document.getElementById('tower-thr');
  const wS = document.getElementById('tower-w');
  const cxV = document.getElementById('val-tower-cx');
  const cyV = document.getElementById('val-tower-cy');
  const sizeV = document.getElementById('val-tower-size');
  const thrV = document.getElementById('val-tower-thr');
  const wV = document.getElementById('val-tower-w');

  const G = 70;

  function draw() {
    const cx = parseFloat(cxS.value);
    const cy = parseFloat(cyS.value);
    const size = parseFloat(sizeS.value);
    const thr = parseFloat(thrS.value);
    const W = parseFloat(wS.value);
    cxV.textContent = cx.toFixed(2);
    cyV.textContent = cy.toFixed(2);
    sizeV.textContent = size.toFixed(2);
    thrV.textContent = thr.toFixed(2);
    wV.textContent = W.toFixed(0);

    const x0 = cx - size / 2, x1 = cx + size / 2;
    const y0 = cy - size / 2, y1 = cy + size / 2;
    const K = 18; // output sigmoid sharpness for the threshold

    const sumField = new Float32Array(G * G);
    const outField = new Float32Array(G * G);
    for (let gy = 0; gy < G; gy++) {
      const y = (gy + 0.5) / G;
      const by = sigmoid(W * (y - y0)) - sigmoid(W * (y - y1));
      for (let gx = 0; gx < G; gx++) {
        const x = (gx + 0.5) / G;
        const bx = sigmoid(W * (x - x0)) - sigmoid(W * (x - x1));
        const s = bx + by; // 0..2
        sumField[gy * G + gx] = s;
        outField[gy * G + gx] = sigmoid(K * (s - thr));
      }
    }
    paintField(sumCanvas, sumField, G, 0, 2);
    paintField(outCanvas, outField, G, 0, 1);
  }

  [cxS, cyS, sizeS, thrS, wS].forEach((el) => el.addEventListener('input', draw));
  draw();
}

// ============================================================
// STEP 6 — tile towers into a surface
// ============================================================
const SURFACE_TARGETS = {
  peaks: {
    label: 'Two peaks',
    raw: (x, y) =>
      Math.exp(-(Math.pow(x - 0.32, 2) + Math.pow(y - 0.34, 2)) / 0.025) -
      0.7 * Math.exp(-(Math.pow(x - 0.7, 2) + Math.pow(y - 0.68, 2)) / 0.03),
  },
  ripple: {
    label: 'Ripples',
    raw: (x, y) => {
      const r = Math.sqrt(Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2));
      return Math.cos(16 * r) * Math.exp(-2.2 * r);
    },
  },
  saddle: { label: 'Saddle', raw: (x, y) => Math.pow(x - 0.5, 2) - Math.pow(y - 0.5, 2) },
  gauss: {
    label: 'Gaussian hill',
    raw: (x, y) => Math.exp(-(Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2)) / 0.04),
  },
};

let surfaceState = { target: 'peaks', res: 6 };

function initSurface() {
  const targetCanvas = document.getElementById('surfaceTargetCanvas');
  const approxCanvas = document.getElementById('surfaceApproxCanvas');
  const resS = document.getElementById('surface-res');
  const resV = document.getElementById('val-surface-res');
  const resV2 = document.getElementById('val-surface-res2');
  const towersStat = document.getElementById('stat-surface-towers');
  const neuronsStat = document.getElementById('stat-surface-neurons');
  const mseStat = document.getElementById('stat-surface-mse');
  const buttons = document.querySelectorAll('#surface-target-buttons [data-surface]');

  const G = 64; // render resolution

  // Normalize a target into [0,1] using its own min/max over the grid.
  function normalizedTarget(key) {
    const raw = SURFACE_TARGETS[key].raw;
    let lo = Infinity, hi = -Infinity;
    const S = 40;
    for (let i = 0; i <= S; i++)
      for (let j = 0; j <= S; j++) {
        const v = raw(i / S, j / S);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    const span = hi - lo || 1;
    return (x, y) => (raw(x, y) - lo) / span;
  }

  function draw() {
    const R = surfaceState.res;
    const f = normalizedTarget(surfaceState.target);
    resV.textContent = R;
    resV2.textContent = R;
    towersStat.textContent = R * R;
    neuronsStat.textContent = 4 * R * R;

    const W = clamp(10 * R, 30, 150); // ridge steepness scales with grid
    const K = 16; // tower threshold sharpness

    // Precompute per-axis bump values: Bx[i][gx], By[j][gy]
    const Bx = [], By = [];
    for (let i = 0; i < R; i++) {
      const x0 = i / R, x1 = (i + 1) / R;
      const row = new Float32Array(G);
      for (let gx = 0; gx < G; gx++) {
        const x = (gx + 0.5) / G;
        row[gx] = sigmoid(W * (x - x0)) - sigmoid(W * (x - x1));
      }
      Bx.push(row);
    }
    for (let j = 0; j < R; j++) {
      const y0 = j / R, y1 = (j + 1) / R;
      const row = new Float32Array(G);
      for (let gy = 0; gy < G; gy++) {
        const y = (gy + 0.5) / G;
        row[gy] = sigmoid(W * (y - y0)) - sigmoid(W * (y - y1));
      }
      By.push(row);
    }

    // Tower heights = target at each cell center
    const heights = [];
    for (let i = 0; i < R; i++) {
      const hr = new Float32Array(R);
      for (let j = 0; j < R; j++) hr[j] = f((i + 0.5) / R, (j + 0.5) / R);
      heights.push(hr);
    }

    const targetField = new Float32Array(G * G);
    const approxField = new Float32Array(G * G);
    let mse = 0;
    for (let gy = 0; gy < G; gy++) {
      const y = (gy + 0.5) / G;
      for (let gx = 0; gx < G; gx++) {
        const x = (gx + 0.5) / G;
        const tv = f(x, y);
        let acc = 0;
        for (let i = 0; i < R; i++) {
          const bxi = Bx[i][gx];
          if (bxi < 0.01) continue; // skip far cells
          for (let j = 0; j < R; j++) {
            const tower = sigmoid(K * (bxi + By[j][gy] - 1.5));
            if (tower > 0.01) acc += heights[i][j] * tower;
          }
        }
        targetField[gy * G + gx] = tv;
        approxField[gy * G + gx] = acc;
        const d = acc - tv;
        mse += d * d;
      }
    }
    mse /= G * G;

    paintField(targetCanvas, targetField, G, 0, 1);
    paintField(approxCanvas, approxField, G, 0, 1);
    mseStat.textContent = mse.toFixed(4);
  }

  resS.addEventListener('input', () => {
    surfaceState.res = parseInt(resS.value, 10);
    draw();
  });
  buttons.forEach((b) => {
    b.addEventListener('click', () => {
      buttons.forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      surfaceState.target = b.dataset.surface;
      draw();
    });
  });
  draw();
}

// ============================================================
// KaTeX
// ============================================================
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-sigmoid': 'a = \\sigma(w\\,x + b)',
    'math-step': 's = -\\,b / w \\quad\\Longrightarrow\\quad a \\approx \\begin{cases} 0 & x < s \\\\ 1 & x > s \\end{cases}',
    'math-bump': 'h\\,\\sigma\\!\\big(w(x - s_1)\\big) \\;-\\; h\\,\\sigma\\!\\big(w(x - s_2)\\big)',
    'math-design': 'H(x) \\;=\\; \\sum_{i=1}^{N} h_i \\,\\big[\\,\\text{bump}_i(x)\\,\\big]',
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try {
      katex.render(blocks[id], el, { displayMode: true, throwOnError: false });
    } catch (_) {}
  });

  if (window.renderMathInElement) {
    try {
      renderMathInElement(document.querySelector('.article'), {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true },
        ],
        throwOnError: false,
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
      });
    } catch (_) {}
  }
}

function waitForKatexAndRender() {
  let tries = 0;
  function tick() {
    if (window.katex && window.renderMathInElement) return renderMath();
    if (window.katex && tries > 20) return renderMath();
    tries++;
    setTimeout(tick, 50);
  }
  tick();
}

// ============================================================
// Boot
// ============================================================
function init() {
  waitForKatexAndRender();
  initNeuron();
  initStep();
  initBump();
  initDesign();
  initTower();
  initSurface();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
