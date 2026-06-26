// ============================================================
// A Visual Proof That One Layer Can Compute Anything
// Nielsen's construction, made interactive. Everything is real:
// sigmoids, steps, bumps and towers evaluated live, with 1D plots
// and 3D surface plots rendered straight to canvas.
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
    const yVal = clamp(ys[i], yRange[0], yRange[1]);
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
  [0.0, [27, 27, 38]],
  [0.22, [40, 64, 110]],
  [0.45, [44, 111, 183]],
  [0.6, [126, 134, 150]],
  [0.74, [217, 98, 43]],
  [0.88, [242, 168, 82]],
  [1.0, [255, 232, 192]],
];
function colormapRGB(t) {
  t = clamp(t, 0, 1);
  for (let i = 1; i < CMAP_STOPS.length; i++) {
    if (t <= CMAP_STOPS[i][0]) {
      const [t0, c0] = CMAP_STOPS[i - 1];
      const [t1, c1] = CMAP_STOPS[i];
      const f = (t - t0) / (t1 - t0 || 1);
      return [
        Math.round(c0[0] + f * (c1[0] - c0[0])),
        Math.round(c0[1] + f * (c1[1] - c0[1])),
        Math.round(c0[2] + f * (c1[2] - c0[2])),
      ];
    }
  }
  return [255, 232, 192];
}

// ============================================================
// 3D SURFACE RENDERER
// Draw a G×G height field as a shaded 3D surface (height field
// viewed from an azimuth/elevation; painter's algorithm).
// ============================================================
function drawSurface(canvas, field, G, opts = {}) {
  const {
    side = 360,
    azimuth = 0.7,
    elevation = 0.52,
    vmin = 0,
    vmax = 1,
    heightScale = 0.62,
    edges = true,
    bg = '#fbfaf6',
  } = opts;

  const { ctx, w, h } = setupCanvas(canvas, side, Math.round(side * 0.86));
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const ca = Math.cos(azimuth), sa = Math.sin(azimuth);
  const ce = Math.cos(elevation), se = Math.sin(elevation);
  const span = vmax - vmin || 1;

  // Project every grid vertex
  const sx = new Float32Array(G * G);
  const sy = new Float32Array(G * G);
  const dep = new Float32Array(G * G);
  const zt = new Float32Array(G * G);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let j = 0; j < G; j++) {
    for (let i = 0; i < G; i++) {
      const x = i / (G - 1) - 0.5;
      const y = j / (G - 1) - 0.5;
      const t = clamp((field[j * G + i] - vmin) / span, 0, 1);
      const z = t * heightScale;
      const X = x * ca - y * sa;
      const Y = x * sa + y * ca;
      const px = X;
      const py = -z * ce + Y * se;
      sx[j * G + i] = px;
      sy[j * G + i] = py;
      dep[j * G + i] = Y * ce + z * se;
      zt[j * G + i] = t;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }
  }

  const pad = 14;
  const scale = Math.min((w - 2 * pad) / (maxX - minX || 1), (h - 2 * pad) / (maxY - minY || 1));
  const offX = (w - (maxX - minX) * scale) / 2 - minX * scale;
  const offY = (h - (maxY - minY) * scale) / 2 - minY * scale;
  const PX = (idx) => sx[idx] * scale + offX;
  const PY = (idx) => sy[idx] * scale + offY;

  // Light direction for lambert shading
  const Lx = -0.4, Ly = -0.5, Lz = 0.78;
  const Llen = Math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz);

  // Build quads with centroid depth, then painter-sort
  const quads = [];
  for (let j = 0; j < G - 1; j++) {
    for (let i = 0; i < G - 1; i++) {
      const a = j * G + i;
      const b = j * G + i + 1;
      const c = (j + 1) * G + i + 1;
      const d = (j + 1) * G + i;
      const depth = (dep[a] + dep[b] + dep[c] + dep[d]) * 0.25;
      const tAvg = (zt[a] + zt[b] + zt[c] + zt[d]) * 0.25;

      // surface normal from height gradient (world units)
      const cell = 1 / (G - 1);
      const dzx = (field[b] - field[a]) / span * heightScale;
      const dzy = (field[d] - field[a]) / span * heightScale;
      let nx = -dzx * cell, ny = -dzy * cell, nz = cell * cell;
      const nlen = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      let lambert = (nx * Lx + ny * Ly + nz * Lz) / (nlen * Llen);
      lambert = clamp(0.55 + 0.5 * lambert, 0.4, 1.12);

      const [r, g, bl] = colormapRGB(tAvg);
      quads.push({
        depth,
        col: `rgb(${Math.round(r * lambert)},${Math.round(g * lambert)},${Math.round(bl * lambert)})`,
        a, b, c, d,
      });
    }
  }
  quads.sort((p, q) => q.depth - p.depth);

  ctx.lineJoin = 'round';
  for (const q of quads) {
    ctx.beginPath();
    ctx.moveTo(PX(q.a), PY(q.a));
    ctx.lineTo(PX(q.b), PY(q.b));
    ctx.lineTo(PX(q.c), PY(q.c));
    ctx.lineTo(PX(q.d), PY(q.d));
    ctx.closePath();
    ctx.fillStyle = q.col;
    ctx.fill();
    if (edges) {
      ctx.strokeStyle = 'rgba(20,18,16,0.10)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }
}

// ============================================================
// STEP 1 — one sigmoid neuron
// ============================================================
function initNeuron() {
  const canvas = document.getElementById('neuronCanvas');
  if (!canvas) return;
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
  if (!canvas) return;
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
  if (!canvas) return;
  const s1S = document.getElementById('bump-s1');
  const s2S = document.getElementById('bump-s2');
  const hS = document.getElementById('bump-h');
  const s1V = document.getElementById('val-bump-s1');
  const s2V = document.getElementById('val-bump-s2');
  const hV = document.getElementById('val-bump-h');
  const margin = { top: 20, right: 30, bottom: 35, left: 45 };
  const xRange = [0, 1];
  const yRange = [-1.6, 1.6];
  const W = 50;

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
// σ⁻¹ mini-illustration (the "undo the squish" aside)
// ============================================================
function initLogitMini() {
  const canvas = document.getElementById('logitMiniCanvas');
  if (!canvas) return;
  const margin = { top: 16, right: 16, bottom: 26, left: 30 };
  const f = (x) => clamp(0.5 + 0.36 * Math.sin(2 * Math.PI * x), 0.04, 0.96);
  const { ctx, w: cw, h: ch } = setupCanvas(canvas, 460, 220);
  ctx.clearRect(0, 0, cw, ch);
  // left axis 0..1, but logit goes wide → use a shared [-4,4] frame with a 0..1 band marked
  const yRange = [-4, 4];
  const xRange = [0, 1];
  drawAxes(ctx, margin, cw, ch, xRange, yRange, 'x', '');
  // shade the 0..1 band (where the final answer must live)
  const [, py1] = dataToPixel(0, 1, margin, cw, ch, xRange, yRange);
  const [, py0] = dataToPixel(0, 0, margin, cw, ch, xRange, yRange);
  ctx.fillStyle = 'rgba(217,98,43,0.07)';
  ctx.fillRect(margin.left, py1, cw - margin.left - margin.right, py0 - py1);

  const xs = [], fy = [], gy = [];
  for (let i = 0; i <= 300; i++) {
    const x = i / 300;
    xs.push(x);
    fy.push(f(x));
    gy.push(logit(f(x)));
  }
  plotCurve(ctx, xs, fy, margin, cw, ch, xRange, yRange, '#d9622b', 3);
  plotCurve(ctx, xs, gy, margin, cw, ch, xRange, yRange, '#7558a8', 2.6, [7, 4]);
  ctx.font = '600 12px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#d9622b';
  ctx.fillText('target  f(x)  — lives in 0…1', margin.left + 8, py1 + 16);
  ctx.fillStyle = '#7558a8';
  ctx.fillText('σ⁻¹(f) — the un-squished goal (stretched out)', margin.left + 8, margin.top + 14);
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
  spike: { label: 'Sharp spike', raw: (x) => 0.1 + 0.85 * Math.exp(-Math.pow((x - 0.5) / 0.05, 2)) },
  plateaus: {
    label: 'Plateaus',
    raw: (x) => 0.2 + 0.32 * sigmoid(26 * (x - 0.33)) + 0.32 * sigmoid(26 * (x - 0.66)),
  },
};
function targetFn(key) {
  const raw = DESIGN_TARGETS[key].raw;
  return (x) => clamp(raw(x), 0.04, 0.96);
}

let designState = { target: 'wave', N: 6, p: [], view: 'fn' };

function initDesign() {
  const canvas = document.getElementById('designCanvas');
  if (!canvas) return;
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

  const steepness = () => 24 * designState.N;
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
      designState.p[i] = clamp(f((i + 0.5) / designState.N), 0.02, 0.98);
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
    btnMath.textContent = logitView ? 'Back to the simple view' : 'Peek: the un-squished goal';

    const yRange = logitView ? [-6, 6] : [-0.05, 1.05];

    const { ctx, w: cw, h: ch } = setupCanvas(canvas, 920, 360);
    ctx.clearRect(0, 0, cw, ch);
    drawAxes(ctx, margin, cw, ch, xRange, yRange, 'x', logitView ? 'σ⁻¹(f)' : 'y');

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

    // --- show the bricks (the individual bumps) ---
    if (!logitView) {
      // function view: each bump is the plateau height it sets on its slab
      const [, pyBase] = dataToPixel(0, 0, margin, cw, ch, xRange, yRange);
      for (let i = 0; i < designState.N; i++) {
        const p = designState.p[i];
        const [px0] = dataToPixel(i / designState.N, 0, margin, cw, ch, xRange, yRange);
        const [px1] = dataToPixel((i + 1) / designState.N, 0, margin, cw, ch, xRange, yRange);
        const [, pyTop] = dataToPixel(0, p, margin, cw, ch, xRange, yRange);
        const [r, g, b] = colormapRGB(p);
        ctx.fillStyle = `rgba(${r},${g},${b},0.16)`;
        ctx.fillRect(px0, pyTop, px1 - px0, pyBase - pyTop);
        ctx.strokeStyle = `rgba(${r},${g},${b},0.55)`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px0 + 1, pyTop);
        ctx.lineTo(px1 - 1, pyTop);
        ctx.stroke();
      }
    } else {
      // logit view: the bumps add up linearly — draw each additive piece
      for (let i = 0; i < designState.N; i++) {
        const xs2 = [], pcs = [];
        for (let k = 0; k <= 160; k++) {
          const x = k / 160;
          xs2.push(x);
          pcs.push(logit(designState.p[i]) * bumpVal(x, i));
        }
        plotCurve(ctx, xs2, pcs, margin, cw, ch, xRange, yRange, 'rgba(44,111,183,0.28)', 1.5);
      }
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
        ? '<span style="color:var(--warm);font-weight:600">Orange</span>: the un-squished goal σ⁻¹(f). <span style="color:var(--accent);font-weight:600">Blue</span>: the bumps&rsquo; total H(x). Same bars, just shown before the final squish.'
        : '<span style="color:var(--warm);font-weight:600">Orange</span>: target f(x). <span style="color:var(--accent);font-weight:600">Blue</span>: your network&rsquo;s output. Drag the bars to make blue hug orange.';
    }
  }

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
// STEP 5 — two inputs: build a tower (staged, in 3D)
// ============================================================
function initTower() {
  const wallC = document.getElementById('towerWallCanvas');
  if (!wallC) return;
  const ridgeC = document.getElementById('towerRidgeCanvas');
  const crossC = document.getElementById('towerCrossCanvas');
  const towerC = document.getElementById('towerTowerCanvas');

  const cxS = document.getElementById('tower-cx');
  const cyS = document.getElementById('tower-cy');
  const sizeS = document.getElementById('tower-size');
  const thrS = document.getElementById('tower-thr');
  const wS = document.getElementById('tower-w');
  const angS = document.getElementById('tower-angle');
  const cxV = document.getElementById('val-tower-cx');
  const cyV = document.getElementById('val-tower-cy');
  const sizeV = document.getElementById('val-tower-size');
  const thrV = document.getElementById('val-tower-thr');
  const wV = document.getElementById('val-tower-w');

  const G = 44;
  let wall, ridge, cross, tower; // cached fields

  function compute() {
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
    const K = 18;

    wall = new Float32Array(G * G);
    ridge = new Float32Array(G * G);
    cross = new Float32Array(G * G);
    tower = new Float32Array(G * G);
    for (let j = 0; j < G; j++) {
      const y = j / (G - 1);
      const byBump = sigmoid(W * (y - y0)) - sigmoid(W * (y - y1));
      for (let i = 0; i < G; i++) {
        const x = i / (G - 1);
        const wallV = sigmoid(W * (x - x0)); // single step in x
        const bxBump = sigmoid(W * (x - x0)) - sigmoid(W * (x - x1));
        const s = bxBump + byBump; // 0..2
        const idx = j * G + i;
        wall[idx] = wallV;
        ridge[idx] = bxBump;
        cross[idx] = s;
        tower[idx] = sigmoid(K * (s - thr));
      }
    }
  }

  function render() {
    const az = parseFloat(angS.value);
    const common = { side: 320, azimuth: az, elevation: 0.5, edges: true };
    drawSurface(wallC, wall, G, { ...common, vmin: 0, vmax: 1 });
    drawSurface(ridgeC, ridge, G, { ...common, vmin: 0, vmax: 1 });
    drawSurface(crossC, cross, G, { ...common, vmin: 0, vmax: 2 });
    drawSurface(towerC, tower, G, { ...common, vmin: 0, vmax: 1 });
  }

  [cxS, cyS, sizeS, thrS, wS].forEach((el) =>
    el.addEventListener('input', () => {
      compute();
      render();
    })
  );
  angS.addEventListener('input', render);

  compute();
  render();
}

// ============================================================
// STEP 6 — tile towers into a surface (in 3D)
// ============================================================
const SURFACE_TARGETS = {
  spike: {
    label: 'Single spike',
    raw: (x, y) => Math.exp(-(Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2)) / 0.008),
  },
  spikes: {
    label: 'Four spikes',
    raw: (x, y) => {
      const c = [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]];
      let v = 0;
      for (const [a, b] of c) v += Math.exp(-(Math.pow(x - a, 2) + Math.pow(y - b, 2)) / 0.006);
      return v;
    },
  },
  peaks: {
    label: 'Peak & pit',
    raw: (x, y) =>
      Math.exp(-(Math.pow(x - 0.32, 2) + Math.pow(y - 0.34, 2)) / 0.02) -
      0.8 * Math.exp(-(Math.pow(x - 0.7, 2) + Math.pow(y - 0.68, 2)) / 0.025),
  },
  ripple: {
    label: 'Ripples',
    raw: (x, y) => {
      const r = Math.sqrt(Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2));
      return Math.cos(16 * r) * Math.exp(-2.2 * r);
    },
  },
};

let surfaceState = { target: 'spike', res: 7 };

function initSurface() {
  const targetCanvas = document.getElementById('surfaceTargetCanvas');
  if (!targetCanvas) return;
  const approxCanvas = document.getElementById('surfaceApproxCanvas');
  const resS = document.getElementById('surface-res');
  const resV = document.getElementById('val-surface-res');
  const resV2 = document.getElementById('val-surface-res2');
  const angS = document.getElementById('surface-angle');
  const towersStat = document.getElementById('stat-surface-towers');
  const neuronsStat = document.getElementById('stat-surface-neurons');
  const mseStat = document.getElementById('stat-surface-mse');
  const buttons = document.querySelectorAll('#surface-target-buttons [data-surface]');

  const G = 52;
  let targetField, approxField;

  function normalizedTarget(key) {
    const raw = SURFACE_TARGETS[key].raw;
    let lo = Infinity, hi = -Infinity;
    const S = 60;
    for (let i = 0; i <= S; i++)
      for (let j = 0; j <= S; j++) {
        const v = raw(i / S, j / S);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    const span = hi - lo || 1;
    return (x, y) => (raw(x, y) - lo) / span;
  }

  function compute() {
    const R = surfaceState.res;
    const f = normalizedTarget(surfaceState.target);
    resV.textContent = R;
    resV2.textContent = R;
    towersStat.textContent = R * R;
    neuronsStat.textContent = 4 * R * R;

    const W = clamp(12 * R, 36, 170);
    const K = 18;

    const Bx = [], By = [];
    for (let i = 0; i < R; i++) {
      const x0 = i / R, x1 = (i + 1) / R;
      const row = new Float32Array(G);
      for (let gx = 0; gx < G; gx++) {
        const x = gx / (G - 1);
        row[gx] = sigmoid(W * (x - x0)) - sigmoid(W * (x - x1));
      }
      Bx.push(row);
    }
    for (let j = 0; j < R; j++) {
      const y0 = j / R, y1 = (j + 1) / R;
      const row = new Float32Array(G);
      for (let gy = 0; gy < G; gy++) {
        const y = gy / (G - 1);
        row[gy] = sigmoid(W * (y - y0)) - sigmoid(W * (y - y1));
      }
      By.push(row);
    }

    const heights = [];
    for (let i = 0; i < R; i++) {
      const hr = new Float32Array(R);
      for (let j = 0; j < R; j++) hr[j] = f((i + 0.5) / R, (j + 0.5) / R);
      heights.push(hr);
    }

    targetField = new Float32Array(G * G);
    approxField = new Float32Array(G * G);
    let mse = 0;
    for (let gy = 0; gy < G; gy++) {
      const y = gy / (G - 1);
      for (let gx = 0; gx < G; gx++) {
        const x = gx / (G - 1);
        const tv = f(x, y);
        let acc = 0;
        for (let i = 0; i < R; i++) {
          const bxi = Bx[i][gx];
          if (bxi < 0.01) continue;
          for (let j = 0; j < R; j++) {
            const tw = sigmoid(K * (bxi + By[j][gy] - 1.5));
            if (tw > 0.01) acc += heights[i][j] * tw;
          }
        }
        targetField[gy * G + gx] = tv;
        approxField[gy * G + gx] = acc;
        const d = acc - tv;
        mse += d * d;
      }
    }
    mse /= G * G;
    mseStat.textContent = mse.toFixed(4);
  }

  function render() {
    const az = parseFloat(angS.value);
    const common = { side: 430, azimuth: az, elevation: 0.5, vmin: 0, vmax: 1, edges: true };
    drawSurface(targetCanvas, targetField, G, common);
    drawSurface(approxCanvas, approxField, G, common);
  }

  resS.addEventListener('input', () => {
    surfaceState.res = parseInt(resS.value, 10);
    compute();
    render();
  });
  angS.addEventListener('input', render);
  buttons.forEach((b) => {
    b.addEventListener('click', () => {
      buttons.forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      surfaceState.target = b.dataset.surface;
      compute();
      render();
    });
  });

  compute();
  render();
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
  initLogitMini();
  initDesign();
  initTower();
  initSurface();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
