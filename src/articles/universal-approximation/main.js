// ============================================================
// Watching a Neural Network Become Universal — Enhanced
// All computations are real: live neuron evaluation and a
// working MLP trained by mini-batch SGD in the browser.
// ============================================================

// ---------- Target functions ----------
const TARGETS = {
  sine: {
    fn: (x) => Math.sin(2 * Math.PI * x),
    label: 'sin(2\u03c0x)',
    yRange: [-1.2, 1.2]
  },
  step: {
    fn: (x) => (x < 0.5 ? -0.7 : 0.7),
    label: 'step at x = 0.5',
    yRange: [-1.0, 1.0]
  },
  spike: {
    fn: (x) => Math.exp(-60 * Math.pow(x - 0.5, 2)),
    label: 'Gaussian spike',
    yRange: [-0.2, 1.2]
  },
  zigzag: {
    fn: (x) => {
      const t = 4 * x;
      return 0.8 * (Math.abs((t % 2) - 1) * 2 - 1);
    },
    label: 'zigzag (triangle wave)',
    yRange: [-1.0, 1.0]
  },
  bumpy: {
    fn: (x) => 0.6 * Math.sin(6 * Math.PI * x) * Math.exp(-Math.pow((x - 0.5) * 2.5, 2)),
    label: 'windowed oscillation',
    yRange: [-0.8, 0.8]
  },
  custom: {
    fn: null,
    label: 'Your drawing',
    yRange: [-1.5, 1.5]
  }
};

// ---------- Custom drawing system ----------
const CUSTOM_SAMPLE_COUNT = 400;

function buildCustomTarget(drawnPoints, yRange) {
  const sorted = [...drawnPoints].sort((a, b) => a.x - b.x);
  if (sorted.length < 2) return;

  const N = CUSTOM_SAMPLE_COUNT;
  const samples = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const x = i / (N - 1);
    // Binary search for surrounding points
    let lo = 0, hi = sorted.length - 1;
    if (x <= sorted[0].x) { samples[i] = sorted[0].y; continue; }
    if (x >= sorted[hi].x) { samples[i] = sorted[hi].y; continue; }
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid].x <= x) lo = mid;
      else hi = mid;
    }
    const t = (x - sorted[lo].x) / (sorted[hi].x - sorted[lo].x + 1e-12);
    samples[i] = sorted[lo].y * (1 - t) + sorted[hi].y * t;
  }

  // Compute actual y range with padding
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < N; i++) {
    if (samples[i] < minY) minY = samples[i];
    if (samples[i] > maxY) maxY = samples[i];
  }
  const pad = Math.max(0.3, (maxY - minY) * 0.15);
  const computedRange = [minY - pad, maxY + pad];

  TARGETS.custom.fn = (x) => {
    const idx = x * (N - 1);
    const lo = Math.max(0, Math.floor(idx));
    const hi = Math.min(lo + 1, N - 1);
    const t = idx - lo;
    return samples[lo] * (1 - t) + samples[hi] * t;
  };
  TARGETS.custom.yRange = computedRange;
  TARGETS.custom.samples = samples;
}

function pixelToPlot(canvas, event, margin, logicalW, logicalH, xRange, yRange) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = logicalW / rect.width;
  const scaleY = logicalH / rect.height;
  const px = (event.clientX - rect.left) * scaleX;
  const py = (event.clientY - rect.top) * scaleY;

  const plotW = logicalW - margin.left - margin.right;
  const plotH = logicalH - margin.top - margin.bottom;

  const x = xRange[0] + ((px - margin.left) / plotW) * (xRange[1] - xRange[0]);
  const y = yRange[1] - ((py - margin.top) / plotH) * (yRange[1] - yRange[0]);

  return {
    x: Math.max(xRange[0], Math.min(xRange[1], x)),
    y: Math.max(yRange[0], Math.min(yRange[1], y))
  };
}

function getTouchPos(canvas, touch) {
  return { clientX: touch.clientX, clientY: touch.clientY };
}

// ---------- Canvas helper ----------
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

  // Zero line (y = 0) highlighted
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
  ctx.moveTo(margin.left, margin.top + plotH);
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
    let yVal = ys[i];
    if (yVal > yRange[1]) yVal = yRange[1];
    if (yVal < yRange[0]) yVal = yRange[0];
    const py = margin.top + plotH - ((yVal - yRange[0]) / (yRange[1] - yRange[0])) * plotH;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawInstructionText(ctx, text, cw, ch) {
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(154, 145, 127, 0.6)';
  ctx.font = '600 18px system-ui, sans-serif';
  ctx.fillText(text, cw / 2, ch / 2);
  ctx.restore();
}

// ---------- Activation functions ----------
const ACT = {
  relu: {
    f: (z) => (z > 0 ? z : 0),
    df: (z) => (z > 0 ? 1 : 0),
    label: 'ReLU'
  },
  tanh: {
    f: (z) => Math.tanh(z),
    df: (z) => 1 - Math.tanh(z) ** 2,
    label: 'tanh'
  },
  sigmoid: {
    f: (z) => 1 / (1 + Math.exp(-z)),
    df: (z) => {
      const s = 1 / (1 + Math.exp(-z));
      return s * (1 - s);
    },
    label: 'sigmoid'
  }
};

// ============================================================
// STEP 1: Single neuron
// ============================================================
function initSingleNeuron() {
  const canvas = document.getElementById('singleCanvas');
  const wSlider = document.getElementById('single-w');
  const bSlider = document.getElementById('single-b');
  const wVal = document.getElementById('val-single-w');
  const bVal = document.getElementById('val-single-b');
  const formula = document.getElementById('single-formula');

  function draw() {
    const w = parseFloat(wSlider.value);
    const b = parseFloat(bSlider.value);
    wVal.textContent = w.toFixed(1);
    bVal.textContent = b.toFixed(1);
    if (formula) {
      formula.textContent = `y = max(0, ${w.toFixed(1)}\u00b7x + ${b.toFixed(1)})`;
    }

    const { ctx, w: cw, h: ch } = setupCanvas(canvas, 920, 300);
    ctx.clearRect(0, 0, cw, ch);

    const margin = { top: 20, right: 30, bottom: 35, left: 45 };
    const xRange = [-3, 3];
    const yRange = [-0.2, 3.2];
    drawAxes(ctx, margin, cw, ch, xRange, yRange);

    const xs = [], ys = [];
    for (let i = 0; i <= 200; i++) {
      const x = xRange[0] + ((xRange[1] - xRange[0]) * i) / 200;
      xs.push(x);
      ys.push(ACT.relu.f(w * x + b));
    }
    plotCurve(ctx, xs, ys, margin, cw, ch, xRange, yRange, '#2c6fb7', 3);

    // Mark kink position
    if (Math.abs(w) > 1e-6) {
      const kinkX = -b / w;
      if (kinkX >= xRange[0] && kinkX <= xRange[1]) {
        const plotW = cw - margin.left - margin.right;
        const plotH = ch - margin.top - margin.bottom;
        const px = margin.left + ((kinkX - xRange[0]) / (xRange[1] - xRange[0])) * plotW;
        const py = margin.top + plotH - ((0 - yRange[0]) / (yRange[1] - yRange[0])) * plotH;
        ctx.fillStyle = '#d9622b';
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#d9622b';
        ctx.font = '600 12px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`kink at x = ${kinkX.toFixed(2)}`, px, py - 12);
      }
    }
  }

  wSlider.addEventListener('input', draw);
  bSlider.addEventListener('input', draw);
  draw();
}

// ============================================================
// STEP 2: Two-neuron bump
// ============================================================
function initBump() {
  const canvas = document.getElementById('bumpCanvas');
  const b1S = document.getElementById('b1');
  const b2S = document.getElementById('b2');
  const c1S = document.getElementById('c1');
  const c2S = document.getElementById('c2');
  const b1V = document.getElementById('val-b1');
  const b2V = document.getElementById('val-b2');
  const c1V = document.getElementById('val-c1');
  const c2V = document.getElementById('val-c2');

  function draw() {
    const b1 = parseFloat(b1S.value);
    const b2 = parseFloat(b2S.value);
    const c1 = parseFloat(c1S.value);
    const c2 = parseFloat(c2S.value);
    b1V.textContent = b1.toFixed(1);
    b2V.textContent = b2.toFixed(1);
    c1V.textContent = c1.toFixed(1);
    c2V.textContent = c2.toFixed(1);

    const { ctx, w: cw, h: ch } = setupCanvas(canvas, 920, 320);
    ctx.clearRect(0, 0, cw, ch);

    const margin = { top: 20, right: 30, bottom: 35, left: 45 };
    const xRange = [-3, 3];
    const yRange = [-2.5, 2.5];
    drawAxes(ctx, margin, cw, ch, xRange, yRange);

    const xs = [];
    const h1s = [], h2s = [], ysum = [];
    const N = 240;
    for (let i = 0; i <= N; i++) {
      const x = xRange[0] + ((xRange[1] - xRange[0]) * i) / N;
      xs.push(x);
      const n1 = ACT.relu.f(2 * x + b1);
      const n2 = ACT.relu.f(2 * x + b2);
      h1s.push(c1 * n1);
      h2s.push(c2 * n2);
      ysum.push(c1 * n1 + c2 * n2);
    }

    plotCurve(ctx, xs, h1s, margin, cw, ch, xRange, yRange, 'rgba(44, 111, 183, 0.45)', 2, [6, 4]);
    plotCurve(ctx, xs, h2s, margin, cw, ch, xRange, yRange, 'rgba(30, 119, 112, 0.45)', 2, [6, 4]);
    plotCurve(ctx, xs, ysum, margin, cw, ch, xRange, yRange, '#d9622b', 3);

    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(44, 111, 183, 0.8)';
    ctx.fillText('c\u2081\u00b7ReLU(2x + b\u2081)', margin.left + 8, margin.top + 14);
    ctx.fillStyle = 'rgba(30, 119, 112, 0.8)';
    ctx.fillText('c\u2082\u00b7ReLU(2x + b\u2082)', margin.left + 8, margin.top + 30);
    ctx.fillStyle = '#d9622b';
    ctx.fillText('Sum', margin.left + 8, margin.top + 46);
  }

  [b1S, b2S, c1S, c2S].forEach((el) => el.addEventListener('input', draw));
  draw();
}

// ============================================================
// STEP 3: Hand-placed bumps fitting a target
// ============================================================
let manualState = {
  target: 'sine',
  width: 8
};

let manualDrawState = {
  isDrawing: false,
  points: [],
  hasDrawn: false
};

function manualFit(targetKey, width) {
  const t = TARGETS[targetKey];
  if (!t.fn) return null;
  const kinks = new Array(width);
  const targetVals = new Array(width);
  for (let i = 0; i < width; i++) {
    const k = (i + 0.5) / width;
    kinks[i] = k;
    targetVals[i] = t.fn(k);
  }

  const inputW = 30;
  const c = new Array(width);
  c[0] = targetVals[0];
  for (let i = 1; i < width; i++) {
    c[i] = targetVals[i] - targetVals[i - 1];
  }
  const b0 = 0;

  function predict(x) {
    let y = b0;
    for (let i = 0; i < width; i++) {
      const z = inputW * (x - kinks[i]);
      y += c[i] * ACT.relu.f(z);
    }
    return y;
  }

  return { predict, kinks, inputW, c, b0 };
}

function manualMSE(model, targetKey, n = 300) {
  const t = TARGETS[targetKey];
  if (!t.fn || !model) return Infinity;
  let mse = 0;
  for (let i = 0; i < n; i++) {
    const x = i / (n - 1);
    const diff = model.predict(x) - t.fn(x);
    mse += diff * diff;
  }
  return mse / n;
}

function initManual() {
  const canvas = document.getElementById('manualCanvas');
  const widthSlider = document.getElementById('manual-width');
  const widthVal = document.getElementById('val-manual-width');
  const buttons = document.querySelectorAll('#manual-scenario-buttons [data-target]');
  const neuronsStat = document.getElementById('stat-manual-neurons');
  const paramsStat = document.getElementById('stat-manual-params');
  const mseStat = document.getElementById('stat-manual-mse');
  const tableBody = document.getElementById('manual-table-body');
  const clearBtn = document.getElementById('btn-manual-clear-draw');

  const manualLogicalW = 920, manualLogicalH = 360;
  const manualMargin = { top: 20, right: 30, bottom: 35, left: 50 };
  const manualXRange = [0, 1];

  function paramCount(N) {
    return 3 * N + 1;
  }

  function draw() {
    const { target, width } = manualState;
    widthVal.textContent = width;
    neuronsStat.textContent = width;
    paramsStat.textContent = paramCount(width);

    const { ctx, w: cw, h: ch } = setupCanvas(canvas, manualLogicalW, manualLogicalH);
    ctx.clearRect(0, 0, cw, ch);

    // Handle custom drawing mode
    if (target === 'custom') {
      if (clearBtn) clearBtn.style.display = manualDrawState.hasDrawn ? '' : 'none';
      canvas.classList.toggle('draw-mode', !manualDrawState.hasDrawn);

      if (!manualDrawState.hasDrawn) {
        const yRange = [-1.5, 1.5];
        drawAxes(ctx, manualMargin, cw, ch, manualXRange, yRange);

        if (manualDrawState.isDrawing && manualDrawState.points.length > 1) {
          // Draw the in-progress stroke
          const pts = manualDrawState.points;
          const plotW = cw - manualMargin.left - manualMargin.right;
          const plotH = ch - manualMargin.top - manualMargin.bottom;
          ctx.strokeStyle = '#d9622b';
          ctx.lineWidth = 3;
          ctx.beginPath();
          for (let i = 0; i < pts.length; i++) {
            const px = manualMargin.left + ((pts[i].x - manualXRange[0]) / (manualXRange[1] - manualXRange[0])) * plotW;
            const py = manualMargin.top + plotH - ((pts[i].y - yRange[0]) / (yRange[1] - yRange[0])) * plotH;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          ctx.stroke();
        } else {
          drawInstructionText(ctx, 'Click and drag to draw a curve', cw, ch);
        }
        mseStat.textContent = '\u2014';
        renderManualTable(target);
        return;
      }
    } else {
      if (clearBtn) clearBtn.style.display = 'none';
      canvas.classList.remove('draw-mode');
    }

    const t = TARGETS[target];
    if (!t.fn) { mseStat.textContent = '\u2014'; return; }

    const yRange = t.yRange;
    drawAxes(ctx, manualMargin, cw, ch, manualXRange, yRange);

    const model = manualFit(target, width);
    const mse = model ? manualMSE(model, target) : Infinity;
    mseStat.textContent = isFinite(mse) ? mse.toFixed(4) : '\u2014';

    // Individual neurons (faint)
    if (model && width <= 30) {
      for (let i = 0; i < width; i++) {
        const xs = [], ys = [];
        for (let j = 0; j <= 200; j++) {
          const x = j / 200;
          xs.push(x);
          ys.push(model.c[i] * ACT.relu.f(model.inputW * (x - model.kinks[i])));
        }
        plotCurve(ctx, xs, ys, manualMargin, cw, ch, manualXRange, yRange, 'rgba(140, 130, 110, 0.4)', 1.2);
      }
    }

    // Target
    const targetXs = [], targetYs = [];
    for (let i = 0; i <= 400; i++) {
      const x = i / 400;
      targetXs.push(x);
      targetYs.push(t.fn(x));
    }
    plotCurve(ctx, targetXs, targetYs, manualMargin, cw, ch, manualXRange, yRange, '#d9622b', 3);

    // Network output
    if (model) {
      const netXs = [], netYs = [];
      for (let i = 0; i <= 400; i++) {
        const x = i / 400;
        netXs.push(x);
        netYs.push(model.predict(x));
      }
      plotCurve(ctx, netXs, netYs, manualMargin, cw, ch, manualXRange, yRange, '#2c6fb7', 2.5);
    }

    renderManualTable(target);
  }

  function renderManualTable(target) {
    if (target === 'custom' && !manualDrawState.hasDrawn) {
      tableBody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:#9a917f;">Draw a target above to see the table</td></tr>';
      return;
    }
    const t = TARGETS[target];
    if (!t.fn) return;
    const rows = [2, 4, 8, 16, 32, 60];
    const descriptions = {
      2: 'A single bump + linear base \u2014 can barely match a sine',
      4: 'Two bumps \u2014 roughly the positive and negative halves',
      8: 'Coarse staircase \u2014 visible kinks at every sample',
      16: 'Smooth-looking fit for most targets',
      32: 'Near-perfect on sine; sharp transitions on step/spike',
      60: 'Indistinguishable from the target on smooth functions'
    };
    let html = '';
    for (const N of rows) {
      const model = manualFit(target, N);
      const mse = model ? manualMSE(model, target) : Infinity;
      html += `
        <tr>
          <td><strong>N = ${N}</strong></td>
          <td>${descriptions[N]}</td>
          <td><code>${isFinite(mse) ? mse.toExponential(2) : '\u2014'}</code></td>
        </tr>
      `;
    }
    tableBody.innerHTML = html;
  }

  // --- Drawing handlers for manual canvas ---
  function handleDrawStart(e) {
    if (manualState.target !== 'custom' || manualDrawState.hasDrawn) return;
    e.preventDefault();
    manualDrawState.isDrawing = true;
    manualDrawState.points = [];
    const pos = e.touches ? getTouchPos(canvas, e.touches[0]) : e;
    const p = pixelToPlot(canvas, pos, manualMargin, manualLogicalW, manualLogicalH, manualXRange, [-1.5, 1.5]);
    manualDrawState.points.push(p);
    draw();
  }

  function handleDrawMove(e) {
    if (!manualDrawState.isDrawing) return;
    e.preventDefault();
    const pos = e.touches ? getTouchPos(canvas, e.touches[0]) : e;
    const p = pixelToPlot(canvas, pos, manualMargin, manualLogicalW, manualLogicalH, manualXRange, [-1.5, 1.5]);
    manualDrawState.points.push(p);
    draw();
  }

  function handleDrawEnd(e) {
    if (!manualDrawState.isDrawing) return;
    manualDrawState.isDrawing = false;
    if (manualDrawState.points.length > 3) {
      buildCustomTarget(manualDrawState.points, [-1.5, 1.5]);
      manualDrawState.hasDrawn = true;
    }
    draw();
  }

  canvas.addEventListener('mousedown', handleDrawStart);
  canvas.addEventListener('mousemove', handleDrawMove);
  canvas.addEventListener('mouseup', handleDrawEnd);
  canvas.addEventListener('mouseleave', handleDrawEnd);
  canvas.addEventListener('touchstart', handleDrawStart, { passive: false });
  canvas.addEventListener('touchmove', handleDrawMove, { passive: false });
  canvas.addEventListener('touchend', handleDrawEnd);

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      manualDrawState.hasDrawn = false;
      manualDrawState.points = [];
      TARGETS.custom.fn = null;
      draw();
    });
  }

  widthSlider.addEventListener('input', () => {
    manualState.width = parseInt(widthSlider.value, 10);
    draw();
  });
  buttons.forEach((b) => {
    b.addEventListener('click', () => {
      buttons.forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      manualState.target = b.dataset.target;
      if (b.dataset.target !== 'custom') {
        manualDrawState.hasDrawn = false;
        manualDrawState.points = [];
      }
      draw();
    });
  });
  draw();
}

// ============================================================
// STEP 5: Real trainable MLP
// ============================================================
function createMLP(width, activation) {
  const act = ACT[activation];
  const W1 = new Array(width);
  const b1 = new Array(width);
  for (let i = 0; i < width; i++) {
    if (activation === 'relu') {
      W1[i] = (Math.random() * 2 - 1) * 6;
      b1[i] = -((i + 0.5) / width) * W1[i] + (Math.random() - 0.5) * 1.5;
    } else {
      W1[i] = (Math.random() * 2 - 1) * 4;
      b1[i] = -((i + 0.5) / width) * W1[i] + (Math.random() - 0.5) * 0.5;
    }
  }
  const W2 = new Array(width);
  for (let i = 0; i < width; i++) W2[i] = (Math.random() * 2 - 1) * (1 / Math.sqrt(width));
  let b2 = 0;

  function forward(x) {
    const z1 = new Array(width);
    const h = new Array(width);
    for (let i = 0; i < width; i++) {
      z1[i] = W1[i] * x + b1[i];
      h[i] = act.f(z1[i]);
    }
    let yhat = b2;
    for (let i = 0; i < width; i++) yhat += W2[i] * h[i];
    return { yhat, z1, h };
  }

  function predict(x) {
    return forward(x).yhat;
  }

  function trainStep(batchX, batchY, lr) {
    const B = batchX.length;
    const dW1 = new Array(width).fill(0);
    const db1 = new Array(width).fill(0);
    const dW2 = new Array(width).fill(0);
    let db2 = 0;
    let loss = 0;
    for (let k = 0; k < B; k++) {
      const x = batchX[k];
      const y = batchY[k];
      const { yhat, z1, h } = forward(x);
      const err = yhat - y;
      loss += err * err;
      const dY = 2 * err;
      db2 += dY;
      for (let i = 0; i < width; i++) {
        dW2[i] += dY * h[i];
        const dh = dY * W2[i];
        const dz = dh * act.df(z1[i]);
        dW1[i] += dz * x;
        db1[i] += dz;
      }
    }
    const inv = 1 / B;
    for (let i = 0; i < width; i++) {
      W1[i] -= lr * dW1[i] * inv;
      b1[i] -= lr * db1[i] * inv;
      W2[i] -= lr * dW2[i] * inv;
    }
    b2 -= lr * db2 * inv;
    return loss / B;
  }

  // Compute each neuron's individual contribution curve
  function neuronOutputs(xArr) {
    const results = [];
    for (let j = 0; j < width; j++) {
      const ys = new Array(xArr.length);
      for (let i = 0; i < xArr.length; i++) {
        const z = W1[j] * xArr[i] + b1[j];
        ys[i] = W2[j] * act.f(z);
      }
      results.push(ys);
    }
    return results;
  }

  // Get kink positions (for ReLU: -b1/W1)
  function kinkPositions() {
    const kinks = [];
    for (let j = 0; j < width; j++) {
      if (Math.abs(W1[j]) < 1e-8) continue;
      const kx = -b1[j] / W1[j];
      kinks.push({ x: kx, weight: W2[j], idx: j });
    }
    return kinks;
  }

  return {
    get width() { return width; },
    get activation() { return activation; },
    get W1() { return W1; },
    get b1() { return b1; },
    get W2() { return W2; },
    get b2() { return b2; },
    forward,
    predict,
    trainStep,
    neuronOutputs,
    kinkPositions
  };
}

let trainState = {
  target: 'sine',
  width: 16,
  activation: 'relu',
  model: null,
  epoch: 0,
  losses: [],
  running: false,
  dataX: null,
  dataY: null,
  lr: 0.05,
  showNeurons: false,
  showComparison: false
};

let trainDrawState = {
  isDrawing: false,
  points: [],
  hasDrawn: false
};

function buildData(targetKey) {
  const t = TARGETS[targetKey];
  if (!t.fn) return { xs: [], ys: [] };
  const n = 128;
  const xs = new Array(n);
  const ys = new Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = i / (n - 1);
    ys[i] = t.fn(xs[i]);
  }
  return { xs, ys };
}

function initTrainer() {
  const canvas = document.getElementById('trainCanvas');
  const lossCanvas = document.getElementById('lossCanvas');
  const widthSlider = document.getElementById('train-width');
  const widthVal = document.getElementById('val-train-width');
  const lrSlider = document.getElementById('train-lr');
  const lrVal = document.getElementById('val-train-lr');
  const activationChips = document.querySelectorAll('#activation-chips [data-act]');
  const activationLabel = document.getElementById('val-train-activation');
  const buttons = document.querySelectorAll('#train-scenario-buttons [data-target]');
  const btnTrain = document.getElementById('btn-train');
  const btnStep = document.getElementById('btn-step');
  const btnReset = document.getElementById('btn-reset');
  const btnShowNeurons = document.getElementById('btn-show-neurons');
  const btnShowComparison = document.getElementById('btn-show-comparison');
  const clearDrawBtn = document.getElementById('btn-train-clear-draw');
  const epochStat = document.getElementById('stat-epoch');
  const paramsStat = document.getElementById('stat-train-params');
  const mseStat = document.getElementById('stat-train-mse');
  const revealN = document.getElementById('reveal-n');
  const revealParams = document.getElementById('reveal-params');
  const revealBreakdown = document.getElementById('reveal-breakdown');

  const trainLogicalW = 920, trainLogicalH = 340;
  const trainMargin = { top: 20, right: 30, bottom: 35, left: 50 };
  const trainXRange = [0, 1];

  function paramCount(N) { return 3 * N + 1; }

  function resetModel() {
    trainState.model = createMLP(trainState.width, trainState.activation);
    trainState.epoch = 0;
    trainState.losses = [];
    const data = buildData(trainState.target);
    trainState.dataX = data.xs;
    trainState.dataY = data.ys;
    drawAll();
  }

  // --- Neuron decomposition colors ---
  function neuronColor(idx, total, alpha) {
    const hue = (idx / total) * 300;
    return `hsla(${hue}, 65%, 50%, ${alpha})`;
  }

  function drawAll() {
    widthVal.textContent = trainState.width;
    activationLabel.textContent = ACT[trainState.activation].label;
    epochStat.textContent = trainState.epoch;
    paramsStat.textContent = paramCount(trainState.width);

    if (lrSlider && lrVal) {
      lrVal.textContent = trainState.lr.toFixed(3);
    }

    revealN.textContent = trainState.width;
    const N = trainState.width;
    revealParams.textContent = paramCount(N);
    revealBreakdown.textContent =
      `${N} input weights + ${N} hidden biases + ${N} output weights + 1 output bias`;

    // Toggle button states
    if (btnShowNeurons) {
      btnShowNeurons.classList.toggle('is-active', trainState.showNeurons);
    }
    if (btnShowComparison) {
      btnShowComparison.classList.toggle('is-active', trainState.showComparison);
    }
    if (clearDrawBtn) {
      clearDrawBtn.style.display = (trainState.target === 'custom' && trainDrawState.hasDrawn) ? '' : 'none';
    }

    // --- Fit canvas ---
    const t = TARGETS[trainState.target];
    const { ctx, w: cw, h: ch } = setupCanvas(canvas, trainLogicalW, trainLogicalH);
    ctx.clearRect(0, 0, cw, ch);

    // Handle custom draw mode
    canvas.classList.toggle('draw-mode',
      trainState.target === 'custom' && !trainDrawState.hasDrawn);
    if (trainState.target === 'custom' && !trainDrawState.hasDrawn) {
      const yRange = [-1.5, 1.5];
      drawAxes(ctx, trainMargin, cw, ch, trainXRange, yRange);

      if (trainDrawState.isDrawing && trainDrawState.points.length > 1) {
        const pts = trainDrawState.points;
        const plotW = cw - trainMargin.left - trainMargin.right;
        const plotH = ch - trainMargin.top - trainMargin.bottom;
        ctx.strokeStyle = '#d9622b';
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i < pts.length; i++) {
          const px = trainMargin.left + ((pts[i].x - trainXRange[0]) / (trainXRange[1] - trainXRange[0])) * plotW;
          const py = trainMargin.top + plotH - ((pts[i].y - yRange[0]) / (yRange[1] - yRange[0])) * plotH;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      } else {
        drawInstructionText(ctx, 'Click and drag to draw a target curve', cw, ch);
      }
      mseStat.textContent = '\u2014';
      drawLossCurve();
      return;
    }

    if (!t.fn) {
      mseStat.textContent = '\u2014';
      drawLossCurve();
      return;
    }

    const yRange = t.yRange;
    drawAxes(ctx, trainMargin, cw, ch, trainXRange, yRange);

    // --- Show individual neuron contributions ---
    if (trainState.showNeurons && trainState.model && trainState.width <= 40) {
      const xArr = [];
      for (let i = 0; i <= 300; i++) xArr.push(i / 300);
      const neuronYs = trainState.model.neuronOutputs(xArr);
      for (let j = 0; j < trainState.width; j++) {
        plotCurve(ctx, xArr, neuronYs[j], trainMargin, cw, ch, trainXRange, yRange,
          neuronColor(j, trainState.width, 0.4), 1.3);
      }
    }

    // --- Hand-placed comparison overlay ---
    if (trainState.showComparison) {
      const compModel = manualFit(trainState.target, trainState.width);
      if (compModel) {
        const compXs = [], compYs = [];
        for (let i = 0; i <= 400; i++) {
          const x = i / 400;
          compXs.push(x);
          compYs.push(compModel.predict(x));
        }
        plotCurve(ctx, compXs, compYs, trainMargin, cw, ch, trainXRange, yRange,
          'rgba(30, 119, 112, 0.6)', 2.5, [8, 5]);
      }
    }

    // Target curve
    const targetXs = [], targetYs = [];
    for (let i = 0; i <= 400; i++) {
      const x = i / 400;
      targetXs.push(x);
      targetYs.push(t.fn(x));
    }
    plotCurve(ctx, targetXs, targetYs, trainMargin, cw, ch, trainXRange, yRange, '#d9622b', 3);

    // Training data dots
    if (trainState.dataX) {
      const plotW = cw - trainMargin.left - trainMargin.right;
      const plotH = ch - trainMargin.top - trainMargin.bottom;
      ctx.fillStyle = 'rgba(44, 111, 183, 0.25)';
      for (let i = 0; i < trainState.dataX.length; i++) {
        const x = trainState.dataX[i];
        let y = trainState.dataY[i];
        if (y > yRange[1]) y = yRange[1];
        if (y < yRange[0]) y = yRange[0];
        const px = trainMargin.left + ((x - trainXRange[0]) / (trainXRange[1] - trainXRange[0])) * plotW;
        const py = trainMargin.top + plotH - ((y - yRange[0]) / (yRange[1] - yRange[0])) * plotH;
        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Model prediction
    if (trainState.model) {
      const xs = [], ys = [];
      for (let i = 0; i <= 400; i++) {
        const x = i / 400;
        xs.push(x);
        ys.push(trainState.model.predict(x));
      }
      plotCurve(ctx, xs, ys, trainMargin, cw, ch, trainXRange, yRange, '#2c6fb7', 2.5);

      // --- Kink position markers ---
      if (trainState.showNeurons && trainState.activation === 'relu') {
        const kinks = trainState.model.kinkPositions();
        const plotW = cw - trainMargin.left - trainMargin.right;
        const plotH = ch - trainMargin.top - trainMargin.bottom;
        const axisY = trainMargin.top + plotH;

        for (const k of kinks) {
          if (k.x < trainXRange[0] || k.x > trainXRange[1]) continue;
          const px = trainMargin.left + ((k.x - trainXRange[0]) / (trainXRange[1] - trainXRange[0])) * plotW;
          ctx.fillStyle = neuronColor(k.idx, trainState.width, 0.7);
          ctx.beginPath();
          ctx.moveTo(px, axisY);
          ctx.lineTo(px - 4, axisY + 10);
          ctx.lineTo(px + 4, axisY + 10);
          ctx.closePath();
          ctx.fill();
        }
      }

      // Current MSE
      let mse = 0;
      for (let i = 0; i < trainState.dataX.length; i++) {
        const d = trainState.model.predict(trainState.dataX[i]) - trainState.dataY[i];
        mse += d * d;
      }
      mse /= trainState.dataX.length;
      mseStat.textContent = mse.toFixed(4);
    } else {
      mseStat.textContent = '\u2014';
    }

    // --- Legend overlay ---
    const legendX = cw - trainMargin.right - 200;
    const legendY = trainMargin.top + 8;
    ctx.font = '600 11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    // Target label
    ctx.strokeStyle = '#d9622b'; ctx.lineWidth = 2.5; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(legendX, legendY + 5); ctx.lineTo(legendX + 20, legendY + 5); ctx.stroke();
    ctx.fillStyle = '#d9622b'; ctx.fillText('Target', legendX + 25, legendY + 9);
    // Network label
    ctx.strokeStyle = '#2c6fb7'; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(legendX, legendY + 21); ctx.lineTo(legendX + 20, legendY + 21); ctx.stroke();
    ctx.fillStyle = '#2c6fb7'; ctx.fillText('Trained network', legendX + 25, legendY + 25);
    if (trainState.showComparison) {
      ctx.strokeStyle = 'rgba(30, 119, 112, 0.6)'; ctx.lineWidth = 2.5; ctx.setLineDash([8, 5]);
      ctx.beginPath(); ctx.moveTo(legendX, legendY + 37); ctx.lineTo(legendX + 20, legendY + 37); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(30, 119, 112, 0.8)'; ctx.fillText('Hand-placed', legendX + 25, legendY + 41);
    }

    drawLossCurve();
  }

  function drawLossCurve() {
    const { ctx, w: cw, h: ch } = setupCanvas(lossCanvas, 920, 180);
    ctx.clearRect(0, 0, cw, ch);
    const margin = { top: 20, right: 30, bottom: 30, left: 50 };
    const plotW = cw - margin.left - margin.right;
    const plotH = ch - margin.top - margin.bottom;

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

    ctx.fillStyle = '#9a917f';
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('log MSE', margin.left + 4, margin.top + 12);

    if (!trainState.losses.length) {
      ctx.textAlign = 'center';
      ctx.fillStyle = '#9a917f';
      ctx.font = '13px system-ui, sans-serif';
      ctx.fillText('Click Train to start', margin.left + plotW / 2, margin.top + plotH / 2);
      return;
    }

    // Full loss history (no truncation)
    const logs = trainState.losses.map((v) => Math.log10(Math.max(v, 1e-8)));
    let minL = logs[0], maxL = logs[0];
    for (let i = 1; i < logs.length; i++) {
      if (logs[i] < minL) minL = logs[i];
      if (logs[i] > maxL) maxL = logs[i];
    }
    const pad = 0.2;
    const loLog = minL - pad;
    const hiLog = maxL + pad;

    ctx.strokeStyle = '#2c6fb7';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < logs.length; i++) {
      const t = i / Math.max(1, logs.length - 1);
      const px = margin.left + t * plotW;
      const py = margin.top + plotH - ((logs[i] - loLog) / (hiLog - loLog)) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Epoch labels on x-axis
    ctx.fillStyle = '#9a917f';
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('0', margin.left, margin.top + plotH + 14);
    ctx.fillText(String(trainState.losses.length), margin.left + plotW, margin.top + plotH + 14);
    ctx.fillText('Epoch', margin.left + plotW / 2, ch - 4);

    ctx.textAlign = 'right';
    ctx.fillText('10^' + hiLog.toFixed(1), margin.left - 6, margin.top + 4);
    ctx.fillText('10^' + loLog.toFixed(1), margin.left - 6, margin.top + plotH + 4);
  }

  function shuffleIdx(n) {
    const arr = Array.from({ length: n }, (_, i) => i);
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  function oneEpoch() {
    if (!trainState.dataX || !trainState.dataX.length) return;
    const batchSize = 32;
    const n = trainState.dataX.length;
    const idx = shuffleIdx(n);
    let epochLoss = 0;
    let batchCount = 0;
    for (let start = 0; start < n; start += batchSize) {
      const end = Math.min(start + batchSize, n);
      const bx = new Array(end - start);
      const by = new Array(end - start);
      for (let k = start; k < end; k++) {
        bx[k - start] = trainState.dataX[idx[k]];
        by[k - start] = trainState.dataY[idx[k]];
      }
      const l = trainState.model.trainStep(bx, by, trainState.lr);
      epochLoss += l;
      batchCount++;
    }
    trainState.epoch += 1;
    trainState.losses.push(epochLoss / batchCount);
    // No truncation — keep full training history
  }

  let rafId = null;
  function trainLoop() {
    if (!trainState.running) return;
    for (let k = 0; k < 3; k++) {
      oneEpoch();
    }
    drawAll();
    if (trainState.running) rafId = requestAnimationFrame(trainLoop);
  }

  // --- Drawing handlers for train canvas ---
  function handleTrainDrawStart(e) {
    if (trainState.target !== 'custom' || trainDrawState.hasDrawn) return;
    e.preventDefault();
    trainDrawState.isDrawing = true;
    trainDrawState.points = [];
    const pos = e.touches ? getTouchPos(canvas, e.touches[0]) : e;
    const p = pixelToPlot(canvas, pos, trainMargin, trainLogicalW, trainLogicalH, trainXRange, [-1.5, 1.5]);
    trainDrawState.points.push(p);
    drawAll();
  }

  function handleTrainDrawMove(e) {
    if (!trainDrawState.isDrawing) return;
    e.preventDefault();
    const pos = e.touches ? getTouchPos(canvas, e.touches[0]) : e;
    const p = pixelToPlot(canvas, pos, trainMargin, trainLogicalW, trainLogicalH, trainXRange, [-1.5, 1.5]);
    trainDrawState.points.push(p);
    drawAll();
  }

  function handleTrainDrawEnd(e) {
    if (!trainDrawState.isDrawing) return;
    trainDrawState.isDrawing = false;
    if (trainDrawState.points.length > 3) {
      buildCustomTarget(trainDrawState.points, [-1.5, 1.5]);
      trainDrawState.hasDrawn = true;
      resetModel();
    }
    drawAll();
  }

  canvas.addEventListener('mousedown', handleTrainDrawStart);
  canvas.addEventListener('mousemove', handleTrainDrawMove);
  canvas.addEventListener('mouseup', handleTrainDrawEnd);
  canvas.addEventListener('mouseleave', handleTrainDrawEnd);
  canvas.addEventListener('touchstart', handleTrainDrawStart, { passive: false });
  canvas.addEventListener('touchmove', handleTrainDrawMove, { passive: false });
  canvas.addEventListener('touchend', handleTrainDrawEnd);

  if (clearDrawBtn) {
    clearDrawBtn.addEventListener('click', () => {
      trainState.running = false;
      btnTrain.textContent = 'Train';
      if (rafId) cancelAnimationFrame(rafId);
      trainDrawState.hasDrawn = false;
      trainDrawState.points = [];
      TARGETS.custom.fn = null;
      resetModel();
    });
  }

  // --- Slider bindings ---
  widthSlider.addEventListener('input', () => {
    trainState.width = parseInt(widthSlider.value, 10);
    resetModel();
  });

  if (lrSlider) {
    lrSlider.addEventListener('input', () => {
      trainState.lr = parseFloat(lrSlider.value);
      if (lrVal) lrVal.textContent = trainState.lr.toFixed(3);
    });
  }

  // --- Activation chips ---
  activationChips.forEach((c) => {
    c.addEventListener('click', () => {
      activationChips.forEach((cc) => cc.classList.remove('is-active'));
      c.classList.add('is-active');
      trainState.activation = c.dataset.act;
      resetModel();
    });
  });

  // --- Target buttons ---
  buttons.forEach((b) => {
    b.addEventListener('click', () => {
      buttons.forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      trainState.target = b.dataset.target;
      if (b.dataset.target !== 'custom') {
        trainDrawState.hasDrawn = false;
        trainDrawState.points = [];
      }
      trainState.running = false;
      btnTrain.textContent = 'Train';
      if (rafId) cancelAnimationFrame(rafId);
      resetModel();
    });
  });

  // --- Toggle buttons ---
  if (btnShowNeurons) {
    btnShowNeurons.addEventListener('click', () => {
      trainState.showNeurons = !trainState.showNeurons;
      drawAll();
    });
  }
  if (btnShowComparison) {
    btnShowComparison.addEventListener('click', () => {
      trainState.showComparison = !trainState.showComparison;
      drawAll();
    });
  }

  // --- Train / Step / Reset ---
  btnTrain.addEventListener('click', () => {
    if (trainState.target === 'custom' && !trainDrawState.hasDrawn) return;
    if (trainState.running) {
      trainState.running = false;
      btnTrain.textContent = 'Train';
      if (rafId) cancelAnimationFrame(rafId);
    } else {
      trainState.running = true;
      btnTrain.textContent = 'Stop';
      trainLoop();
    }
  });
  btnStep.addEventListener('click', () => {
    if (trainState.target === 'custom' && !trainDrawState.hasDrawn) return;
    if (!trainState.model) resetModel();
    oneEpoch();
    drawAll();
  });
  btnReset.addEventListener('click', () => {
    trainState.running = false;
    btnTrain.textContent = 'Train';
    if (rafId) cancelAnimationFrame(rafId);
    resetModel();
  });

  // --- Keyboard shortcut: space to toggle train ---
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && document.activeElement === document.body) {
      const step5 = document.getElementById('step-5');
      if (!step5) return;
      const rect = step5.getBoundingClientRect();
      if (rect.top < window.innerHeight && rect.bottom > 0) {
        e.preventDefault();
        btnTrain.click();
      }
    }
  });

  resetModel();
}

// ============================================================
// KaTeX
// ============================================================
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-single-neuron': 'y = \\phi(w\\,x + b)',
    'math-two-neurons':
      'y = c_1\\,\\phi(w_1 x + b_1) + c_2\\,\\phi(w_2 x + b_2)'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try {
      katex.render(blocks[id], el, { displayMode: true, throwOnError: false });
    } catch (_) { /* no-op */ }
  });

  // Auto-render inline $...$ and $$...$$ math across the whole article
  if (window.renderMathInElement) {
    try {
      renderMathInElement(document.querySelector('.article'), {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true }
        ],
        throwOnError: false,
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option']
      });
    } catch (_) { /* no-op */ }
  }
}

// ============================================================
// Boot
// ============================================================
function waitForKatexAndRender() {
  let tries = 0;
  function tick() {
    if (window.katex && window.renderMathInElement) {
      renderMath();
      return;
    }
    if (window.katex && tries > 20) {
      // auto-render didn't show up; render at least the block math
      renderMath();
      return;
    }
    tries++;
    setTimeout(tick, 50);
  }
  tick();
}

function init() {
  waitForKatexAndRender();
  initSingleNeuron();
  initBump();
  initManual();
  initTrainer();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
