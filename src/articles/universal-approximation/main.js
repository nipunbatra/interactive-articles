// ============================================================
// Watching a Neural Network Become Universal
// All computations are real: live neuron evaluation and a
// working MLP trained by mini-batch SGD in the browser.
// ============================================================

// ---------- Target functions ----------
const TARGETS = {
  sine: {
    fn: (x) => Math.sin(2 * Math.PI * x),
    label: 'sin(2πx)',
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
  }
};

function sampleTarget(targetKey, n = 200) {
  const t = TARGETS[targetKey];
  const xs = new Array(n);
  const ys = new Array(n);
  for (let i = 0; i < n; i++) {
    const x = i / (n - 1);
    xs[i] = x;
    ys[i] = t.fn(x);
  }
  return { xs, ys };
}

// ---------- Canvas helper ----------
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
      formula.textContent = `y = max(0, ${w.toFixed(1)}·x + ${b.toFixed(1)})`;
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
      const n1 = ACT.relu.f(2 * x + b1); // fixed weight 2
      const n2 = ACT.relu.f(2 * x + b2);
      h1s.push(c1 * n1);
      h2s.push(c2 * n2);
      ysum.push(c1 * n1 + c2 * n2);
    }

    // Individual neurons dashed
    plotCurve(ctx, xs, h1s, margin, cw, ch, xRange, yRange, 'rgba(44, 111, 183, 0.45)', 2, [6, 4]);
    plotCurve(ctx, xs, h2s, margin, cw, ch, xRange, yRange, 'rgba(30, 119, 112, 0.45)', 2, [6, 4]);
    // Sum solid
    plotCurve(ctx, xs, ysum, margin, cw, ch, xRange, yRange, '#d9622b', 3);

    // Legend
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(44, 111, 183, 0.8)';
    ctx.fillText('c₁·ReLU(2x + b₁)', margin.left + 8, margin.top + 14);
    ctx.fillStyle = 'rgba(30, 119, 112, 0.8)';
    ctx.fillText('c₂·ReLU(2x + b₂)', margin.left + 8, margin.top + 30);
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

function manualFit(targetKey, width) {
  // Hand-placed network: place each neuron's kink at an equally spaced
  // point in [0, 1] and use the difference of successive neighbors as the
  // output weight so that the sum interpolates the target values at the
  // kink points.
  const t = TARGETS[targetKey];
  const kinks = new Array(width);
  const targetVals = new Array(width);
  for (let i = 0; i < width; i++) {
    const k = (i + 0.5) / width;
    kinks[i] = k;
    targetVals[i] = t.fn(k);
  }

  // Input weights: fixed sharpness; sign so ReLU activates above kink.
  const inputW = 30; // sharpness
  // Each neuron: phi(inputW * (x - kink_i))
  // Output c_i chosen via finite differences: c_0 = t(k_0), c_i = t(k_i) - t(k_{i-1})
  // This gives a staircase-style approximation.
  const c = new Array(width);
  c[0] = targetVals[0];
  for (let i = 1; i < width; i++) {
    c[i] = targetVals[i] - targetVals[i - 1];
  }
  const b0 = 0; // output bias

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

  function paramCount(N) {
    // input weights (N) + input biases (N) + output weights (N) + output bias (1)
    return 3 * N + 1;
  }

  function draw() {
    const { target, width } = manualState;
    widthVal.textContent = width;
    neuronsStat.textContent = width;
    paramsStat.textContent = paramCount(width);

    const model = manualFit(target, width);
    const mse = manualMSE(model, target);
    mseStat.textContent = mse.toFixed(4);

    const t = TARGETS[target];
    const { ctx, w: cw, h: ch } = setupCanvas(canvas, 920, 360);
    ctx.clearRect(0, 0, cw, ch);

    const margin = { top: 20, right: 30, bottom: 35, left: 50 };
    const xRange = [0, 1];
    const yRange = t.yRange;
    drawAxes(ctx, margin, cw, ch, xRange, yRange);

    // Individual neurons (faint)
    if (width <= 30) {
      for (let i = 0; i < width; i++) {
        const xs = [], ys = [];
        for (let j = 0; j <= 200; j++) {
          const x = j / 200;
          xs.push(x);
          ys.push(model.c[i] * ACT.relu.f(model.inputW * (x - model.kinks[i])));
        }
        plotCurve(ctx, xs, ys, margin, cw, ch, xRange, yRange, 'rgba(140, 130, 110, 0.4)', 1.2);
      }
    }

    // Target
    const targetXs = [], targetYs = [];
    for (let i = 0; i <= 400; i++) {
      const x = i / 400;
      targetXs.push(x);
      targetYs.push(t.fn(x));
    }
    plotCurve(ctx, targetXs, targetYs, margin, cw, ch, xRange, yRange, '#d9622b', 3);

    // Network output
    const netXs = [], netYs = [];
    for (let i = 0; i <= 400; i++) {
      const x = i / 400;
      netXs.push(x);
      netYs.push(model.predict(x));
    }
    plotCurve(ctx, netXs, netYs, margin, cw, ch, xRange, yRange, '#2c6fb7', 2.5);

    // Update table live
    renderManualTable(target);
  }

  function renderManualTable(target) {
    const rows = [2, 4, 8, 16, 32, 60];
    const descriptions = {
      2: 'A single bump + linear base — can barely match a sine',
      4: 'Two bumps — roughly the positive and negative halves',
      8: 'Coarse staircase — visible kinks at every sample',
      16: 'Smooth-looking fit for most targets',
      32: 'Near-perfect on sine; sharp transitions on step/spike',
      60: 'Indistinguishable from the target on smooth functions'
    };
    let html = '';
    for (const N of rows) {
      const model = manualFit(target, N);
      const mse = manualMSE(model, target);
      html += `
        <tr>
          <td><strong>N = ${N}</strong></td>
          <td>${descriptions[N]}</td>
          <td><code>${mse.toExponential(2)}</code></td>
        </tr>
      `;
    }
    tableBody.innerHTML = html;
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
  // Initialize with a scheme appropriate for the activation
  const W1 = new Array(width);
  const b1 = new Array(width);
  for (let i = 0; i < width; i++) {
    // Spread initial biases across the input range; scale weights so that
    // activations don't all saturate from epoch 0.
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
      // dL/dyhat = 2*err
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

  return {
    get width() { return width; },
    get activation() { return activation; },
    forward,
    predict,
    trainStep
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
  lr: 0.05
};

function buildData(targetKey) {
  const n = 128;
  const xs = new Array(n);
  const ys = new Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = i / (n - 1);
    ys[i] = TARGETS[targetKey].fn(xs[i]);
  }
  return { xs, ys };
}

function initTrainer() {
  const canvas = document.getElementById('trainCanvas');
  const lossCanvas = document.getElementById('lossCanvas');
  const widthSlider = document.getElementById('train-width');
  const widthVal = document.getElementById('val-train-width');
  const activationChips = document.querySelectorAll('#activation-chips [data-act]');
  const activationLabel = document.getElementById('val-train-activation');
  const buttons = document.querySelectorAll('#train-scenario-buttons [data-target]');
  const btnTrain = document.getElementById('btn-train');
  const btnStep = document.getElementById('btn-step');
  const btnReset = document.getElementById('btn-reset');
  const epochStat = document.getElementById('stat-epoch');
  const paramsStat = document.getElementById('stat-train-params');
  const mseStat = document.getElementById('stat-train-mse');
  const revealN = document.getElementById('reveal-n');
  const revealParams = document.getElementById('reveal-params');
  const revealBreakdown = document.getElementById('reveal-breakdown');

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

  function drawAll() {
    widthVal.textContent = trainState.width;
    activationLabel.textContent = ACT[trainState.activation].label;
    epochStat.textContent = trainState.epoch;
    paramsStat.textContent = paramCount(trainState.width);

    revealN.textContent = trainState.width;
    const N = trainState.width;
    revealParams.textContent = paramCount(N);
    revealBreakdown.textContent =
      `${N} input weights + ${N} hidden biases + ${N} output weights + 1 output bias`;

    // Fit canvas
    const t = TARGETS[trainState.target];
    const { ctx, w: cw, h: ch } = setupCanvas(canvas, 920, 340);
    ctx.clearRect(0, 0, cw, ch);
    const margin = { top: 20, right: 30, bottom: 35, left: 50 };
    const xRange = [0, 1];
    const yRange = t.yRange;
    drawAxes(ctx, margin, cw, ch, xRange, yRange);

    // Target curve
    const targetXs = [], targetYs = [];
    for (let i = 0; i <= 400; i++) {
      const x = i / 400;
      targetXs.push(x);
      targetYs.push(t.fn(x));
    }
    plotCurve(ctx, targetXs, targetYs, margin, cw, ch, xRange, yRange, '#d9622b', 3);

    // Training data dots
    if (trainState.dataX) {
      const plotW = cw - margin.left - margin.right;
      const plotH = ch - margin.top - margin.bottom;
      ctx.fillStyle = 'rgba(44, 111, 183, 0.25)';
      for (let i = 0; i < trainState.dataX.length; i++) {
        const x = trainState.dataX[i];
        let y = trainState.dataY[i];
        if (y > yRange[1]) y = yRange[1];
        if (y < yRange[0]) y = yRange[0];
        const px = margin.left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * plotW;
        const py = margin.top + plotH - ((y - yRange[0]) / (yRange[1] - yRange[0])) * plotH;
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
      plotCurve(ctx, xs, ys, margin, cw, ch, xRange, yRange, '#2c6fb7', 2.5);

      // Current MSE
      let mse = 0;
      for (let i = 0; i < trainState.dataX.length; i++) {
        const d = trainState.model.predict(trainState.dataX[i]) - trainState.dataY[i];
        mse += d * d;
      }
      mse /= trainState.dataX.length;
      mseStat.textContent = mse.toFixed(4);
    } else {
      mseStat.textContent = '—';
    }

    drawLossCurve();
  }

  function drawLossCurve() {
    const { ctx, w: cw, h: ch } = setupCanvas(lossCanvas, 920, 180);
    ctx.clearRect(0, 0, cw, ch);
    const margin = { top: 20, right: 30, bottom: 30, left: 50 };
    const plotW = cw - margin.left - margin.right;
    const plotH = ch - margin.top - margin.bottom;

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

    // Log scale
    const logs = trainState.losses.map((v) => Math.log10(Math.max(v, 1e-8)));
    const minL = Math.min(...logs);
    const maxL = Math.max(...logs);
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

    ctx.fillStyle = '#9a917f';
    ctx.font = '11px system-ui, sans-serif';
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
    if (trainState.losses.length > 500) {
      trainState.losses.shift();
    }
  }

  let rafId = null;
  function trainLoop() {
    if (!trainState.running) return;
    // Run a few epochs per frame for speed
    for (let k = 0; k < 3; k++) {
      oneEpoch();
      if (trainState.epoch >= 400) {
        trainState.running = false;
        btnTrain.textContent = 'Train';
        break;
      }
    }
    drawAll();
    if (trainState.running) rafId = requestAnimationFrame(trainLoop);
  }

  widthSlider.addEventListener('input', () => {
    trainState.width = parseInt(widthSlider.value, 10);
    resetModel();
  });
  activationChips.forEach((c) => {
    c.addEventListener('click', () => {
      activationChips.forEach((cc) => cc.classList.remove('is-active'));
      c.classList.add('is-active');
      trainState.activation = c.dataset.act;
      resetModel();
    });
  });
  buttons.forEach((b) => {
    b.addEventListener('click', () => {
      buttons.forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      trainState.target = b.dataset.target;
      resetModel();
    });
  });

  btnTrain.addEventListener('click', () => {
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
    if (!trainState.model) resetModel();
    oneEpoch();
    drawAll();
  });
  btnReset.addEventListener('click', () => {
    trainState.running = false;
    btnTrain.textContent = 'Train';
    resetModel();
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
}

// ============================================================
// Boot
// ============================================================
function init() {
  if (window.katex) {
    renderMath();
  } else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
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
