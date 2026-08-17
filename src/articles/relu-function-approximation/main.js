(() => {
  'use strict';

  const COLORS = {
    ink: '#232b32', inkSoft: '#3f4b54', muted: '#66737d', grid: '#d7dde0',
    blue: '#2c67a0', blueSoft: '#e4eef7', rust: '#bd5f35', rustSoft: '#f5e8e0',
    teal: '#28756e', red: '#a74943', paper: '#ffffff', canvas: '#f7f8f8'
  };
  const GRID_COUNT = 401;
  const TRAIN_COUNT = 121;
  const CUSTOM_COUNT = 201;
  const Y_RANGE = [-1.35, 1.35];

  const $ = (id) => document.getElementById(id);
  const els = {
    target: $('targetSelect'), drawHelp: $('drawHelp'), resetTarget: $('resetTargetBtn'),
    complexity: $('complexityRange'), complexityLabel: $('complexityLabel'),
    complexityValue: $('complexityValue'), complexityMeta: $('complexityMeta'), knotInputList: $('knotInputList'),
    epsilon: $('epsilonRange'), epsilonValue: $('epsilonValue'),
    learningRate: $('learningRateSelect'), seed: $('seedInput'), newSeed: $('newSeedBtn'),
    run: $('runBtn'), step: $('stepBtn'), batch: $('batchBtn'), reset: $('resetBtn'),
    actionIndex: $('actionIndex'), status: $('statusText'), provenance: $('provenanceBadge'),
    workspaceEyebrow: $('workspaceEyebrow'), workspaceTitle: $('workspaceTitle'),
    tolerance: $('toleranceBadge'), trainMse: $('trainMseValue'), trainMseNote: $('trainMseNote'),
    denseMse: $('denseMseValue'), maxGap: $('maxGapValue'), steps: $('stepsValue'),
    parameterNote: $('parameterNote'), main: $('mainCanvas'), residual: $('residualCanvas'),
    modelFormula: $('modelFormula'), maxGapAt: $('maxGapAt'), modelSummary: $('modelSummaryTitle'),
    sourceFact: $('sourceFact'), biasFact: $('biasFact'), objectiveFact: $('objectiveFact'),
    boundaryNote: $('boundaryNote'), sharedScale: $('sharedScale'), contributionGrid: $('contributionGrid'),
    plotInstruction: $('plotInstruction')
  };

  const TARGETS = {
    sine: (x) => Math.sin(Math.PI * x),
    tent: (x) => Math.max(0, 1 - Math.abs(1.7 * x)),
    twoBumps: (x) => 0.9 * Math.exp(-18 * (x + 0.42) ** 2) - 0.68 * Math.exp(-26 * (x - 0.38) ** 2),
    smoothStep: (x) => 0.78 * Math.tanh(3.2 * x)
  };
  const TARGET_NAMES = {
    sine: 'Sine', tent: 'Tent', twoBumps: 'Two bumps', smoothStep: 'Smooth step', custom: 'Drawn target'
  };

  const state = {
    lane: 'construct',
    target: 'sine',
    constructHinges: 5,
    constructKnots: Array.from({ length: 5 }, (_, index) => -1 + 2 * (index + 1) / 6),
    trainWidth: 6,
    epsilon: 0.05,
    learningRate: 0.01,
    seed: 11,
    model: null,
    steps: 0,
    running: false,
    provenance: 'constructed',
    statusMessage: 'Fixed-knot interpolant loaded. No optimizer was used.',
    customTarget: Array(CUSTOM_COUNT).fill(0),
    drawActive: false,
    lastDrawIndex: null,
    knotDragIndex: null,
    mainFrame: null,
    lastFrame: 0
  };

  const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
  const relu = (value) => Math.max(0, value);
  const lerp = (a, b, t) => a + (b - a) * t;
  const finite = (value, fallback = 0) => Number.isFinite(value) ? value : fallback;

  function resetUniformKnots() {
    const segments = state.constructHinges + 1;
    state.constructKnots = Array.from(
      { length: state.constructHinges },
      (_, index) => -1 + 2 * (index + 1) / segments
    );
  }

  function knotsAreUniform() {
    const segments = state.constructHinges + 1;
    return state.constructKnots.every((value, index) => (
      Math.abs(value - (-1 + 2 * (index + 1) / segments)) < 1e-8
    ));
  }

  function assignKnotPosition(index, value) {
    const left = index === 0 ? -1 : state.constructKnots[index - 1];
    const right = index === state.constructKnots.length - 1 ? 1 : state.constructKnots[index + 1];
    state.constructKnots[index] = clamp(Number(value), left + 0.025, right - 0.025);
    state.model = makeConstructedModel(state.constructHinges);
  }

  function mulberry32(seed) {
    let value = seed >>> 0;
    return () => {
      value |= 0;
      value = value + 0x6D2B79F5 | 0;
      let t = Math.imul(value ^ value >>> 15, 1 | value);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function gaussian(rng) {
    const u = Math.max(1e-9, rng());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }

  function targetAt(x) {
    if (state.target !== 'custom') return TARGETS[state.target](x);
    const position = clamp((x + 1) * (CUSTOM_COUNT - 1) / 2, 0, CUSTOM_COUNT - 1);
    const lo = Math.floor(position);
    const hi = Math.ceil(position);
    return lerp(state.customTarget[lo], state.customTarget[hi], position - lo);
  }

  function makeConstructedModel(interiorHinges) {
    const segments = interiorHinges + 1;
    const interior = state.constructKnots.length === interiorHinges
      ? state.constructKnots.slice()
      : Array.from({ length: interiorHinges }, (_, index) => -1 + 2 * (index + 1) / segments);
    const knots = [-1, ...interior, 1];
    const values = knots.map(targetAt);
    const slopes = Array.from({ length: segments }, (_, index) => (
      (values[index + 1] - values[index]) / (knots[index + 1] - knots[index])
    ));
    const W = Array(segments).fill(1);
    const b = knots.slice(0, -1).map((knot) => -knot);
    const v = slopes.map((slope, index) => index === 0 ? slope : slope - slopes[index - 1]);
    return { kind: 'constructed', width: segments, W, b, v, c: values[0], knots, values };
  }

  function makeTrainModel(width, seed) {
    const rng = mulberry32(seed * 3571 + width * 97);
    const W = [];
    const b = [];
    const v = [];
    for (let index = 0; index < width; index++) {
      const w = 0.72 + 0.56 * rng();
      const base = -0.88 + 1.76 * (index + 0.5) / width;
      const hinge = clamp(base + 0.1 * gaussian(rng), -0.96, 0.96);
      W.push(w);
      b.push(-w * hinge);
      v.push(0.28 * gaussian(rng) / Math.sqrt(Math.max(1, width / 4)));
    }
    return { kind: 'train', width, W, b, v, c: 0, knots: [] };
  }

  function forward(model, x) {
    let y = model.c;
    for (let index = 0; index < model.width; index++) {
      y += model.v[index] * relu(model.W[index] * x + model.b[index]);
    }
    return y;
  }

  function contribution(model, index, x) {
    return model.v[index] * relu(model.W[index] * x + model.b[index]);
  }

  function trainSampleMse() {
    let sum = 0;
    for (let index = 0; index < TRAIN_COUNT; index++) {
      const x = -1 + 2 * index / (TRAIN_COUNT - 1);
      const error = forward(state.model, x) - targetAt(x);
      sum += error * error;
    }
    return sum / TRAIN_COUNT;
  }

  function denseMetrics() {
    let squared = 0;
    let maxGap = -1;
    let maxX = -1;
    const gaps = [];
    for (let index = 0; index < GRID_COUNT; index++) {
      const x = -1 + 2 * index / (GRID_COUNT - 1);
      const gap = Math.abs(targetAt(x) - forward(state.model, x));
      gaps.push({ x, gap });
      squared += gap * gap;
      if (gap > maxGap) {
        maxGap = gap;
        maxX = x;
      }
    }
    return { mse: squared / GRID_COUNT, maxGap, maxX, gaps };
  }

  function parametersFinite() {
    const values = [state.model.c, ...state.model.W, ...state.model.b, ...state.model.v];
    return values.every(Number.isFinite);
  }

  function trainOneStep() {
    if (state.lane !== 'train') return false;
    const model = state.model;
    const gW = Array(model.width).fill(0);
    const gb = Array(model.width).fill(0);
    const gv = Array(model.width).fill(0);
    let gc = 0;

    for (let sample = 0; sample < TRAIN_COUNT; sample++) {
      const x = -1 + 2 * sample / (TRAIN_COUNT - 1);
      const target = targetAt(x);
      const activations = [];
      const preactivations = [];
      let prediction = model.c;
      for (let index = 0; index < model.width; index++) {
        const u = model.W[index] * x + model.b[index];
        const h = relu(u);
        preactivations.push(u);
        activations.push(h);
        prediction += model.v[index] * h;
      }
      const dy = 2 * (prediction - target);
      gc += dy;
      for (let index = 0; index < model.width; index++) {
        gv[index] += dy * activations[index];
        const du = preactivations[index] > 0 ? dy * model.v[index] : 0;
        gW[index] += du * x;
        gb[index] += du;
      }
    }

    const rate = state.learningRate / TRAIN_COUNT;
    model.c -= rate * gc;
    for (let index = 0; index < model.width; index++) {
      model.W[index] -= rate * gW[index];
      model.b[index] -= rate * gb[index];
      model.v[index] -= rate * gv[index];
    }
    state.steps += 1;

    if (!parametersFinite()) {
      stopRunning();
      state.statusMessage = 'Training became numerically unstable. Reset and choose a smaller learning rate.';
      return false;
    }
    return true;
  }

  function trainSteps(count) {
    for (let index = 0; index < count; index++) {
      if (!trainOneStep()) break;
    }
    if (parametersFinite()) {
      state.provenance = state.steps > 0 ? 'trained' : 'random';
      state.statusMessage = state.running
        ? `Training from seed ${state.seed}… ${state.steps} full-batch steps completed.`
        : `${count === 1 ? 'One' : count} gradient ${count === 1 ? 'step' : 'steps'} completed.`;
    }
  }

  function parameterCount() {
    return 3 * state.model.width + 1;
  }

  function setupCanvas(canvas, minHeight) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(280, Math.round(rect.width || canvas.parentElement.clientWidth));
    const height = Math.max(minHeight, Math.round(rect.height || minHeight));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function tickText(value) {
    if (Math.abs(value) < 1e-9) return '0';
    return Math.abs(value) >= 1 ? value.toFixed(1) : value.toFixed(2);
  }

  function plotFrame(ctx, width, height, xRange, yRange, labels = { x: 'x', y: 'value' }) {
    const margin = { left: 58, right: 20, top: 19, bottom: 43 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const px = (x) => margin.left + (x - xRange[0]) / (xRange[1] - xRange[0]) * plotWidth;
    const py = (y) => margin.top + (yRange[1] - y) / (yRange[1] - yRange[0]) * plotHeight;

    ctx.fillStyle = COLORS.paper;
    ctx.fillRect(margin.left, margin.top, plotWidth, plotHeight);
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let index = 0; index <= 4; index++) {
      const x = margin.left + index * plotWidth / 4;
      const y = margin.top + index * plotHeight / 4;
      ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotHeight); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotWidth, y); ctx.stroke();
    }
    ctx.strokeStyle = COLORS.ink;
    ctx.lineWidth = 1.35;
    ctx.strokeRect(margin.left, margin.top, plotWidth, plotHeight);

    ctx.fillStyle = COLORS.muted;
    ctx.font = '12px "Avenir Next", "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    for (let index = 0; index <= 4; index++) {
      const value = xRange[0] + (xRange[1] - xRange[0]) * index / 4;
      ctx.fillText(tickText(value), margin.left + index * plotWidth / 4, height - 19);
    }
    ctx.textAlign = 'right';
    for (let index = 0; index <= 4; index++) {
      const value = yRange[1] - (yRange[1] - yRange[0]) * index / 4;
      ctx.fillText(tickText(value), margin.left - 8, margin.top + index * plotHeight / 4 + 4);
    }
    ctx.fillStyle = COLORS.ink;
    ctx.font = '700 13px "Avenir Next", "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(labels.x, margin.left + plotWidth / 2, height - 3);
    ctx.save();
    ctx.translate(14, margin.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(labels.y, 0, 0);
    ctx.restore();
    return { margin, plotWidth, plotHeight, width, height, xRange, yRange, px, py };
  }

  function curvePath(ctx, frame, fn, color, lineWidth, samples = 400) {
    ctx.beginPath();
    for (let index = 0; index <= samples; index++) {
      const x = frame.xRange[0] + (frame.xRange[1] - frame.xRange[0]) * index / samples;
      const y = finite(fn(x));
      if (index === 0) ctx.moveTo(frame.px(x), frame.py(y));
      else ctx.lineTo(frame.px(x), frame.py(y));
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();
  }

  function drawMain(metrics) {
    const { ctx, width, height } = setupCanvas(els.main, 430);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.canvas;
    ctx.fillRect(0, 0, width, height);
    const frame = plotFrame(ctx, width, height, [-1, 1], Y_RANGE, { x: 'x', y: 'function value' });
    state.mainFrame = frame;
    const { margin, plotWidth, plotHeight, px, py } = frame;

    ctx.save();
    ctx.beginPath();
    ctx.rect(margin.left, margin.top, plotWidth, plotHeight);
    ctx.clip();

    ctx.beginPath();
    for (let index = 0; index <= 400; index++) {
      const x = -1 + 2 * index / 400;
      const y = targetAt(x) + state.epsilon;
      if (index === 0) ctx.moveTo(px(x), py(y)); else ctx.lineTo(px(x), py(y));
    }
    for (let index = 400; index >= 0; index--) {
      const x = -1 + 2 * index / 400;
      ctx.lineTo(px(x), py(targetAt(x) - state.epsilon));
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(44, 103, 160, 0.13)';
    ctx.fill();

    if (state.lane === 'train') {
      for (let index = 0; index < TRAIN_COUNT; index++) {
        const x = -1 + 2 * index / (TRAIN_COUNT - 1);
        ctx.beginPath();
        ctx.arc(px(x), py(targetAt(x)), 2.2, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.paper;
        ctx.fill();
        ctx.strokeStyle = 'rgba(44, 103, 160, 0.75)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    curvePath(ctx, frame, targetAt, COLORS.ink, 2.5);
    curvePath(ctx, frame, (x) => forward(state.model, x), COLORS.blue, 3.2);

    if (state.lane === 'construct') {
      for (let knotIndex = 0; knotIndex < state.model.knots.length; knotIndex++) {
        const knot = state.model.knots[knotIndex];
        const interior = knotIndex > 0 && knotIndex < state.model.knots.length - 1;
        if (interior) {
          ctx.strokeStyle = 'rgba(189, 95, 53, 0.34)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 4]);
          ctx.beginPath();
          ctx.moveTo(px(knot), margin.top);
          ctx.lineTo(px(knot), margin.top + plotHeight);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        const point = [px(knot), py(forward(state.model, knot))];
        ctx.save();
        ctx.translate(point[0], point[1]);
        ctx.rotate(Math.PI / 4);
        ctx.fillStyle = interior ? COLORS.rust : COLORS.paper;
        ctx.strokeStyle = COLORS.rust;
        ctx.lineWidth = 2;
        const radius = interior ? 5 : 4;
        ctx.fillRect(-radius, -radius, radius * 2, radius * 2);
        ctx.strokeRect(-radius, -radius, radius * 2, radius * 2);
        ctx.restore();
      }
    }

    const targetY = targetAt(metrics.maxX);
    const modelY = forward(state.model, metrics.maxX);
    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(px(metrics.maxX), py(targetY));
    ctx.lineTo(px(metrics.maxX), py(modelY));
    ctx.stroke();
    ctx.setLineDash([]);
    for (const y of [targetY, modelY]) {
      ctx.beginPath();
      ctx.arc(px(metrics.maxX), py(y), 3.3, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.red;
      ctx.fill();
    }

    if (state.target === 'custom') {
      ctx.fillStyle = 'rgba(35, 43, 50, 0.72)';
      ctx.font = '800 12px "SFMono-Regular", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('DRAG TO DRAW TARGET', margin.left + 12, margin.top + 22);
    }
    ctx.restore();

    const laneText = state.lane === 'construct' ? 'fixed-knot construction with draggable interior knots' : `${state.steps}-step trained model`;
    els.main.setAttribute('aria-label', `${TARGET_NAMES[state.target]} target, ${laneText}, tolerance ${state.epsilon.toFixed(2)}, and maximum 401-point grid gap ${metrics.maxGap.toFixed(3)}.`);
  }

  function drawResidual(metrics) {
    const { ctx, width, height } = setupCanvas(els.residual, 300);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.canvas;
    ctx.fillRect(0, 0, width, height);
    const yMax = Math.max(0.08, state.epsilon * 1.22, metrics.maxGap * 1.12);
    const frame = plotFrame(ctx, width, height, [-1, 1], [0, yMax], { x: 'x', y: 'absolute gap' });
    const { margin, plotWidth, plotHeight, px, py } = frame;
    ctx.save();
    ctx.beginPath();
    ctx.rect(margin.left, margin.top, plotWidth, plotHeight);
    ctx.clip();

    ctx.beginPath();
    ctx.moveTo(px(-1), py(0));
    for (const point of metrics.gaps) ctx.lineTo(px(point.x), py(point.gap));
    ctx.lineTo(px(1), py(0));
    ctx.closePath();
    ctx.fillStyle = 'rgba(189, 95, 53, 0.18)';
    ctx.fill();

    curvePath(ctx, frame, (x) => Math.abs(targetAt(x) - forward(state.model, x)), COLORS.rust, 2.5);
    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 1.6;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(px(-1), py(state.epsilon));
    ctx.lineTo(px(1), py(state.epsilon));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.red;
    ctx.font = '700 12px "Avenir Next", "Segoe UI", sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`ε = ${state.epsilon.toFixed(2)}`, px(0.97), py(state.epsilon) - 7);

    ctx.beginPath();
    ctx.arc(px(metrics.maxX), py(metrics.maxGap), 4, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.rust;
    ctx.fill();
    ctx.restore();
    els.residual.setAttribute('aria-label', `Pointwise absolute error on a 401-point grid. Maximum sampled gap ${metrics.maxGap.toFixed(3)} at x ${metrics.maxX.toFixed(2)}; tolerance ${state.epsilon.toFixed(2)}.`);
  }

  function niceSharedScale(raw) {
    const value = Math.max(0.1, raw);
    const power = 10 ** Math.floor(Math.log10(value));
    const unit = value / power;
    const rounded = unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10;
    return rounded * power;
  }

  function contributionScale() {
    let maxAbs = 0;
    for (let unit = 0; unit < state.model.width; unit++) {
      for (let sample = 0; sample <= 200; sample++) {
        const x = -1 + 2 * sample / 200;
        maxAbs = Math.max(maxAbs, Math.abs(contribution(state.model, unit, x)));
      }
    }
    return niceSharedScale(maxAbs);
  }

  function drawContributionCanvas(canvas, unit, scale) {
    const { ctx, width, height } = setupCanvas(canvas, 145);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.paper;
    ctx.fillRect(0, 0, width, height);
    const margin = { left: 32, right: 9, top: 10, bottom: 25 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const px = (x) => margin.left + (x + 1) / 2 * plotWidth;
    const py = (y) => margin.top + (scale - y) / (2 * scale) * plotHeight;

    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(margin.left, py(0)); ctx.lineTo(margin.left + plotWidth, py(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px(0), margin.top); ctx.lineTo(px(0), margin.top + plotHeight); ctx.stroke();

    const w = state.model.W[unit];
    const b = state.model.b[unit];
    const hinge = Math.abs(w) > 1e-10 ? -b / w : NaN;
    if (hinge >= -1 && hinge <= 1) {
      ctx.strokeStyle = 'rgba(189, 95, 53, 0.55)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(px(hinge), margin.top); ctx.lineTo(px(hinge), margin.top + plotHeight); ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(px(-1), py(0));
    for (let sample = 0; sample <= 160; sample++) {
      const x = -1 + 2 * sample / 160;
      ctx.lineTo(px(x), py(contribution(state.model, unit, x)));
    }
    ctx.lineTo(px(1), py(0));
    ctx.closePath();
    const positive = state.model.v[unit] >= 0;
    ctx.fillStyle = positive ? 'rgba(44, 103, 160, 0.13)' : 'rgba(189, 95, 53, 0.14)';
    ctx.fill();

    ctx.beginPath();
    for (let sample = 0; sample <= 160; sample++) {
      const x = -1 + 2 * sample / 160;
      const y = contribution(state.model, unit, x);
      if (sample === 0) ctx.moveTo(px(x), py(y)); else ctx.lineTo(px(x), py(y));
    }
    ctx.strokeStyle = positive ? COLORS.blue : COLORS.rust;
    ctx.lineWidth = 2.4;
    ctx.lineJoin = 'round';
    ctx.stroke();

    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left, margin.top, plotWidth, plotHeight);
    ctx.fillStyle = COLORS.muted;
    ctx.font = '11px "Avenir Next", "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('−1', px(-1), height - 8);
    ctx.fillText('0', px(0), height - 8);
    ctx.fillText('1', px(1), height - 8);
    ctx.textAlign = 'right';
    ctx.fillText(scale.toPrecision(2), margin.left - 4, margin.top + 4);
    ctx.fillText(`−${scale.toPrecision(2)}`, margin.left - 4, margin.top + plotHeight + 4);
  }

  function signedTerm(value, variable = '') {
    const sign = value >= 0 ? '+' : '−';
    return `${sign} ${Math.abs(value).toFixed(2)}${variable}`;
  }

  function renderContributions() {
    const scale = contributionScale();
    els.sharedScale.textContent = `Shared vertical scale ±${scale.toPrecision(2)}`;
    els.contributionGrid.replaceChildren();
    for (let unit = 0; unit < state.model.width; unit++) {
      const card = document.createElement('section');
      card.className = 'contribution-card';
      const header = document.createElement('header');
      const title = document.createElement('strong');
      title.textContent = `Contribution ${unit + 1}`;
      const meta = document.createElement('span');
      const w = state.model.W[unit];
      const b = state.model.b[unit];
      const v = state.model.v[unit];
      const hinge = Math.abs(w) > 1e-10 ? -b / w : NaN;
      if (state.lane === 'construct' && unit === 0) meta.textContent = 'left-edge slope';
      else if (hinge >= -1 && hinge <= 1) meta.textContent = `hinge x=${hinge.toFixed(2)}`;
      else meta.textContent = 'hinge off domain';
      header.append(title, meta);

      const canvas = document.createElement('canvas');
      canvas.setAttribute('role', 'img');
      canvas.setAttribute('aria-label', `Signed output contribution ${unit + 1}, with output weight ${v.toFixed(2)} and hinge ${Number.isFinite(hinge) ? hinge.toFixed(2) : 'undefined'}.`);
      const equation = document.createElement('p');
      equation.textContent = `c${unit + 1}(x) = ${v.toFixed(2)} · ReLU(${w.toFixed(2)}x ${signedTerm(b)})`;
      const note = document.createElement('small');
      note.textContent = `v=${v.toFixed(2)} · w=${w.toFixed(2)} · b=${b.toFixed(2)}`;
      card.append(header, canvas, equation, note);
      els.contributionGrid.append(card);
      drawContributionCanvas(canvas, unit, scale);
    }
  }

  function provenanceText() {
    if (state.provenance === 'constructed') return 'Constructed · fixed knots';
    if (state.provenance === 'trained') return `Trained · ${state.steps} steps · seed ${state.seed}`;
    return `Random initialization · seed ${state.seed}`;
  }

  function renderKnotInputs() {
    els.knotInputList.replaceChildren();
    for (let index = 0; index < state.constructKnots.length; index++) {
      const label = document.createElement('label');
      label.textContent = `t${index + 1}`;
      const input = document.createElement('input');
      input.type = 'number';
      input.step = '0.01';
      input.min = String((index === 0 ? -1 : state.constructKnots[index - 1]) + 0.025);
      input.max = String((index === state.constructKnots.length - 1 ? 1 : state.constructKnots[index + 1]) - 0.025);
      input.value = state.constructKnots[index].toFixed(2);
      input.setAttribute('aria-label', `Interior knot ${index + 1} position`);
      input.addEventListener('change', () => {
        assignKnotPosition(index, Number(input.value));
        state.statusMessage = `Interior knot ${index + 1} moved to x = ${state.constructKnots[index].toFixed(2)}.`;
        render();
      });
      label.append(input);
      els.knotInputList.append(label);
    }
  }

  function syncControls() {
    const constructing = state.lane === 'construct';
    document.querySelectorAll('[data-lane]').forEach((button) => {
      const active = button.dataset.lane === state.lane;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    document.querySelectorAll('.construction-only').forEach((element) => { element.hidden = !constructing; });
    document.querySelectorAll('.train-only').forEach((element) => { element.hidden = constructing; });

    els.target.value = state.target;
    els.drawHelp.hidden = state.target !== 'custom';
    els.main.classList.toggle('is-drawable', state.target === 'custom');
    els.main.classList.toggle('is-knot-editable', constructing);
    els.plotInstruction.textContent = constructing
      ? (state.target === 'custom'
        ? 'Drag an orange interior knot horizontally; drag elsewhere to draw the target.'
        : 'Drag an orange interior knot left or right; endpoint knots stay fixed.')
      : 'Training uses 121 fixed target samples; the error metrics use a separate 401-point grid.';
    els.complexity.min = constructing ? '1' : '2';
    els.complexity.max = constructing ? '15' : '16';
    els.complexity.value = constructing ? String(state.constructHinges) : String(state.trainWidth);
    els.complexityLabel.textContent = constructing ? 'Interior hinges' : 'Hidden ReLU units';
    els.complexityValue.value = els.complexity.value;
    if (constructing) {
      const segments = state.constructHinges + 1;
      els.complexityMeta.textContent = `${segments} segments · ${segments} ReLU units · ${segments + 1} knots`;
    } else {
      els.complexityMeta.textContent = `${state.trainWidth} trainable signed hinge functions`;
    }
    renderKnotInputs();
    els.epsilon.value = String(state.epsilon);
    els.epsilonValue.value = state.epsilon.toFixed(2);
    els.learningRate.value = String(state.learningRate);
    els.seed.value = String(state.seed);
    els.actionIndex.textContent = constructing ? '3' : '4';
    els.run.disabled = constructing;
    els.step.disabled = constructing;
    els.batch.disabled = constructing;
    els.run.textContent = state.running ? 'Pause' : 'Run';
    els.run.classList.toggle('is-running', state.running);
    els.run.setAttribute('aria-pressed', String(state.running));
    els.status.textContent = state.statusMessage;
    els.provenance.textContent = provenanceText();
    els.provenance.dataset.kind = state.provenance;
  }

  function updateText(metrics) {
    const constructing = state.lane === 'construct';
    const targetName = TARGET_NAMES[state.target];
    const pass = metrics.maxGap < state.epsilon;
    els.workspaceEyebrow.textContent = constructing ? 'CONSTRUCTED WITNESS' : 'OPTIMIZER RUN';
    els.workspaceTitle.textContent = constructing
      ? `${targetName} · fixed-knot interpolation`
      : `${targetName} · trained ReLU approximation`;
    els.tolerance.textContent = pass ? 'Grid gap < ε' : 'Grid gap ≥ ε';
    els.tolerance.classList.toggle('is-pass', pass);
    els.tolerance.classList.toggle('is-fail', !pass);
    els.trainMse.textContent = constructing ? '—' : trainSampleMse().toFixed(5);
    els.trainMseNote.textContent = constructing ? 'not trained' : `${TRAIN_COUNT} fixed samples`;
    els.denseMse.textContent = metrics.mse.toFixed(5);
    els.maxGap.textContent = metrics.maxGap.toFixed(3);
    els.steps.textContent = String(state.steps);
    els.parameterNote.textContent = `${parameterCount()} parameters`;
    els.maxGapAt.textContent = `max at x = ${metrics.maxX.toFixed(2)}`;
    els.modelFormula.textContent = `ĝ(x) = ${state.model.c.toFixed(2)} + Σ vⱼ ReLU(wⱼx + bⱼ)`;
    els.modelSummary.textContent = `One input → ${state.model.width} ReLUs → one output`;
    els.sourceFact.textContent = constructing
      ? (knotsAreUniform() ? 'Fixed uniform knots' : 'Fixed learner-moved knots')
      : provenanceText();
    els.biasFact.textContent = state.model.c.toFixed(3);
    els.objectiveFact.textContent = constructing ? 'No optimization' : `MSE on ${TRAIN_COUNT} samples`;
    els.boundaryNote.textContent = constructing
      ? 'This construction is a witness for this target and width. It does not show that gradient descent would discover the same weights.'
      : 'This curve is one deterministic optimizer run. Low training MSE does not by itself establish a uniform guarantee or generalization.';
  }

  function render() {
    syncControls();
    const metrics = denseMetrics();
    updateText(metrics);
    drawMain(metrics);
    drawResidual(metrics);
    renderContributions();
  }

  function stopRunning() {
    state.running = false;
  }

  function animationFrame(time) {
    if (!state.running) return;
    if (time - state.lastFrame >= 85) {
      trainSteps(8);
      state.lastFrame = time;
      render();
    }
    requestAnimationFrame(animationFrame);
  }

  function toggleRunning() {
    if (state.lane !== 'train') return;
    if (state.running) {
      stopRunning();
      state.statusMessage = `Paused after ${state.steps} full-batch steps.`;
      render();
      return;
    }
    state.running = true;
    state.lastFrame = performance.now();
    state.statusMessage = `Training from seed ${state.seed}…`;
    render();
    requestAnimationFrame(animationFrame);
  }

  function rebuildModel(message) {
    stopRunning();
    state.steps = 0;
    if (state.lane === 'construct') {
      state.model = makeConstructedModel(state.constructHinges);
      state.provenance = 'constructed';
      state.statusMessage = message || `Fixed-knot interpolant with ${state.constructHinges} interior hinges. No optimizer was used.`;
    } else {
      state.model = makeTrainModel(state.trainWidth, state.seed);
      state.provenance = 'random';
      state.statusMessage = message || `Deterministic random weights loaded from seed ${state.seed}.`;
    }
    render();
  }

  function setLane(lane) {
    if (!['construct', 'train'].includes(lane) || lane === state.lane) return;
    state.lane = lane;
    rebuildModel(lane === 'construct'
      ? 'Fixed-knot construction loaded. No optimizer was used.'
      : `Random initialization loaded from seed ${state.seed}.`);
  }

  function setTarget(target) {
    if (!Object.prototype.hasOwnProperty.call(TARGET_NAMES, target)) return;
    state.target = target;
    rebuildModel(target === 'custom'
      ? 'Draw inside the target plot; the model resets when the target changes.'
      : `${TARGET_NAMES[target]} target loaded; model reset.`);
  }

  function setComplexity(value) {
    const numeric = Math.round(Number(value));
    if (state.lane === 'construct') {
      state.constructHinges = clamp(numeric, 1, 15);
      resetUniformKnots();
    } else state.trainWidth = clamp(numeric, 2, 16);
    rebuildModel(state.lane === 'construct'
      ? `${state.constructHinges} interior hinges define ${state.constructHinges + 1} linear segments.`
      : `${state.trainWidth} random ReLU units loaded at seed ${state.seed}.`);
  }

  function eventToCanvas(event) {
    const frame = state.mainFrame;
    if (!frame) return null;
    const rect = els.main.getBoundingClientRect();
    const screenX = (event.clientX - rect.left) * frame.width / rect.width;
    const screenY = (event.clientY - rect.top) * frame.height / rect.height;
    return { frame, screenX, screenY };
  }

  function eventToPlot(event) {
    const location = eventToCanvas(event);
    if (!location) return null;
    const { frame, screenX, screenY } = location;
    const inside = screenX >= frame.margin.left
      && screenX <= frame.margin.left + frame.plotWidth
      && screenY >= frame.margin.top
      && screenY <= frame.margin.top + frame.plotHeight;
    if (!inside) return null;
    const x = frame.xRange[0] + (screenX - frame.margin.left) / frame.plotWidth * (frame.xRange[1] - frame.xRange[0]);
    const y = frame.yRange[1] - (screenY - frame.margin.top) / frame.plotHeight * (frame.yRange[1] - frame.yRange[0]);
    return { x: clamp(x, -1, 1), y: clamp(y, -1.2, 1.2) };
  }

  function knotHitIndex(event) {
    if (state.lane !== 'construct') return null;
    const location = eventToCanvas(event);
    if (!location) return null;
    const { frame, screenX, screenY } = location;
    let best = null;
    let bestDistance = 15;
    for (let index = 0; index < state.constructKnots.length; index++) {
      const knot = state.constructKnots[index];
      const knotX = frame.px(knot);
      const knotY = frame.py(forward(state.model, knot));
      const distance = Math.hypot(screenX - knotX, screenY - knotY);
      if (distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    }
    return best;
  }

  function updateDraggedKnot(event) {
    if (state.knotDragIndex === null) return;
    const point = eventToPlot(event);
    if (!point) return;
    const index = state.knotDragIndex;
    assignKnotPosition(index, point.x);
    state.statusMessage = `Interior knot ${index + 1} moved to x = ${state.constructKnots[index].toFixed(2)}.`;
    render();
  }

  function updateCustomTarget(event) {
    const point = eventToPlot(event);
    if (!point) return;
    const index = Math.round((point.x + 1) * (CUSTOM_COUNT - 1) / 2);
    if (state.lastDrawIndex === null) {
      state.customTarget[index] = point.y;
    } else {
      const startIndex = state.lastDrawIndex;
      const startValue = state.customTarget[startIndex];
      const distance = Math.abs(index - startIndex);
      const direction = index >= startIndex ? 1 : -1;
      for (let offset = 0; offset <= distance; offset++) {
        const t = distance === 0 ? 1 : offset / distance;
        state.customTarget[startIndex + direction * offset] = lerp(startValue, point.y, t);
      }
    }
    state.lastDrawIndex = index;
    if (state.lane === 'construct') state.model = makeConstructedModel(state.constructHinges);
    state.statusMessage = 'Drawn target updated. The target remains continuous between sampled points.';
    render();
  }

  function beginCustomDraw(event) {
    if (state.target !== 'custom') return;
    event.preventDefault();
    stopRunning();
    state.drawActive = true;
    state.lastDrawIndex = null;
    state.steps = 0;
    if (state.lane === 'train') {
      state.model = makeTrainModel(state.trainWidth, state.seed);
      state.provenance = 'random';
    }
    els.main.setPointerCapture?.(event.pointerId);
    updateCustomTarget(event);
  }

  function handleMainPointerDown(event) {
    const knotIndex = knotHitIndex(event);
    if (knotIndex !== null) {
      event.preventDefault();
      stopRunning();
      state.knotDragIndex = knotIndex;
      els.main.setPointerCapture?.(event.pointerId);
      updateDraggedKnot(event);
      return;
    }
    beginCustomDraw(event);
  }

  function endCustomDraw() {
    state.drawActive = false;
    state.lastDrawIndex = null;
    state.knotDragIndex = null;
  }

  document.querySelectorAll('[data-lane]').forEach((button) => {
    button.addEventListener('click', () => setLane(button.dataset.lane));
  });
  document.querySelectorAll('[data-hinges]').forEach((button) => {
    button.addEventListener('click', () => {
      state.constructHinges = Number(button.dataset.hinges);
      resetUniformKnots();
      rebuildModel(`${state.constructHinges} interior hinges define ${state.constructHinges + 1} linear segments.`);
    });
  });
  els.target.addEventListener('change', () => setTarget(els.target.value));
  els.complexity.addEventListener('input', () => setComplexity(els.complexity.value));
  els.epsilon.addEventListener('input', () => {
    state.epsilon = Number(els.epsilon.value);
    state.statusMessage = `Tolerance set to ε = ${state.epsilon.toFixed(2)}.`;
    render();
  });
  els.learningRate.addEventListener('change', () => {
    state.learningRate = Number(els.learningRate.value);
    state.statusMessage = `Learning rate set to ${state.learningRate}. Current weights are unchanged.`;
    render();
  });
  els.seed.addEventListener('change', () => {
    state.seed = clamp(Math.round(Number(els.seed.value) || 11), 1, 9999);
    if (state.lane === 'train') rebuildModel(`Random initialization loaded from seed ${state.seed}.`);
    else render();
  });
  els.newSeed.addEventListener('click', () => {
    state.seed = state.seed % 9999 + 1;
    if (state.lane === 'train') rebuildModel(`Random initialization loaded from seed ${state.seed}.`);
    else render();
  });
  els.run.addEventListener('click', toggleRunning);
  els.step.addEventListener('click', () => { trainSteps(1); render(); });
  els.batch.addEventListener('click', () => { trainSteps(100); render(); });
  els.reset.addEventListener('click', () => {
    if (state.lane === 'construct') resetUniformKnots();
    rebuildModel(state.lane === 'construct' ? 'Uniform knot positions restored.' : undefined);
  });
  els.resetTarget.addEventListener('click', () => {
    state.customTarget.fill(0);
    rebuildModel('Drawn target cleared to zero; model reset.');
  });
  els.main.addEventListener('pointerdown', handleMainPointerDown);
  els.main.addEventListener('pointermove', (event) => {
    if (state.knotDragIndex !== null) updateDraggedKnot(event);
    else if (state.drawActive) updateCustomTarget(event);
  });
  els.main.addEventListener('pointerup', endCustomDraw);
  els.main.addEventListener('pointercancel', endCustomDraw);
  els.main.addEventListener('lostpointercapture', endCustomDraw);
  window.addEventListener('resize', render);

  function snapshot() {
    const metrics = denseMetrics();
    return {
      lane: state.lane,
      target: state.target,
      provenance: state.provenance,
      interiorHinges: state.lane === 'construct' ? state.constructHinges : null,
      interiorKnotPositions: state.lane === 'construct' ? state.constructKnots.slice() : null,
      width: state.model.width,
      parameters: parameterCount(),
      seed: state.seed,
      learningRate: state.learningRate,
      epsilon: state.epsilon,
      steps: state.steps,
      trainingSampleMse: state.lane === 'train' ? trainSampleMse() : null,
      denseGridMse: metrics.mse,
      maxGridGap: metrics.maxGap,
      maxGapX: metrics.maxX,
      gridPassesTolerance: metrics.maxGap < state.epsilon,
      weightSignature: [state.model.c, ...state.model.W, ...state.model.b, ...state.model.v]
        .map((value) => value.toFixed(6)).join('|')
    };
  }

  window.ReLUFunctionLab = Object.freeze({
    snapshot,
    setLane,
    setTarget,
    setComplexity,
    setInteriorKnots(knots) {
      if (state.lane !== 'construct' || !Array.isArray(knots) || knots.length !== state.constructHinges) return;
      const sorted = knots.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
      if (sorted.length !== state.constructHinges) return;
      for (let index = 0; index < sorted.length; index++) {
        const left = index === 0 ? -1 : sorted[index - 1];
        sorted[index] = clamp(sorted[index], left + 0.025, 1 - 0.025 * (sorted.length - index));
      }
      state.constructKnots = sorted;
      rebuildModel('Interior knots updated directly.');
    },
    setSeed(seed) {
      state.seed = clamp(Math.round(Number(seed) || 11), 1, 9999);
      if (state.lane === 'train') rebuildModel(`Random initialization loaded from seed ${state.seed}.`);
      else render();
    },
    setEpsilon(value) {
      state.epsilon = clamp(Number(value), 0.01, 0.5);
      render();
    },
    step(count = 1) {
      if (state.lane !== 'train') return;
      trainSteps(Math.max(1, Math.round(count)));
      render();
    }
  });

  state.model = makeConstructedModel(state.constructHinges);
  render();
})();
