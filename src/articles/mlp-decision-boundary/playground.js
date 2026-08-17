(() => {
  'use strict';

  const COLORS = {
    ink: '#1f2933', muted: '#637080', grid: '#d7dde2', blue: '#2463a7',
    blueSoft: '#dceafa', orange: '#d56b2d', orangeSoft: '#f7e2d5',
    teal: '#21766f', red: '#b84843', paper: '#ffffff', canvas: '#f4f6f7'
  };
  const $ = (id) => document.getElementById(id);
  const els = {
    app: $('app'), dataset: $('datasetSelect'), target: $('targetSelect'), depth: $('depthSelect'),
    width: $('widthRange'), widthOutput: $('widthOutput'), widthLabel: $('widthLabel'),
    lr: $('learningRateSelect'), seed: $('seedInput'), newSeed: $('newSeedBtn'),
    run: $('runBtn'), step: $('stepBtn'), reset: $('resetBtn'), status: $('statusText'),
    main: $('mainCanvas'), network: $('networkCanvas'), surface: $('surfaceCanvas'), residual: $('residualCanvas'),
    featureGrid: $('featureGrid'), parameterCount: $('parameterCount'), networkTitle: $('networkTitle'),
    workspaceEyebrow: $('workspaceEyebrow'), workspaceTitle: $('workspaceTitle'),
    metric1Label: $('metric1Label'), metric1Value: $('metric1Value'),
    metric2Label: $('metric2Label'), metric2Value: $('metric2Value'),
    metric3Label: $('metric3Label'), metric3Value: $('metric3Value'), stepsValue: $('stepsValue'),
    classData: $('classificationDataControls'), targetControls: $('targetControls'),
    classModel: $('classificationModelControls'), approxModel: $('approximationModelControls'),
    lrField: $('learningRateField'), classPresets: $('classificationPresets'), approxPresets: $('approximationPresets'),
    customTools: $('customPointTools'), clearPoints: $('clearPointsBtn'), drawNote: $('drawTargetNote'),
    classViewSwitch: $('classificationViewSwitch'), classLegend: $('classificationLegend'), approxLegend: $('approximationLegend'),
    hiddenSwitch: $('hiddenLayerSwitch'), surfacePanel: $('surfacePanel'), residualPanel: $('residualPanel'),
    probeControls: $('probeControls'), probeX: $('probeX'), probeY: $('probeY'),
    probeXOutput: $('probeXOutput'), probeYOutput: $('probeYOutput'), probeReadout: $('probeReadout'),
    classPrompts: $('classificationPrompts'), approxPrompts: $('approximationPrompts'),
    cornerProof: $('cornerProofBtn'), fieldProof: $('fieldProofBtn')
  };

  const state = {
    mode: 'classification', dataset: 'xor4', target: 'tent', seed: 11,
    classDepth: 1, classWidth: 2, approxWidth: 5, approxMethod: 'construct',
    learningRate: 0.03, classView: 'probability', hiddenLayer: 0,
    pointClass: 0, customPoints: [], customTarget: null, data: [], model: null,
    steps: 0, running: false, history: [], lastFrame: 0, drawActive: false,
    lastDrawIndex: null, mainTransform: null, renderTick: 0
  };

  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function gaussian(rng) {
    const u = Math.max(1e-9, rng());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const sigmoid = (z) => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  const relu = (x) => Math.max(0, x);
  const lerp = (a, b, t) => a + (b - a) * t;
  const pct = (x) => Number.isFinite(x) ? `${(100 * x).toFixed(x > .995 ? 1 : 0)}%` : '—';
  const finite = (x, fallback = 0) => Number.isFinite(x) ? x : fallback;

  function truthFor(dataset, x, y) {
    if (dataset === 'xor4' || dataset === 'xorField') return ((x > .5) !== (y > .5)) ? 1 : 0;
    if (dataset === 'blobs') return x + .12 * Math.sin(5 * y) > .98 ? 1 : 0;
    if (dataset === 'circles') return Math.hypot(x - .5, y - .5) < .245 ? 1 : 0;
    if (dataset === 'moons') {
      const d0 = Math.abs(Math.hypot(x - .36, y - .50) - .24);
      const d1 = Math.abs(Math.hypot(x - .64, y - .46) - .24);
      return d1 < d0 ? 1 : 0;
    }
    return null;
  }

  function makeDataset(key, seed) {
    const rng = mulberry32(seed * 997 + 17);
    if (key === 'xor4') return [
      { x: 0, y: 0, label: 0 }, { x: 0, y: 1, label: 1 },
      { x: 1, y: 0, label: 1 }, { x: 1, y: 1, label: 0 }
    ];
    if (key === 'custom') {
      if (!state.customPoints.length) state.customPoints = [
        { x: .22, y: .25, label: 0 }, { x: .33, y: .35, label: 0 },
        { x: .72, y: .68, label: 1 }, { x: .82, y: .78, label: 1 }
      ];
      return state.customPoints.map(p => ({ ...p }));
    }
    const n = key === 'xorField' ? 256 : 220;
    const points = [];
    if (key === 'xorField') {
      while (points.length < n) {
        const x = rng(), y = rng();
        if (Math.abs(x - .5) < .025 || Math.abs(y - .5) < .025) continue;
        points.push({ x, y, label: truthFor(key, x, y) });
      }
    } else if (key === 'blobs') {
      for (let i = 0; i < n; i++) {
        const label = i % 2;
        points.push({
          x: clamp((label ? .70 : .30) + .12 * gaussian(rng), .02, .98),
          y: clamp((label ? .65 : .35) + .13 * gaussian(rng), .02, .98), label
        });
      }
    } else if (key === 'circles') {
      for (let i = 0; i < n; i++) {
        const label = i % 2;
        const angle = 2 * Math.PI * rng();
        const radius = (label ? .17 : .37) + .025 * gaussian(rng);
        points.push({ x: clamp(.5 + radius * Math.cos(angle), .01, .99), y: clamp(.5 + radius * Math.sin(angle), .01, .99), label });
      }
    } else if (key === 'moons') {
      for (let i = 0; i < n; i++) {
        const label = i % 2, angle = Math.PI * rng();
        let x, y;
        if (!label) { x = .19 + .47 * (1 + Math.cos(angle)) / 2; y = .48 + .24 * Math.sin(angle); }
        else { x = .34 + .47 * (1 - Math.cos(angle)) / 2; y = .50 - .24 * Math.sin(angle); }
        points.push({ x: clamp(x + .025 * gaussian(rng), .01, .99), y: clamp(y + .025 * gaussian(rng), .01, .99), label });
      }
    } else if (key === 'spirals') {
      for (let i = 0; i < n; i++) {
        const label = i % 2, t = (i / n) * 2.6 * Math.PI + .12 * gaussian(rng);
        const r = .06 + .39 * (i / n);
        const a = t + label * Math.PI;
        points.push({ x: clamp(.5 + r * Math.cos(a), .01, .99), y: clamp(.5 + r * Math.sin(a), .01, .99), label });
      }
    }
    return points;
  }

  function initClassModel(depth, width, seed) {
    const rng = mulberry32(seed * 7919 + depth * 97 + width);
    const rand = (fanIn) => gaussian(rng) * Math.sqrt(2 / Math.max(1, fanIn));
    const model = { depth, width, W1: [], b1: [], W2: [], b2: [], Wo: [], bo: 0 };
    if (depth === 0) {
      model.Wo = [rand(2), rand(2)];
    } else {
      model.W1 = Array.from({ length: width }, () => [rand(2), rand(2)]);
      model.b1 = Array(width).fill(.03);
      if (depth === 2) {
        model.W2 = Array.from({ length: width }, () => Array.from({ length: width }, () => rand(width)));
        model.b2 = Array(width).fill(.03);
      }
      model.Wo = Array.from({ length: width }, () => rand(width));
    }
    return model;
  }

  function loadCornerRule() {
    state.classDepth = 1; state.classWidth = 2;
    state.model = {
      depth: 1, width: 2,
      W1: [[1, 1], [1, 1]], b1: [0, -1], W2: [], b2: [], Wo: [2, -4], bo: -1
    };
    state.steps = 0; state.history = [];
    state.statusMessage = 'Hand-chosen weights: exact on the four Boolean inputs.';
    syncControls(); render(true);
  }

  function loadFieldRule() {
    state.classDepth = 1; state.classWidth = 4;
    state.model = {
      depth: 1, width: 4,
      W1: [[1, -1], [-1, 1], [1, 1], [-1, -1]],
      b1: [0, 0, -1, 1], W2: [], b2: [], Wo: [1.7, 1.7, -1.7, -1.7], bo: 0
    };
    state.steps = 0; state.history = [];
    state.statusMessage = 'Hand-chosen weights: exact on the filled XOR regions away from the axes.';
    syncControls(); render(true);
  }

  function forwardClass(model, x, y) {
    if (model.depth === 0) {
      const z = model.Wo[0] * x + model.Wo[1] * y + model.bo;
      return { input: [x, y], u1: [], h1: [], u2: [], h2: [], z, p: sigmoid(z) };
    }
    const u1 = model.W1.map((w, j) => w[0] * x + w[1] * y + model.b1[j]);
    const h1 = u1.map(relu);
    let u2 = [], h2 = [], final = h1;
    if (model.depth === 2) {
      u2 = model.W2.map((row, j) => row.reduce((sum, w, k) => sum + w * h1[k], model.b2[j]));
      h2 = u2.map(relu); final = h2;
    }
    const z = model.Wo.reduce((sum, w, j) => sum + w * final[j], model.bo);
    return { input: [x, y], u1, h1, u2, h2, z, p: sigmoid(z) };
  }

  function trainClassStep() {
    const m = state.model, n = Math.max(1, state.data.length);
    if (!state.data.length) return;
    if (m.depth === 0) {
      const gW = [0, 0]; let gb = 0;
      for (const d of state.data) {
        const f = forwardClass(m, d.x, d.y), dz = f.p - d.label;
        gW[0] += dz * d.x; gW[1] += dz * d.y; gb += dz;
      }
      m.Wo[0] -= state.learningRate * gW[0] / n; m.Wo[1] -= state.learningRate * gW[1] / n; m.bo -= state.learningRate * gb / n;
    } else {
      const gW1 = m.W1.map(() => [0, 0]), gb1 = m.b1.map(() => 0);
      const gW2 = m.depth === 2 ? m.W2.map(row => row.map(() => 0)) : [];
      const gb2 = m.depth === 2 ? m.b2.map(() => 0) : [];
      const gWo = m.Wo.map(() => 0); let gbo = 0;
      for (const d of state.data) {
        const f = forwardClass(m, d.x, d.y), dz = f.p - d.label;
        const final = m.depth === 2 ? f.h2 : f.h1;
        for (let j = 0; j < m.width; j++) gWo[j] += dz * final[j];
        gbo += dz;
        if (m.depth === 2) {
          const dh2 = m.Wo.map(w => dz * w);
          const du2 = dh2.map((g, j) => f.u2[j] > 0 ? g : 0);
          for (let j = 0; j < m.width; j++) {
            gb2[j] += du2[j];
            for (let k = 0; k < m.width; k++) gW2[j][k] += du2[j] * f.h1[k];
          }
          const dh1 = Array(m.width).fill(0);
          for (let k = 0; k < m.width; k++) for (let j = 0; j < m.width; j++) dh1[k] += m.W2[j][k] * du2[j];
          for (let j = 0; j < m.width; j++) {
            const du1 = f.u1[j] > 0 ? dh1[j] : 0;
            gW1[j][0] += du1 * d.x; gW1[j][1] += du1 * d.y; gb1[j] += du1;
          }
        } else {
          for (let j = 0; j < m.width; j++) {
            const du1 = f.u1[j] > 0 ? dz * m.Wo[j] : 0;
            gW1[j][0] += du1 * d.x; gW1[j][1] += du1 * d.y; gb1[j] += du1;
          }
        }
      }
      const lr = state.learningRate / n;
      for (let j = 0; j < m.width; j++) {
        m.Wo[j] -= lr * gWo[j];
        m.W1[j][0] -= lr * gW1[j][0]; m.W1[j][1] -= lr * gW1[j][1]; m.b1[j] -= lr * gb1[j];
        if (m.depth === 2) {
          m.b2[j] -= lr * gb2[j];
          for (let k = 0; k < m.width; k++) m.W2[j][k] -= lr * gW2[j][k];
        }
      }
      m.bo -= lr * gbo;
    }
    state.steps++;
    if (state.steps % 5 === 0 || state.steps < 5) state.history.push(classMetrics().loss);
  }

  function classMetrics() {
    if (!state.data.length) return { accuracy: NaN, loss: NaN, field: NaN };
    let correct = 0, loss = 0;
    for (const d of state.data) {
      const p = clamp(forwardClass(state.model, d.x, d.y).p, 1e-7, 1 - 1e-7);
      correct += (p >= .5 ? 1 : 0) === d.label ? 1 : 0;
      loss += -(d.label * Math.log(p) + (1 - d.label) * Math.log(1 - p));
    }
    let field = NaN, total = 0, agree = 0;
    if (state.dataset !== 'custom' && state.dataset !== 'spirals') {
      for (let iy = 0; iy < 55; iy++) for (let ix = 0; ix < 55; ix++) {
        const x = (ix + .5) / 55, y = (iy + .5) / 55;
        if ((state.dataset === 'xor4' || state.dataset === 'xorField') && (Math.abs(x - .5) < .012 || Math.abs(y - .5) < .012)) continue;
        const truth = truthFor(state.dataset, x, y);
        if (truth === null) continue;
        agree += ((forwardClass(state.model, x, y).p >= .5 ? 1 : 0) === truth) ? 1 : 0; total++;
      }
      field = total ? agree / total : NaN;
    }
    return { accuracy: correct / state.data.length, loss: loss / state.data.length, field };
  }

  const TARGETS = {
    tent: (x) => Math.max(0, 1 - Math.abs(1.7 * x)),
    sine: (x) => .82 * Math.sin(Math.PI * x),
    twoBumps: (x) => .9 * Math.exp(-18 * (x + .42) ** 2) - .68 * Math.exp(-26 * (x - .38) ** 2),
    smoothStep: (x) => .78 * Math.tanh(3.2 * x)
  };

  function ensureCustomTarget() {
    if (state.customTarget) return;
    state.customTarget = Array.from({ length: 201 }, (_, i) => TARGETS.twoBumps(-1 + 2 * i / 200));
  }
  function targetAt(x) {
    if (state.target !== 'custom') return TARGETS[state.target](x);
    ensureCustomTarget();
    const t = clamp((x + 1) * 100, 0, 200), lo = Math.floor(t), hi = Math.ceil(t);
    return lerp(state.customTarget[lo], state.customTarget[hi], t - lo);
  }

  function makeConstructedApprox(segments) {
    const knots = Array.from({ length: segments + 1 }, (_, i) => -1 + 2 * i / segments);
    const values = knots.map(targetAt);
    const slopes = Array.from({ length: segments }, (_, i) => (values[i + 1] - values[i]) / (knots[i + 1] - knots[i]));
    const W = Array(segments).fill(1), b = knots.slice(0, -1).map(t => -t);
    const v = slopes.map((s, i) => i === 0 ? s : s - slopes[i - 1]);
    return { type: 'approx', method: 'construct', width: segments, W, b, v, c: values[0], knots, values };
  }

  function initTrainApprox(width, seed) {
    const rng = mulberry32(seed * 3457 + width * 13);
    const W = Array.from({ length: width }, () => gaussian(rng) * .85);
    const b = Array.from({ length: width }, () => -.65 + 1.3 * rng());
    const v = Array.from({ length: width }, () => gaussian(rng) * Math.sqrt(1 / width));
    return { type: 'approx', method: 'train', width, W, b, v, c: 0, knots: [] };
  }
  function forwardApprox(model, x) {
    const u = model.W.map((w, j) => w * x + model.b[j]);
    const h = u.map(relu);
    const y = model.v.reduce((sum, v, j) => sum + v * h[j], model.c);
    return { u, h, y };
  }
  function trainApproxStep() {
    const m = state.model, n = 121;
    const gW = m.W.map(() => 0), gb = m.b.map(() => 0), gv = m.v.map(() => 0); let gc = 0;
    for (let i = 0; i < n; i++) {
      const x = -1 + 2 * i / (n - 1), target = targetAt(x), f = forwardApprox(m, x), dy = 2 * (f.y - target);
      gc += dy;
      for (let j = 0; j < m.width; j++) {
        gv[j] += dy * f.h[j];
        const du = f.u[j] > 0 ? dy * m.v[j] : 0;
        gW[j] += du * x; gb[j] += du;
      }
    }
    const lr = state.learningRate / n;
    m.c -= lr * gc;
    for (let j = 0; j < m.width; j++) {
      m.v[j] -= lr * gv[j]; m.W[j] -= lr * gW[j]; m.b[j] -= lr * gb[j];
    }
    state.steps++;
    if (state.steps % 5 === 0 || state.steps < 5) state.history.push(approxMetrics().mse);
  }
  function approxMetrics() {
    let mse = 0, maxGap = 0;
    for (let i = 0; i <= 400; i++) {
      const x = -1 + 2 * i / 400, gap = Math.abs(targetAt(x) - forwardApprox(state.model, x).y);
      mse += gap * gap; maxGap = Math.max(maxGap, gap);
    }
    return { mse: mse / 401, maxGap };
  }

  function resetModel(message = 'Random weights loaded.') {
    stopRunning(); state.steps = 0; state.history = [];
    if (state.mode === 'classification') {
      state.model = initClassModel(state.classDepth, state.classWidth, state.seed);
    } else if (state.approxMethod === 'construct') {
      state.model = makeConstructedApprox(state.approxWidth);
      message = `Fixed-knot construction with ${state.approxWidth} segments.`;
    } else {
      state.model = initTrainApprox(state.approxWidth, state.seed);
    }
    state.statusMessage = message; render(true);
  }

  function setMode(mode) {
    if (mode === state.mode) return;
    stopRunning(); state.mode = mode; state.steps = 0; state.history = [];
    if (mode === 'classification') {
      state.data = makeDataset(state.dataset, state.seed);
      state.model = initClassModel(state.classDepth, state.classWidth, state.seed);
      state.statusMessage = 'Random classifier weights loaded.';
    } else {
      state.model = state.approxMethod === 'construct' ? makeConstructedApprox(state.approxWidth) : initTrainApprox(state.approxWidth, state.seed);
      state.statusMessage = state.approxMethod === 'construct' ? 'Fixed-knot construction loaded.' : 'Random approximation weights loaded.';
    }
    syncControls(); render(true);
  }

  function setDataset(key) {
    stopRunning(); state.dataset = key; state.data = makeDataset(key, state.seed);
    state.classDepth = key === 'spirals' ? 2 : 1;
    state.classWidth = key === 'spirals' ? 8 : (key === 'xor4' ? 2 : 4);
    state.model = initClassModel(state.classDepth, state.classWidth, state.seed);
    state.steps = 0; state.history = [];
    state.statusMessage = `${datasetName(key)} loaded; weights reset.`;
    syncControls(); render(true);
  }

  function setTarget(key) {
    stopRunning(); state.target = key;
    if (key === 'custom') ensureCustomTarget();
    state.steps = 0; state.history = [];
    state.model = state.approxMethod === 'construct' ? makeConstructedApprox(state.approxWidth) : initTrainApprox(state.approxWidth, state.seed);
    state.statusMessage = key === 'custom' ? 'Drag on the plot to draw a target.' : `${targetName(key)} target loaded.`;
    syncControls(); render(true);
  }

  const datasetName = (key) => ({ xor4: 'Four-point XOR', xorField: 'Filled XOR', blobs: 'Two blobs', circles: 'Concentric circles', moons: 'Two moons', spirals: 'Two spirals', custom: 'Custom points' }[key]);
  const targetName = (key) => ({ tent: 'Tent', sine: 'Sine', twoBumps: 'Two bumps', smoothStep: 'Smooth step', custom: 'Custom' }[key]);

  function parameterCount() {
    const m = state.model;
    if (state.mode === 'approximation') return 3 * m.width + 1;
    if (m.depth === 0) return 3;
    return m.depth === 1 ? 4 * m.width + 1 : m.width * m.width + 5 * m.width + 1;
  }

  function syncControls() {
    document.querySelectorAll('[data-mode]').forEach(btn => {
      const active = btn.dataset.mode === state.mode; btn.classList.toggle('is-active', active); btn.setAttribute('aria-pressed', String(active));
    });
    const classification = state.mode === 'classification';
    els.classData.hidden = !classification; els.targetControls.hidden = classification;
    els.classModel.hidden = !classification; els.approxModel.hidden = classification;
    els.classPresets.hidden = !classification; els.approxPresets.hidden = classification;
    els.classViewSwitch.hidden = !classification; els.classLegend.hidden = !classification; els.approxLegend.hidden = classification;
    els.surfacePanel.hidden = !classification; els.residualPanel.hidden = classification;
    els.classPrompts.hidden = !classification; els.approxPrompts.hidden = classification;
    els.probeControls.hidden = !classification;
    els.dataset.value = state.dataset; els.target.value = state.target; els.depth.value = String(state.classDepth);
    els.customTools.hidden = !(classification && state.dataset === 'custom');
    els.drawNote.hidden = !(state.mode === 'approximation' && state.target === 'custom');
    els.width.min = classification ? 1 : 2; els.width.max = 16;
    els.width.value = classification ? state.classWidth : state.approxWidth;
    els.widthOutput.value = els.width.value;
    els.widthLabel.textContent = classification ? 'Hidden width' : (state.approxMethod === 'construct' ? 'Segments / ReLUs' : 'Hidden width');
    els.lrField.hidden = !classification && state.approxMethod === 'construct';
    els.run.disabled = !classification && state.approxMethod === 'construct';
    els.step.disabled = !classification && state.approxMethod === 'construct';
    els.lr.value = String(state.learningRate); els.seed.value = String(state.seed);
    document.querySelectorAll('[data-approx-method]').forEach(btn => {
      const active = btn.dataset.approxMethod === state.approxMethod; btn.classList.toggle('is-active', active); btn.setAttribute('aria-pressed', String(active));
    });
    document.querySelectorAll('[data-point-class]').forEach(btn => {
      const active = Number(btn.dataset.pointClass) === state.pointClass; btn.classList.toggle('is-active', active); btn.setAttribute('aria-pressed', String(active));
    });
    document.querySelectorAll('[data-class-view]').forEach(btn => {
      const active = btn.dataset.classView === state.classView; btn.classList.toggle('is-active', active); btn.setAttribute('aria-pressed', String(active));
    });
    document.querySelectorAll('[data-hidden-layer]').forEach(btn => {
      const index = Number(btn.dataset.hiddenLayer), enabled = classification && state.classDepth === 2 ? true : index === 0;
      btn.disabled = !enabled; if (!enabled && index === 1) btn.classList.remove('is-active');
      const active = index === state.hiddenLayer && enabled; btn.classList.toggle('is-active', active); btn.setAttribute('aria-pressed', String(active));
    });
  }

  function setupCanvas(canvas, minHeight = 120) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(280, Math.round(rect.width || canvas.parentElement.clientWidth));
    const height = Math.max(minHeight, Math.round(rect.height || minHeight));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function rgb(hex) { return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)]; }
  function blend(a, b, t) {
    const A = rgb(a), B = rgb(b);
    return `rgb(${Math.round(lerp(A[0],B[0],t))},${Math.round(lerp(A[1],B[1],t))},${Math.round(lerp(A[2],B[2],t))})`;
  }
  function plotFrame(ctx, width, height, xMin, xMax, yMin, yMax, labelX = 'x₁', labelY = 'x₂') {
    const m = { left: 48, right: 20, top: 18, bottom: 38 };
    const pw = width - m.left - m.right, ph = height - m.top - m.bottom;
    const px = x => m.left + (x - xMin) / (xMax - xMin) * pw;
    const py = y => m.top + (yMax - y) / (yMax - yMin) * ph;
    ctx.fillStyle = COLORS.paper; ctx.fillRect(m.left, m.top, pw, ph);
    ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const tx = m.left + i * pw / 4, ty = m.top + i * ph / 4;
      ctx.beginPath(); ctx.moveTo(tx, m.top); ctx.lineTo(tx, m.top + ph); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(m.left, ty); ctx.lineTo(m.left + pw, ty); ctx.stroke();
    }
    ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.4; ctx.strokeRect(m.left, m.top, pw, ph);
    ctx.fillStyle = COLORS.muted; ctx.font = '11px ui-sans-serif, sans-serif'; ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) ctx.fillText((xMin + (xMax - xMin) * i / 4).toFixed(i === 0 || i === 4 ? 1 : 2), m.left + i * pw / 4, height - 17);
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) ctx.fillText((yMax - (yMax - yMin) * i / 4).toFixed(i === 0 || i === 4 ? 1 : 2), m.left - 7, m.top + i * ph / 4 + 4);
    ctx.fillStyle = COLORS.ink; ctx.font = '700 12px ui-sans-serif, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(labelX, m.left + pw / 2, height - 2);
    ctx.save(); ctx.translate(12, m.top + ph / 2); ctx.rotate(-Math.PI/2); ctx.fillText(labelY, 0, 0); ctx.restore();
    return { m, pw, ph, px, py, xMin, xMax, yMin, yMax };
  }

  function drawMainClassification() {
    const { ctx, width, height } = setupCanvas(els.main, 360);
    ctx.clearRect(0, 0, width, height);
    const frame = plotFrame(ctx, width, height, 0, 1, 0, 1);
    const { m, pw, ph, px, py } = frame; state.mainTransform = frame;
    ctx.save(); ctx.beginPath(); ctx.rect(m.left, m.top, pw, ph); ctx.clip();
    const nx = Math.max(45, Math.round(pw / 8)), ny = Math.max(45, Math.round(ph / 8));
    for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
      const x = (ix + .5) / nx, y = 1 - (iy + .5) / ny, p = forwardClass(state.model, x, y).p;
      let fill;
      if (state.classView === 'class') fill = p >= .5 ? COLORS.blueSoft : COLORS.orangeSoft;
      else fill = p >= .5 ? blend('#ffffff', COLORS.blue, .12 + .42 * (p - .5) * 2) : blend('#ffffff', COLORS.orange, .12 + .42 * (.5 - p) * 2);
      ctx.fillStyle = fill; ctx.fillRect(m.left + ix * pw / nx, m.top + iy * ph / ny, pw / nx + 1, ph / ny + 1);
    }
    drawBoundary(ctx, frame);
    if (state.dataset === 'xor4' || state.dataset === 'xorField') {
      ctx.save(); ctx.setLineDash([5,5]); ctx.strokeStyle = '#8996a1'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px(.5), m.top); ctx.lineTo(px(.5), m.top + ph); ctx.moveTo(m.left, py(.5)); ctx.lineTo(m.left + pw, py(.5)); ctx.stroke(); ctx.restore();
    }
    for (const d of state.data) drawPoint(ctx, px(d.x), py(d.y), d.label, state.data.length <= 20 ? 7 : 4.1);
    ctx.restore();
    els.main.setAttribute('aria-label', `${datasetName(state.dataset)} with ${state.data.length} points and the current model ${state.classView} field.`);
  }

  function drawBoundary(ctx, frame) {
    const { m, pw, ph } = frame, n = 52;
    const value = (ix, iy) => forwardClass(state.model, ix / n, 1 - iy / n).z;
    ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.35; ctx.beginPath();
    for (let iy = 0; iy < n; iy++) for (let ix = 0; ix < n; ix++) {
      const vals = [value(ix,iy), value(ix+1,iy), value(ix+1,iy+1), value(ix,iy+1)];
      const points = [];
      const corners = [[ix,iy],[ix+1,iy],[ix+1,iy+1],[ix,iy+1]];
      for (let e = 0; e < 4; e++) {
        const j = (e + 1) % 4, a = vals[e], b = vals[j];
        if ((a >= 0) === (b >= 0) || a === b) continue;
        const t = a / (a - b), c0 = corners[e], c1 = corners[j];
        points.push([lerp(c0[0], c1[0], t), lerp(c0[1], c1[1], t)]);
      }
      if (points.length >= 2) {
        ctx.moveTo(m.left + points[0][0] * pw / n, m.top + points[0][1] * ph / n);
        ctx.lineTo(m.left + points[1][0] * pw / n, m.top + points[1][1] * ph / n);
      }
    }
    ctx.stroke();
  }

  function drawPoint(ctx, x, y, label, size) {
    ctx.save(); ctx.translate(x, y); ctx.fillStyle = label ? COLORS.blue : COLORS.orange; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.7;
    ctx.beginPath();
    if (label) { ctx.moveTo(0, -size); ctx.lineTo(size, 0); ctx.lineTo(0, size); ctx.lineTo(-size, 0); ctx.closePath(); }
    else ctx.arc(0, 0, size * .86, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke(); ctx.restore();
  }

  function approxYRange() {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i <= 200; i++) {
      const x = -1 + 2 * i / 200, t = targetAt(x), y = forwardApprox(state.model, x).y;
      lo = Math.min(lo, t, y); hi = Math.max(hi, t, y);
    }
    const span = Math.max(.4, hi - lo), pad = .14 * span;
    return [Math.max(-2.2, lo - pad), Math.min(2.2, hi + pad)];
  }
  function drawMainApproximation() {
    const { ctx, width, height } = setupCanvas(els.main, 360); ctx.clearRect(0,0,width,height);
    const [yMin, yMax] = approxYRange(), frame = plotFrame(ctx, width, height, -1, 1, yMin, yMax, 'x', 'value');
    const { m, pw, ph, px, py } = frame; state.mainTransform = frame;
    ctx.save(); ctx.beginPath(); ctx.rect(m.left,m.top,pw,ph); ctx.clip();
    if (yMin < 0 && yMax > 0) { ctx.strokeStyle = '#9aa5ae'; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(m.left,py(0)); ctx.lineTo(m.left+pw,py(0)); ctx.stroke(); ctx.setLineDash([]); }
    drawCurve(ctx, px, py, x => targetAt(x), COLORS.ink, 2.2);
    drawCurve(ctx, px, py, x => forwardApprox(state.model, x).y, COLORS.blue, 3);
    const knots = state.model.method === 'construct' ? state.model.knots : state.model.W.map((w,j) => Math.abs(w) > 1e-8 ? -state.model.b[j] / w : NaN).filter(t => t >= -1 && t <= 1);
    for (const x of knots) {
      const y = forwardApprox(state.model, x).y;
      ctx.save(); ctx.translate(px(x),py(y)); ctx.rotate(Math.PI/4); ctx.fillStyle = COLORS.orange; ctx.fillRect(-4,-4,8,8); ctx.restore();
    }
    ctx.restore();
    els.main.setAttribute('aria-label', `${targetName(state.target)} target and ${state.model.width}-ReLU ${state.model.method === 'construct' ? 'fixed-knot construction' : 'trained approximation'}.`);
  }
  function drawCurve(ctx, px, py, fn, color, width) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    for (let i = 0; i <= 300; i++) {
      const x = -1 + 2 * i / 300, y = finite(fn(x));
      if (i) ctx.lineTo(px(x),py(y)); else ctx.moveTo(px(x),py(y));
    }
    ctx.stroke();
  }

  function drawNetwork() {
    const { ctx, width, height } = setupCanvas(els.network, 130); ctx.clearRect(0,0,width,height);
    const layers = state.mode === 'classification'
      ? (state.classDepth === 0 ? [2,1] : state.classDepth === 1 ? [2,state.classWidth,1] : [2,state.classWidth,state.classWidth,1])
      : [1,state.approxWidth,1];
    const maxShow = 9, xPositions = layers.map((_,i) => 38 + i * (width - 76) / Math.max(1,layers.length-1));
    const nodePositions = layers.map((count,li) => {
      const shown = Math.min(count,maxShow), gap = Math.min(25,(height-28)/Math.max(1,shown-1));
      return Array.from({length:shown},(_,i)=>({x:xPositions[li],y:height/2+(i-(shown-1)/2)*gap}));
    });
    ctx.lineWidth = .8; ctx.strokeStyle = '#c6cfd6';
    for (let li=0;li<nodePositions.length-1;li++) for (const a of nodePositions[li]) for (const b of nodePositions[li+1]) { ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke(); }
    for (let li=0;li<nodePositions.length;li++) for (const p of nodePositions[li]) {
      ctx.beginPath();ctx.arc(p.x,p.y,li===0||li===nodePositions.length-1?8:7,0,Math.PI*2);
      ctx.fillStyle = li===0?'#fff':li===nodePositions.length-1?COLORS.blueSoft:COLORS.tealSoft; ctx.fill(); ctx.strokeStyle = li===nodePositions.length-1?COLORS.blue:li===0?COLORS.lineStrong||'#abb6c0':COLORS.teal; ctx.lineWidth=1.5;ctx.stroke();
    }
    ctx.fillStyle=COLORS.muted;ctx.font='700 10px ui-sans-serif, sans-serif';ctx.textAlign='center';
    const labels = state.mode==='classification'
      ? (state.classDepth===0?['x₁, x₂','score z']:state.classDepth===1?['x₁, x₂',`${state.classWidth} ReLUs`,'score z']:['x₁, x₂',`${state.classWidth} ReLUs`,`${state.classWidth} ReLUs`,'score z'])
      : ['x',`${state.approxWidth} ReLU hinges`,'f̂(x)'];
    labels.forEach((label,i)=>ctx.fillText(label,xPositions[i],height-4));
    if (layers.some(n=>n>maxShow)) { ctx.fillStyle=COLORS.orange;ctx.font='700 10px ui-monospace, monospace'; layers.forEach((n,i)=>{if(n>maxShow)ctx.fillText(`+${n-maxShow}`,xPositions[i]+23,height/2+3);}); }
    els.network.setAttribute('aria-label', labels.join(' to '));
  }

  function drawFeatureGrid(force = false) {
    if (!force && state.running && state.renderTick % 4 !== 0) return;
    els.featureGrid.replaceChildren();
    if (state.mode === 'classification' && state.classDepth === 0) {
      const empty = document.createElement('div'); empty.className='empty-feature'; empty.textContent='A linear model has no hidden ReLU features.'; els.featureGrid.append(empty); return;
    }
    const count = state.mode === 'classification' ? state.classWidth : state.approxWidth;
    const layer = state.mode === 'classification' ? state.hiddenLayer : 0;
    for (let j=0;j<count;j++) {
      const card=document.createElement('article');card.className='feature-card';
      const head=document.createElement('header'), strong=document.createElement('strong'), meta=document.createElement('span');
      strong.textContent=`ReLU ${j+1}`;
      meta.textContent=state.mode==='classification'?(layer===0?'layer 1':'layer 2'):(state.model.method==='construct'?'fixed knot':'learned');
      head.append(strong,meta);
      const canvas=document.createElement('canvas');canvas.width=160;canvas.height=160;canvas.setAttribute('role','img');
      const equation=document.createElement('p');
      card.append(head,canvas,equation);els.featureGrid.append(card);
      if(state.mode==='classification') drawClassFeature(canvas,j,layer,equation); else drawApproxFeature(canvas,j,equation);
    }
  }

  function drawClassFeature(canvas,index,layer,equation) {
    const {ctx,width,height}=setupCanvas(canvas,120);ctx.clearRect(0,0,width,height);
    const n=32,vals=[];let max=1e-9;
    for(let iy=0;iy<n;iy++)for(let ix=0;ix<n;ix++){
      const f=forwardClass(state.model,(ix+.5)/n,1-(iy+.5)/n),v=layer===0?f.h1[index]:f.h2[index];vals.push(v);max=Math.max(max,v);
    }
    for(let iy=0;iy<n;iy++)for(let ix=0;ix<n;ix++){const v=vals[iy*n+ix]/max;ctx.fillStyle=blend('#ffffff',COLORS.teal,.08+.82*v);ctx.fillRect(ix*width/n,iy*height/n,width/n+1,height/n+1);}
    ctx.strokeStyle=COLORS.line;ctx.strokeRect(.5,.5,width-1,height-1);
    if(layer===0){const w=state.model.W1[index],b=state.model.b1[index];equation.textContent=`ReLU(${w[0].toFixed(2)}x₁ ${w[1]>=0?'+':'−'} ${Math.abs(w[1]).toFixed(2)}x₂ ${b>=0?'+':'−'} ${Math.abs(b).toFixed(2)})`;}
    else equation.textContent=`ReLU(weighted sum of layer 1)`;
    canvas.setAttribute('aria-label',`Activation map for hidden ReLU ${index+1} in layer ${layer+1}.`);
  }
  function drawApproxFeature(canvas,index,equation) {
    const {ctx,width,height}=setupCanvas(canvas,120);ctx.clearRect(0,0,width,height);
    const w=state.model.W[index],b=state.model.b[index],v=state.model.v[index];
    let max=1e-9;for(let i=0;i<=100;i++)max=Math.max(max,relu(w*(-1+2*i/100)+b));
    const px=x=>(x+1)*width/2,py=y=>height-8-(height-18)*y/max;
    ctx.strokeStyle=COLORS.grid;ctx.beginPath();ctx.moveTo(0,height-8);ctx.lineTo(width,height-8);ctx.stroke();
    ctx.strokeStyle=COLORS.teal;ctx.lineWidth=2;ctx.beginPath();for(let i=0;i<=100;i++){const x=-1+2*i/100,y=relu(w*x+b);if(i)ctx.lineTo(px(x),py(y));else ctx.moveTo(px(x),py(y));}ctx.stroke();
    const hinge=Math.abs(w)>1e-8?-b/w:NaN;if(hinge>=-1&&hinge<=1){ctx.fillStyle=COLORS.orange;ctx.beginPath();ctx.arc(px(hinge),height-8,3.5,0,Math.PI*2);ctx.fill();}
    ctx.strokeStyle=COLORS.line;ctx.strokeRect(.5,.5,width-1,height-1);
    equation.textContent=`${v.toFixed(2)} · ReLU(${w.toFixed(2)}x ${b>=0?'+':'−'} ${Math.abs(b).toFixed(2)})`;
    canvas.setAttribute('aria-label',`One-dimensional activation of ReLU ${index+1}.`);
  }

  function drawSurface() {
    const {ctx,width,height}=setupCanvas(els.surface,250);ctx.clearRect(0,0,width,height);ctx.fillStyle=COLORS.paper;ctx.fillRect(0,0,width,height);
    const n=17,vals=[];let maxAbs=.4;
    for(let iy=0;iy<n;iy++){vals[iy]=[];for(let ix=0;ix<n;ix++){const z=forwardClass(state.model,ix/(n-1),iy/(n-1)).z;vals[iy][ix]=z;maxAbs=Math.max(maxAbs,Math.abs(z));}}
    const sx=Math.min(width*.42,210),sy=Math.min(height*.19,54),sz=Math.min(height*.27,78)/maxAbs,cx=width*.5,cy=height*.58;
    const proj=(x,y,z)=>[cx+(x-y)*sx,cy+(x+y-1)*sy-z*sz];
    const plane=[proj(0,0,0),proj(1,0,0),proj(1,1,0),proj(0,1,0)];ctx.fillStyle='rgba(213,107,45,.10)';ctx.strokeStyle='rgba(213,107,45,.55)';ctx.beginPath();plane.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill();ctx.stroke();
    for(let iy=0;iy<n;iy++){ctx.beginPath();for(let ix=0;ix<n;ix++){const p=proj(ix/(n-1),iy/(n-1),vals[iy][ix]);ix?ctx.lineTo(...p):ctx.moveTo(...p);}ctx.strokeStyle=iy%2?blend(COLORS.orange,COLORS.blue,.45):COLORS.blue;ctx.globalAlpha=.72;ctx.lineWidth=.8;ctx.stroke();}
    for(let ix=0;ix<n;ix++){ctx.beginPath();for(let iy=0;iy<n;iy++){const p=proj(ix/(n-1),iy/(n-1),vals[iy][ix]);iy?ctx.lineTo(...p):ctx.moveTo(...p);}ctx.strokeStyle=COLORS.ink;ctx.globalAlpha=.34;ctx.lineWidth=.7;ctx.stroke();}ctx.globalAlpha=1;
    const x=Number(els.probeX.value),y=Number(els.probeY.value),f=forwardClass(state.model,x,y),base=proj(x,y,0),tip=proj(x,y,f.z);
    ctx.strokeStyle=COLORS.red;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...base);ctx.lineTo(...tip);ctx.stroke();ctx.fillStyle=COLORS.red;ctx.beginPath();ctx.arc(...tip,4,0,Math.PI*2);ctx.fill();
    ctx.fillStyle=COLORS.muted;ctx.font='700 10px ui-sans-serif, sans-serif';ctx.textAlign='center';ctx.fillText('x₁',width-29,height-12);ctx.fillText('x₂',29,height-12);ctx.fillStyle=COLORS.orange;ctx.fillText('z = 0 plane',width-55,18);
    els.probeXOutput.value=x.toFixed(2);els.probeYOutput.value=y.toFixed(2);els.probeReadout.textContent=`z = ${f.z.toFixed(2)} · p = ${f.p.toFixed(2)} · class ${f.p>=.5?1:0}`;
    els.surface.setAttribute('aria-label',`Three-dimensional logit surface. Probe at x one ${x.toFixed(2)}, x two ${y.toFixed(2)} has score ${f.z.toFixed(2)}.`);
  }

  function drawResidual() {
    const {ctx,width,height}=setupCanvas(els.residual,250);ctx.clearRect(0,0,width,height);
    let max=.05;for(let i=0;i<=300;i++){const x=-1+2*i/300;max=Math.max(max,Math.abs(targetAt(x)-forwardApprox(state.model,x).y));}
    const frame=plotFrame(ctx,width,height,-1,1,0,max*1.08,'x','gap');
    ctx.save();ctx.beginPath();ctx.rect(frame.m.left,frame.m.top,frame.pw,frame.ph);ctx.clip();
    ctx.fillStyle='rgba(213,107,45,.20)';ctx.beginPath();ctx.moveTo(frame.px(-1),frame.py(0));for(let i=0;i<=300;i++){const x=-1+2*i/300,g=Math.abs(targetAt(x)-forwardApprox(state.model,x).y);ctx.lineTo(frame.px(x),frame.py(g));}ctx.lineTo(frame.px(1),frame.py(0));ctx.closePath();ctx.fill();
    ctx.strokeStyle=COLORS.orange;ctx.lineWidth=2;ctx.beginPath();for(let i=0;i<=300;i++){const x=-1+2*i/300,g=Math.abs(targetAt(x)-forwardApprox(state.model,x).y);i?ctx.lineTo(frame.px(x),frame.py(g)):ctx.moveTo(frame.px(x),frame.py(g));}ctx.stroke();ctx.restore();
    els.residual.setAttribute('aria-label',`Pointwise absolute approximation error, with maximum ${max.toFixed(3)}.`);
  }

  function updateText() {
    els.parameterCount.textContent=`${parameterCount()} parameters`;
    els.stepsValue.textContent=String(state.steps);
    els.status.textContent=state.statusMessage || '';
    if(state.mode==='classification'){
      const m=classMetrics();els.metric1Label.textContent='Data accuracy';els.metric1Value.textContent=pct(m.accuracy);
      els.metric2Label.textContent=state.dataset==='xor4'?'Filled XOR agreement':'Field agreement';els.metric2Value.textContent=pct(m.field);
      els.metric3Label.textContent='Cross-entropy';els.metric3Value.textContent=Number.isFinite(m.loss)?m.loss.toFixed(3):'—';
      els.workspaceEyebrow.textContent='INPUT SPACE';
      els.workspaceTitle.textContent=state.dataset==='xor4'?'Four points do not specify four regions':`${datasetName(state.dataset)} · learned decision field`;
      els.networkTitle.textContent=state.classDepth===0?'Two inputs → one score':state.classDepth===1?`Two inputs → ${state.classWidth} ReLUs → one score`:`Two inputs → ${state.classWidth} + ${state.classWidth} ReLUs → one score`;
    }else{
      const m=approxMetrics();els.metric1Label.textContent='Mean squared error';els.metric1Value.textContent=m.mse.toFixed(4);
      els.metric2Label.textContent='Largest gap';els.metric2Value.textContent=m.maxGap.toFixed(3);
      els.metric3Label.textContent=state.approxMethod==='construct'?'Segments':'Training loss';els.metric3Value.textContent=state.approxMethod==='construct'?String(state.approxWidth):m.mse.toFixed(4);
      els.workspaceEyebrow.textContent='FUNCTION ON A COMPACT INTERVAL';
      els.workspaceTitle.textContent=state.approxMethod==='construct'?`${targetName(state.target)} · fixed-knot construction`:`${targetName(state.target)} · trained ReLU approximation`;
      els.networkTitle.textContent=`One input → ${state.approxWidth} ReLU hinges → one output`;
    }
  }

  function render(force=false){state.renderTick++;syncControls();updateText();if(state.mode==='classification'){drawMainClassification();drawSurface();}else{drawMainApproximation();drawResidual();}drawNetwork();drawFeatureGrid(force);}

  function step(count=1){if(state.mode==='approximation'&&state.approxMethod==='construct')return;for(let i=0;i<count;i++){state.mode==='classification'?trainClassStep():trainApproxStep();}state.statusMessage=`${count===1?'One':count} gradient ${count===1?'step':'steps'} completed.`;render(false);}
  function animationFrame(time){if(!state.running)return;if(time-state.lastFrame>30){step(4);state.lastFrame=time;}requestAnimationFrame(animationFrame);}
  function startRunning(){if(state.running||(state.mode==='approximation'&&state.approxMethod==='construct'))return;state.running=true;els.run.classList.add('is-running');els.run.textContent='Pause';els.run.setAttribute('aria-pressed','true');state.statusMessage='Training…';requestAnimationFrame(animationFrame);}
  function stopRunning(){state.running=false;els.run.classList.remove('is-running');els.run.textContent='Run';els.run.setAttribute('aria-pressed','false');}
  function toggleRunning(){state.running?stopRunning():startRunning();render(false);}

  function eventToPlot(event){const rect=els.main.getBoundingClientRect(),frame=state.mainTransform;if(!frame)return null;const sx=(event.clientX-rect.left)*frame.width/rect.width,sy=(event.clientY-rect.top)*frame.height/rect.height;const x=frame.xMin+(sx-frame.m.left)/frame.pw*(frame.xMax-frame.xMin),y=frame.yMax-(sy-frame.m.top)/frame.ph*(frame.yMax-frame.yMin);return{x:clamp(x,frame.xMin,frame.xMax),y:clamp(y,frame.yMin,frame.yMax),inside:sx>=frame.m.left&&sx<=frame.m.left+frame.pw&&sy>=frame.m.top&&sy<=frame.m.top+frame.ph};}
  function handleMainPointer(event){const p=eventToPlot(event);if(!p||!p.inside)return;if(state.mode==='classification'&&state.dataset==='custom'){
      if(event.type==='pointerdown'){state.customPoints.push({x:p.x,y:p.y,label:state.pointClass});state.data=state.customPoints.map(q=>({...q}));state.statusMessage=`Added a class ${state.pointClass} point.`;render(true);}
    }else if(state.mode==='approximation'&&state.target==='custom'&&state.drawActive){
      ensureCustomTarget();const index=Math.round((p.x+1)*100),value=clamp(p.y,-1.6,1.6);
      if(state.lastDrawIndex===null)state.customTarget[index]=value;else{const a=state.lastDrawIndex,b=index,start=state.customTarget[a];for(let k=Math.min(a,b);k<=Math.max(a,b);k++){const t=a===b?1:(k-a)/(b-a);state.customTarget[k]=lerp(start,value,t);}}
      state.lastDrawIndex=index;if(state.approxMethod==='construct')state.model=makeConstructedApprox(state.approxWidth);state.steps=0;state.statusMessage='Custom target updated.';render(true);
    }}

  document.querySelectorAll('[data-mode]').forEach(btn=>btn.addEventListener('click',()=>setMode(btn.dataset.mode)));
  els.dataset.addEventListener('change',()=>setDataset(els.dataset.value));els.target.addEventListener('change',()=>setTarget(els.target.value));
  els.depth.addEventListener('change',()=>{state.classDepth=Number(els.depth.value);state.hiddenLayer=0;resetModel();});
  els.width.addEventListener('input',()=>{const value=Number(els.width.value);if(state.mode==='classification')state.classWidth=value;else state.approxWidth=value;els.widthOutput.value=value;});
  els.width.addEventListener('change',()=>resetModel(state.mode==='classification'?'Width changed; random weights loaded.':state.approxMethod==='construct'?'Fixed-knot construction rebuilt.':'Width changed; random weights loaded.'));
  els.lr.addEventListener('change',()=>{state.learningRate=Number(els.lr.value);state.statusMessage=`Learning rate set to ${state.learningRate}.`;render(false);});
  els.seed.addEventListener('change',()=>{state.seed=clamp(Math.round(Number(els.seed.value)||11),1,9999);if(state.mode==='classification')state.data=makeDataset(state.dataset,state.seed);resetModel('Seed changed; data and weights reset.');});
  els.newSeed.addEventListener('click',()=>{state.seed=state.seed%9999+1;els.seed.value=state.seed;if(state.mode==='classification')state.data=makeDataset(state.dataset,state.seed);resetModel('New deterministic seed loaded.');});
  els.run.addEventListener('click',toggleRunning);els.step.addEventListener('click',()=>step(1));els.reset.addEventListener('click',()=>resetModel());
  els.cornerProof.addEventListener('click',loadCornerRule);els.fieldProof.addEventListener('click',loadFieldRule);
  document.querySelectorAll('[data-approx-method]').forEach(btn=>btn.addEventListener('click',()=>{state.approxMethod=btn.dataset.approxMethod;resetModel(state.approxMethod==='construct'?'Fixed-knot construction loaded.':'Random weights loaded for training.');}));
  document.querySelectorAll('[data-segments]').forEach(btn=>btn.addEventListener('click',()=>{state.approxWidth=Number(btn.dataset.segments);state.approxMethod='construct';resetModel();}));
  document.querySelectorAll('[data-point-class]').forEach(btn=>btn.addEventListener('click',()=>{state.pointClass=Number(btn.dataset.pointClass);syncControls();}));
  els.clearPoints.addEventListener('click',()=>{state.customPoints=[];state.data=[];resetModel('Custom points cleared. Click the plot to add data.');});
  document.querySelectorAll('[data-class-view]').forEach(btn=>btn.addEventListener('click',()=>{state.classView=btn.dataset.classView;render(false);}));
  document.querySelectorAll('[data-hidden-layer]').forEach(btn=>btn.addEventListener('click',()=>{if(btn.disabled)return;state.hiddenLayer=Number(btn.dataset.hiddenLayer);render(true);}));
  [els.probeX,els.probeY].forEach(input=>input.addEventListener('input',()=>drawSurface()));
  els.main.addEventListener('pointerdown',event=>{state.drawActive=true;state.lastDrawIndex=null;els.main.setPointerCapture?.(event.pointerId);handleMainPointer(event);});
  els.main.addEventListener('pointermove',event=>{if(state.drawActive)handleMainPointer(event);});
  const endDraw=()=>{state.drawActive=false;state.lastDrawIndex=null;};els.main.addEventListener('pointerup',endDraw);els.main.addEventListener('pointercancel',endDraw);
  window.addEventListener('resize',()=>render(true));

  function snapshot(){
    if(state.mode==='classification'){
      const m=classMetrics();return{mode:state.mode,dataset:state.dataset,dataSize:state.data.length,depth:state.classDepth,width:state.classWidth,parameters:parameterCount(),accuracy:m.accuracy,fieldAgreement:m.field,loss:m.loss,steps:state.steps,featureCount:state.classDepth?state.classWidth:0,dataSignature:state.data.map(d=>`${d.x.toFixed(4)},${d.y.toFixed(4)},${d.label}`).join('|')};
    }
    const m=approxMetrics();return{mode:state.mode,target:state.target,approxMethod:state.approxMethod,width:state.approxWidth,segments:state.approxMethod==='construct'?state.approxWidth:null,parameters:parameterCount(),mse:m.mse,maxGap:m.maxGap,steps:state.steps,featureCount:state.approxWidth};
  }
  window.ReLULab=Object.freeze({
    snapshot,setMode,setDataset,setTarget,loadCornerRule,loadFieldRule,step,
    setSeed(seed){state.seed=clamp(Math.round(seed),1,9999);if(state.mode==='classification')state.data=makeDataset(state.dataset,state.seed);resetModel();},
    setArchitecture({depth=state.classDepth,width=state.classWidth}={}){state.classDepth=Number(depth);state.classWidth=Number(width);resetModel();},
    setApproximation({method=state.approxMethod,width=state.approxWidth,target=state.target}={}){state.approxMethod=method;state.approxWidth=Number(width);state.target=target;if(target==='custom')ensureCustomTarget();resetModel();},
    addCustomPoint(x,y,label){state.dataset='custom';state.customPoints.push({x:clamp(x,0,1),y:clamp(y,0,1),label:Number(label)?1:0});state.data=makeDataset('custom',state.seed);render(true);},
    clearCustomPoints(){state.customPoints=[];state.dataset='custom';state.data=[];resetModel();}
  });

  state.data=makeDataset(state.dataset,state.seed);loadCornerRule();
})();
