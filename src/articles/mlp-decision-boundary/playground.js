(() => {
  'use strict';

  const C = {
    ink: '#1f2933', muted: '#637080', line: '#d7dde2', lineStrong: '#abb6c0',
    blue: '#2463a7', blueSoft: '#e5eef8', orange: '#d56b2d', orangeSoft: '#f8e9df',
    teal: '#21766f', tealSoft: '#e1f0ed', red: '#b84843', paper: '#ffffff', canvas: '#f4f6f7'
  };
  const DATASETS = new Set(['xor4', 'xorField', 'blobs', 'circles', 'moons', 'spirals', 'custom']);
  const VIEWS = new Set(['probability', 'logit', 'class']);
  const GRID = 49;
  const CAMERA_PRESETS = {
    iso: { azimuth: -Math.PI / 4, elevation: 0.58, zoom: 1 },
    top: { azimuth: 0, elevation: Math.PI / 2 - 0.015, zoom: 1 },
    front: { azimuth: -Math.PI / 2, elevation: 0.07, zoom: 1.04 }
  };
  const $ = (id) => document.getElementById(id);
  const layout = {
    grid: document.querySelector('.lab-grid'), controls: document.querySelector('.controls'),
    workspace: document.querySelector('.workspace'), primaryVisual: document.querySelector('.primary-visual')
  };
  const el = {
    dataset: $('datasetSelect'), customTools: $('customPointTools'), clear: $('clearPointsBtn'), undo: $('undoPointBtn'),
    customCount: $('customCount'), datasetNote: $('datasetNote'), depth: $('depthSelect'), width: $('widthRange'),
    widthOutput: $('widthOutput'), lr: $('learningRateSelect'), dataSeed: $('dataSeedInput'), newDataSeed: $('newDataSeedBtn'),
    weightSeed: $('weightSeedInput'), newWeightSeed: $('newWeightSeedBtn'),
    recommended: $('recommendedSetupBtn'), recommendedText: $('recommendedSetupText'), random: $('randomInitBtn'),
    corner: $('cornerProofBtn'), field: $('fieldProofBtn'), run: $('runBtn'),
    step: $('stepBtn'), reset: $('resetBtn'), status: $('statusText'), main: $('mainCanvas'), network: $('networkCanvas'),
    history: $('historyCanvas'), surface: $('surfaceCanvas'), features: $('featureGrid'), metric1Label: $('metric1Label'),
    metric1: $('metric1Value'), metric2Label: $('metric2Label'), metric2: $('metric2Value'),
    metric3Label: $('metric3Label'), metric3: $('metric3Value'), steps: $('stepsValue'), title: $('workspaceTitle'),
    params: $('parameterCount'), networkTitle: $('networkTitle'), provenance: $('provenanceBadge'),
    probeX: $('probeX'), probeY: $('probeY'), probeXOut: $('probeXOutput'), probeYOut: $('probeYOutput'),
    probeReadout: $('probeReadout'), surfaceReset: $('surfaceResetBtn'), hiddenSwitch: $('hiddenLayerSwitch')
  };

  const state = {
    dataset: 'xor4', customPoints: [], undo: [], pointClass: 0, data: [], dataSeed: 11, weightSeed: 11, depth: 1, width: 2,
    lr: 0.03, model: null, provenance: 'constructed', initializer: 'corner', steps: 0, history: [], view: 'logit', running: false,
    raf: null, lastFrame: 0, revision: 0, grid: null, metrics: null, renderCount: 0, mainFrame: null,
    probe: { x: 0.25, y: 0.75 }, camera: { ...CAMERA_PRESETS.iso }, drag: null, status: '', featureSig: ''
  };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const finite = (v, fallback = 0) => Number.isFinite(v) ? v : fallback;
  const relu = (v) => Math.max(0, v);
  const text = (node, value) => { if (node) node.textContent = String(value); };
  const hidden = (node, value) => { if (node) node.hidden = Boolean(value); };
  const pct = (v) => Number.isFinite(v) ? `${(v * 100).toFixed(v > 0.995 ? 1 : 0)}%` : '—';
  const signed = (v) => `${v < 0 ? '−' : '+'} ${Math.abs(v).toFixed(2)}`;
  const DOMAIN_ANCHORS = Array.from({ length: 25 }, (_, i) => ({ x: 0.05 + (i % 5) * 0.225, y: 0.05 + Math.floor(i / 5) * 0.225 }));
  const RECOMMENDED_SETUPS = Object.freeze({
    xor4: Object.freeze({ depth: 1, width: 8, lr: 0.1, weightSeed: 23 }),
    xorField: Object.freeze({ depth: 2, width: 8, lr: 0.1, weightSeed: 23 }),
    blobs: Object.freeze({ depth: 0, width: 1, lr: 0.1, weightSeed: 23 }),
    circles: Object.freeze({ depth: 2, width: 8, lr: 0.1, weightSeed: 23 }),
    moons: Object.freeze({ depth: 2, width: 8, lr: 0.1, weightSeed: 23 }),
    spirals: Object.freeze({ depth: 2, width: 16, lr: 0.1, weightSeed: 23 }),
    custom: Object.freeze({ depth: 1, width: 8, lr: 0.03, weightSeed: 23 })
  });

  function rngFor(seed) {
    let a = seed >>> 0;
    return () => {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function gaussian(rng) {
    const u = Math.max(1e-12, rng());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z); return e / (1 + e);
  }
  function bceFromLogit(z, label) {
    return Math.max(z, 0) - z * label + Math.log1p(Math.exp(-Math.abs(z)));
  }
  function blend(a, b, t) {
    const parse = (x) => [1, 3, 5].map((i) => parseInt(x.slice(i, i + 2), 16));
    const x = parse(a), y = parse(b), q = clamp(t, 0, 1);
    return `#${[0, 1, 2].map((i) => Math.round(x[i] + (y[i] - x[i]) * q).toString(16).padStart(2, '0')).join('')}`;
  }
  function datasetName(key) {
    return ({ xor4: 'Four-point XOR', xorField: 'Filled XOR regions', blobs: 'Two diagonal blobs', circles: 'Concentric circles', moons: 'Two interlocking moons', spirals: 'Two spirals', custom: 'Custom points' })[key] || key;
  }
  function datasetNote(key) {
    return ({
      xor4: 'Only the four Boolean corners are labelled. Many fields can fit them.',
      xorField: 'Samples fill all four XOR regions, outside a narrow ambiguous band.',
      blobs: 'A balanced, nearly linear classification task.', circles: 'The positive class sits inside a closed ring.',
      moons: 'Two curved classes need several learned hinges.', spirals: 'A hard task suited to a deeper, wider model.',
      custom: 'Click to add. Shift-click a nearby point to remove it. Seed changes preserve every point.'
    })[key] || '';
  }
  function truth(key, x, y) {
    if (key === 'xor4' || key === 'xorField') return (x > 0.5) !== (y > 0.5) ? 1 : 0;
    if (key === 'blobs') return x + y >= 1 ? 1 : 0;
    if (key === 'circles') return Math.hypot(x - 0.5, y - 0.5) < 0.25 ? 1 : 0;
    if (key === 'moons') {
      const d0 = Math.abs(Math.hypot(x - 0.36, y - 0.48) - 0.28) + Math.max(0, 0.48 - y) * 1.6;
      const d1 = Math.abs(Math.hypot(x - 0.64, y - 0.52) - 0.28) + Math.max(0, y - 0.52) * 1.6;
      return d1 < d0 ? 1 : 0;
    }
    return null;
  }
  function makeDataset(key, seed, custom) {
    if (key === 'custom') return custom.map((p) => ({ ...p }));
    if (key === 'xor4') return [{ x: 0, y: 0, label: 0 }, { x: 0, y: 1, label: 1 }, { x: 1, y: 0, label: 1 }, { x: 1, y: 1, label: 0 }];
    const rng = rngFor(seed * 997 + 17), out = [];
    if (key === 'xorField') {
      // Reflect each lower-left sample into all four quadrants. The resulting
      // evidence is exactly balanced and symmetric about both XOR boundaries.
      for (let i = 0; i < 64; i++) {
        const x = 0.475 * rng(), y = 0.475 * rng();
        out.push({ x, y, label: 0 }, { x, y: 1 - y, label: 1 }, { x: 1 - x, y, label: 1 }, { x: 1 - x, y: 1 - y, label: 0 });
      }
    } else if (key === 'blobs') {
      for (let label = 0; label < 2; label++) {
        while (out.filter((p) => p.label === label).length < 110) {
          const x = clamp((label ? 0.7 : 0.3) + 0.115 * gaussian(rng), 0.015, 0.985);
          const y = clamp((label ? 0.68 : 0.32) + 0.12 * gaussian(rng), 0.015, 0.985);
          if (truth(key, x, y) === label) out.push({ x, y, label });
        }
      }
    } else if (key === 'circles') {
      for (let label = 0; label < 2; label++) for (let i = 0; i < 110;) {
        const a = Math.PI * 2 * rng(), r = (label ? 0.17 : 0.37) + 0.023 * gaussian(rng);
        const x = clamp(0.5 + r * Math.cos(a), 0.01, 0.99), y = clamp(0.5 + r * Math.sin(a), 0.01, 0.99);
        if (truth(key, x, y) === label) { out.push({ x, y, label }); i++; }
      }
    } else if (key === 'moons') {
      for (let label = 0; label < 2; label++) for (let i = 0; i < 110;) {
        const a = label ? Math.PI + Math.PI * rng() : Math.PI * rng();
        const x = clamp((label ? 0.64 : 0.36) + 0.28 * Math.cos(a) + 0.018 * gaussian(rng), 0.01, 0.99);
        const y = clamp((label ? 0.52 : 0.48) + 0.28 * Math.sin(a) + 0.018 * gaussian(rng), 0.01, 0.99);
        if (truth(key, x, y) === label) { out.push({ x, y, label }); i++; }
      }
    } else if (key === 'spirals') {
      for (let label = 0; label < 2; label++) for (let i = 0; i < 110; i++) {
        const q = i / 109, r = 0.035 + 0.43 * q, a = 0.35 + q * 2.55 * Math.PI + label * Math.PI + 0.055 * gaussian(rng);
        out.push({ x: clamp(0.5 + r * Math.cos(a) + 0.008 * gaussian(rng), 0.01, 0.99), y: clamp(0.5 + r * Math.sin(a) + 0.008 * gaussian(rng), 0.01, 0.99), label });
      }
    }
    return out;
  }

  const validSeed = (v) => clamp(Math.round(finite(Number(v), 11)), 1, 9999);
  const validDepth = (v) => clamp(Math.round(finite(Number(v), 1)), 0, 2);
  const validWidth = (v) => clamp(Math.round(finite(Number(v), 2)), 1, 16);
  function median(values) {
    const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }
  function randomModel(depth, width, seed) {
    const rng = rngFor(seed * 7919 + depth * 97 + width * 13);
    const rw = (fan) => gaussian(rng) * Math.sqrt(2 / Math.max(1, fan));
    const m = { depth, width, W1: [], b1: [], W2: [], b2: [], Wo: [], bo: 0 };
    if (!depth) { m.Wo = [rw(2), rw(2)]; return m; }
    m.W1 = Array.from({ length: width }, () => [rw(2), rw(2)]);
    // Place each first-layer hinge through the input domain instead of near
    // the origin. Every initialized ReLU is active for some anchor points and
    // inactive for others, so it can receive a useful gradient immediately.
    m.b1 = m.W1.map((w) => -median(DOMAIN_ANCHORS.map((p) => w[0] * p.x + w[1] * p.y)));
    if (depth === 2) {
      m.W2 = Array.from({ length: width }, () => Array.from({ length: width }, () => rw(width)));
      const h1Anchors = DOMAIN_ANCHORS.map((p) => m.W1.map((w, j) => relu(w[0] * p.x + w[1] * p.y + m.b1[j])));
      m.b2 = m.W2.map((row) => -median(h1Anchors.map((h) => row.reduce((sum, w, j) => sum + w * h[j], 0))));
    }
    m.Wo = Array.from({ length: width }, () => rw(width));
    return m;
  }
  const cornerModel = () => ({ depth: 1, width: 2, W1: [[1, 1], [1, 1]], b1: [0, -1], W2: [], b2: [], Wo: [2, -4], bo: -1 });
  const fieldModel = () => ({ depth: 1, width: 4, W1: [[1, -1], [-1, 1], [1, 1], [-1, -1]], b1: [0, 0, -1, 1], W2: [], b2: [], Wo: [1.7, 1.7, -1.7, -1.7], bo: 0 });
  function forward(m, x, y) {
    if (!m.depth) { const z = m.Wo[0] * x + m.Wo[1] * y + m.bo; return { u1: [], h1: [], u2: [], h2: [], final: [x, y], z, p: sigmoid(z) }; }
    const u1 = m.W1.map((w, j) => w[0] * x + w[1] * y + m.b1[j]), h1 = u1.map(relu);
    let u2 = [], h2 = [], final = h1;
    if (m.depth === 2) { u2 = m.W2.map((row, j) => row.reduce((s, w, k) => s + w * h1[k], m.b2[j])); h2 = u2.map(relu); final = h2; }
    const z = m.Wo.reduce((s, w, j) => s + w * final[j], m.bo);
    return { u1, h1, u2, h2, final, z, p: sigmoid(z) };
  }
  function dataLoss() {
    if (!state.data.length) return NaN;
    return state.data.reduce((sum, p) => sum + bceFromLogit(forward(state.model, p.x, p.y).z, p.label), 0) / state.data.length;
  }
  function changed() { state.revision++; state.grid = null; state.metrics = null; }
  function stop() {
    state.running = false;
    if (state.raf !== null) { cancelAnimationFrame(state.raf); state.raf = null; }
    state.lastFrame = 0; syncRun(); return true;
  }
  function commit(model, provenance, message) {
    stop(); state.model = model; state.depth = model.depth; state.width = model.width; state.provenance = provenance; state.steps = 0; changed();
    const loss = dataLoss(); state.history = Number.isFinite(loss) ? [{ step: 0, loss }] : []; state.status = state.data.length ? message : `${message} Add at least one custom point before training.`; render(true);
  }
  function constructionMatchesDataset() {
    return (state.initializer === 'corner' && state.dataset === 'xor4') || (state.initializer === 'field' && state.dataset === 'xorField');
  }
  function reconcileProvenanceAfterDataChange() {
    if (state.initializer === 'corner' || state.initializer === 'field') state.provenance = constructionMatchesDataset() ? 'constructed' : 'carried';
    else if (state.provenance === 'trained' || state.provenance === 'carried') state.provenance = 'carried';
  }
  function dataChanged(message) {
    stop(); reconcileProvenanceAfterDataChange(); state.steps = 0; changed(); const loss = dataLoss(); state.history = Number.isFinite(loss) ? [{ step: 0, loss }] : [];
    state.status = state.provenance === 'carried' ? `${message} Choose Recommended or Random before training.` : message; render(true); return snapshot();
  }
  function loadCornerRule() {
    state.initializer = 'corner';
    const matches = state.dataset === 'xor4';
    commit(cornerModel(), matches ? 'constructed' : 'carried', matches ? 'Loaded the exact two-ReLU corner construction for inspection. Choose Recommended or Random before training.' : 'Loaded the corner construction on different evidence for comparison. Choose Recommended or Random before training.'); return snapshot();
  }
  function loadFieldRule() {
    state.initializer = 'field';
    const matches = state.dataset === 'xorField';
    commit(fieldModel(), matches ? 'constructed' : 'carried', matches ? 'Loaded the exact four-ReLU XOR field construction for inspection. Choose Recommended or Random before training.' : 'Loaded the XOR field construction on different evidence for comparison. Choose Recommended or Random before training.'); return snapshot();
  }
  function recommendationFor(key = state.dataset) {
    if (!DATASETS.has(key)) throw new RangeError(`Unknown dataset: ${key}`);
    return { ...RECOMMENDED_SETUPS[key] };
  }
  function setupDescription(setup) {
    const architecture = setup.depth ? `${setup.depth} ${setup.depth === 1 ? 'layer' : 'layers'} · width ${setup.width}` : 'linear model';
    return `${architecture} · learning rate ${setup.lr} · weight seed ${setup.weightSeed}`;
  }
  function loadRecommendedSetup() {
    const setup = recommendationFor();
    state.lr = setup.lr; state.weightSeed = setup.weightSeed; state.initializer = 'random';
    commit(randomModel(setup.depth, setup.width, state.weightSeed), 'random', `Recommended random start loaded for ${datasetName(state.dataset)}: ${setupDescription(setup)}.`);
    return snapshot();
  }
  function initialize(options = 'random') {
    let o = typeof options === 'string' ? { provenance: options } : (options || {}), kind = o.provenance || o.preset || 'random';
    if (kind === 'recommended') return loadRecommendedSetup();
    if (kind === 'corner' || kind === 'cornerRule') return loadCornerRule();
    if (kind === 'field' || kind === 'fieldRule') return loadFieldRule();
    if (kind === 'constructed') return state.dataset === 'xor4' ? loadCornerRule() : loadFieldRule();
    if (o.dataset !== undefined) { if (!DATASETS.has(o.dataset)) throw new RangeError(`Unknown dataset: ${o.dataset}`); state.dataset = o.dataset; }
    if (o.dataSeed !== undefined) state.dataSeed = validSeed(o.dataSeed);
    if (o.weightSeed !== undefined || o.seed !== undefined) state.weightSeed = validSeed(o.weightSeed ?? o.seed);
    if (o.depth !== undefined) state.depth = validDepth(o.depth);
    if (o.width !== undefined) state.width = validWidth(o.width);
    state.initializer = 'random'; state.data = makeDataset(state.dataset, state.dataSeed, state.customPoints);
    commit(randomModel(state.depth, state.width, state.weightSeed), kind === 'trained' ? 'trained' : 'random', 'Loaded deterministic random weights; data is unchanged.'); return snapshot();
  }
  function setDataset(key) {
    if (!DATASETS.has(key)) throw new RangeError(`Unknown dataset: ${key}`);
    state.dataset = key; state.data = makeDataset(key, state.dataSeed, state.customPoints);
    return dataChanged(key === 'custom' && !state.data.length ? 'Custom dataset is empty. Model weights are unchanged.' : `${datasetName(key)} loaded; model weights preserved.`);
  }
  function setArchitecture(depthOrOptions, maybeWidth) {
    const o = depthOrOptions && typeof depthOrOptions === 'object' ? depthOrOptions : { depth: depthOrOptions, width: maybeWidth };
    const depth = o.depth === undefined ? state.depth : validDepth(o.depth), width = o.width === undefined ? state.width : validWidth(o.width);
    state.initializer = 'random'; commit(randomModel(depth, width, state.weightSeed), 'random', depth ? `${depth} hidden ${depth === 1 ? 'layer' : 'layers'}, width ${width}; weight-seeded random initialization loaded.` : 'Linear model loaded from the weight seed.'); return snapshot();
  }
  function setDataSeed(value) {
    state.dataSeed = validSeed(value); if (state.dataset !== 'custom') state.data = makeDataset(state.dataset, state.dataSeed, state.customPoints);
    return dataChanged(state.dataset === 'custom' ? `Data seed ${state.dataSeed} recorded; custom points and model weights preserved.` : `Data seed ${state.dataSeed} regenerated data; model weights preserved.`);
  }
  function setWeightSeed(value) {
    state.weightSeed = validSeed(value); state.initializer = 'random';
    commit(randomModel(state.depth, state.width, state.weightSeed), 'random', `Weight seed ${state.weightSeed} loaded; data preserved.`); return snapshot();
  }
  function setSeed(value) {
    state.dataSeed = validSeed(value); state.weightSeed = validSeed(value);
    if (state.dataset !== 'custom') state.data = makeDataset(state.dataset, state.dataSeed, state.customPoints);
    state.initializer = 'random'; commit(randomModel(state.depth, state.width, state.weightSeed), 'random', `Compatibility seed ${state.weightSeed} applied to data and weights.`); return snapshot();
  }
  const clipG = (v) => clamp(finite(v), -12, 12), clipP = (v) => clamp(finite(v), -60, 60);
  const canTrain = () => Boolean(state.data.length) && (state.provenance === 'random' || state.provenance === 'trained');
  function explainTrainingBlock() {
    if (!state.data.length) return 'Add at least one custom point before training.';
    return state.provenance === 'constructed'
      ? 'This exact construction is for inspection. Choose Recommended or Random before training.'
      : 'These weights were carried from different evidence. Choose Recommended or Random before training.';
  }
  function trainOne() {
    const m = state.model, n = state.data.length; if (!n || !canTrain()) return false;
    if (!m.depth) {
      const gw = [0, 0]; let gb = 0;
      for (const p of state.data) { const f = forward(m, p.x, p.y), dz = f.p - p.label; gw[0] += dz * p.x; gw[1] += dz * p.y; gb += dz; }
      m.Wo[0] = clipP(m.Wo[0] - state.lr * clipG(gw[0] / n)); m.Wo[1] = clipP(m.Wo[1] - state.lr * clipG(gw[1] / n)); m.bo = clipP(m.bo - state.lr * clipG(gb / n));
    } else {
      const g1 = m.W1.map(() => [0, 0]), gb1 = m.b1.map(() => 0), g2 = m.depth === 2 ? m.W2.map((r) => r.map(() => 0)) : [], gb2 = m.depth === 2 ? m.b2.map(() => 0) : [], go = m.Wo.map(() => 0); let gbo = 0;
      for (const p of state.data) {
        const f = forward(m, p.x, p.y), dz = f.p - p.label; gbo += dz;
        for (let j = 0; j < m.width; j++) go[j] += dz * f.final[j];
        if (m.depth === 2) {
          const du2 = m.Wo.map((w, j) => f.u2[j] > 0 ? dz * w : 0);
          for (let j = 0; j < m.width; j++) { gb2[j] += du2[j]; for (let k = 0; k < m.width; k++) g2[j][k] += du2[j] * f.h1[k]; }
          for (let k = 0; k < m.width; k++) { let dh = 0; for (let j = 0; j < m.width; j++) dh += m.W2[j][k] * du2[j]; const du = f.u1[k] > 0 ? dh : 0; g1[k][0] += du * p.x; g1[k][1] += du * p.y; gb1[k] += du; }
        } else for (let j = 0; j < m.width; j++) { const du = f.u1[j] > 0 ? dz * m.Wo[j] : 0; g1[j][0] += du * p.x; g1[j][1] += du * p.y; gb1[j] += du; }
      }
      for (let j = 0; j < m.width; j++) {
        m.Wo[j] = clipP(m.Wo[j] - state.lr * clipG(go[j] / n)); m.W1[j][0] = clipP(m.W1[j][0] - state.lr * clipG(g1[j][0] / n)); m.W1[j][1] = clipP(m.W1[j][1] - state.lr * clipG(g1[j][1] / n)); m.b1[j] = clipP(m.b1[j] - state.lr * clipG(gb1[j] / n));
        if (m.depth === 2) { m.b2[j] = clipP(m.b2[j] - state.lr * clipG(gb2[j] / n)); for (let k = 0; k < m.width; k++) m.W2[j][k] = clipP(m.W2[j][k] - state.lr * clipG(g2[j][k] / n)); }
      }
      m.bo = clipP(m.bo - state.lr * clipG(gbo / n));
    }
    state.steps++; state.provenance = 'trained'; changed(); state.history.push({ step: state.steps, loss: dataLoss() }); if (state.history.length > 2400) state.history.shift(); return true;
  }
  function step(count = 1) {
    const n = clamp(Math.round(finite(Number(count), 1)), 1, 10000);
    if (!canTrain()) { state.status = explainTrainingBlock(); render(false); return snapshot(); }
    for (let i = 0; i < n; i++) trainOne(); state.status = `${n} full-batch gradient ${n === 1 ? 'step' : 'steps'} completed.`; render(false); return snapshot();
  }
  function schedule() { if (state.running && state.raf === null) state.raf = requestAnimationFrame(frame); }
  function frame(time) { state.raf = null; if (!state.running) return; if (!state.lastFrame || time - state.lastFrame > 32) { for (let i = 0; i < 4; i++) trainOne(); state.lastFrame = time; state.status = 'Training with full-batch gradient descent…'; render(false); } schedule(); }
  function start() { if (state.running) return false; if (!canTrain()) { state.status = explainTrainingBlock(); render(false); return false; } state.running = true; state.lastFrame = 0; syncRun(); schedule(); return true; }

  function pushUndo() { state.undo.push(state.customPoints.map((p) => ({ ...p }))); if (state.undo.length > 60) state.undo.shift(); }
  function customReset(message) { state.dataset = 'custom'; state.data = makeDataset('custom', state.dataSeed, state.customPoints); dataChanged(`${message} Model weights preserved.`); }
  function addCustomPoint(x, y, label = state.pointClass) { x = Number(x); y = Number(y); if (!Number.isFinite(x) || !Number.isFinite(y)) throw new TypeError('Point coordinates must be finite.'); pushUndo(); state.customPoints.push({ x: clamp(x, 0, 1), y: clamp(y, 0, 1), label: Number(label) ? 1 : 0 }); customReset('Custom point added.'); return state.customPoints.length; }
  function nearest(x, y) { let index = -1, d = Infinity; state.customPoints.forEach((p, i) => { const q = (p.x - x) ** 2 + (p.y - y) ** 2; if (q < d) { d = q; index = i; } }); return index; }
  function removeCustomPoint(value) { if (!state.customPoints.length) return false; let i = typeof value === 'number' ? Math.round(value) : value && Number.isFinite(Number(value.x)) && Number.isFinite(Number(value.y)) ? nearest(Number(value.x), Number(value.y)) : state.customPoints.length - 1; if (i < 0 || i >= state.customPoints.length) return false; pushUndo(); state.customPoints.splice(i, 1); customReset('Custom point removed.'); return true; }
  function clearCustomPoints() { if (state.customPoints.length) pushUndo(); state.customPoints = []; customReset('Custom points cleared. Add at least one custom point before training; seed changes keep this dataset empty.'); return snapshot(); }
  function undoCustomEdit() { if (!state.undo.length) return false; state.customPoints = state.undo.pop().map((p) => ({ ...p })); customReset('Previous custom dataset restored.'); return true; }

  function parameterCount() { return !state.depth ? 3 : state.depth === 1 ? 4 * state.width + 1 : state.width * state.width + 5 * state.width + 1; }
  function getGrid() {
    if (state.grid?.revision === state.revision) return state.grid;
    const z = new Float64Array(GRID * GRID); let maxAbs = 0.35, min = Infinity, max = -Infinity;
    for (let iy = 0; iy < GRID; iy++) for (let ix = 0; ix < GRID; ix++) { const v = forward(state.model, ix / (GRID - 1), iy / (GRID - 1)).z, i = iy * GRID + ix; z[i] = v; maxAbs = Math.max(maxAbs, Math.abs(v)); min = Math.min(min, v); max = Math.max(max, v); }
    state.grid = { revision: state.revision, size: GRID, z, maxAbs, min, max }; return state.grid;
  }
  function getMetrics() {
    if (state.metrics?.revision === state.revision && state.metrics.data === state.data) return state.metrics.value;
    if (!state.data.length) return { accuracy: NaN, agreement: NaN, loss: NaN };
    let right = 0, loss = 0;
    for (const p of state.data) { const f = forward(state.model, p.x, p.y); right += (f.z >= 0 ? 1 : 0) === p.label; loss += bceFromLogit(f.z, p.label); }
    let agreement = NaN;
    if (truth(state.dataset, 0.2, 0.3) !== null) {
      const g = getGrid(), totals = [0, 0], correct = [0, 0];
      for (let y = 0; y < GRID - 1; y++) for (let x = 0; x < GRID - 1; x++) {
        const px = (x + 0.5) / (GRID - 1), py = (y + 0.5) / (GRID - 1);
        if ((state.dataset === 'xor4' || state.dataset === 'xorField') && (Math.abs(px - 0.5) < 0.012 || Math.abs(py - 0.5) < 0.012)) continue;
        const label = truth(state.dataset, px, py), i = y * GRID + x;
        const z = (g.z[i] + g.z[i + 1] + g.z[i + GRID] + g.z[i + GRID + 1]) / 4;
        totals[label]++; correct[label] += (z >= 0 ? 1 : 0) === label;
      }
      agreement = (correct[0] / totals[0] + correct[1] / totals[1]) / 2;
    } else if (state.dataset === 'spirals') {
      const holdout = makeDataset('spirals', state.dataSeed + 104729, []);
      agreement = holdout.reduce((sum, p) => sum + ((forward(state.model, p.x, p.y).z >= 0 ? 1 : 0) === p.label), 0) / holdout.length;
    }
    const value = { accuracy: right / state.data.length, agreement, loss: loss / state.data.length }; state.metrics = { revision: state.revision, data: state.data, value }; return value;
  }
  function dataCounts() {
    const classCounts = [0, 0], quadrantCounts = state.dataset === 'xorField' ? [0, 0, 0, 0] : null;
    for (const p of state.data) {
      classCounts[p.label ? 1 : 0]++;
      if (quadrantCounts) quadrantCounts[(p.x >= 0.5 ? 2 : 0) + (p.y >= 0.5 ? 1 : 0)]++;
    }
    return { classCounts, quadrantCounts };
  }
  function activationDiagnostics() {
    if (!state.depth) return { deadUnits: 0, deadUnitsByLayer: [] };
    const active1 = Array.from({ length: state.width }, () => 0), active2 = Array.from({ length: state.depth === 2 ? state.width : 0 }, () => 0);
    for (const p of DOMAIN_ANCHORS) {
      const f = forward(state.model, p.x, p.y);
      f.u1.forEach((v, j) => { if (v > 0) active1[j]++; });
      f.u2.forEach((v, j) => { if (v > 0) active2[j]++; });
    }
    const deadUnitsByLayer = [active1.filter((count) => count === 0).length];
    if (state.depth === 2) deadUnitsByLayer.push(active2.filter((count) => count === 0).length);
    return { deadUnits: deadUnitsByLayer.reduce((sum, count) => sum + count, 0), deadUnitsByLayer };
  }

  function setupCanvas(canvas, minHeight = 120) {
    const rect = canvas.getBoundingClientRect(), width = Math.max(260, Math.round(rect.width || canvas.parentElement?.clientWidth || 300));
    const height = Math.max(minHeight, Math.round(rect.height || minHeight)), dpr = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx, width, height };
  }
  function diverging(value, maxAbs, strength = 0.82) {
    const amount = 0.06 + strength * clamp(Math.abs(value) / Math.max(1e-9, maxAbs), 0, 1);
    return blend('#ffffff', value >= 0 ? C.blue : C.orange, amount);
  }
  function inputFrame(ctx, width, height) {
    const m = { left: 49, right: 20, top: 17, bottom: 39 }, inset = 12;
    const left = m.left + inset, top = m.top + inset, pw = width - m.left - m.right - 2 * inset, ph = height - m.top - m.bottom - 2 * inset;
    const px = (x) => left + x * pw, py = (y) => top + (1 - y) * ph;
    ctx.fillStyle = C.paper; ctx.fillRect(m.left, m.top, width - m.left - m.right, height - m.top - m.bottom);
    ctx.strokeStyle = C.line; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) { const x = left + i * pw / 4, y = top + i * ph / 4; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.stroke(); }
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.strokeRect(left, top, pw, ph);
    ctx.fillStyle = C.muted; ctx.font = '11px ui-sans-serif, sans-serif'; ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) ctx.fillText((i / 4).toFixed(i % 4 ? 2 : 1), left + i * pw / 4, height - 17);
    ctx.textAlign = 'right'; for (let i = 0; i <= 4; i++) ctx.fillText((1 - i / 4).toFixed(i % 4 ? 2 : 1), left - 7, top + i * ph / 4 + 4);
    ctx.fillStyle = C.ink; ctx.font = '700 12px ui-sans-serif, sans-serif'; ctx.textAlign = 'center'; ctx.fillText('x₁', left + pw / 2, height - 2);
    ctx.save(); ctx.translate(13, top + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('x₂', 0, 0); ctx.restore();
    return { left, top, pw, ph, px, py, width, height };
  }
  function contour(ctx, values, size, map, color = C.ink, lineWidth = 1.5, dash = []) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.setLineDash(dash); ctx.beginPath();
    for (let iy = 0; iy < size - 1; iy++) for (let ix = 0; ix < size - 1; ix++) {
      const a0 = iy * size + ix, vals = [values[a0], values[a0 + 1], values[a0 + size + 1], values[a0 + size]];
      const corners = [[ix / (size - 1), iy / (size - 1)], [(ix + 1) / (size - 1), iy / (size - 1)], [(ix + 1) / (size - 1), (iy + 1) / (size - 1)], [ix / (size - 1), (iy + 1) / (size - 1)]], points = [];
      for (let edge = 0; edge < 4; edge++) { const next = (edge + 1) % 4, a = vals[edge], b = vals[next]; if ((a >= 0) === (b >= 0) || a === b) continue; const t = a / (a - b); points.push([corners[edge][0] + (corners[next][0] - corners[edge][0]) * t, corners[edge][1] + (corners[next][1] - corners[edge][1]) * t]); }
      if (points.length >= 2) { const p = map(...points[0]), q = map(...points[1]); ctx.moveTo(...p); ctx.lineTo(...q); }
    }
    ctx.stroke(); ctx.restore();
  }
  function drawPoint(ctx, x, y, label, radius) {
    const path = () => { ctx.beginPath(); if (label) { ctx.moveTo(0, -radius); ctx.lineTo(radius, 0); ctx.lineTo(0, radius); ctx.lineTo(-radius, 0); ctx.closePath(); } else ctx.arc(0, 0, radius * 0.88, 0, Math.PI * 2); };
    ctx.save(); ctx.translate(x, y); ctx.lineJoin = 'round'; path(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 5; ctx.stroke(); path(); ctx.fillStyle = label ? C.blue : C.orange; ctx.strokeStyle = C.ink; ctx.lineWidth = 1.25; ctx.fill(); ctx.stroke(); ctx.restore();
  }
  function drawMain() {
    const { ctx, width, height } = setupCanvas(el.main, 400); ctx.clearRect(0, 0, width, height); const frame = inputFrame(ctx, width, height), g = getGrid(), n = g.size; state.mainFrame = frame;
    ctx.save(); ctx.beginPath(); ctx.rect(frame.left, frame.top, frame.pw, frame.ph); ctx.clip();
    for (let iy = 0; iy < n - 1; iy++) for (let ix = 0; ix < n - 1; ix++) {
      const z = g.z[iy * n + ix], p = sigmoid(z); let fill;
      if (state.view === 'class') fill = p >= 0.5 ? C.blueSoft : C.orangeSoft;
      else if (state.view === 'logit') fill = diverging(z, g.maxAbs, 0.72);
      else fill = p >= 0.5 ? blend('#ffffff', C.blue, 0.10 + 0.36 * (p - 0.5) * 2) : blend('#ffffff', C.orange, 0.10 + 0.36 * (0.5 - p) * 2);
      ctx.fillStyle = fill; const x = frame.px(ix / (n - 1)), y = frame.py((iy + 1) / (n - 1)); ctx.fillRect(x, y, frame.pw / (n - 1) + 1, frame.ph / (n - 1) + 1);
    }
    contour(ctx, g.z, n, (x, y) => [frame.px(x), frame.py(y)], C.ink, 1.8);
    if (state.dataset === 'xor4' || state.dataset === 'xorField') { ctx.save(); ctx.strokeStyle = 'rgba(51,68,83,.5)'; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.moveTo(frame.px(0.5), frame.top); ctx.lineTo(frame.px(0.5), frame.top + frame.ph); ctx.moveTo(frame.left, frame.py(0.5)); ctx.lineTo(frame.left + frame.pw, frame.py(0.5)); ctx.stroke(); ctx.restore(); }
    const radius = state.data.length <= 20 ? 8.5 : 5.4; for (const p of state.data) drawPoint(ctx, frame.px(p.x), frame.py(p.y), p.label, radius);
    const qx = frame.px(state.probe.x), qy = frame.py(state.probe.y); ctx.strokeStyle = '#fff'; ctx.lineWidth = 5; ctx.beginPath(); ctx.arc(qx, qy, 7, 0, Math.PI * 2); ctx.stroke(); ctx.fillStyle = '#f0b93f'; ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(qx, qy, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.restore();
    el.main.setAttribute('aria-label', `${datasetName(state.dataset)} with ${state.data.length} labeled points over the learned ${state.view} field.`);
  }
  function drawNetwork() {
    const { ctx, width, height } = setupCanvas(el.network, 125); ctx.clearRect(0, 0, width, height);
    const layers = !state.depth ? [2, 1] : state.depth === 1 ? [2, state.width, 1] : [2, state.width, state.width, 1], maxShow = 9;
    const xs = layers.map((_, i) => 38 + i * (width - 76) / Math.max(1, layers.length - 1));
    const nodes = layers.map((count, li) => { const shown = Math.min(count, maxShow), gap = Math.min(23, (height - 30) / Math.max(1, shown - 1)); return Array.from({ length: shown }, (_, i) => ({ x: xs[li], y: height / 2 + (i - (shown - 1) / 2) * gap })); });
    ctx.strokeStyle = '#c7d1d6'; ctx.lineWidth = 0.8; for (let li = 0; li < nodes.length - 1; li++) for (const a of nodes[li]) for (const b of nodes[li + 1]) { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
    nodes.forEach((layer, li) => layer.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, li === 0 || li === nodes.length - 1 ? 7.5 : 6.5, 0, Math.PI * 2); ctx.fillStyle = li === 0 ? '#fff' : li === nodes.length - 1 ? C.blueSoft : C.tealSoft; ctx.strokeStyle = li === 0 ? C.lineStrong : li === nodes.length - 1 ? C.blue : C.teal; ctx.lineWidth = 1.5; ctx.fill(); ctx.stroke(); }));
    const labels = !state.depth ? ['x₁, x₂', 'score z'] : state.depth === 1 ? ['x₁, x₂', `${state.width} ReLUs`, 'score z'] : ['x₁, x₂', `${state.width} ReLUs`, `${state.width} ReLUs`, 'score z'];
    ctx.fillStyle = C.muted; ctx.font = '700 10px ui-sans-serif, sans-serif'; ctx.textAlign = 'center'; labels.forEach((label, i) => ctx.fillText(label, xs[i], height - 4));
    el.network.setAttribute('aria-label', labels.join(' to '));
  }
  function drawHistory() {
    const { ctx, width, height } = setupCanvas(el.history, 125); ctx.clearRect(0, 0, width, height); ctx.fillStyle = C.canvas; ctx.fillRect(0, 0, width, height);
    const m = { left: 36, right: 10, top: 10, bottom: 24 }, pw = width - m.left - m.right, ph = height - m.top - m.bottom; ctx.strokeStyle = C.line; ctx.strokeRect(m.left, m.top, pw, ph);
    const valid = state.history.filter((h) => Number.isFinite(h.loss) && h.loss > 0); if (valid.length < 2) { ctx.fillStyle = C.muted; ctx.font = '11px ui-sans-serif, sans-serif'; ctx.textAlign = 'center'; ctx.fillText('Run or step to trace optimization', m.left + pw / 2, m.top + ph / 2 + 4); return; }
    const maxStep = Math.max(1, valid.at(-1).step), logs = valid.map((h) => Math.log10(h.loss)); let lo = Math.min(...logs), hi = Math.max(...logs); if (hi - lo < 0.15) { lo -= 0.075; hi += 0.075; }
    const px = (x) => m.left + x / maxStep * pw, py = (y) => m.top + (hi - y) / (hi - lo) * ph; ctx.strokeStyle = C.teal; ctx.lineWidth = 2; ctx.beginPath(); valid.forEach((h, i) => i ? ctx.lineTo(px(h.step), py(Math.log10(h.loss))) : ctx.moveTo(px(h.step), py(Math.log10(h.loss)))); ctx.stroke();
    ctx.fillStyle = C.muted; ctx.font = '9px ui-monospace, monospace'; ctx.fillText('0', m.left, height - 7); ctx.textAlign = 'right'; ctx.fillText(String(maxStep), m.left + pw, height - 7); el.history.setAttribute('aria-label', `Cross-entropy history over ${maxStep} steps.`);
  }
  function contributionFields(size = 29) {
    const count = state.depth ? state.width : 0, values = Array.from({ length: count }, () => new Float64Array(size * size)), hinges = Array.from({ length: count }, () => new Float64Array(size * size)); let maxAbs = 0.05;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) { const f = forward(state.model, x / (size - 1), y / (size - 1)), at = y * size + x; for (let j = 0; j < count; j++) { values[j][at] = state.model.Wo[j] * f.final[j]; hinges[j][at] = state.depth === 1 ? f.u1[j] : f.u2[j]; maxAbs = Math.max(maxAbs, Math.abs(values[j][at])); } }
    return { size, values, hinges, maxAbs };
  }
  function drawFeatures(force = false) {
    if (!force && state.running && state.renderCount % 5 !== 0) return; el.features.replaceChildren();
    if (!state.depth) { const empty = document.createElement('div'); empty.className = 'empty-feature'; empty.textContent = 'A linear classifier has no hidden ReLU terms. Its score is one plane: w₁x₁ + w₂x₂ + b.'; el.features.append(empty); return; }
    const bundle = contributionFields();
    for (let j = 0; j < state.width; j++) {
      const card = document.createElement('article'); card.className = 'feature-card'; const head = document.createElement('header'), strong = document.createElement('strong'), meta = document.createElement('span'); strong.textContent = `v${j + 1}h${j + 1}`; meta.textContent = `v = ${state.model.Wo[j].toFixed(2)}`; head.append(strong, meta);
      const canvas = document.createElement('canvas'); canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `Signed output contribution from ReLU ${j + 1}.`); const eq = document.createElement('p');
      if (state.depth === 1) { const w = state.model.W1[j]; eq.textContent = `${state.model.Wo[j].toFixed(2)} · ReLU(${w[0].toFixed(2)}x₁ ${signed(w[1])}x₂ ${signed(state.model.b1[j])})`; } else eq.textContent = `${state.model.Wo[j].toFixed(2)} · ReLU(weighted layer-1 sum ${signed(state.model.b2[j])})`;
      card.append(head, canvas, eq); el.features.append(card); const { ctx, width, height } = setupCanvas(canvas, 130), n = bundle.size;
      for (let y = 0; y < n - 1; y++) for (let x = 0; x < n - 1; x++) { ctx.fillStyle = diverging(bundle.values[j][y * n + x], bundle.maxAbs, 0.86); ctx.fillRect(x * width / (n - 1), height - (y + 1) * height / (n - 1), width / (n - 1) + 1, height / (n - 1) + 1); }
      contour(ctx, bundle.hinges[j], n, (x, y) => [x * width, (1 - y) * height], C.ink, 1.25, [3, 2]); ctx.strokeStyle = C.lineStrong; ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
    }
  }

  function drawSurface() {
    const { ctx, width, height } = setupCanvas(el.surface, 310); ctx.clearRect(0, 0, width, height); ctx.fillStyle = C.canvas; ctx.fillRect(0, 0, width, height);
    const size = 25, total = size * size, raw = new Float64Array(total); let maxAbs = 0.35;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) { const z = forward(state.model, x / (size - 1), y / (size - 1)).z; raw[y * size + x] = z; maxAbs = Math.max(maxAbs, Math.abs(z)); }
    const ca = Math.cos(state.camera.azimuth), sa = Math.sin(state.camera.azimuth), ce = Math.cos(state.camera.elevation), se = Math.sin(state.camera.elevation);
    const rotate = (x, y, z) => { const X = x - 0.5, Y = y - 0.5, rx = X * ca - Y * sa, ry = X * sa + Y * ca; return [rx, -z * ce + ry * se, ry * ce + z * se]; };
    const sx = new Float64Array(total), sy = new Float64Array(total), dep = new Float64Array(total); let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) { const i = y * size + x, q = rotate(x / (size - 1), y / (size - 1), raw[i] / maxAbs * 0.48); sx[i] = q[0]; sy[i] = q[1]; dep[i] = q[2]; minX = Math.min(minX, q[0]); maxX = Math.max(maxX, q[0]); minY = Math.min(minY, q[1]); maxY = Math.max(maxY, q[1]); }
    const planeRaw = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([x, y]) => rotate(x, y, 0)); for (const q of planeRaw) { minX = Math.min(minX, q[0]); maxX = Math.max(maxX, q[0]); minY = Math.min(minY, q[1]); maxY = Math.max(maxY, q[1]); }
    const pad = 22, scale = Math.min((width - 2 * pad) / Math.max(0.1, maxX - minX), (height - 2 * pad) / Math.max(0.1, maxY - minY)) * state.camera.zoom;
    const ox = width / 2 - (minX + maxX) * scale / 2, oy = height / 2 - (minY + maxY) * scale / 2, PX = (i) => ox + sx[i] * scale, PY = (i) => oy + sy[i] * scale, P = (q) => [ox + q[0] * scale, oy + q[1] * scale];
    const quads = []; for (let y = 0; y < size - 1; y++) for (let x = 0; x < size - 1; x++) { const a = y * size + x, b = a + 1, d = a + size, c = d + 1; quads.push({ a, b, c, d, depth: (dep[a] + dep[b] + dep[c] + dep[d]) / 4, z: (raw[a] + raw[b] + raw[c] + raw[d]) / 4 }); } quads.sort((a, b) => b.depth - a.depth);
    ctx.lineJoin = 'round'; for (const q of quads) { ctx.beginPath(); ctx.moveTo(PX(q.a), PY(q.a)); ctx.lineTo(PX(q.b), PY(q.b)); ctx.lineTo(PX(q.c), PY(q.c)); ctx.lineTo(PX(q.d), PY(q.d)); ctx.closePath(); ctx.fillStyle = diverging(q.z, maxAbs, 0.82); ctx.fill(); ctx.strokeStyle = 'rgba(24,36,47,.1)'; ctx.lineWidth = 0.45; ctx.stroke(); }
    const plane = planeRaw.map(P); ctx.beginPath(); plane.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.closePath(); ctx.fillStyle = 'rgba(213,107,45,.12)'; ctx.strokeStyle = 'rgba(213,107,45,.78)'; ctx.lineWidth = 1.2; ctx.fill(); ctx.stroke();
    const f = forward(state.model, state.probe.x, state.probe.y), base = P(rotate(state.probe.x, state.probe.y, 0)), tip = P(rotate(state.probe.x, state.probe.y, f.z / maxAbs * 0.48)); ctx.strokeStyle = C.ink; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(...base); ctx.lineTo(...tip); ctx.stroke(); ctx.fillStyle = '#f0b93f'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(...tip, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#854329'; ctx.font = '700 10px ui-sans-serif, sans-serif'; ctx.textAlign = 'right'; ctx.fillText('z = 0', width - 12, 16);
    el.surface.setAttribute('aria-label', `Interactive logit surface at azimuth ${state.camera.azimuth.toFixed(2)}, elevation ${state.camera.elevation.toFixed(2)}, zoom ${state.camera.zoom.toFixed(2)}. Probe score ${f.z.toFixed(2)}.`);
  }
  function syncRun() {
    if (!el.run) return;
    el.run.classList.toggle('is-running', state.running); text(el.run, state.running ? 'Pause' : 'Run'); el.run.setAttribute('aria-pressed', String(state.running));
    el.run.disabled = !state.running && !canTrain();
    if (el.step) el.step.disabled = !canTrain();
  }
  function syncControls() {
    el.dataset.value = state.dataset; el.depth.value = String(state.depth); el.width.value = String(state.width); el.widthOutput.value = String(state.width); el.width.disabled = state.depth === 0; el.lr.value = String(state.lr);
    el.dataSeed.value = String(state.dataSeed); el.weightSeed.value = String(state.weightSeed); hidden(el.customTools, state.dataset !== 'custom'); el.customCount.value = `${state.customPoints.length} ${state.customPoints.length === 1 ? 'point' : 'points'}`; text(el.datasetNote, datasetNote(state.dataset));
    const recommended = recommendationFor(); text(el.recommendedText, setupDescription(recommended));
    document.querySelectorAll('[data-point-class]').forEach((b) => { const active = Number(b.dataset.pointClass) === state.pointClass; b.classList.toggle('is-active', active); b.setAttribute('aria-pressed', String(active)); });
    document.querySelectorAll('[data-class-view]').forEach((b) => { const active = b.dataset.classView === state.view; b.classList.toggle('is-active', active); b.setAttribute('aria-pressed', String(active)); });
    el.provenance.className = `provenance-badge is-${state.provenance}`; text(el.provenance, state.provenance); syncRun();
  }
  function updateText() {
    const m = getMetrics(), f = forward(state.model, state.probe.x, state.probe.y); text(el.metric1, pct(m.accuracy)); text(el.metric2, pct(m.agreement)); text(el.metric3, Number.isFinite(m.loss) ? m.loss.toFixed(3) : '—'); text(el.steps, state.steps);
    text(el.metric1Label, 'Training accuracy');
    text(el.metric2Label, state.dataset === 'spirals' ? 'Holdout accuracy' : state.dataset === 'custom' ? 'Field accuracy unavailable' : 'Balanced field accuracy');
    text(el.metric3Label, 'Training BCE');
    text(el.title, state.dataset === 'xor4' ? 'Four points do not specify four regions' : `${datasetName(state.dataset)} · learned ${state.view} field`); text(el.params, `${parameterCount()} parameters`);
    text(el.networkTitle, !state.depth ? 'Two inputs → one linear score' : state.depth === 1 ? `Two inputs → ${state.width} ReLUs → one score` : `Two inputs → ${state.width} + ${state.width} ReLUs → one score`); text(el.status, state.status);
    el.probeX.value = String(state.probe.x); el.probeY.value = String(state.probe.y); el.probeXOut.value = state.probe.x.toFixed(2); el.probeYOut.value = state.probe.y.toFixed(2); text(el.probeReadout, `z = ${f.z.toFixed(2)} · p = ${f.p.toFixed(2)} · class ${f.p >= 0.5 ? 1 : 0}`);
  }
  function render(force = false) { state.renderCount++; syncControls(); updateText(); drawMain(); drawNetwork(); drawHistory(); drawSurface(); drawFeatures(force); }
  function setView(view) { if (!VIEWS.has(view)) throw new RangeError(`Unknown view: ${view}`); state.view = view; render(false); return snapshot(); }
  function setProbe(x, y) { state.probe.x = clamp(finite(Number(x), state.probe.x), 0, 1); state.probe.y = clamp(finite(Number(y), state.probe.y), 0, 1); render(false); return { ...state.probe }; }
  function markCameraCustom() { document.querySelectorAll('[data-surface-view]').forEach((b) => { b.classList.remove('is-active'); b.setAttribute('aria-pressed', 'false'); }); }
  function setSurfaceView(name) { if (!CAMERA_PRESETS[name]) throw new RangeError(`Unknown surface view: ${name}`); Object.assign(state.camera, CAMERA_PRESETS[name]); document.querySelectorAll('[data-surface-view]').forEach((b) => { const active = b.dataset.surfaceView === name; b.classList.toggle('is-active', active); b.setAttribute('aria-pressed', String(active)); }); drawSurface(); return { ...state.camera }; }
  function setSurfaceCamera(camera = {}) { if (Number.isFinite(camera.azimuth)) state.camera.azimuth = camera.azimuth; if (Number.isFinite(camera.elevation)) state.camera.elevation = clamp(camera.elevation, 0.05, 1.55); if (Number.isFinite(camera.zoom)) state.camera.zoom = clamp(camera.zoom, 0.65, 2.6); markCameraCustom(); drawSurface(); return { ...state.camera }; }
  function decomposition(x = state.probe.x, y = state.probe.y) { const f = forward(state.model, clamp(Number(x), 0, 1), clamp(Number(y), 0, 1)), contributions = state.depth ? state.model.Wo.map((w, i) => w * f.final[i]) : []; return { bias: state.model.bo, contributions, sum: f.z, reconstructed: state.depth ? state.model.bo + contributions.reduce((a, b) => a + b, 0) : f.z }; }
  function eventToInput(event) { const f = state.mainFrame, r = el.main.getBoundingClientRect(); if (!f || !r.width || !r.height) return null; const sx = (event.clientX - r.left) * f.width / r.width, sy = (event.clientY - r.top) * f.height / r.height; return { x: clamp((sx - f.left) / f.pw, 0, 1), y: clamp(1 - (sy - f.top) / f.ph, 0, 1), inside: sx >= f.left && sx <= f.left + f.pw && sy >= f.top && sy <= f.top + f.ph }; }
  function modelSignature() { const m = state.model, numbers = [m.depth, m.width, ...m.W1.flat(), ...m.b1, ...m.W2.flat(), ...m.b2, ...m.Wo, m.bo]; return numbers.map((v) => Number(v).toFixed(7)).join('|'); }
  function dataSignature() { return state.data.map((p) => `${p.x.toFixed(4)},${p.y.toFixed(4)},${p.label}`).join('|'); }
  function resetWeights() { return setWeightSeed(state.weightSeed); }
  function setLearningRate(value) {
    const next = Number(value); if (!Number.isFinite(next) || next <= 0) throw new RangeError('Learning rate must be a positive finite number.');
    state.lr = clamp(next, 0.0001, 1); state.status = `Learning rate set to ${state.lr}.`; render(false); return snapshot();
  }

  el.main.addEventListener('pointerdown', (event) => { const p = eventToInput(event); if (!p?.inside) return; state.probe = { x: p.x, y: p.y }; if (state.dataset === 'custom') { if (event.shiftKey && state.customPoints.length) removeCustomPoint({ x: p.x, y: p.y }); else addCustomPoint(p.x, p.y, state.pointClass); } else render(false); });
  el.dataset.addEventListener('change', () => setDataset(el.dataset.value)); el.depth.addEventListener('change', () => setArchitecture({ depth: Number(el.depth.value), width: state.width })); el.width.addEventListener('input', () => { el.widthOutput.value = el.width.value; }); el.width.addEventListener('change', () => setArchitecture({ depth: state.depth, width: Number(el.width.value) })); el.lr.addEventListener('change', () => setLearningRate(el.lr.value));
  el.dataSeed.addEventListener('change', () => setDataSeed(el.dataSeed.value)); el.newDataSeed.addEventListener('click', () => setDataSeed(state.dataSeed % 9999 + 1)); el.weightSeed.addEventListener('change', () => setWeightSeed(el.weightSeed.value)); el.newWeightSeed.addEventListener('click', () => setWeightSeed(state.weightSeed % 9999 + 1));
  el.recommended?.addEventListener('click', loadRecommendedSetup); el.random.addEventListener('click', resetWeights); el.corner.addEventListener('click', loadCornerRule); el.field.addEventListener('click', loadFieldRule); el.run.addEventListener('click', () => state.running ? (stop(), state.status = 'Training paused.', render(false)) : start()); el.step.addEventListener('click', () => { stop(); step(1); }); el.reset.addEventListener('click', resetWeights); el.clear.addEventListener('click', clearCustomPoints); el.undo.addEventListener('click', undoCustomEdit);
  document.querySelectorAll('[data-point-class]').forEach((b) => b.addEventListener('click', () => { state.pointClass = Number(b.dataset.pointClass); syncControls(); })); document.querySelectorAll('[data-class-view]').forEach((b) => b.addEventListener('click', () => setView(b.dataset.classView))); [el.probeX, el.probeY].forEach((node) => node.addEventListener('input', () => setProbe(el.probeX.value, el.probeY.value))); document.querySelectorAll('[data-surface-view]').forEach((b) => b.addEventListener('click', () => setSurfaceView(b.dataset.surfaceView))); el.surfaceReset.addEventListener('click', () => setSurfaceView('iso'));
  el.surface.addEventListener('pointerdown', (event) => { state.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, azimuth: state.camera.azimuth, elevation: state.camera.elevation }; el.surface.setPointerCapture?.(event.pointerId); });
  el.surface.addEventListener('pointermove', (event) => { if (!state.drag || state.drag.id !== event.pointerId) return; state.camera.azimuth = state.drag.azimuth - (event.clientX - state.drag.x) * 0.012; state.camera.elevation = clamp(state.drag.elevation + (event.clientY - state.drag.y) * 0.008, 0.05, 1.55); markCameraCustom(); drawSurface(); });
  const endDrag = (event) => { if (!state.drag || (event.pointerId !== undefined && event.pointerId !== state.drag.id)) return; state.drag = null; }; el.surface.addEventListener('pointerup', endDrag); el.surface.addEventListener('pointercancel', endDrag); el.surface.addEventListener('lostpointercapture', endDrag);
  el.surface.addEventListener('wheel', (event) => { event.preventDefault(); state.camera.zoom = clamp(state.camera.zoom * Math.exp(-event.deltaY * 0.0012), 0.65, 2.6); markCameraCustom(); drawSurface(); }, { passive: false });
  el.surface.addEventListener('keydown', (event) => { let handled = true; if (event.key === 'ArrowLeft') state.camera.azimuth += 0.12; else if (event.key === 'ArrowRight') state.camera.azimuth -= 0.12; else if (event.key === 'ArrowUp') state.camera.elevation = clamp(state.camera.elevation - 0.08, 0.05, 1.55); else if (event.key === 'ArrowDown') state.camera.elevation = clamp(state.camera.elevation + 0.08, 0.05, 1.55); else if (event.key === '+' || event.key === '=') state.camera.zoom = clamp(state.camera.zoom * 1.1, 0.65, 2.6); else if (event.key === '-') state.camera.zoom = clamp(state.camera.zoom / 1.1, 0.65, 2.6); else if (event.key === '0') { setSurfaceView('iso'); return; } else handled = false; if (handled) { event.preventDefault(); markCameraCustom(); drawSurface(); } });
  function syncResponsiveLayout() {
    const mobile = window.matchMedia('(max-width: 800px)').matches;
    if (mobile && layout.controls.parentElement !== layout.workspace) layout.primaryVisual.after(layout.controls);
    if (!mobile && layout.controls.parentElement !== layout.grid) layout.grid.insertBefore(layout.controls, layout.workspace);
  }
  let resizeTimer = null; window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { syncResponsiveLayout(); render(true); }, 90); });

  function snapshot() {
    const m = getMetrics(), counts = dataCounts(), activity = activationDiagnostics(), historyLastLoss = state.history.at(-1)?.loss;
    return { mode: 'classification', dataset: state.dataset, dataSize: state.data.length, dataSeed: state.dataSeed, weightSeed: state.weightSeed, depth: state.depth, width: state.width, parameters: parameterCount(), accuracy: m.accuracy, fieldAgreement: m.agreement, loss: m.loss, steps: state.steps, featureCount: state.depth ? state.width : 0, provenance: state.provenance, initializer: state.initializer, view: state.view, running: state.running, trainable: canTrain(), learningRate: state.lr, historyLength: state.history.length, historyLastLoss: Number.isFinite(historyLastLoss) ? historyLastLoss : null, classCounts: counts.classCounts, quadrantCounts: counts.quadrantCounts, deadUnits: activity.deadUnits, deadUnitsByLayer: activity.deadUnitsByLayer, probe: { ...state.probe }, camera: { ...state.camera }, dataSignature: dataSignature(), weightSignature: modelSignature() };
  }
  window.ReLUClassificationLab = Object.freeze({ snapshot, setDataset, setArchitecture, setDataSeed, setWeightSeed, setSeed, setLearningRate, initialize, getRecommendedSetup: recommendationFor, loadRecommendedSetup, loadCornerRule, loadFieldRule, step, start, stop, reset: resetWeights, setView, setProbe, addCustomPoint, removeCustomPoint, clearCustomPoints, undoCustomEdit, getContributionDecomposition: decomposition, setSurfaceCamera, setSurfaceView });
  syncResponsiveLayout(); state.data = makeDataset(state.dataset, state.dataSeed, state.customPoints); state.model = cornerModel(); state.status = 'Constructed two-ReLU corner rule loaded for inspection. Choose Recommended or Random before training.'; changed(); const initialLoss = dataLoss(); state.history = [{ step: 0, loss: initialLoss }]; render(true);
})();
