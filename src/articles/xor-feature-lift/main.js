(() => {
  'use strict';

  const BOX = 1.25;
  const LEARNING_RATE = 0.5;
  const MAX_STEPS = 4000;
  const STEPS_PER_FRAME = 10;
  const SCENE_HALF_HEIGHT = 1;
  const CHANCE_LOSS = Math.LN2;
  const SLIDER_LIMIT = 8;
  const LIFT_DURATION = 850;
  const CAMERA_DURATION = 450;
  const INITIAL_WEIGHTS = Object.freeze({ w1: 0.7, w2: -0.45, w3: 0, b: 0.15 });
  const WEIGHT_KEYS = ['w1', 'w2', 'w3', 'b'];
  const CLASS_COLORS = ['#2c67a0', '#bd5f35'];
  const CLASS_ZERO_TINT = [214, 230, 244];
  const CLASS_ONE_TINT = [246, 219, 204];
  const INK = '#232b32';
  const reducedMotion = Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  const FEATURES = {
    none: {
      third: null,
      symbol: '',
      axis: 'φ = 0 (nothing lifted)',
      zRange: [-1.6, 1.6],
      fn: () => 0,
      boundary: 'straight line',
      eyebrow: 'TWO FEATURES: x₁ AND x₂',
      title: 'Can one straight line separate XOR?',
      help: 'Logistic regression on the raw inputs. Whatever the weights, the boundary z = 0 is a straight line in the (x₁, x₂) plane.',
      loaded: 'Try the sliders first. Can any line put both blue corners on one side and both orange corners on the other?'
    },
    product: {
      third: 'x₁x₂',
      symbol: 'x₁x₂',
      axis: 'φ = x₁x₂',
      zRange: [-1.6, 1.6],
      fn: (x1, x2) => x1 * x2,
      boundary: 'hyperbola',
      eyebrow: 'THREE FEATURES: x₁, x₂ AND x₁x₂',
      title: 'Lift the data by x₁x₂. Which flat plane separates it?',
      help: 'A third column, x₁x₂, lifts every point to height x₁x₂. Blue corners rise to +1 and orange corners sink to −1.',
      loaded: 'The points now sit on a saddle. Run gradient descent and watch a flat plane in feature space become a curve in input space.'
    },
    radius: {
      third: 'x₁² + x₂²',
      symbol: '(x₁² + x₂²)',
      axis: 'φ = x₁² + x₂²',
      zRange: [0, 3.2],
      fn: (x1, x2) => x1 * x1 + x2 * x2,
      boundary: 'circle',
      eyebrow: 'A DECOY FEATURE: x₁² + x₂²',
      title: 'Does any extra feature work?',
      help: 'The squared radius is 2 at every corner, so this lift raises blue and orange corners by the same amount.',
      loaded: 'Every corner is lifted to the same height. Predict the accuracy before you press Run.'
    },
    relu: {
      third: 'ReLU(x₁ + x₂ − 1)',
      symbol: 'ReLU(x₁ + x₂ − 1)',
      axis: 'φ = ReLU(x₁ + x₂ − 1)',
      zRange: [-0.3, 1.5],
      fn: (x1, x2) => Math.max(0, x1 + x2 - 1),
      boundary: 'bent line',
      eyebrow: 'ONE HAND-MADE HIDDEN NEURON',
      title: 'What if the feature is a ReLU unit?',
      help: 'This feature is one hidden ReLU neuron with hand-picked weights (1, 1) and bias −1. It is zero everywhere except near the (1, 1) corner.',
      loaded: 'Only the (1, 1) corner is lifted. Run gradient descent: one ReLU plus a line is enough for XOR.'
    }
  };

  const VIEW_PRESETS = {
    orbit: { azimuth: -Math.PI / 2 + 0.45, elevation: 0.55, zoom: 1 },
    top: { azimuth: -Math.PI / 2, elevation: Math.PI / 2, zoom: 1 }
  };

  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const normalize = (a) => {
    const length = norm(a);
    return length < 1e-12 ? [0, 0, 0] : [a[0] / length, a[1] / length, a[2] / length];
  };
  const wrapAngle = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));
  const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - ((-2 * t + 2) ** 2) / 2);
  const $ = (id) => document.getElementById(id);

  const els = {
    featureHelp: $('featureHelp'),
    sliders: { w1: $('w1Slider'), w2: $('w2Slider'), w3: $('w3Slider'), b: $('bSlider') },
    outputs: { w1: $('w1Value'), w2: $('w2Value'), w3: $('w3Value'), b: $('bValue') },
    w3Label: $('w3Label'),
    run: $('runBtn'),
    step: $('stepBtn'),
    reset: $('resetBtn'),
    stepValue: $('stepValue'),
    status: $('statusText'),
    workspaceEyebrow: $('workspaceEyebrow'),
    workspaceTitle: $('workspaceTitle'),
    datasetBadge: $('datasetBadge'),
    loss: $('lossValue'),
    lossNote: $('lossNote'),
    accuracy: $('accuracyValue'),
    accuracyNote: $('accuracyNote'),
    boundary: $('boundaryValue'),
    dimension: $('dimensionValue'),
    dimensionNote: $('dimensionNote'),
    input: $('inputCanvas'),
    inputSummary: $('inputSummary'),
    boundaryEquation: $('boundaryEquation'),
    feature: $('featureCanvas'),
    planeEquation: $('planeEquation'),
    history: $('historyCanvas'),
    historySummary: $('historySummary')
  };

  // ---------- Data ----------
  function seededRandom(seed) {
    let value = seed >>> 0;
    return () => {
      value = (1664525 * value + 1013904223) >>> 0;
      return value / 4294967296;
    };
  }

  function makeDataset(kind) {
    // Class 1 (orange) is the XOR-true pair of corners, where the signs differ.
    const corners = [[1, 1, 0], [-1, -1, 0], [1, -1, 1], [-1, 1, 1]];
    if (kind === 'corners') return corners.map(([x1, x2, y]) => ({ x1, x2, y }));
    const random = seededRandom(7);
    const points = [];
    for (const [cx, cy, y] of corners) {
      for (let index = 0; index < 25; index += 1) {
        const radius = Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12)));
        const angle = 2 * Math.PI * random();
        points.push({
          x1: clamp(cx + 0.15 * radius * Math.cos(angle), -1.15, 1.15),
          x2: clamp(cy + 0.15 * radius * Math.sin(angle), -1.15, 1.15),
          y
        });
      }
    }
    return points;
  }

  const state = {
    dataset: 'clusters',
    data: makeDataset('clusters'),
    feature: 'none',
    weights: { ...INITIAL_WEIGHTS },
    steps: 0,
    running: false,
    loop: null,
    history: [],
    lastAction: 'load',
    statusMessage: FEATURES.none.loaded,
    view: 'orbit',
    camera: { ...VIEW_PRESETS.orbit },
    cameraTween: null,
    drag: null,
    lift: { from: 'none', to: 'none', t: 1, start: 0, duration: 0 }
  };

  // ---------- Model ----------
  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const expZ = Math.exp(z);
    return expZ / (1 + expZ);
  }

  function binaryCrossEntropy(logitValue, target) {
    return Math.max(logitValue, 0) - target * logitValue + Math.log1p(Math.exp(-Math.abs(logitValue)));
  }

  function phi(x1, x2, key = state.feature) {
    return key === 'none' ? 0 : FEATURES[key].fn(x1, x2);
  }

  function logit(x1, x2, weights = state.weights, key = state.feature) {
    const lifted = key === 'none' ? 0 : weights.w3 * phi(x1, x2, key);
    return weights.b + weights.w1 * x1 + weights.w2 * x2 + lifted;
  }

  function measure(weights = state.weights) {
    // A point with z exactly 0 (p = 0.5) is a coin flip, so it counts as half correct.
    let loss = 0;
    let correct = 0;
    let ties = 0;
    for (const point of state.data) {
      const z = logit(point.x1, point.x2, weights);
      loss += binaryCrossEntropy(z, point.y);
      if (Math.abs(z) < 1e-9) {
        ties += 1;
        correct += 0.5;
      } else {
        correct += Number((z > 0 ? 1 : 0) === point.y);
      }
    }
    return { loss: loss / state.data.length, accuracy: correct / state.data.length, ties };
  }

  function gradientStep() {
    const count = state.data.length;
    let gradient1 = 0;
    let gradient2 = 0;
    let gradient3 = 0;
    let gradientB = 0;
    for (const point of state.data) {
      const feature = phi(point.x1, point.x2);
      const error = (sigmoid(logit(point.x1, point.x2)) - point.y) / count;
      gradientB += error;
      gradient1 += error * point.x1;
      gradient2 += error * point.x2;
      gradient3 += error * feature;
    }
    state.weights.b -= LEARNING_RATE * gradientB;
    state.weights.w1 -= LEARNING_RATE * gradient1;
    state.weights.w2 -= LEARNING_RATE * gradient2;
    if (state.feature !== 'none') state.weights.w3 -= LEARNING_RATE * gradient3;
    state.steps += 1;
    state.history.push(measure().loss);
  }

  function trainSteps(count) {
    const requested = Math.max(0, Math.floor(Number(count) || 0));
    const actual = Math.min(requested, Math.max(0, MAX_STEPS - state.steps));
    for (let index = 0; index < actual; index += 1) gradientStep();
    if (actual > 0) state.lastAction = 'train';
    if (state.steps >= MAX_STEPS) stopRunning();
    return actual;
  }

  function resetHistory() {
    state.history = [measure().loss];
  }

  // ---------- Boundary (marching squares on the live logit) ----------
  function boundarySegments(cells = 64) {
    const nodes = cells + 1;
    const values = new Float64Array(nodes * nodes);
    const coordinate = (index) => -BOX + (2 * BOX * index) / cells;
    for (let row = 0; row < nodes; row += 1) {
      for (let column = 0; column < nodes; column += 1) {
        values[row * nodes + column] = logit(coordinate(column), coordinate(row));
      }
    }
    const positive = (value) => value >= 0;
    const crossing = (xa, ya, va, xb, yb, vb) => {
      const t = va / (va - vb);
      return [xa + (xb - xa) * t, ya + (yb - ya) * t];
    };
    const segments = [];
    for (let row = 0; row < cells; row += 1) {
      for (let column = 0; column < cells; column += 1) {
        const x0 = coordinate(column);
        const x1 = coordinate(column + 1);
        const y0 = coordinate(row);
        const y1 = coordinate(row + 1);
        const v00 = values[row * nodes + column];
        const v10 = values[row * nodes + column + 1];
        const v01 = values[(row + 1) * nodes + column];
        const v11 = values[(row + 1) * nodes + column + 1];
        const points = [];
        if (positive(v00) !== positive(v10)) points.push(crossing(x0, y0, v00, x1, y0, v10));
        if (positive(v10) !== positive(v11)) points.push(crossing(x1, y0, v10, x1, y1, v11));
        if (positive(v11) !== positive(v01)) points.push(crossing(x1, y1, v11, x0, y1, v01));
        if (positive(v01) !== positive(v00)) points.push(crossing(x0, y1, v01, x0, y0, v00));
        if (points.length === 2) {
          segments.push(points);
        } else if (points.length === 4) {
          // Saddle cell: the centre value decides which corners connect.
          const centre = (v00 + v10 + v01 + v11) / 4;
          if (positive(centre) === positive(v00)) {
            segments.push([points[0], points[1]], [points[2], points[3]]);
          } else {
            segments.push([points[0], points[3]], [points[1], points[2]]);
          }
        }
      }
    }
    return segments;
  }

  function boundaryKind() {
    const { w1, w2, w3 } = state.weights;
    if (state.feature === 'none' || Math.abs(w3) < 1e-9) {
      return Math.abs(w1) < 1e-9 && Math.abs(w2) < 1e-9 ? 'none (z is constant)' : 'straight line';
    }
    return FEATURES[state.feature].boundary;
  }

  // ---------- Lift bookkeeping ----------
  function liftMix() {
    return state.lift.t >= 1 ? 1 : easeInOut(state.lift.t);
  }

  function liftedValue(x1, x2) {
    const mix = liftMix();
    const target = phi(x1, x2, state.lift.to);
    if (mix >= 1) return target;
    return (1 - mix) * phi(x1, x2, state.lift.from) + mix * target;
  }

  function liftedRange() {
    const mix = liftMix();
    const from = FEATURES[state.lift.from].zRange;
    const to = FEATURES[state.lift.to].zRange;
    return [from[0] + (to[0] - from[0]) * mix, from[1] + (to[1] - from[1]) * mix];
  }

  function zMapper() {
    const [low, high] = liftedRange();
    const mid = (low + high) / 2;
    const half = (high - low) / 2;
    return { mid, half, toScene: (value) => ((value - mid) / half) * SCENE_HALF_HEIGHT };
  }

  function beginLift(from, to, animate) {
    const duration = animate && !reducedMotion ? LIFT_DURATION : 0;
    state.lift = { from, to, t: duration > 0 ? 0 : 1, start: performance.now(), duration };
    if (duration > 0) ensureLoop();
  }

  // ---------- Camera ----------
  function cameraBasis(camera) {
    const ca = Math.cos(camera.azimuth);
    const sa = Math.sin(camera.azimuth);
    const ce = Math.cos(camera.elevation);
    const se = Math.sin(camera.elevation);
    return {
      toward: [ce * ca, ce * sa, se],
      right: [-sa, ca, 0],
      up: [-se * ca, -se * sa, ce]
    };
  }

  function sceneNormal(mapper = zMapper()) {
    const { w1, w2, w3 } = state.weights;
    const third = state.feature === 'none' ? 0 : (w3 * mapper.half) / SCENE_HALF_HEIGHT;
    return [w1, w2, third];
  }

  function edgeOnCamera(current) {
    // Find a view direction that lies inside the plane, so the plane projects to a line.
    const normal = sceneNormal();
    const planar = Math.hypot(normal[0], normal[1]);
    if (planar < 1e-9) return { azimuth: current.azimuth, elevation: 0 };
    let elevation = 0.1;
    if (Math.tan(elevation) * Math.abs(normal[2]) > planar) elevation = 0.9 * Math.atan(planar / Math.abs(normal[2]));
    const target = -Math.tan(elevation) * normal[2];
    const theta = Math.atan2(normal[1], normal[0]);
    const delta = Math.acos(clamp(target / planar, -1, 1));
    const candidates = [theta + delta, theta - delta];
    const nearest = candidates.reduce((best, candidate) => (
      Math.abs(wrapAngle(candidate - current.azimuth)) < Math.abs(wrapAngle(best - current.azimuth)) ? candidate : best
    ));
    return { azimuth: current.azimuth + wrapAngle(nearest - current.azimuth), elevation };
  }

  function animateCamera(target, animate) {
    const to = {
      azimuth: state.camera.azimuth + wrapAngle(target.azimuth - state.camera.azimuth),
      elevation: target.elevation,
      zoom: target.zoom ?? state.camera.zoom
    };
    if (!animate || reducedMotion) {
      Object.assign(state.camera, to);
      state.cameraTween = null;
      return;
    }
    state.cameraTween = { from: { ...state.camera }, to, start: performance.now(), duration: CAMERA_DURATION };
    ensureLoop();
  }

  function setView(name, animate = true) {
    if (name === 'orbit' || name === 'top') {
      animateCamera(VIEW_PRESETS[name], animate);
    } else if (name === 'edge') {
      animateCamera({ ...edgeOnCamera(state.camera), zoom: 1 }, animate);
    } else if (name !== 'custom') {
      throw new RangeError(`Unknown view: ${name}`);
    }
    state.view = name;
    render();
  }

  function markCameraCustom() {
    state.view = 'custom';
    state.cameraTween = null;
  }

  // ---------- Canvas helpers ----------
  function setupCanvas(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const context = canvas.getContext('2d');
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    return { context, width, height };
  }

  function probabilityColor(probability, shade = 1, alpha = 1) {
    const channel = (index) => Math.round((CLASS_ZERO_TINT[index] * (1 - probability) + CLASS_ONE_TINT[index] * probability) * shade);
    return alpha >= 1
      ? `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`
      : `rgba(${channel(0)}, ${channel(1)}, ${channel(2)}, ${alpha})`;
  }

  function drawMarker(context, x, y, label, size) {
    context.fillStyle = CLASS_COLORS[label];
    context.strokeStyle = '#ffffff';
    context.lineWidth = 1.2;
    context.beginPath();
    if (label === 0) context.arc(x, y, size, 0, Math.PI * 2);
    else context.rect(x - size, y - size, size * 2, size * 2);
    context.fill();
    context.stroke();
  }

  function makeTransform(width, height) {
    const padding = { left: 40, right: 14, top: 14, bottom: 34 };
    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;
    return {
      padding,
      innerWidth,
      innerHeight,
      x: (value) => padding.left + ((value + BOX) / (2 * BOX)) * innerWidth,
      y: (value) => padding.top + ((BOX - value) / (2 * BOX)) * innerHeight
    };
  }

  function drawAxes(context, width, height, transform) {
    const { padding, x, y } = transform;
    context.strokeStyle = '#aeb9bf';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x(0), padding.top);
    context.lineTo(x(0), height - padding.bottom);
    context.moveTo(padding.left, y(0));
    context.lineTo(width - padding.right, y(0));
    context.stroke();

    context.fillStyle = '#66737d';
    context.font = '11px SFMono-Regular, Consolas, monospace';
    context.textAlign = 'center';
    context.textBaseline = 'top';
    for (const tick of [-1, -0.5, 0.5, 1]) context.fillText(tick.toFixed(1), x(tick), height - padding.bottom + 6);
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    for (const tick of [-1, -0.5, 0.5, 1]) context.fillText(tick.toFixed(1), padding.left - 7, y(tick));
    context.fillStyle = '#3f4b54';
    context.font = '700 12px Avenir Next, Segoe UI, sans-serif';
    context.textAlign = 'right';
    context.textBaseline = 'bottom';
    context.fillText('x₁', width - padding.right, height - 6);
    context.textAlign = 'center';
    context.textBaseline = 'bottom';
    context.fillText('x₂', x(0), padding.top - 2);
  }

  // ---------- Input space ----------
  function drawInputSpace(segments) {
    const { context, width, height } = setupCanvas(els.input);
    const transform = makeTransform(width, height);
    const { padding, innerWidth, innerHeight, x, y } = transform;
    const cells = 56;
    const domain = 2 * BOX;
    const cellWidth = innerWidth / cells;
    const cellHeight = innerHeight / cells;
    for (let row = 0; row < cells; row += 1) {
      for (let column = 0; column < cells; column += 1) {
        const x1 = -BOX + ((column + 0.5) / cells) * domain;
        const x2 = BOX - ((row + 0.5) / cells) * domain;
        context.fillStyle = probabilityColor(sigmoid(logit(x1, x2)));
        context.fillRect(padding.left + column * cellWidth, padding.top + row * cellHeight, cellWidth + 0.6, cellHeight + 0.6);
      }
    }
    drawAxes(context, width, height, transform);

    context.strokeStyle = INK;
    context.lineWidth = 2.2;
    context.lineCap = 'round';
    context.beginPath();
    for (const [p, q] of segments) {
      context.moveTo(x(p[0]), y(p[1]));
      context.lineTo(x(q[0]), y(q[1]));
    }
    context.stroke();

    const size = state.dataset === 'corners' ? 6.5 : 3.6;
    for (const point of state.data) drawMarker(context, x(point.x1), y(point.x2), point.y, size);
  }

  // ---------- Feature space ----------
  function planePolygon(normal, offset) {
    const H = SCENE_HALF_HEIGHT;
    const corners = [];
    for (const x of [-BOX, BOX]) for (const y of [-BOX, BOX]) for (const z of [-H, H]) corners.push([x, y, z]);
    const evaluate = (p) => dot(normal, p) + offset;
    const points = [];
    const push = (p) => {
      if (!points.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) < 1e-9)) points.push(p);
    };
    for (let i = 0; i < corners.length; i += 1) {
      for (let j = i + 1; j < corners.length; j += 1) {
        const a = corners[i];
        const b = corners[j];
        const differing = Number(a[0] !== b[0]) + Number(a[1] !== b[1]) + Number(a[2] !== b[2]);
        if (differing !== 1) continue;
        const fa = evaluate(a);
        const fb = evaluate(b);
        if (fa === 0) push(a);
        if (fb === 0) push(b);
        if ((fa < 0 && fb > 0) || (fa > 0 && fb < 0)) {
          const t = fa / (fa - fb);
          push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
        }
      }
    }
    if (points.length < 3) return null;
    const centroid = points.reduce((sum, p) => [sum[0] + p[0], sum[1] + p[1], sum[2] + p[2]], [0, 0, 0]).map((v) => v / points.length);
    const unit = normalize(normal);
    let e1 = normalize(sub(points[0], centroid));
    if (norm(e1) < 1e-9) e1 = normalize(sub(points[1], centroid));
    const e2 = normalize(cross(unit, e1));
    return points
      .map((p) => ({ p, angle: Math.atan2(dot(sub(p, centroid), e2), dot(sub(p, centroid), e1)) }))
      .sort((a, b) => a.angle - b.angle)
      .map((entry) => entry.p);
  }

  function clipHalf(polygon, inside) {
    const output = [];
    for (let index = 0; index < polygon.length; index += 1) {
      const current = polygon[index];
      const previous = polygon[(index + polygon.length - 1) % polygon.length];
      const insideCurrent = inside(current);
      const insidePrevious = inside(previous);
      const intersection = () => {
        const t = insidePrevious / (insidePrevious - insideCurrent);
        return [previous[0] + (current[0] - previous[0]) * t, previous[1] + (current[1] - previous[1]) * t];
      };
      if (insideCurrent >= 0) {
        if (insidePrevious < 0) output.push(intersection());
        output.push(current);
      } else if (insidePrevious >= 0) {
        output.push(intersection());
      }
    }
    return output;
  }

  function tessellatePlane(polygon, normal, cell = 0.15) {
    const centroid = polygon.reduce((sum, p) => [sum[0] + p[0], sum[1] + p[1], sum[2] + p[2]], [0, 0, 0]).map((v) => v / polygon.length);
    const unit = normalize(normal);
    let e1 = normalize(sub(polygon[0], centroid));
    if (norm(e1) < 1e-9) e1 = normalize(sub(polygon[1], centroid));
    const e2 = normalize(cross(unit, e1));
    const local = polygon.map((p) => [dot(sub(p, centroid), e1), dot(sub(p, centroid), e2)]);
    let sMin = Infinity;
    let sMax = -Infinity;
    let tMin = Infinity;
    let tMax = -Infinity;
    for (const [s, t] of local) {
      sMin = Math.min(sMin, s);
      sMax = Math.max(sMax, s);
      tMin = Math.min(tMin, t);
      tMax = Math.max(tMax, t);
    }
    const pieces = [];
    for (let s0 = Math.floor(sMin / cell) * cell; s0 < sMax; s0 += cell) {
      for (let t0 = Math.floor(tMin / cell) * cell; t0 < tMax; t0 += cell) {
        let piece = clipHalf(local, (p) => p[0] - s0);
        if (piece.length >= 3) piece = clipHalf(piece, (p) => s0 + cell - p[0]);
        if (piece.length >= 3) piece = clipHalf(piece, (p) => p[1] - t0);
        if (piece.length >= 3) piece = clipHalf(piece, (p) => t0 + cell - p[1]);
        if (piece.length >= 3) {
          pieces.push(piece.map(([s, t]) => [
            centroid[0] + s * e1[0] + t * e2[0],
            centroid[1] + s * e1[1] + t * e2[1],
            centroid[2] + s * e1[2] + t * e2[2]
          ]));
        }
      }
    }
    return pieces;
  }

  function drawFeatureSpace(segments) {
    const { context, width, height } = setupCanvas(els.feature);
    if (state.view === 'edge' && !state.cameraTween) Object.assign(state.camera, edgeOnCamera(state.camera));
    const basis = cameraBasis(state.camera);
    const mapper = zMapper();
    const H = SCENE_HALF_HEIGHT;

    const boxCorners = [];
    for (const x of [-BOX, BOX]) for (const y of [-BOX, BOX]) for (const z of [-H, H]) boxCorners.push([x, y, z]);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const corner of boxCorners) {
      const sx = dot(corner, basis.right);
      const sy = -dot(corner, basis.up);
      minX = Math.min(minX, sx);
      maxX = Math.max(maxX, sx);
      minY = Math.min(minY, sy);
      maxY = Math.max(maxY, sy);
    }
    const pad = 36;
    const scale = Math.min((width - 2 * pad) / Math.max(0.1, maxX - minX), (height - 2 * pad) / Math.max(0.1, maxY - minY)) * state.camera.zoom;
    const originX = width / 2 - ((minX + maxX) / 2) * scale;
    const originY = height / 2 - ((minY + maxY) / 2) * scale;
    const project = (p) => ({ x: originX + dot(p, basis.right) * scale, y: originY - dot(p, basis.up) * scale, depth: dot(p, basis.toward) });

    const items = [];
    const addPolygon = (points, fill, stroke, lineWidth, bias = 0) => {
      const projected = points.map(project);
      const depth = projected.reduce((sum, p) => sum + p.depth, 0) / projected.length + bias;
      items.push({
        depth,
        draw(ctx) {
          ctx.beginPath();
          projected.forEach((p, index) => (index ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
          ctx.closePath();
          if (fill) {
            ctx.fillStyle = fill;
            ctx.fill();
          }
          if (stroke) {
            ctx.strokeStyle = stroke;
            ctx.lineWidth = lineWidth;
            ctx.stroke();
          }
        }
      });
    };
    const addSegment = (a, b, stroke, lineWidth, dash = [], bias = 0) => {
      const pa = project(a);
      const pb = project(b);
      items.push({
        depth: (pa.depth + pb.depth) / 2 + bias,
        draw(ctx) {
          ctx.setLineDash(dash);
          ctx.strokeStyle = stroke;
          ctx.lineWidth = lineWidth;
          ctx.beginPath();
          ctx.moveTo(pa.x, pa.y);
          ctx.lineTo(pb.x, pb.y);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
    };

    const floorZ = mapper.toScene(0);
    const liftDone = state.lift.t >= 1;

    // Decorations sit exactly on the surface or the floor, so depth sorting against
    // the coplanar quads is unreliable. They are drawn after all quads instead, with a
    // ray march toward the camera deciding whether the surface hides them.
    const decorations = [];
    const occluded = (p) => {
      const [dx, dy, dz] = basis.toward;
      let x = p[0] + dx * 0.03;
      let y = p[1] + dy * 0.03;
      let z = p[2] + dz * 0.03;
      for (let index = 0; index < 90; index += 1) {
        if (Math.abs(x) > BOX || Math.abs(y) > BOX || z > H) return false;
        if (z < mapper.toScene(liftedValue(x, y)) - 0.012) return true;
        x += dx * 0.05;
        y += dy * 0.05;
        z += dz * 0.05;
      }
      return false;
    };
    const addDecorationSegment = (a, b, stroke, lineWidth, dash, order, ghostAlpha) => {
      const hidden = occluded([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]);
      if (hidden && ghostAlpha <= 0) return;
      const pa = project(a);
      const pb = project(b);
      decorations.push({
        hidden,
        order,
        draw(ctx) {
          ctx.globalAlpha = hidden ? ghostAlpha : 1;
          ctx.setLineDash(dash);
          ctx.strokeStyle = stroke;
          ctx.lineWidth = lineWidth;
          ctx.beginPath();
          ctx.moveTo(pa.x, pa.y);
          ctx.lineTo(pb.x, pb.y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        }
      });
    };

    // Floor grid at φ = 0, where the un-lifted data lives.
    const gridN = 10;
    for (let i = 0; i <= gridN; i += 1) {
      const v = -BOX + (2 * BOX * i) / gridN;
      const strong = i === 0 || i === gridN;
      const style = strong ? 'rgba(35, 43, 50, 0.5)' : 'rgba(35, 43, 50, 0.16)';
      for (let k = 0; k < gridN; k += 1) {
        const a = -BOX + (2 * BOX * k) / gridN;
        const b = a + (2 * BOX) / gridN;
        addDecorationSegment([v, a, floorZ], [v, b, floorZ], style, strong ? 1.2 : 0.8, [], 0, 0);
        addDecorationSegment([a, v, floorZ], [b, v, floorZ], style, strong ? 1.2 : 0.8, [], 0, 0);
      }
    }

    // Lifted surface z = φ(x₁, x₂), coloured exactly like the input-space map.
    const n = 26;
    const nodes = [];
    for (let row = 0; row <= n; row += 1) {
      const line = [];
      for (let column = 0; column <= n; column += 1) {
        const x1 = -BOX + (2 * BOX * column) / n;
        const x2 = -BOX + (2 * BOX * row) / n;
        line.push([x1, x2, mapper.toScene(liftedValue(x1, x2))]);
      }
      nodes.push(line);
    }
    const light = normalize([0.35, -0.55, 0.75]);
    for (let row = 0; row < n; row += 1) {
      for (let column = 0; column < n; column += 1) {
        const a = nodes[row][column];
        const b = nodes[row][column + 1];
        const c = nodes[row + 1][column + 1];
        const d = nodes[row + 1][column];
        const probability = sigmoid(logit((a[0] + c[0]) / 2, (a[1] + c[1]) / 2));
        const facet = normalize(cross(sub(c, a), sub(d, b)));
        const shade = 0.84 + 0.16 * Math.abs(dot(facet, light));
        addPolygon([a, b, c, d], probabilityColor(probability, shade, 0.95), 'rgba(35, 43, 50, 0.09)', 0.5);
      }
    }

    // Decision plane, clipped to the box and tessellated so it sorts against the surface.
    const normal = sceneNormal(mapper);
    const offset = state.weights.b + (state.feature === 'none' ? 0 : state.weights.w3 * mapper.mid);
    let planeDrawn = false;
    const overhead = state.camera.elevation / (Math.PI / 2);
    const planeAlpha = 0.18 * (1 - 0.82 * overhead * overhead);
    if (liftDone && norm(normal) > 1e-6) {
      const polygon = planePolygon(normal, offset);
      if (polygon) {
        planeDrawn = true;
        for (const piece of tessellatePlane(polygon, normal)) addPolygon(piece, `rgba(35, 43, 50, ${planeAlpha.toFixed(3)})`, null, 0, 0.001);
        for (let index = 0; index < polygon.length; index += 1) {
          addSegment(polygon[index], polygon[(index + 1) % polygon.length], 'rgba(35, 43, 50, 0.55)', 1.1, [], 0.004);
        }
      }
    }

    // The boundary: plane ∩ surface, and its shadow on the floor.
    if (liftDone) {
      for (const [p, q] of segments) {
        const pz = mapper.toScene(phi(p[0], p[1]));
        const qz = mapper.toScene(phi(q[0], q[1]));
        addDecorationSegment([p[0], p[1], pz], [q[0], q[1], qz], INK, 2.4, [], 2, 0.3);
        if (state.feature !== 'none') addDecorationSegment([p[0], p[1], floorZ], [q[0], q[1], floorZ], 'rgba(35, 43, 50, 0.72)', 1.3, [4, 3], 1, 0.25);
      }
    }

    // Data, lifted, with stems back to the floor.
    const size = state.dataset === 'corners' ? 6.5 : 3.4;
    for (const point of state.data) {
      const lifted = mapper.toScene(liftedValue(point.x1, point.x2));
      if (Math.abs(lifted - floorZ) > 0.015) {
        addSegment([point.x1, point.x2, floorZ], [point.x1, point.x2, lifted], 'rgba(35, 43, 50, 0.2)', 0.8, [], 0.001);
      }
      const hidden = occluded([point.x1, point.x2, lifted]);
      const projected = project([point.x1, point.x2, lifted]);
      decorations.push({
        hidden,
        order: 3,
        draw(ctx) {
          ctx.globalAlpha = hidden ? 0.3 : 1;
          drawMarker(ctx, projected.x, projected.y, point.y, size);
          ctx.globalAlpha = 1;
        }
      });
    }

    // Vertical axis at the farthest floor corner, labels near the nearest one.
    const floorCorners = [[-BOX, -BOX], [BOX, -BOX], [BOX, BOX], [-BOX, BOX]];
    let far = null;
    let near = null;
    for (const [cx, cy] of floorCorners) {
      const depth = project([cx, cy, floorZ]).depth;
      if (!far || depth < far.depth - 1e-9) far = { cx, cy, depth };
      if (!near || depth > near.depth + 1e-9) near = { cx, cy, depth };
    }
    const axisBottom = project([far.cx, far.cy, -H]);
    const axisTop = project([far.cx, far.cy, H]);
    const showAxis = state.camera.elevation < 1.35;
    if (showAxis) {
      context.strokeStyle = 'rgba(35, 43, 50, 0.5)';
      context.lineWidth = 1.2;
      context.beginPath();
      context.moveTo(axisBottom.x, axisBottom.y);
      context.lineTo(axisTop.x, axisTop.y);
      context.stroke();
      const [low, high] = liftedRange();
      context.fillStyle = '#66737d';
      context.font = '10px SFMono-Regular, Consolas, monospace';
      context.textAlign = 'right';
      context.textBaseline = 'middle';
      for (let tick = Math.ceil(low - 1e-9); tick <= Math.floor(high + 1e-9); tick += 1) {
        const p = project([far.cx, far.cy, mapper.toScene(tick)]);
        context.beginPath();
        context.moveTo(p.x - 4, p.y);
        context.lineTo(p.x + 4, p.y);
        context.stroke();
        context.fillText(String(tick), p.x - 7, p.y);
      }
    }

    items.sort((a, b) => a.depth - b.depth);
    context.lineJoin = 'round';
    for (const item of items) item.draw(context);
    decorations.sort((a, b) => (Number(b.hidden) - Number(a.hidden)) || (a.order - b.order));
    for (const decoration of decorations) decoration.draw(context);

    context.fillStyle = '#3f4b54';
    context.font = '700 12px Avenir Next, Segoe UI, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    const outward = 1 + 0.2 / BOX;
    if (state.camera.elevation > 0.2) {
      const labelX1 = project([0, near.cy * outward, floorZ]);
      const labelX2 = project([near.cx * outward, 0, floorZ]);
      context.fillText('x₁', labelX1.x, labelX1.y);
      context.fillText('x₂', labelX2.x, labelX2.y);
      context.font = '10px SFMono-Regular, Consolas, monospace';
      context.fillStyle = '#66737d';
      for (const tick of [-1, 1]) {
        const p1 = project([tick, near.cy * outward, floorZ]);
        const p2 = project([near.cx * outward, tick, floorZ]);
        context.fillText(String(tick), p1.x, p1.y);
        context.fillText(String(tick), p2.x, p2.y);
      }
    }
    context.fillStyle = '#3f4b54';
    context.font = '700 11px Avenir Next, Segoe UI, sans-serif';
    context.textBaseline = 'bottom';
    if (showAxis) {
      context.textAlign = 'center';
      context.fillText(FEATURES[state.lift.to].axis, clamp(axisTop.x, 60, width - 60), Math.max(14, axisTop.y - 8));
    } else {
      context.textAlign = 'left';
      context.fillText(`top view · ${FEATURES[state.lift.to].axis} points at you`, 10, height - 8);
    }

    return planeDrawn;
  }

  // ---------- Loss history ----------
  function drawHistory() {
    const { context, width, height } = setupCanvas(els.history);
    const padding = { left: 46, right: 16, top: 12, bottom: 24 };
    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;
    const values = state.history;
    let peak = 0.8;
    for (const value of values) peak = Math.max(peak, value);
    const yMax = clamp(peak * 1.05, 0.8, 4);
    const total = Math.max(60, values.length - 1);
    const x = (index) => padding.left + (index / total) * innerWidth;
    const y = (value) => padding.top + (1 - clamp(value, 0, yMax) / yMax) * innerHeight;

    context.strokeStyle = '#d7dde0';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(padding.left, padding.top);
    context.lineTo(padding.left, height - padding.bottom);
    context.lineTo(width - padding.right, height - padding.bottom);
    context.stroke();

    context.setLineDash([5, 4]);
    context.strokeStyle = '#bd5f35';
    context.beginPath();
    context.moveTo(padding.left, y(CHANCE_LOSS));
    context.lineTo(width - padding.right, y(CHANCE_LOSS));
    context.stroke();
    context.setLineDash([]);

    context.fillStyle = '#66737d';
    context.font = '10px SFMono-Regular, Consolas, monospace';
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    context.fillText('0', padding.left - 6, y(0));
    context.fillText(yMax.toFixed(1), padding.left - 6, y(yMax));
    context.fillStyle = '#bd5f35';
    context.fillText('ln 2', padding.left - 6, y(CHANCE_LOSS));
    context.fillStyle = '#66737d';
    context.textAlign = 'center';
    context.textBaseline = 'top';
    context.fillText('0', padding.left, height - padding.bottom + 5);
    context.textAlign = 'right';
    context.fillText(`${total} steps`, width - padding.right, height - padding.bottom + 5);

    if (values.length > 0) {
      context.strokeStyle = '#28756e';
      context.lineWidth = 2;
      context.lineJoin = 'round';
      context.beginPath();
      values.forEach((value, index) => (index ? context.lineTo(x(index), y(value)) : context.moveTo(x(index), y(value))));
      context.stroke();
      const last = values[values.length - 1];
      context.fillStyle = '#28756e';
      context.beginPath();
      context.arc(x(values.length - 1), y(last), 3.2, 0, Math.PI * 2);
      context.fill();
    }
  }

  // ---------- Copy ----------
  function formatWeight(value) {
    return `${value < 0 ? '−' : ''}${Math.abs(value).toFixed(2)}`;
  }

  function signedTerm(value, symbol) {
    return ` ${value < 0 ? '−' : '+'} ${Math.abs(value).toFixed(2)}${symbol}`;
  }

  function boundaryEquationText() {
    const { w1, w2, w3, b } = state.weights;
    const feature = FEATURES[state.feature];
    let text = `z = ${formatWeight(b)}${signedTerm(w1, 'x₁')}${signedTerm(w2, 'x₂')}${state.feature === 'none' ? '' : signedTerm(w3, feature.symbol)} = 0`;
    if (state.feature === 'none') {
      text += '   ·   one weight per input, so nothing can bend the line';
    } else if (state.feature === 'product' && Math.abs(w3) > 1e-9) {
      text += `   ⇒   x₂ = −(${formatWeight(b)}${signedTerm(w1, 'x₁')}) / (${formatWeight(w2)}${signedTerm(w3, 'x₁')})   ·   asymptotes x₁ = ${formatWeight(-w2 / w3)}, x₂ = ${formatWeight(-w1 / w3)}`;
    } else if (state.feature === 'radius' && Math.abs(w3) > 1e-9) {
      text += `   ·   a circle centred at (${formatWeight(-w1 / (2 * w3))}, ${formatWeight(-w2 / (2 * w3))})`;
    } else if (state.feature === 'relu') {
      text += '   ·   one line where the ReLU is off, another where it is on';
    }
    return text;
  }

  function planeEquationText() {
    const { w1, w2, w3, b } = state.weights;
    if (state.feature === 'none') {
      return `plane: ${formatWeight(b)}${signedTerm(w1, 'u₁')}${signedTerm(w2, 'u₂')} + 0.00u₃ = 0   ·   u₃ has weight 0, so the plane stands vertical: its top view is the line on the left`;
    }
    return `plane: ${formatWeight(b)}${signedTerm(w1, 'u₁')}${signedTerm(w2, 'u₂')}${signedTerm(w3, 'u₃')} = 0   ·   normal (${formatWeight(w1)}, ${formatWeight(w2)}, ${formatWeight(w3)})   ·   flat whatever the values`;
  }

  function percent(value) {
    return `${Math.round(value * 100)}%`;
  }

  function defaultStatus(metrics) {
    const lossText = metrics.loss.toFixed(3);
    if (state.lastAction === 'hand') {
      const tail = state.feature === 'none'
        ? 'Whatever you try, at least one corner stays on the wrong side.'
        : 'Press Run to let gradient descent finish the job.';
      return `Weights set by hand: loss ${lossText}, accuracy ${percent(metrics.accuracy)}. ${tail}`;
    }
    if (state.steps === 0) return state.statusMessage;
    const step = `Step ${state.steps.toLocaleString('en-US')}`;
    if (state.feature === 'none') {
      return `${step}: loss ${lossText} against a floor of ln 2 ≈ 0.693. Gradient descent is shrinking the line toward w = 0, the convex optimum. No line beats chance on XOR.`;
    }
    if (state.feature === 'product') {
      return metrics.accuracy === 1
        ? `${step}: every point is on the correct side. The plane is flat in (x₁, x₂, x₁x₂); its trace on the saddle projects to the hyperbola on the left.`
        : `${step}: w₃ is growing negative, tilting the plane so that the raised blue corners fall on its negative side.`;
    }
    if (state.feature === 'radius') {
      return `${step}: accuracy ${percent(metrics.accuracy)}. All four clusters were lifted by about the same amount, so the plane still cannot separate them.`;
    }
    return metrics.accuracy === 1
      ? `${step}: every point is correct. The line handles three corners and the ReLU feature carves out the fourth.`
      : `${step}: the ReLU column is nonzero only near (1, 1); the plane is learning to push that corner down.`;
  }

  // ---------- Render ----------
  function syncButtons(selector, attribute, active) {
    document.querySelectorAll(selector).forEach((button) => {
      const isActive = button.dataset[attribute] === active;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', String(isActive));
    });
  }

  function render() {
    const metrics = measure();
    const feature = FEATURES[state.feature];
    const kind = boundaryKind();

    els.workspaceEyebrow.textContent = feature.eyebrow;
    els.workspaceTitle.textContent = feature.title;
    els.featureHelp.textContent = feature.help;
    els.datasetBadge.textContent = state.dataset === 'corners' ? '4 points · the Boolean corners' : '100 points · 25 per corner';
    els.stepValue.textContent = state.steps.toLocaleString('en-US');
    els.loss.textContent = metrics.loss.toFixed(3);
    els.lossNote.textContent = state.feature === 'none' ? 'floor for a line: ln 2 = 0.693' : 'chance = ln 2 = 0.693';
    els.accuracy.textContent = percent(metrics.accuracy);
    els.accuracyNote.textContent = metrics.ties > 0
      ? `${metrics.ties} ${metrics.ties === 1 ? 'point sits' : 'points sit'} at p = 0.5, counted as half`
      : (state.dataset === 'corners' ? 'on the 4 XOR corners' : 'on 100 points, 25 per corner');
    els.boundary.textContent = kind;
    els.dimension.textContent = state.feature === 'none' ? 'line in 2D' : 'plane in 3D';
    els.dimensionNote.textContent = state.feature === 'none' ? 'flat in (x₁, x₂)' : `flat in (x₁, x₂, ${feature.third})`;
    els.inputSummary.textContent = `step ${state.steps.toLocaleString('en-US')} · ${percent(metrics.accuracy)} correct`;
    els.historySummary.textContent = `${state.steps.toLocaleString('en-US')} steps · loss ${metrics.loss.toFixed(3)}`;
    els.boundaryEquation.textContent = boundaryEquationText();
    els.planeEquation.textContent = planeEquationText();

    for (const key of WEIGHT_KEYS) {
      const value = state.weights[key];
      els.sliders[key].value = String(clamp(value, -SLIDER_LIMIT, SLIDER_LIMIT));
      els.outputs[key].textContent = formatWeight(value);
    }
    const w3Disabled = state.feature === 'none';
    els.sliders.w3.disabled = w3Disabled;
    els.sliders.w3.closest('.slider-row').classList.toggle('is-disabled', w3Disabled);
    els.w3Label.textContent = `w₃ · ${feature.third || 'φ (absent)'}`;

    syncButtons('[data-dataset]', 'dataset', state.dataset);
    syncButtons('[data-feature]', 'feature', state.feature);
    syncButtons('[data-view]', 'view', state.view);

    els.run.textContent = state.running ? 'Pause' : 'Run';
    els.run.classList.toggle('is-running', state.running);
    els.run.setAttribute('aria-pressed', String(state.running));
    els.status.textContent = state.running ? `Running from step ${state.steps.toLocaleString('en-US')}…` : defaultStatus(metrics);

    const segments = boundarySegments(64);
    drawInputSpace(segments);
    drawFeatureSpace(segments);
    drawHistory();

    els.input.setAttribute('aria-label', `Input-space decision map after ${state.steps} gradient steps: ${kind} boundary, ${percent(metrics.accuracy)} accuracy.`);
    els.feature.setAttribute('aria-label', `Feature-space view (${state.view}) of the data lifted by ${feature.third || 'nothing'}, with a ${state.feature === 'none' ? 'vertical' : 'tilted'} flat plane; ${percent(metrics.accuracy)} accuracy.`);
    els.history.setAttribute('aria-label', `Loss history over ${state.steps} steps, currently ${metrics.loss.toFixed(3)} against the chance level ${CHANCE_LOSS.toFixed(3)}.`);
    return metrics;
  }

  // ---------- Animation loop ----------
  function ensureLoop() {
    if (state.loop === null) state.loop = requestAnimationFrame(frame);
  }

  function frame(now) {
    state.loop = null;
    if (state.running) trainSteps(STEPS_PER_FRAME);
    if (state.lift.t < 1) {
      state.lift.t = state.lift.duration > 0 ? clamp((now - state.lift.start) / state.lift.duration, 0, 1) : 1;
    }
    if (state.cameraTween) {
      const tween = state.cameraTween;
      const t = tween.duration > 0 ? clamp((now - tween.start) / tween.duration, 0, 1) : 1;
      const mix = easeInOut(t);
      for (const key of ['azimuth', 'elevation', 'zoom']) state.camera[key] = tween.from[key] + (tween.to[key] - tween.from[key]) * mix;
      if (t >= 1) state.cameraTween = null;
    }
    render();
    if (state.running || state.lift.t < 1 || state.cameraTween) ensureLoop();
  }

  function stopRunning() {
    state.running = false;
    els.run?.setAttribute('aria-pressed', 'false');
  }

  function toggleRunning() {
    if (state.running) {
      stopRunning();
      render();
      return;
    }
    if (state.steps >= MAX_STEPS) resetWeights();
    state.running = true;
    render();
    ensureLoop();
  }

  // ---------- State changes ----------
  function setFeature(key, animate = true) {
    stopRunning();
    const previous = state.feature;
    state.feature = key;
    state.weights.w3 = 0;
    state.steps = 0;
    state.lastAction = 'load';
    state.statusMessage = FEATURES[key].loaded;
    resetHistory();
    beginLift(previous, key, animate && previous !== key);
    render();
  }

  function setDataset(kind) {
    stopRunning();
    state.dataset = kind;
    state.data = makeDataset(kind);
    state.steps = 0;
    state.lastAction = 'load';
    state.statusMessage = kind === 'corners'
      ? 'Only the four Boolean corners. Try to place a line that gets all four right.'
      : 'Four noisy clusters, 25 points each, one per corner.';
    resetHistory();
    render();
  }

  function resetWeights() {
    stopRunning();
    state.weights = { ...INITIAL_WEIGHTS };
    state.steps = 0;
    state.lastAction = 'load';
    state.statusMessage = FEATURES[state.feature].loaded;
    resetHistory();
    render();
  }

  function setWeights(partial) {
    stopRunning();
    for (const key of WEIGHT_KEYS) {
      if (key === 'w3' && state.feature === 'none') continue;
      const value = Number(partial?.[key]);
      if (Number.isFinite(value)) state.weights[key] = value;
    }
    state.lastAction = 'hand';
    render();
  }

  // ---------- Events ----------
  document.querySelectorAll('[data-dataset]').forEach((button) => {
    button.addEventListener('click', () => setDataset(button.dataset.dataset));
  });
  document.querySelectorAll('[data-feature]').forEach((button) => {
    button.addEventListener('click', () => setFeature(button.dataset.feature, true));
  });
  document.querySelectorAll('[data-view]').forEach((button) => {
    button.addEventListener('click', () => setView(button.dataset.view, true));
  });
  for (const key of WEIGHT_KEYS) {
    els.sliders[key].addEventListener('input', () => {
      if (els.sliders[key].disabled) return;
      stopRunning();
      state.weights[key] = Number(els.sliders[key].value);
      state.lastAction = 'hand';
      render();
    });
  }
  els.run.addEventListener('click', toggleRunning);
  els.step.addEventListener('click', () => {
    stopRunning();
    trainSteps(1);
    render();
  });
  els.reset.addEventListener('click', resetWeights);
  window.addEventListener('resize', render);

  els.feature.addEventListener('pointerdown', (event) => {
    state.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, azimuth: state.camera.azimuth, elevation: state.camera.elevation };
    els.feature.setPointerCapture?.(event.pointerId);
  });
  els.feature.addEventListener('pointermove', (event) => {
    if (!state.drag || state.drag.id !== event.pointerId) return;
    state.camera.azimuth = state.drag.azimuth - (event.clientX - state.drag.x) * 0.012;
    state.camera.elevation = clamp(state.drag.elevation + (event.clientY - state.drag.y) * 0.008, 0, Math.PI / 2);
    markCameraCustom();
    render();
  });
  const endDrag = (event) => {
    if (!state.drag || (event.pointerId !== undefined && event.pointerId !== state.drag.id)) return;
    state.drag = null;
  };
  els.feature.addEventListener('pointerup', endDrag);
  els.feature.addEventListener('pointercancel', endDrag);
  els.feature.addEventListener('lostpointercapture', endDrag);
  els.feature.addEventListener('wheel', (event) => {
    event.preventDefault();
    state.camera.zoom = clamp(state.camera.zoom * Math.exp(-event.deltaY * 0.0012), 0.6, 2.6);
    markCameraCustom();
    render();
  }, { passive: false });
  els.feature.addEventListener('keydown', (event) => {
    let handled = true;
    if (event.key === 'ArrowLeft') state.camera.azimuth += 0.12;
    else if (event.key === 'ArrowRight') state.camera.azimuth -= 0.12;
    else if (event.key === 'ArrowUp') state.camera.elevation = clamp(state.camera.elevation + 0.08, 0, Math.PI / 2);
    else if (event.key === 'ArrowDown') state.camera.elevation = clamp(state.camera.elevation - 0.08, 0, Math.PI / 2);
    else if (event.key === '+' || event.key === '=') state.camera.zoom = clamp(state.camera.zoom * 1.1, 0.6, 2.6);
    else if (event.key === '-') state.camera.zoom = clamp(state.camera.zoom / 1.1, 0.6, 2.6);
    else if (event.key === '0') { event.preventDefault(); setView('orbit', true); return; }
    else handled = false;
    if (handled) {
      event.preventDefault();
      markCameraCustom();
      render();
    }
  });

  // ---------- Public test surface ----------
  function snapshot() {
    const metrics = measure();
    const segments = boundarySegments(48);
    const { w1, w2, w3, b } = state.weights;
    let lineDeviation = 0;
    if (state.feature === 'none' && Math.hypot(w1, w2) > 1e-9) {
      for (const segment of segments) {
        for (const [x, y] of segment) lineDeviation = Math.max(lineDeviation, Math.abs(w1 * x + w2 * y + b) / Math.hypot(w1, w2));
      }
    }
    const basis = cameraBasis(state.camera);
    const unitNormal = normalize(sceneNormal());
    return {
      dataset: state.dataset,
      feature: state.feature,
      steps: state.steps,
      running: state.running,
      learningRate: LEARNING_RATE,
      datasetSize: state.data.length,
      classCounts: [state.data.filter((point) => point.y === 0).length, state.data.filter((point) => point.y === 1).length],
      loss: metrics.loss,
      accuracy: metrics.accuracy,
      ties: metrics.ties,
      chanceLoss: CHANCE_LOSS,
      weights: { w1, w2, w3, b },
      boundaryKind: boundaryKind(),
      featureDimension: state.feature === 'none' ? 2 : 3,
      boundarySegmentCount: segments.length,
      boundaryLineDeviation: lineDeviation,
      lift: state.lift.t,
      view: state.view,
      camera: { ...state.camera },
      planeEdgeAlignment: Math.abs(dot(basis.toward, unitNormal)),
      historyLength: state.history.length,
      w3SliderDisabled: els.sliders.w3.disabled,
      lastAction: state.lastAction
    };
  }

  window.XorFeatureLab = Object.freeze({
    snapshot,
    setFeature(key, options = {}) {
      if (!Object.hasOwn(FEATURES, key)) throw new RangeError(`Unknown feature: ${key}`);
      setFeature(key, options.animate !== false);
      return snapshot();
    },
    setDataset(kind) {
      if (kind !== 'clusters' && kind !== 'corners') throw new RangeError(`Unknown dataset: ${kind}`);
      setDataset(kind);
      return snapshot();
    },
    setWeights(partial) {
      setWeights(partial);
      return snapshot();
    },
    step(count = 1) {
      stopRunning();
      trainSteps(Math.max(1, Math.round(Number(count) || 1)));
      render();
      return snapshot();
    },
    reset() {
      resetWeights();
      return snapshot();
    },
    start() {
      if (!state.running) toggleRunning();
      return snapshot();
    },
    stop() {
      stopRunning();
      render();
      return snapshot();
    },
    setView(name, options = {}) {
      setView(name, options.animate !== false);
      return snapshot();
    },
    setCamera(camera = {}) {
      if (Number.isFinite(camera.azimuth)) state.camera.azimuth = camera.azimuth;
      if (Number.isFinite(camera.elevation)) state.camera.elevation = clamp(camera.elevation, 0, Math.PI / 2);
      if (Number.isFinite(camera.zoom)) state.camera.zoom = clamp(camera.zoom, 0.6, 2.6);
      markCameraCustom();
      render();
      return snapshot();
    }
  });

  resetHistory();
  render();
})();
