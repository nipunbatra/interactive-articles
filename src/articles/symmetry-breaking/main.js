(() => {
  'use strict';

  const WIDTH = 4;
  const LEARNING_RATE = 0.20;
  const MAX_STEPS = 5000;
  const DOMAIN = [-1.08, 1.08];
  const NEURON_COLORS = ['#2c67a0', '#bd5f35', '#28756e', '#765c9d'];
  const MODE_COPY = {
    identical: {
      eyebrow: 'IDENTICAL INITIALIZATION',
      title: 'Can four overlaid hinges solve XOR?',
      help: 'All four parameter slots contain the same numbers. They are separate arrays, not shared weights.'
    },
    perturbed: {
      eyebrow: 'SLIGHTLY PERTURBED INITIALIZATION',
      title: 'Do small offsets release the four features?',
      help: 'Each neuron starts near the identical baseline, with a small deterministic offset in its own parameters.'
    },
    independent: {
      eyebrow: 'INDEPENDENT INITIALIZATION',
      title: 'Can distinct directions cover all four XOR regions?',
      help: 'A fixed pseudo-random seed gives every neuron an independent incoming row, bias, and outgoing coefficient.'
    }
  };

  const els = {
    run: document.getElementById('runBtn'),
    step: document.getElementById('stepBtn'),
    reset: document.getElementById('resetBtn'),
    stepValue: document.getElementById('stepValue'),
    status: document.getElementById('statusText'),
    modeHelp: document.getElementById('modeHelp'),
    workspaceEyebrow: document.getElementById('workspaceEyebrow'),
    workspaceTitle: document.getElementById('workspaceTitle'),
    loss: document.getElementById('lossValue'),
    accuracy: document.getElementById('accuracyValue'),
    accuracyNote: document.getElementById('accuracyNote'),
    spread: document.getElementById('spreadValue'),
    rank: document.getElementById('rankValue'),
    surface: document.getElementById('surfaceCanvas'),
    hinge: document.getElementById('hingeCanvas'),
    surfaceSummary: document.getElementById('surfaceSummary'),
    hingeSummary: document.getElementById('hingeSummary'),
    neuronLegend: document.getElementById('neuronLegend')
  };

  function makeDataset() {
    // Four equally sized quadrants, with a margin around the axes. The margin
    // makes the 3-of-4 corner limitation of a single feature easy to see.
    const coordinates = [-0.95, -0.80, -0.65, -0.50, -0.35, 0.35, 0.50, 0.65, 0.80, 0.95];
    const points = [];
    for (const x1 of coordinates) {
      for (const x2 of coordinates) {
        points.push({ x1, x2, y: x1 * x2 < 0 ? 1 : 0 });
      }
    }
    return points;
  }

  const DATA = makeDataset();

  function seededRandom(seed) {
    let value = seed >>> 0;
    return () => {
      value = (1664525 * value + 1013904223) >>> 0;
      return value / 4294967296;
    };
  }

  function cloneModel(model) {
    return {
      w: model.w.map((row) => row.slice()),
      b: model.b.slice(),
      v: model.v.slice(),
      c: model.c
    };
  }

  function makeModel(mode) {
    const base = { w: [0.90, 0.90], b: -0.25, v: -0.18 };
    if (mode === 'identical') {
      return {
        // Every row is cloned, so equal updates demonstrate symmetry rather
        // than accidental JavaScript reference sharing.
        w: Array.from({ length: WIDTH }, () => base.w.slice()),
        b: Array(WIDTH).fill(base.b),
        v: Array(WIDTH).fill(base.v),
        c: 0.35
      };
    }

    if (mode === 'perturbed') {
      const offsets = [
        [0.10, -0.05, 0.040, 0.025],
        [-0.08, 0.07, -0.035, -0.015],
        [0.04, 0.11, 0.020, 0.015],
        [-0.07, -0.09, -0.025, -0.020]
      ];
      return {
        w: offsets.map(([dw1, dw2]) => [base.w[0] + dw1, base.w[1] + dw2]),
        b: offsets.map(([, , db]) => base.b + db),
        v: offsets.map(([, , , dv]) => base.v + dv),
        c: 0.35
      };
    }

    const random = seededRandom(12);
    const model = { w: [], b: [], v: [], c: 0 };
    for (let neuron = 0; neuron < WIDTH; neuron += 1) {
      model.w.push([(random() * 2 - 1) * 1.2, (random() * 2 - 1) * 1.2]);
      model.b.push((random() * 2 - 1) * 0.25);
      model.v.push((random() * 2 - 1) * 0.65);
    }
    return model;
  }

  const state = {
    mode: 'identical',
    model: makeModel('identical'),
    initialModel: null,
    steps: 0,
    running: false,
    animationFrame: null,
    statusMessage: 'Identical initialization loaded. Predict what can change after one step.'
  };
  state.initialModel = cloneModel(state.model);

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const expZ = Math.exp(z);
    return expZ / (1 + expZ);
  }

  function forward(model, x1, x2) {
    const pre = Array(WIDTH);
    const hidden = Array(WIDTH);
    let logit = model.c;
    for (let neuron = 0; neuron < WIDTH; neuron += 1) {
      pre[neuron] = model.w[neuron][0] * x1 + model.w[neuron][1] * x2 + model.b[neuron];
      hidden[neuron] = Math.max(0, pre[neuron]);
      logit += model.v[neuron] * hidden[neuron];
    }
    return { pre, hidden, logit, probability: sigmoid(logit) };
  }

  function binaryCrossEntropyFromLogit(logit, target) {
    return Math.max(logit, 0) - target * logit + Math.log1p(Math.exp(-Math.abs(logit)));
  }

  function jacobiEigenvalues(matrix) {
    const values = matrix.map((row) => row.slice());
    for (let iteration = 0; iteration < 60; iteration += 1) {
      let p = 0;
      let q = 1;
      let largest = 0;
      for (let row = 0; row < WIDTH; row += 1) {
        for (let column = row + 1; column < WIDTH; column += 1) {
          const candidate = Math.abs(values[row][column]);
          if (candidate > largest) {
            largest = candidate;
            p = row;
            q = column;
          }
        }
      }
      if (largest < 1e-12) break;

      const app = values[p][p];
      const aqq = values[q][q];
      const apq = values[p][q];
      const angle = 0.5 * Math.atan2(2 * apq, aqq - app);
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);

      for (let index = 0; index < WIDTH; index += 1) {
        if (index === p || index === q) continue;
        const aip = values[index][p];
        const aiq = values[index][q];
        values[index][p] = values[p][index] = cosine * aip - sine * aiq;
        values[index][q] = values[q][index] = sine * aip + cosine * aiq;
      }
      values[p][p] = cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq;
      values[q][q] = sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq;
      values[p][q] = values[q][p] = 0;
    }
    return values.map((row, index) => Math.max(0, row[index])).sort((a, b) => b - a);
  }

  function hiddenFeatureRank(model) {
    const gram = Array.from({ length: WIDTH }, () => Array(WIDTH).fill(0));
    for (const point of DATA) {
      const hidden = forward(model, point.x1, point.x2).hidden;
      for (let row = 0; row < WIDTH; row += 1) {
        for (let column = 0; column < WIDTH; column += 1) {
          gram[row][column] += hidden[row] * hidden[column] / DATA.length;
        }
      }
    }
    const eigenvalues = jacobiEigenvalues(gram);
    const threshold = Math.max(1e-10, eigenvalues[0] * 1e-6);
    return eigenvalues.filter((value) => value > threshold).length;
  }

  function maximumNeuronSeparation(model) {
    let maximum = 0;
    for (let left = 0; left < WIDTH; left += 1) {
      for (let right = left + 1; right < WIDTH; right += 1) {
        const distance = Math.hypot(
          model.w[left][0] - model.w[right][0],
          model.w[left][1] - model.w[right][1],
          model.b[left] - model.b[right]
        );
        maximum = Math.max(maximum, distance);
      }
    }
    return maximum;
  }

  function distinctDirections(model) {
    const directions = [];
    for (const row of model.w) {
      const norm = Math.hypot(row[0], row[1]);
      if (norm < 1e-10) continue;
      const candidate = [row[0] / norm, row[1] / norm];
      const alreadySeen = directions.some((direction) => {
        const dot = Math.max(-1, Math.min(1, direction[0] * candidate[0] + direction[1] * candidate[1]));
        return Math.acos(dot) < 0.10;
      });
      if (!alreadySeen) directions.push(candidate);
    }
    return directions.length;
  }

  function measure(model = state.model) {
    let loss = 0;
    let correct = 0;
    for (const point of DATA) {
      const prediction = forward(model, point.x1, point.x2);
      loss += binaryCrossEntropyFromLogit(prediction.logit, point.y);
      correct += Number((prediction.logit >= 0 ? 1 : 0) === point.y);
    }
    return {
      loss: loss / DATA.length,
      accuracy: correct / DATA.length,
      separation: maximumNeuronSeparation(model),
      featureRank: hiddenFeatureRank(model),
      distinctDirections: distinctDirections(model)
    };
  }

  function gradientStep() {
    const gradientW = Array.from({ length: WIDTH }, () => [0, 0]);
    const gradientB = Array(WIDTH).fill(0);
    const gradientV = Array(WIDTH).fill(0);
    let gradientC = 0;

    for (const point of DATA) {
      const prediction = forward(state.model, point.x1, point.x2);
      const error = (prediction.probability - point.y) / DATA.length;
      gradientC += error;
      for (let neuron = 0; neuron < WIDTH; neuron += 1) {
        gradientV[neuron] += error * prediction.hidden[neuron];
        if (prediction.pre[neuron] > 0) {
          const hiddenError = error * state.model.v[neuron];
          gradientW[neuron][0] += hiddenError * point.x1;
          gradientW[neuron][1] += hiddenError * point.x2;
          gradientB[neuron] += hiddenError;
        }
      }
    }

    state.model.c -= LEARNING_RATE * gradientC;
    for (let neuron = 0; neuron < WIDTH; neuron += 1) {
      state.model.w[neuron][0] -= LEARNING_RATE * gradientW[neuron][0];
      state.model.w[neuron][1] -= LEARNING_RATE * gradientW[neuron][1];
      state.model.b[neuron] -= LEARNING_RATE * gradientB[neuron];
      state.model.v[neuron] -= LEARNING_RATE * gradientV[neuron];
    }
    state.steps += 1;
  }

  function trainSteps(count) {
    const requested = Math.max(0, Math.floor(Number(count) || 0));
    const available = Math.max(0, MAX_STEPS - state.steps);
    const actual = Math.min(requested, available);
    for (let index = 0; index < actual; index += 1) gradientStep();
    if (state.steps >= MAX_STEPS) stopRunning();
    return actual;
  }

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

  function makeTransform(width, height) {
    const padding = { left: 43, right: 15, top: 16, bottom: 36 };
    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;
    const [minimum, maximum] = DOMAIN;
    return {
      padding,
      innerWidth,
      innerHeight,
      x: (value) => padding.left + (value - minimum) / (maximum - minimum) * innerWidth,
      y: (value) => padding.top + (maximum - value) / (maximum - minimum) * innerHeight
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
    context.textAlign = 'left';
    context.textBaseline = 'top';
    context.fillText('x₂', padding.left + 6, padding.top + 2);
  }

  function probabilityColor(probability) {
    const classZero = [218, 232, 244];
    const classOne = [245, 220, 207];
    const red = Math.round(classZero[0] * (1 - probability) + classOne[0] * probability);
    const green = Math.round(classZero[1] * (1 - probability) + classOne[1] * probability);
    const blue = Math.round(classZero[2] * (1 - probability) + classOne[2] * probability);
    return `rgb(${red}, ${green}, ${blue})`;
  }

  function drawDecisionSurface() {
    const { context, width, height } = setupCanvas(els.surface);
    const transform = makeTransform(width, height);
    const { padding, innerWidth, innerHeight, x, y } = transform;
    const cells = 54;
    const domainWidth = DOMAIN[1] - DOMAIN[0];
    const cellWidth = innerWidth / cells;
    const cellHeight = innerHeight / cells;

    for (let row = 0; row < cells; row += 1) {
      for (let column = 0; column < cells; column += 1) {
        const x1 = DOMAIN[0] + (column + 0.5) / cells * domainWidth;
        const x2 = DOMAIN[1] - (row + 0.5) / cells * domainWidth;
        const probability = forward(state.model, x1, x2).probability;
        context.fillStyle = probabilityColor(probability);
        context.fillRect(padding.left + column * cellWidth, padding.top + row * cellHeight, cellWidth + 0.6, cellHeight + 0.6);
      }
    }

    // Approximate p=0.5 contour by marking cells whose right or lower neighbour
    // changes class. This keeps the rendered boundary tied to real predictions.
    context.fillStyle = 'rgba(35, 43, 50, 0.68)';
    for (let row = 0; row < cells - 1; row += 1) {
      for (let column = 0; column < cells - 1; column += 1) {
        const x1 = DOMAIN[0] + (column + 0.5) / cells * domainWidth;
        const x2 = DOMAIN[1] - (row + 0.5) / cells * domainWidth;
        const here = forward(state.model, x1, x2).logit >= 0;
        const right = forward(state.model, x1 + domainWidth / cells, x2).logit >= 0;
        const below = forward(state.model, x1, x2 - domainWidth / cells).logit >= 0;
        if (here !== right || here !== below) {
          context.fillRect(padding.left + column * cellWidth, padding.top + row * cellHeight, 2, 2);
        }
      }
    }

    drawAxes(context, width, height, transform);
    for (const point of DATA) {
      const px = x(point.x1);
      const py = y(point.x2);
      context.fillStyle = point.y === 0 ? '#2c67a0' : '#bd5f35';
      context.strokeStyle = '#ffffff';
      context.lineWidth = 1.2;
      context.beginPath();
      if (point.y === 0) {
        context.arc(px, py, 3.2, 0, Math.PI * 2);
      } else {
        context.rect(px - 3.0, py - 3.0, 6.0, 6.0);
      }
      context.fill();
      context.stroke();
    }
  }

  function hingeIntersections(row, bias) {
    const [minimum, maximum] = DOMAIN;
    const candidates = [];
    if (Math.abs(row[1]) > 1e-10) {
      for (const xValue of [minimum, maximum]) {
        const yValue = (-bias - row[0] * xValue) / row[1];
        if (yValue >= minimum - 1e-9 && yValue <= maximum + 1e-9) candidates.push([xValue, yValue]);
      }
    }
    if (Math.abs(row[0]) > 1e-10) {
      for (const yValue of [minimum, maximum]) {
        const xValue = (-bias - row[1] * yValue) / row[0];
        if (xValue >= minimum - 1e-9 && xValue <= maximum + 1e-9) candidates.push([xValue, yValue]);
      }
    }
    const unique = candidates.filter((point, index) => (
      candidates.findIndex((other) => Math.hypot(point[0] - other[0], point[1] - other[1]) < 1e-7) === index
    ));
    return unique.slice(0, 2);
  }

  function drawArrow(context, fromX, fromY, toX, toY, color) {
    const angle = Math.atan2(toY - fromY, toX - fromX);
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 2.2;
    context.beginPath();
    context.moveTo(fromX, fromY);
    context.lineTo(toX, toY);
    context.stroke();
    context.beginPath();
    context.moveTo(toX, toY);
    context.lineTo(toX - 8 * Math.cos(angle - Math.PI / 6), toY - 8 * Math.sin(angle - Math.PI / 6));
    context.lineTo(toX - 8 * Math.cos(angle + Math.PI / 6), toY - 8 * Math.sin(angle + Math.PI / 6));
    context.closePath();
    context.fill();
  }

  function drawSingleHinge(context, transform, neuron, color, lineWidth = 2.4, label = true) {
    const row = state.model.w[neuron];
    const bias = state.model.b[neuron];
    const intersections = hingeIntersections(row, bias);
    if (intersections.length === 2) {
      context.strokeStyle = color;
      context.lineWidth = lineWidth;
      context.beginPath();
      context.moveTo(transform.x(intersections[0][0]), transform.y(intersections[0][1]));
      context.lineTo(transform.x(intersections[1][0]), transform.y(intersections[1][1]));
      context.stroke();
    }

    const normSquared = row[0] * row[0] + row[1] * row[1];
    if (normSquared < 1e-10) return;
    const norm = Math.sqrt(normSquared);
    const originX = -bias * row[0] / normSquared;
    const originY = -bias * row[1] / normSquared;
    const tipX = originX + 0.28 * row[0] / norm;
    const tipY = originY + 0.28 * row[1] / norm;
    drawArrow(context, transform.x(originX), transform.y(originY), transform.x(tipX), transform.y(tipY), color);
    if (label) {
      context.fillStyle = color;
      context.font = '800 11px SFMono-Regular, Consolas, monospace';
      context.textAlign = 'left';
      context.textBaseline = 'bottom';
      context.fillText(`h${neuron + 1}`, transform.x(tipX) + 5, transform.y(tipY) - 4);
    }
  }

  function drawHinges(metrics) {
    const { context, width, height } = setupCanvas(els.hinge);
    const transform = makeTransform(width, height);
    const { padding, x, y } = transform;

    // Faint XOR target quadrants are context, not model output.
    context.fillStyle = 'rgba(44, 103, 160, 0.045)';
    context.fillRect(padding.left, padding.top, x(0) - padding.left, y(0) - padding.top);
    context.fillRect(x(0), y(0), width - padding.right - x(0), height - padding.bottom - y(0));
    context.fillStyle = 'rgba(189, 95, 53, 0.045)';
    context.fillRect(x(0), padding.top, width - padding.right - x(0), y(0) - padding.top);
    context.fillRect(padding.left, y(0), x(0) - padding.left, height - padding.bottom - y(0));
    drawAxes(context, width, height, transform);

    if (metrics.separation < 1e-12) {
      drawSingleHinge(context, transform, 0, '#232b32', 4.2, false);
      const row = state.model.w[0];
      const normSquared = row[0] * row[0] + row[1] * row[1];
      const originX = -state.model.b[0] * row[0] / normSquared;
      const originY = -state.model.b[0] * row[1] / normSquared;
      context.fillStyle = '#232b32';
      context.font = '800 11px SFMono-Regular, Consolas, monospace';
      context.textAlign = 'left';
      context.fillText('h1 = h2 = h3 = h4', x(originX) + 9, y(originY) - 9);
    } else {
      for (let neuron = 0; neuron < WIDTH; neuron += 1) {
        drawSingleHinge(context, transform, neuron, NEURON_COLORS[neuron]);
      }
    }

    context.fillStyle = '#66737d';
    context.font = '10px Avenir Next, Segoe UI, sans-serif';
    context.textAlign = 'right';
    context.textBaseline = 'bottom';
    context.fillText('faint quadrants show the XOR target; lines show the learned hidden features', width - padding.right, height - 7);
  }

  function formatSeparation(value) {
    if (value < 5e-8) return '0.0000';
    return value < 0.01 ? value.toExponential(2) : value.toFixed(4);
  }

  function updateNeuronLegend() {
    els.neuronLegend.replaceChildren();
    for (let neuron = 0; neuron < WIDTH; neuron += 1) {
      const chip = document.createElement('span');
      chip.className = 'neuron-chip';
      chip.style.setProperty('--neuron-color', NEURON_COLORS[neuron]);
      const key = document.createElement('i');
      key.setAttribute('aria-hidden', 'true');
      const text = document.createElement('code');
      const row = state.model.w[neuron];
      text.textContent = `h${neuron + 1}: w=(${row[0].toFixed(2)}, ${row[1].toFixed(2)}), b=${state.model.b[neuron].toFixed(2)}`;
      chip.append(key, text);
      els.neuronLegend.append(chip);
    }
  }

  function defaultStatus(metrics) {
    if (state.steps === 0) return state.statusMessage;
    if (state.mode === 'identical') {
      if (state.steps >= 500) return `After ${state.steps} steps, separation is still exactly zero and accuracy is ${Math.round(metrics.accuracy * 100)}%. Width four is still effective width one.`;
      return `Step ${state.steps}: equal parameters received equal gradients, so all four hidden neurons remain identical.`;
    }
    if (state.mode === 'perturbed') {
      return `Step ${state.steps}: the equality constraint is broken; measured separation is ${formatSeparation(metrics.separation)} and feature rank is ${metrics.featureRank}.`;
    }
    if (metrics.accuracy === 1) return `Step ${state.steps}: the independent features now classify all 100 XOR points correctly.`;
    return `Step ${state.steps}: distinct hinges are moving under the same full-batch gradient descent rule.`;
  }

  function render() {
    const metrics = measure();
    const copy = MODE_COPY[state.mode];
    els.workspaceEyebrow.textContent = copy.eyebrow;
    els.workspaceTitle.textContent = copy.title;
    els.modeHelp.textContent = copy.help;
    els.stepValue.textContent = state.steps.toLocaleString('en-US');
    els.loss.textContent = metrics.loss.toFixed(4);
    els.accuracy.textContent = `${(metrics.accuracy * 100).toFixed(0)}%`;
    els.accuracyNote.textContent = state.mode === 'identical' && state.steps >= 500 ? 'near the 3-of-4 ceiling' : 'balanced XOR';
    els.spread.textContent = formatSeparation(metrics.separation);
    els.rank.textContent = `${metrics.featureRank} / 4`;
    els.surfaceSummary.textContent = `step ${state.steps.toLocaleString('en-US')} · ${(metrics.accuracy * 100).toFixed(0)}% correct`;
    els.hingeSummary.textContent = metrics.separation < 1e-12
      ? '4 exact overlaps'
      : `${metrics.distinctDirections} direction${metrics.distinctDirections === 1 ? '' : 's'} · rank ${metrics.featureRank}`;
    els.run.textContent = state.running ? 'Pause' : 'Run';
    els.run.classList.toggle('is-running', state.running);
    els.run.setAttribute('aria-pressed', String(state.running));
    els.status.textContent = state.running ? `Running from step ${state.steps.toLocaleString('en-US')}…` : defaultStatus(metrics);

    document.querySelectorAll('[data-mode]').forEach((button) => {
      const active = button.dataset.mode === state.mode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });

    drawDecisionSurface();
    drawHinges(metrics);
    updateNeuronLegend();
    els.surface.setAttribute('aria-label', `XOR decision surface after ${state.steps} gradient steps; ${Math.round(metrics.accuracy * 100)} percent accuracy in ${state.mode} mode.`);
    els.hinge.setAttribute('aria-label', metrics.separation < 1e-12
      ? 'All four hidden ReLU hinge lines and active-direction arrows overlap exactly.'
      : `Four hidden ReLU hinge lines with ${metrics.distinctDirections} distinct active directions and feature rank ${metrics.featureRank}.`);
    return metrics;
  }

  function stopRunning() {
    state.running = false;
    if (state.animationFrame !== null) cancelAnimationFrame(state.animationFrame);
    state.animationFrame = null;
    els.run?.setAttribute('aria-pressed', 'false');
  }

  function runFrame() {
    if (!state.running) return;
    trainSteps(8);
    render();
    if (state.running) state.animationFrame = requestAnimationFrame(runFrame);
  }

  function toggleRunning() {
    if (state.running) {
      stopRunning();
      render();
      return;
    }
    if (state.steps >= MAX_STEPS) resetMode(state.mode);
    state.running = true;
    render();
    state.animationFrame = requestAnimationFrame(runFrame);
  }

  function resetMode(mode = state.mode) {
    stopRunning();
    state.mode = mode;
    state.model = makeModel(mode);
    state.initialModel = cloneModel(state.model);
    state.steps = 0;
    state.statusMessage = mode === 'identical'
      ? 'Identical initialization loaded. Predict what can change after one step.'
      : mode === 'perturbed'
        ? 'Small deterministic offsets loaded. The four hinges no longer coincide exactly.'
        : 'Independent seed 12 loaded. Run gradient descent and watch the four XOR regions.';
    render();
  }

  document.querySelectorAll('[data-mode]').forEach((button) => {
    button.addEventListener('click', () => resetMode(button.dataset.mode));
  });
  els.run.addEventListener('click', toggleRunning);
  els.step.addEventListener('click', () => {
    stopRunning();
    trainSteps(1);
    render();
  });
  els.reset.addEventListener('click', () => resetMode(state.mode));
  window.addEventListener('resize', render);

  function parameterSignature(model = state.model) {
    return [model.c, ...model.w.flat(), ...model.b, ...model.v]
      .map((value) => value.toFixed(12))
      .join('|');
  }

  function snapshot() {
    const metrics = measure();
    return {
      mode: state.mode,
      steps: state.steps,
      running: state.running,
      learningRate: LEARNING_RATE,
      datasetSize: DATA.length,
      classCounts: [DATA.filter((point) => point.y === 0).length, DATA.filter((point) => point.y === 1).length],
      loss: metrics.loss,
      accuracy: metrics.accuracy,
      separation: metrics.separation,
      featureRank: metrics.featureRank,
      distinctDirections: metrics.distinctDirections,
      incomingRows: state.model.w.map((row) => row.slice()),
      hiddenBiases: state.model.b.slice(),
      outgoingCoefficients: state.model.v.slice(),
      outputBias: state.model.c,
      separateRowStorage: state.model.w.every((row, index, rows) => rows.findIndex((candidate) => candidate === row) === index),
      parameterSignature: parameterSignature()
    };
  }

  window.SymmetryBreakingLab = Object.freeze({
    snapshot,
    setMode(mode) {
      if (Object.hasOwn(MODE_COPY, mode)) resetMode(mode);
    },
    step(count = 1) {
      stopRunning();
      trainSteps(Math.max(1, Math.round(Number(count) || 1)));
      render();
    },
    reset() {
      resetMode(state.mode);
    },
    start() {
      if (!state.running) toggleRunning();
    },
    stop() {
      stopRunning();
      render();
    }
  });

  render();
})();
