/* ============================================================
   TEXT DIFFUSION — interactive explainer
   A discrete denoising diffusion model trained live in the browser
   on Indian names. No libraries, no server, no GPU.
   ============================================================ */

// ── Data ────────────────────────────────────────────────────

const NAMES = [
  'aarav', 'aditi', 'aisha', 'ananya', 'anika', 'arjun', 'avni',
  'dev', 'dhruv', 'diya', 'isha', 'ishaan', 'kabir', 'karan',
  'kavya', 'kiara', 'krish', 'meera', 'myra',
  'navya', 'neel', 'neha', 'nikhil', 'nisha', 'pari', 'pooja',
  'priya', 'rahul', 'reyansh', 'riya', 'rohan', 'saanvi',
  'sara', 'sia', 'sneha', 'tara', 'tanvi', 'vanya',
  'vivaan', 'zara',
];

const PAD = '_';
const MASK = '?';
const chars = [...new Set(NAMES.join(''))].sort();
const itos = [PAD, MASK, ...chars];
const stoi = {};
itos.forEach((c, i) => (stoi[c] = i));
const pad_id = stoi[PAD];
const mask_id = stoi[MASK];
const V = itos.length;
const L = Math.max(...NAMES.map((n) => n.length));

function encode(name) {
  const ids = new Array(L);
  for (let i = 0; i < L; i++) ids[i] = i < name.length ? stoi[name[i]] : pad_id;
  return ids;
}

function decode(ids) {
  return ids
    .map((i) => itos[i])
    .join('')
    .replace(/_+$/, '');
}

const DATA = NAMES.map(encode);

// ── Neural network ──────────────────────────────────────────

const D_EMB = 8;
const D_HID = 128;
const INPUT_DIM = L * D_EMB + 1;
const OUTPUT_DIM = L * V;

function boxMuller() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
}

function initWeight(n, scale) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = boxMuller() * scale;
  return w;
}

let embed, W1, b1, W2, b2;

function initModel() {
  embed = initWeight(V * D_EMB, 0.3);
  W1 = initWeight(INPUT_DIM * D_HID, Math.sqrt(2 / INPUT_DIM));
  b1 = new Float64Array(D_HID);
  W2 = initWeight(D_HID * OUTPUT_DIM, Math.sqrt(2 / D_HID));
  b2 = new Float64Array(OUTPUT_DIM);
}

// Forward pass (single example).
function forward(xt_ids, t, x0_ids) {
  const h_flat = new Float64Array(INPUT_DIM);
  for (let p = 0; p < L; p++) {
    const tok = xt_ids[p];
    const eOff = tok * D_EMB;
    const hOff = p * D_EMB;
    for (let d = 0; d < D_EMB; d++) h_flat[hOff + d] = embed[eOff + d];
  }
  h_flat[L * D_EMB] = t;

  const h1_pre = new Float64Array(D_HID);
  for (let j = 0; j < D_HID; j++) {
    let s = b1[j];
    for (let i = 0; i < INPUT_DIM; i++) s += h_flat[i] * W1[i * D_HID + j];
    h1_pre[j] = s;
  }
  const h1 = new Float64Array(D_HID);
  for (let j = 0; j < D_HID; j++) h1[j] = h1_pre[j] > 0 ? h1_pre[j] : 0;

  const logits = new Float64Array(OUTPUT_DIM);
  for (let k = 0; k < OUTPUT_DIM; k++) {
    let s = b2[k];
    for (let j = 0; j < D_HID; j++) s += h1[j] * W2[j * OUTPUT_DIM + k];
    logits[k] = s;
  }

  const probs = new Float64Array(OUTPUT_DIM);
  let loss = 0;
  for (let p = 0; p < L; p++) {
    const off = p * V;
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[off + v] > mx) mx = logits[off + v];
    let sm = 0;
    for (let v = 0; v < V; v++) {
      const e = Math.exp(logits[off + v] - mx);
      probs[off + v] = e;
      sm += e;
    }
    for (let v = 0; v < V; v++) probs[off + v] /= sm;
    if (x0_ids) loss -= Math.log(probs[off + x0_ids[p]] + 1e-10);
  }
  if (x0_ids) loss /= L;

  return { logits, probs, loss, cache: { h_flat, h1_pre, h1, xt_ids } };
}

// Inference-only forward (no cache, no loss).
function forwardInfer(xt_ids, t) {
  const h_flat = new Float64Array(INPUT_DIM);
  for (let p = 0; p < L; p++) {
    const tok = xt_ids[p];
    const eOff = tok * D_EMB;
    const hOff = p * D_EMB;
    for (let d = 0; d < D_EMB; d++) h_flat[hOff + d] = embed[eOff + d];
  }
  h_flat[L * D_EMB] = t;

  const h1 = new Float64Array(D_HID);
  for (let j = 0; j < D_HID; j++) {
    let s = b1[j];
    for (let i = 0; i < INPUT_DIM; i++) s += h_flat[i] * W1[i * D_HID + j];
    h1[j] = s > 0 ? s : 0;
  }

  const logits = new Float64Array(OUTPUT_DIM);
  for (let k = 0; k < OUTPUT_DIM; k++) {
    let s = b2[k];
    for (let j = 0; j < D_HID; j++) s += h1[j] * W2[j * OUTPUT_DIM + k];
    logits[k] = s;
  }
  return logits;
}

// Backward pass + SGD update (single example).
function backward(probs, cache, x0_ids, lr) {
  const { h_flat, h1_pre, h1, xt_ids } = cache;

  const d_out = new Float64Array(OUTPUT_DIM);
  for (let p = 0; p < L; p++) {
    const off = p * V;
    for (let v = 0; v < V; v++) d_out[off + v] = probs[off + v] / L;
    d_out[off + x0_ids[p]] -= 1.0 / L;
  }

  // Layer 2 backward
  const d_h1 = new Float64Array(D_HID);
  for (let j = 0; j < D_HID; j++) {
    let s = 0;
    const base = j * OUTPUT_DIM;
    for (let k = 0; k < OUTPUT_DIM; k++) s += d_out[k] * W2[base + k];
    d_h1[j] = s;
  }
  for (let j = 0; j < D_HID; j++) {
    const h1j = h1[j];
    const base = j * OUTPUT_DIM;
    for (let k = 0; k < OUTPUT_DIM; k++) W2[base + k] -= lr * h1j * d_out[k];
  }
  for (let k = 0; k < OUTPUT_DIM; k++) b2[k] -= lr * d_out[k];

  // ReLU backward
  for (let j = 0; j < D_HID; j++) if (h1_pre[j] <= 0) d_h1[j] = 0;

  // Layer 1 backward
  const d_hf = new Float64Array(INPUT_DIM);
  for (let i = 0; i < INPUT_DIM; i++) {
    let s = 0;
    const base = i * D_HID;
    for (let j = 0; j < D_HID; j++) s += d_h1[j] * W1[base + j];
    d_hf[i] = s;
  }
  for (let i = 0; i < INPUT_DIM; i++) {
    const hfi = h_flat[i];
    const base = i * D_HID;
    for (let j = 0; j < D_HID; j++) W1[base + j] -= lr * hfi * d_h1[j];
  }
  for (let j = 0; j < D_HID; j++) b1[j] -= lr * d_h1[j];

  // Embedding backward
  for (let p = 0; p < L; p++) {
    const tok = xt_ids[p];
    const eOff = tok * D_EMB;
    const hOff = p * D_EMB;
    for (let d = 0; d < D_EMB; d++) embed[eOff + d] -= lr * d_hf[hOff + d];
  }
}

// ── Corruption ──────────────────────────────────────────────

function corrupt(x0_ids, t) {
  const xt = x0_ids.slice();
  for (let i = 0; i < L; i++) {
    if (Math.random() < t) xt[i] = mask_id;
  }
  return xt;
}

// Deterministic corruption for visualizer (smooth slider).
function seededShuffle(n, seed) {
  const idx = Array.from({ length: n }, (_, i) => i);
  let s = seed | 0;
  for (let i = n - 1; i > 0; i--) {
    s = (Math.imul(s, 1664525) + 1013904223) | 0;
    const j = ((s >>> 0) % (i + 1)) | 0;
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return idx;
}

function deterministicCorrupt(x0_ids, t, seed) {
  const xt = x0_ids.slice();
  const order = seededShuffle(L, seed);
  const numMask = Math.round(t * L);
  for (let i = 0; i < numMask; i++) xt[order[i]] = mask_id;
  return xt;
}

// ── Training ────────────────────────────────────────────────

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

function trainEpoch(lr) {
  const indices = DATA.map((_, i) => i);
  shuffle(indices);
  let totalLoss = 0;
  for (const idx of indices) {
    const x0 = DATA[idx];
    const t = Math.random();
    const xt = corrupt(x0, t);
    const { probs, loss, cache } = forward(xt, t, x0);
    totalLoss += loss;
    backward(probs, cache, x0, lr);
  }
  return totalLoss / DATA.length;
}

// ── Sampling ────────────────────────────────────────────────

function sampleFromLogits(logits, pos, temperature) {
  const off = pos * V;
  let mx = -Infinity;
  for (let v = 0; v < V; v++) {
    const val = logits[off + v] / temperature;
    if (val > mx) mx = val;
  }
  let sm = 0;
  const p = new Float64Array(V);
  for (let v = 0; v < V; v++) {
    p[v] = Math.exp(logits[off + v] / temperature - mx);
    sm += p[v];
  }
  let r = Math.random() * sm;
  for (let v = 0; v < V; v++) {
    r -= p[v];
    if (r <= 0) return v;
  }
  return V - 1;
}

function sample(numSteps, temperature) {
  const x = new Array(L).fill(mask_id);
  const history = [x.slice()];

  for (let s = numSteps - 1; s >= 0; s--) {
    const t = (s + 1) / numSteps;
    const logits = forwardInfer(x, t);

    const pred = new Array(L);
    for (let p = 0; p < L; p++) pred[p] = sampleFromLogits(logits, p, temperature);

    const revealProb = 1.0 / (s + 1);
    for (let p = 0; p < L; p++) {
      if (x[p] === mask_id && Math.random() < revealProb) x[p] = pred[p];
    }
    history.push(x.slice());
  }

  // Final clean pass at t=0 (argmax)
  const finalLogits = forwardInfer(x, 0);
  for (let p = 0; p < L; p++) {
    let bestV = 0,
      bestS = -Infinity;
    for (let v = 0; v < V; v++) {
      if (finalLogits[p * V + v] > bestS) {
        bestS = finalLogits[p * V + v];
        bestV = v;
      }
    }
    x[p] = bestV;
  }
  history.push(x.slice());

  return { name: decode(x), history };
}

// ── UI state ────────────────────────────────────────────────

let selectedEpochBatch = 500;

let trainingState = {
  epoch: 0,
  targetEpoch: 0,
  losses: [],
  isTraining: false,
  trained: false,
};

// Names to show in corruption visualizer: dev(7), diya(9), priya(26), ananya(3), reyansh(28), ishaan(11)
const SHOW_NAMES = [7, 9, 26, 3, 28, 11];

// Names to rotate in live preview during training
const PREVIEW_NAMES = [26, 3, 28, 11, 0, 14, 22]; // priya, ananya, reyansh, ishaan, aarav, kavya, nikhil
let previewRotation = 0;

// ── UI: Progress bar & TOC ──────────────────────────────────

function updateProgress() {
  const scrollY = window.scrollY;
  const docH = document.documentElement.scrollHeight - window.innerHeight;
  const pct = docH > 0 ? Math.min(scrollY / docH, 1) : 0;
  document.getElementById('progressFill').style.width = pct * 100 + '%';
}

function updateToc() {
  const sections = document.querySelectorAll('section[id]');
  const links = document.querySelectorAll('.toc__link');
  let activeIdx = 0;
  const offset = window.innerHeight * 0.35;
  sections.forEach((sec, i) => {
    if (sec.getBoundingClientRect().top < offset) activeIdx = i;
  });
  links.forEach((l, i) => l.classList.toggle('is-active', i === activeIdx));
}

window.addEventListener(
  'scroll',
  () => {
    updateProgress();
    updateToc();
  },
  { passive: true },
);

// ── UI: KaTeX ───────────────────────────────────────────────

function renderAllKatex() {
  if (!window.katex) return;
  document.querySelectorAll('[data-katex-display]').forEach((el) => {
    const tex = el.textContent
      .replace(/^\s*\\\[/, '')
      .replace(/\\\]\s*$/, '')
      .trim();
    try {
      window.katex.render(tex, el, { displayMode: true, throwOnError: false });
    } catch (_) {}
  });
}

// ── UI: Character cells ─────────────────────────────────────

function charCellHTML(charId, extraClass) {
  const ch = itos[charId];
  let cls = 'char-cell';
  if (extraClass) cls += ' ' + extraClass;
  else if (charId === mask_id) cls += ' char-cell--mask';
  else if (charId === pad_id) cls += ' char-cell--pad';
  else cls += ' char-cell--char';
  return '<div class="' + cls + '">' + ch + '</div>';
}

function renderCharRow(container, ids, classPerPos) {
  let html = '';
  for (let i = 0; i < ids.length; i++) {
    html += charCellHTML(ids[i], classPerPos ? classPerPos[i] : null);
  }
  container.innerHTML = html;
}

// ── UI: Dataset listing ─────────────────────────────────────

function renderDatasetListing() {
  const container = document.getElementById('datasetNames');
  container.innerHTML = NAMES.map(
    (n) => '<span class="name-pill name-pill--small">' + n + '</span>',
  ).join('');
}

// ── UI: Corruption visualizer ───────────────────────────────

function renderCorruption() {
  const t = parseInt(document.getElementById('corruptionSlider').value) / 100;
  document.getElementById('corruptionSliderValue').textContent = t.toFixed(2);

  const grid = document.getElementById('corruptionGrid');
  let html = '';
  SHOW_NAMES.forEach((idx, row) => {
    const x0 = DATA[idx];
    const xt = deterministicCorrupt(x0, t, (row + 1) * 7919);
    html += '<div class="corrupt-row">';
    html += '<span class="corrupt-name">' + NAMES[idx] + '</span>';
    html += '<div class="char-row">';
    for (let i = 0; i < L; i++) html += charCellHTML(xt[i]);
    html += '</div></div>';
  });
  grid.innerHTML = html;
}

// ── UI: Architecture diagram ────────────────────────────────

function renderArchDiagram() {
  // Show "priya" corrupted at t=0.5
  const priyaIdx = NAMES.indexOf('priya');
  const x0 = DATA[priyaIdx];
  const xt = deterministicCorrupt(x0, 0.5, 123);
  renderCharRow(document.getElementById('archInput'), xt);
  renderCharRow(document.getElementById('archOutput'), x0);
}

// ── UI: Training ────────────────────────────────────────────

function renderLossChart() {
  const canvas = document.getElementById('lossCanvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const H = rect.height;
  const pad = { top: 12, bottom: 24, left: 6, right: 6 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  const { losses } = trainingState;
  if (losses.length < 2) return;

  const maxLoss = Math.max(losses[0] * 1.05, 0.5);
  const minLoss = 0;
  const totalEpochs = Math.max(trainingState.targetEpoch, losses.length);

  // Grid lines + labels
  ctx.fillStyle = 'rgba(32,39,51,0.35)';
  ctx.font = '10px Manrope, sans-serif';
  ctx.strokeStyle = 'rgba(32,39,51,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const val = maxLoss * (1 - i / 4);
    const y = pad.top + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(1), pad.left + 2, y - 3);
  }

  // Loss curve
  ctx.beginPath();
  ctx.strokeStyle = '#3c68cf';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  for (let i = 0; i < losses.length; i++) {
    const x = pad.left + (i / (totalEpochs - 1)) * plotW;
    const y = pad.top + plotH - ((losses[i] - minLoss) / (maxLoss - minLoss)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Current loss dot
  if (losses.length > 0) {
    const lastI = losses.length - 1;
    const cx = pad.left + (lastI / (totalEpochs - 1)) * plotW;
    const cy =
      pad.top + plotH - ((losses[lastI] - minLoss) / (maxLoss - minLoss)) * plotH;
    ctx.fillStyle = '#3c68cf';
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function getCurrentPreviewIdx() {
  return PREVIEW_NAMES[previewRotation % PREVIEW_NAMES.length];
}

function renderPreview() {
  if (!W1) return;
  const nameIdx = getCurrentPreviewIdx();
  const x0 = DATA[nameIdx];
  const xt = deterministicCorrupt(x0, 0.5, 42 + previewRotation);

  renderCharRow(document.getElementById('previewInput'), xt);
  renderCharRow(document.getElementById('previewTarget'), x0);

  document.getElementById('previewName').textContent = NAMES[nameIdx];

  const logits = forwardInfer(xt, 0.5);
  const pred = new Array(L);
  for (let p = 0; p < L; p++) {
    let best = 0,
      bestS = -Infinity;
    for (let v = 0; v < V; v++) {
      if (logits[p * V + v] > bestS) {
        bestS = logits[p * V + v];
        best = v;
      }
    }
    pred[p] = best;
  }

  const classes = pred.map((p, i) =>
    p === x0[i] ? 'char-cell--correct' : 'char-cell--wrong',
  );
  renderCharRow(document.getElementById('previewOutput'), pred, classes);
}

function getLR() {
  return parseInt(document.getElementById('lrSlider').value) / 1000;
}

function updateTrainingUI() {
  const { epoch, losses } = trainingState;
  document.getElementById('epochDisplay').textContent = String(epoch);
  document.getElementById('lossDisplay').textContent =
    losses.length > 0 ? losses[losses.length - 1].toFixed(4) : '\u2014';
  renderLossChart();
  renderPreview();
}

function trainLoop() {
  if (!trainingState.isTraining) return;

  const lr = getLR();
  const epochsPerFrame = 5;

  for (let i = 0; i < epochsPerFrame; i++) {
    if (trainingState.epoch >= trainingState.targetEpoch) {
      trainingState.isTraining = false;
      trainingState.trained = true;
      document.getElementById('trainBtn').textContent = 'Train';
      document.getElementById('trainBtn').disabled = false;
      document.getElementById('trainMoreBtn').style.display = '';
      document.getElementById('generateBtn').disabled = false;
      updateTrainingUI();
      return;
    }
    const loss = trainEpoch(lr);
    trainingState.epoch++;
    trainingState.losses.push(loss);

    // Rotate preview name every 50 epochs
    if (trainingState.epoch % 50 === 0) previewRotation++;
  }

  updateTrainingUI();
  requestAnimationFrame(trainLoop);
}

function startTraining() {
  if (trainingState.isTraining) return;

  // Fresh start
  initModel();
  trainingState.epoch = 0;
  trainingState.losses = [];
  trainingState.targetEpoch = selectedEpochBatch;
  trainingState.isTraining = true;
  trainingState.trained = false;
  previewRotation = 0;

  document.getElementById('trainBtn').textContent = 'Training\u2026';
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('trainMoreBtn').style.display = 'none';
  document.getElementById('generateBtn').disabled = true;

  requestAnimationFrame(trainLoop);
}

function trainMore() {
  if (trainingState.isTraining) return;

  trainingState.targetEpoch = trainingState.epoch + 500;
  trainingState.isTraining = true;
  trainingState.trained = false;

  document.getElementById('trainBtn').textContent = 'Training\u2026';
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('trainMoreBtn').style.display = 'none';

  requestAnimationFrame(trainLoop);
}

function resetTraining() {
  trainingState.isTraining = false;
  trainingState.epoch = 0;
  trainingState.targetEpoch = 0;
  trainingState.losses = [];
  trainingState.trained = false;
  previewRotation = 0;
  initModel();

  document.getElementById('trainBtn').textContent = 'Train';
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('trainMoreBtn').style.display = 'none';
  document.getElementById('generateBtn').disabled = true;
  updateTrainingUI();

  document.getElementById('timeline').innerHTML = '';
  document.getElementById('namesList').innerHTML =
    '<span class="names-placeholder">Train the model first, then click Generate.</span>';
}

// ── UI: Sampling ────────────────────────────────────────────

function renderTimeline(history) {
  const container = document.getElementById('timeline');
  const numSteps = history.length;

  let html = '';
  for (let s = 0; s < numSteps; s++) {
    const ids = history[s];
    const prevIds = s > 0 ? history[s - 1] : null;
    const isLast = s === numSteps - 1;

    let label;
    if (s === 0) label = 't = 1.0';
    else if (isLast) label = 't = 0.0';
    else {
      const t = 1.0 - s / (numSteps - 2);
      label = 't \u2248 ' + t.toFixed(1);
    }

    html += '<div class="timeline-row">';
    html += '<span class="timeline-step">' + label + '</span>';
    html += '<div class="char-row">';
    for (let i = 0; i < L; i++) {
      let cls = null;
      if (prevIds && prevIds[i] === mask_id && ids[i] !== mask_id) {
        cls = 'char-cell--reveal';
      }
      html += charCellHTML(ids[i], cls);
    }
    html += '</div></div>';
  }
  container.innerHTML = html;
}

function doGenerate() {
  if (!trainingState.trained) return;

  const numSteps = parseInt(document.getElementById('stepsSlider').value);
  const temperature = parseInt(document.getElementById('tempSlider').value) / 100;

  // Generate one name with timeline
  const first = sample(numSteps, temperature);
  renderTimeline(first.history);

  // Generate more names
  const names = [first.name];
  for (let i = 1; i < 15; i++) {
    names.push(sample(numSteps, temperature).name);
  }

  const list = document.getElementById('namesList');
  list.innerHTML = names
    .map((n) => '<span class="name-pill">' + (n || '\u00a0') + '</span>')
    .join('');
}

// ── Event bindings ──────────────────────────────────────────

function bindEvents() {
  // Corruption slider
  document
    .getElementById('corruptionSlider')
    .addEventListener('input', renderCorruption);

  // Epoch chips (names section only)
  document.querySelectorAll('[data-epochs]').forEach((chip) => {
    chip.addEventListener('click', () => {
      document
        .querySelectorAll('[data-epochs]')
        .forEach((c) => c.classList.remove('is-active'));
      chip.classList.add('is-active');
      selectedEpochBatch = parseInt(chip.dataset.epochs);
    });
  });

  // Training
  document.getElementById('trainBtn').addEventListener('click', startTraining);
  document.getElementById('trainMoreBtn').addEventListener('click', trainMore);
  document.getElementById('resetBtn').addEventListener('click', resetTraining);

  // LR slider
  document.getElementById('lrSlider').addEventListener('input', () => {
    document.getElementById('lrValue').textContent = getLR().toFixed(3);
  });

  // Sampling
  document.getElementById('generateBtn').addEventListener('click', doGenerate);
  document.getElementById('stepsSlider').addEventListener('input', () => {
    document.getElementById('stepsValue').textContent =
      document.getElementById('stepsSlider').value;
  });
  document.getElementById('tempSlider').addEventListener('input', () => {
    document.getElementById('tempValue').textContent = (
      parseInt(document.getElementById('tempSlider').value) / 100
    ).toFixed(2);
  });
}

// ══════════════════════════════════════════════════════════════
// QA MODEL — conditional diffusion (English → Hindi word pairs)
// ══════════════════════════════════════════════════════════════

const QA_PAIRS = [
  ['hello', 'namaste'], ['bye', 'alvida'], ['yes', 'haan'], ['no', 'nahi'],
  ['water', 'paani'], ['food', 'khana'], ['come', 'aao'], ['go', 'jao'],
  ['good', 'accha'], ['bad', 'bura'], ['big', 'bada'], ['hot', 'garam'],
  ['cold', 'thanda'], ['love', 'pyaar'], ['name', 'naam'], ['book', 'kitab'],
  ['home', 'ghar'], ['one', 'ek'], ['two', 'do'], ['three', 'teen'],
  ['four', 'chaar'], ['five', 'paanch'], ['sun', 'suraj'], ['moon', 'chand'],
  ['rain', 'baarish'], ['friend', 'dost'], ['sleep', 'neend'], ['eat', 'khao'],
  ['run', 'bhaago'], ['sit', 'baitho'],
];

// Build QA vocabulary
const QA_SEP = '>';
const qa_raw = QA_PAIRS.map(([q, a]) => q + QA_SEP + a).join('');
const qa_chars = [...new Set(qa_raw)].sort();
const qa_itos = [PAD, MASK, ...qa_chars];
const qa_stoi = {};
qa_itos.forEach((c, i) => (qa_stoi[c] = i));
const qa_pad_id = qa_stoi[PAD];
const qa_mask_id = qa_stoi[MASK];
const qa_sep_id = qa_stoi[QA_SEP];
const QA_V = qa_itos.length;
const QA_L = Math.max(...QA_PAIRS.map(([q, a]) => q.length + 1 + a.length));

function qaEncode(str) {
  const ids = new Array(QA_L);
  for (let i = 0; i < QA_L; i++) ids[i] = i < str.length ? qa_stoi[str[i]] : qa_pad_id;
  return ids;
}

function qaDecode(ids) {
  return ids.map((i) => qa_itos[i]).join('').replace(/_+$/, '');
}

const QA_DATA = QA_PAIRS.map(([q, a]) => ({
  ids: qaEncode(q + QA_SEP + a),
  sepPos: q.length,
  q,
  a,
}));

// QA Neural network (separate weights from names model)
const QA_D_EMB = 8;
const QA_D_HID = 96;
const QA_INPUT_DIM = QA_L * QA_D_EMB + 1;
const QA_OUTPUT_DIM = QA_L * QA_V;

let qa_embed, qa_W1, qa_b1, qa_W2, qa_b2;

function qaInitModel() {
  qa_embed = initWeight(QA_V * QA_D_EMB, 0.3);
  qa_W1 = initWeight(QA_INPUT_DIM * QA_D_HID, Math.sqrt(2 / QA_INPUT_DIM));
  qa_b1 = new Float64Array(QA_D_HID);
  qa_W2 = initWeight(QA_D_HID * QA_OUTPUT_DIM, Math.sqrt(2 / QA_D_HID));
  qa_b2 = new Float64Array(QA_OUTPUT_DIM);
}

function qaForward(xt_ids, t, x0_ids, sepPos) {
  const h_flat = new Float64Array(QA_INPUT_DIM);
  for (let p = 0; p < QA_L; p++) {
    const tok = xt_ids[p];
    const eOff = tok * QA_D_EMB;
    const hOff = p * QA_D_EMB;
    for (let d = 0; d < QA_D_EMB; d++) h_flat[hOff + d] = qa_embed[eOff + d];
  }
  h_flat[QA_L * QA_D_EMB] = t;

  const h1_pre = new Float64Array(QA_D_HID);
  for (let j = 0; j < QA_D_HID; j++) {
    let s = qa_b1[j];
    for (let i = 0; i < QA_INPUT_DIM; i++) s += h_flat[i] * qa_W1[i * QA_D_HID + j];
    h1_pre[j] = s;
  }
  const h1 = new Float64Array(QA_D_HID);
  for (let j = 0; j < QA_D_HID; j++) h1[j] = h1_pre[j] > 0 ? h1_pre[j] : 0;

  const logits = new Float64Array(QA_OUTPUT_DIM);
  for (let k = 0; k < QA_OUTPUT_DIM; k++) {
    let s = qa_b2[k];
    for (let j = 0; j < QA_D_HID; j++) s += h1[j] * qa_W2[j * QA_OUTPUT_DIM + k];
    logits[k] = s;
  }

  const probs = new Float64Array(QA_OUTPUT_DIM);
  let loss = 0;
  let count = 0;
  for (let p = 0; p < QA_L; p++) {
    const off = p * QA_V;
    let mx = -Infinity;
    for (let v = 0; v < QA_V; v++) if (logits[off + v] > mx) mx = logits[off + v];
    let sm = 0;
    for (let v = 0; v < QA_V; v++) {
      const e = Math.exp(logits[off + v] - mx);
      probs[off + v] = e;
      sm += e;
    }
    for (let v = 0; v < QA_V; v++) probs[off + v] /= sm;
    // Loss only on answer positions (after separator)
    if (x0_ids && p > sepPos) {
      loss -= Math.log(probs[off + x0_ids[p]] + 1e-10);
      count++;
    }
  }
  if (count > 0) loss /= count;

  return { logits, probs, loss, cache: { h_flat, h1_pre, h1, xt_ids }, count };
}

function qaForwardInfer(xt_ids, t) {
  const h_flat = new Float64Array(QA_INPUT_DIM);
  for (let p = 0; p < QA_L; p++) {
    const tok = xt_ids[p];
    const eOff = tok * QA_D_EMB;
    const hOff = p * QA_D_EMB;
    for (let d = 0; d < QA_D_EMB; d++) h_flat[hOff + d] = qa_embed[eOff + d];
  }
  h_flat[QA_L * QA_D_EMB] = t;

  const h1 = new Float64Array(QA_D_HID);
  for (let j = 0; j < QA_D_HID; j++) {
    let s = qa_b1[j];
    for (let i = 0; i < QA_INPUT_DIM; i++) s += h_flat[i] * qa_W1[i * QA_D_HID + j];
    h1[j] = s > 0 ? s : 0;
  }

  const logits = new Float64Array(QA_OUTPUT_DIM);
  for (let k = 0; k < QA_OUTPUT_DIM; k++) {
    let s = qa_b2[k];
    for (let j = 0; j < QA_D_HID; j++) s += h1[j] * qa_W2[j * QA_OUTPUT_DIM + k];
    logits[k] = s;
  }
  return logits;
}

function qaBackward(probs, cache, x0_ids, sepPos, count, lr) {
  const { h_flat, h1_pre, h1, xt_ids } = cache;
  if (count === 0) return;

  // Gradient only on answer positions
  const d_out = new Float64Array(QA_OUTPUT_DIM);
  for (let p = sepPos + 1; p < QA_L; p++) {
    const off = p * QA_V;
    for (let v = 0; v < QA_V; v++) d_out[off + v] = probs[off + v] / count;
    d_out[off + x0_ids[p]] -= 1.0 / count;
  }

  // Layer 2 backward
  const d_h1 = new Float64Array(QA_D_HID);
  for (let j = 0; j < QA_D_HID; j++) {
    let s = 0;
    const base = j * QA_OUTPUT_DIM;
    for (let k = 0; k < QA_OUTPUT_DIM; k++) s += d_out[k] * qa_W2[base + k];
    d_h1[j] = s;
  }
  for (let j = 0; j < QA_D_HID; j++) {
    const h1j = h1[j];
    const base = j * QA_OUTPUT_DIM;
    for (let k = 0; k < QA_OUTPUT_DIM; k++) qa_W2[base + k] -= lr * h1j * d_out[k];
  }
  for (let k = 0; k < QA_OUTPUT_DIM; k++) qa_b2[k] -= lr * d_out[k];

  // ReLU backward
  for (let j = 0; j < QA_D_HID; j++) if (h1_pre[j] <= 0) d_h1[j] = 0;

  // Layer 1 backward
  const d_hf = new Float64Array(QA_INPUT_DIM);
  for (let i = 0; i < QA_INPUT_DIM; i++) {
    let s = 0;
    const base = i * QA_D_HID;
    for (let j = 0; j < QA_D_HID; j++) s += d_h1[j] * qa_W1[base + j];
    d_hf[i] = s;
  }
  for (let i = 0; i < QA_INPUT_DIM; i++) {
    const hfi = h_flat[i];
    const base = i * QA_D_HID;
    for (let j = 0; j < QA_D_HID; j++) qa_W1[base + j] -= lr * hfi * d_h1[j];
  }
  for (let j = 0; j < QA_D_HID; j++) qa_b1[j] -= lr * d_h1[j];

  // Embedding backward
  for (let p = 0; p < QA_L; p++) {
    const tok = xt_ids[p];
    const eOff = tok * QA_D_EMB;
    const hOff = p * QA_D_EMB;
    for (let d = 0; d < QA_D_EMB; d++) qa_embed[eOff + d] -= lr * d_hf[hOff + d];
  }
}

// QA corruption: only mask answer positions (after separator)
function qaCorrupt(x0_ids, t, sepPos) {
  const xt = x0_ids.slice();
  for (let i = sepPos + 1; i < QA_L; i++) {
    if (Math.random() < t) xt[i] = qa_mask_id;
  }
  return xt;
}

// QA training
function qaTrainEpoch(lr) {
  const indices = QA_DATA.map((_, i) => i);
  shuffle(indices);
  let totalLoss = 0;
  for (const idx of indices) {
    const { ids, sepPos } = QA_DATA[idx];
    const t = Math.random();
    const xt = qaCorrupt(ids, t, sepPos);
    const { probs, loss, cache, count } = qaForward(xt, t, ids, sepPos);
    totalLoss += loss;
    qaBackward(probs, cache, ids, sepPos, count, lr);
  }
  return totalLoss / QA_DATA.length;
}

// QA sampling: fix question, denoise answer
function qaSample(questionStr, numSteps, temperature) {
  const sepPos = questionStr.length;
  const x = qaEncode(questionStr + QA_SEP);
  // Mask all answer positions
  for (let i = sepPos + 1; i < QA_L; i++) x[i] = qa_mask_id;
  const history = [{ ids: x.slice(), sepPos }];

  for (let s = numSteps - 1; s >= 0; s--) {
    const t = (s + 1) / numSteps;
    const logits = qaForwardInfer(x, t);

    const pred = new Array(QA_L);
    for (let p = 0; p < QA_L; p++) {
      const off = p * QA_V;
      let mx2 = -Infinity;
      for (let v = 0; v < QA_V; v++) {
        const val = logits[off + v] / temperature;
        if (val > mx2) mx2 = val;
      }
      let sm2 = 0;
      const pr = new Float64Array(QA_V);
      for (let v = 0; v < QA_V; v++) {
        pr[v] = Math.exp(logits[off + v] / temperature - mx2);
        sm2 += pr[v];
      }
      let r2 = Math.random() * sm2;
      pred[p] = QA_V - 1;
      for (let v = 0; v < QA_V; v++) {
        r2 -= pr[v];
        if (r2 <= 0) { pred[p] = v; break; }
      }
    }

    const revealProb = 1.0 / (s + 1);
    for (let p = sepPos + 1; p < QA_L; p++) {
      if (x[p] === qa_mask_id && Math.random() < revealProb) x[p] = pred[p];
    }
    history.push({ ids: x.slice(), sepPos });
  }

  // Final clean pass at t=0 (argmax on answer positions only)
  const finalLogits = qaForwardInfer(x, 0);
  for (let p = sepPos + 1; p < QA_L; p++) {
    let bestV = 0, bestS = -Infinity;
    for (let v = 0; v < QA_V; v++) {
      if (finalLogits[p * QA_V + v] > bestS) { bestS = finalLogits[p * QA_V + v]; bestV = v; }
    }
    x[p] = bestV;
  }
  history.push({ ids: x.slice(), sepPos });

  // Extract answer
  const answerIds = x.slice(sepPos + 1);
  const answer = answerIds.map((i) => qa_itos[i]).join('').replace(/_+$/, '');

  return { answer, history };
}

// ── QA UI state ─────────────────────────────────────────────

let qaSelectedEpochBatch = 1000;

let qaState = {
  epoch: 0,
  targetEpoch: 0,
  losses: [],
  isTraining: false,
  trained: false,
};

let qaPreviewRotation = 0;

// ── QA UI rendering ─────────────────────────────────────────

function qaCharCellHTML(charId, pos, sepPos, extraClass) {
  const ch = qa_itos[charId];
  let cls = 'char-cell';
  if (extraClass) cls += ' ' + extraClass;
  else if (pos < sepPos) cls += ' char-cell--question';
  else if (pos === sepPos) cls += ' char-cell--sep';
  else if (charId === qa_mask_id) cls += ' char-cell--mask';
  else if (charId === qa_pad_id) cls += ' char-cell--pad';
  else cls += ' char-cell--answer';
  return '<div class="' + cls + '">' + ch + '</div>';
}

function renderQaCharRow(container, ids, sepPos, classPerPos) {
  let html = '';
  for (let i = 0; i < ids.length; i++) {
    html += qaCharCellHTML(ids[i], i, sepPos, classPerPos ? classPerPos[i] : null);
  }
  container.innerHTML = html;
}

function renderQaFormatDemo() {
  // Show "hello>namaste" clean and corrupted
  const pair = QA_DATA[0]; // hello>namaste
  const clean = pair.ids;
  const corrupted = qaCorrupt(clean, 0.6, pair.sepPos);
  // Use deterministic corruption for demo
  const corr = clean.slice();
  const answerLen = QA_L - pair.sepPos - 1;
  const order = seededShuffle(answerLen, 777);
  const numMask = Math.round(0.6 * answerLen);
  for (let i = 0; i < numMask; i++) corr[pair.sepPos + 1 + order[i]] = qa_mask_id;

  renderQaCharRow(document.getElementById('qaFormatClean'), clean, pair.sepPos);
  renderQaCharRow(document.getElementById('qaFormatCorrupt'), corr, pair.sepPos);
}

function renderQaDatasetList() {
  const container = document.getElementById('qaDatasetList');
  container.innerHTML = QA_PAIRS.map(
    ([q, a]) =>
      '<span class="name-pill name-pill--small">' + q + ' \u2192 ' + a + '</span>',
  ).join('');
}

function renderQaWordSelect() {
  const select = document.getElementById('qaWordSelect');
  QA_PAIRS.forEach(([q]) => {
    const opt = document.createElement('option');
    opt.value = q;
    opt.textContent = q;
    select.appendChild(opt);
  });
}

function renderQaLossChart() {
  const canvas = document.getElementById('qaLossCanvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const H = rect.height;
  const pd = { top: 12, bottom: 24, left: 6, right: 6 };
  const plotW = W - pd.left - pd.right;
  const plotH = H - pd.top - pd.bottom;

  ctx.clearRect(0, 0, W, H);

  const { losses } = qaState;
  if (losses.length < 2) return;

  const maxLoss = Math.max(losses[0] * 1.05, 0.5);
  const minLoss = 0;
  const totalEpochs = Math.max(qaState.targetEpoch, losses.length);

  ctx.fillStyle = 'rgba(32,39,51,0.35)';
  ctx.font = '10px Manrope, sans-serif';
  ctx.strokeStyle = 'rgba(32,39,51,0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const val = maxLoss * (1 - i / 4);
    const y = pd.top + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(pd.left, y);
    ctx.lineTo(W - pd.right, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(1), pd.left + 2, y - 3);
  }

  ctx.beginPath();
  ctx.strokeStyle = '#1e7770';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  for (let i = 0; i < losses.length; i++) {
    const x = pd.left + (i / (totalEpochs - 1)) * plotW;
    const y = pd.top + plotH - ((losses[i] - minLoss) / (maxLoss - minLoss)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (losses.length > 0) {
    const lastI = losses.length - 1;
    const cx = pd.left + (lastI / (totalEpochs - 1)) * plotW;
    const cy = pd.top + plotH - ((losses[lastI] - minLoss) / (maxLoss - minLoss)) * plotH;
    ctx.fillStyle = '#1e7770';
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function renderQaPreview() {
  if (!qa_W1) return;
  const pairIdx = qaPreviewRotation % QA_DATA.length;
  const { ids, sepPos } = QA_DATA[pairIdx];

  // Corrupt with t=0.7 (heavily masked answer)
  const xt = ids.slice();
  const ansLen = QA_L - sepPos - 1;
  const order = seededShuffle(ansLen, 99 + qaPreviewRotation);
  const numMask = Math.round(0.7 * ansLen);
  for (let i = 0; i < numMask; i++) xt[sepPos + 1 + order[i]] = qa_mask_id;

  renderQaCharRow(document.getElementById('qaPreviewInput'), xt, sepPos);
  renderQaCharRow(document.getElementById('qaPreviewTarget'), ids, sepPos);

  const logits = qaForwardInfer(xt, 0.7);
  const pred = new Array(QA_L);
  for (let p = 0; p < QA_L; p++) {
    let best = 0, bestS = -Infinity;
    for (let v = 0; v < QA_V; v++) {
      if (logits[p * QA_V + v] > bestS) { bestS = logits[p * QA_V + v]; best = v; }
    }
    pred[p] = best;
  }

  const classes = new Array(QA_L).fill(null);
  for (let p = sepPos + 1; p < QA_L; p++) {
    classes[p] = pred[p] === ids[p] ? 'char-cell--correct' : 'char-cell--wrong';
  }
  renderQaCharRow(document.getElementById('qaPreviewOutput'), pred, sepPos, classes);
}

function updateQaTrainingUI() {
  document.getElementById('qaEpochDisplay').textContent = String(qaState.epoch);
  document.getElementById('qaLossDisplay').textContent =
    qaState.losses.length > 0
      ? qaState.losses[qaState.losses.length - 1].toFixed(4)
      : '\u2014';
  renderQaLossChart();
  renderQaPreview();
}

function qaTrainLoop() {
  if (!qaState.isTraining) return;

  const lr = getLR();
  const epochsPerFrame = 3;

  for (let i = 0; i < epochsPerFrame; i++) {
    if (qaState.epoch >= qaState.targetEpoch) {
      qaState.isTraining = false;
      qaState.trained = true;
      document.getElementById('qaTrainBtn').textContent = 'Train';
      document.getElementById('qaTrainBtn').disabled = false;
      document.getElementById('qaTrainMoreBtn').style.display = '';
      document.getElementById('qaTranslateBtn').disabled = false;
      updateQaTrainingUI();
      return;
    }
    const loss = qaTrainEpoch(lr);
    qaState.epoch++;
    qaState.losses.push(loss);
    if (qaState.epoch % 50 === 0) qaPreviewRotation++;
  }

  updateQaTrainingUI();
  requestAnimationFrame(qaTrainLoop);
}

function startQaTraining() {
  if (qaState.isTraining) return;
  qaInitModel();
  qaState.epoch = 0;
  qaState.losses = [];
  qaState.targetEpoch = qaSelectedEpochBatch;
  qaState.isTraining = true;
  qaState.trained = false;
  qaPreviewRotation = 0;

  document.getElementById('qaTrainBtn').textContent = 'Training\u2026';
  document.getElementById('qaTrainBtn').disabled = true;
  document.getElementById('qaTrainMoreBtn').style.display = 'none';
  document.getElementById('qaTranslateBtn').disabled = true;

  requestAnimationFrame(qaTrainLoop);
}

function qaTrainMore() {
  if (qaState.isTraining) return;
  qaState.targetEpoch = qaState.epoch + 500;
  qaState.isTraining = true;
  qaState.trained = false;
  document.getElementById('qaTrainBtn').textContent = 'Training\u2026';
  document.getElementById('qaTrainBtn').disabled = true;
  document.getElementById('qaTrainMoreBtn').style.display = 'none';
  requestAnimationFrame(qaTrainLoop);
}

function qaReset() {
  qaState.isTraining = false;
  qaState.epoch = 0;
  qaState.targetEpoch = 0;
  qaState.losses = [];
  qaState.trained = false;
  qaPreviewRotation = 0;
  qaInitModel();

  document.getElementById('qaTrainBtn').textContent = 'Train';
  document.getElementById('qaTrainBtn').disabled = false;
  document.getElementById('qaTrainMoreBtn').style.display = 'none';
  document.getElementById('qaTranslateBtn').disabled = true;
  updateQaTrainingUI();

  document.getElementById('qaTimeline').innerHTML = '';
  document.getElementById('qaResultBox').style.display = 'none';
}

function renderQaTimeline(history) {
  const container = document.getElementById('qaTimeline');
  const numSteps = history.length;
  let html = '';

  for (let s = 0; s < numSteps; s++) {
    const { ids, sepPos } = history[s];
    const prevIds = s > 0 ? history[s - 1].ids : null;
    const isLast = s === numSteps - 1;

    let label;
    if (s === 0) label = 't = 1.0';
    else if (isLast) label = 't = 0.0';
    else label = 'step ' + s;

    html += '<div class="timeline-row">';
    html += '<span class="timeline-step">' + label + '</span>';
    html += '<div class="char-row">';
    for (let i = 0; i < QA_L; i++) {
      let cls = null;
      if (prevIds && i > sepPos && prevIds[i] === qa_mask_id && ids[i] !== qa_mask_id) {
        cls = 'char-cell--reveal';
      }
      html += qaCharCellHTML(ids[i], i, sepPos, cls);
    }
    html += '</div></div>';
  }
  container.innerHTML = html;
}

function doQaTranslate() {
  if (!qaState.trained) return;
  const word = document.getElementById('qaWordSelect').value;
  if (!word) return;

  const result = qaSample(word, 8, 0.7);
  renderQaTimeline(result.history);

  document.getElementById('qaResultBox').style.display = '';
  document.getElementById('qaResultText').textContent = result.answer;
}

function bindQaEvents() {
  document.getElementById('qaTrainBtn').addEventListener('click', startQaTraining);
  document.getElementById('qaTrainMoreBtn').addEventListener('click', qaTrainMore);
  document.getElementById('qaResetBtn').addEventListener('click', qaReset);
  document.getElementById('qaTranslateBtn').addEventListener('click', doQaTranslate);

  document.querySelectorAll('[data-qa-epochs]').forEach((chip) => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('[data-qa-epochs]').forEach((c) => c.classList.remove('is-active'));
      chip.classList.add('is-active');
      qaSelectedEpochBatch = parseInt(chip.dataset.qaEpochs);
    });
  });
}

function initQa() {
  qaInitModel();
  bindQaEvents();
  renderQaFormatDemo();
  renderQaDatasetList();
  renderQaWordSelect();
  updateQaTrainingUI();
}

// ── Init ────────────────────────────────────────────────────

function init() {
  initModel();
  bindEvents();
  renderDatasetListing();
  renderCorruption();
  renderArchDiagram();
  updateTrainingUI();
  updateProgress();
  updateToc();

  // QA section
  initQa();

  // Wait for KaTeX to load
  if (window.katex) {
    renderAllKatex();
  } else {
    const katexScript = document.querySelector('script[src*="katex"]');
    if (katexScript) katexScript.addEventListener('load', renderAllKatex);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
