// ============================================================
// CLIP Zero-Shot, from the Embeddings Up
// Runs the real OpenAI CLIP-ViT-B/32 locally via transformers.js
// (ONNX Runtime Web). Encodes the user's photo and labels into a
// shared 512-dim space, computes real cosine similarities, softmax.
// ============================================================

import {
  AutoTokenizer,
  AutoProcessor,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  env
} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Allow remote model hub (default) but prefer quantized for a smaller download.
env.allowLocalModels = false;

const MODEL_ID = 'Xenova/clip-vit-base-patch32';

const SAMPLES = [
  { key: 'dog',   label: 'Puppy',             urls: ['https://picsum.photos/id/237/480/480'] },
  { key: 'corgi', label: 'Corgi',             urls: ['https://picsum.photos/id/1025/480/480'] },
  { key: 'mountain', label: 'Mountain',       urls: ['https://picsum.photos/id/29/480/480'] },
  { key: 'person', label: 'Person',           urls: ['https://picsum.photos/id/64/480/480'] },
  { key: 'bike',  label: 'Bike / urban',      urls: ['https://picsum.photos/id/145/480/480'] },
  { key: 'still', label: 'Still life',        urls: ['https://picsum.photos/id/292/480/480'] }
];

const TEMPLATES = {
  photo:   (label) => `a photo of a ${label}`,
  picture: (label) => `a picture of a ${label}`,
  image:   (label) => `an image of a ${label}`,
  bare:    (label) => label
};

const DEFAULT_LABELS = ['dog', 'cat', 'landscape', 'person', 'food', 'car'];

const state = {
  processor: null,
  visionModel: null,
  tokenizer: null,
  textModel: null,
  modelLoading: true,
  image: null,            // HTMLImageElement or HTMLCanvasElement
  imageEmbedding: null,   // Float32Array [512], unit-normalized
  labels: [...DEFAULT_LABELS],
  templateKey: 'photo',
  textEmbeddings: [],     // Array of Float32Array [512], unit-normalized
  similarities: [],       // cosine similarities per label
  tau: 0.01,
  probabilities: [],
  inferenceMs: 0,
  lastEmbedImageMs: 0
};

// ---------- Model loading ----------
function setModelStatus(cls, text) {
  const el = document.getElementById('model-status');
  const txt = document.getElementById('model-status-text');
  if (!el) return;
  el.classList.remove('is-loading', 'is-ready', 'is-error');
  el.classList.add(cls);
  txt.textContent = text;
}

function setProgress(file, pct) {
  const bar = document.getElementById('progress-bar-fill');
  const fl = document.getElementById('progress-file');
  if (bar) bar.style.width = `${pct}%`;
  if (fl) fl.textContent = file ? `${file} — ${pct.toFixed(0)}%` : '';
}

function clearProgress() {
  const bar = document.getElementById('progress-bar-fill');
  const fl = document.getElementById('progress-file');
  if (bar) bar.style.width = '100%';
  if (fl) fl.textContent = '';
  // Fade out
  setTimeout(() => {
    if (bar) bar.style.width = '0%';
  }, 1500);
}

async function loadModels() {
  const progress = (data) => {
    if (data.status === 'progress' && data.file) {
      const pct = data.total ? 100 * data.loaded / data.total : 0;
      setProgress(data.file, pct);
    }
  };
  try {
    setModelStatus('is-loading', 'Downloading processor & tokenizer\u2026');
    state.processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: progress });
    state.tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: progress });
    setModelStatus('is-loading', 'Downloading vision encoder\u2026');
    state.visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, {
      progress_callback: progress,
      quantized: true
    });
    setModelStatus('is-loading', 'Downloading text encoder\u2026');
    state.textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID, {
      progress_callback: progress,
      quantized: true
    });
    state.modelLoading = false;
    clearProgress();
    setModelStatus('is-ready', 'CLIP ready (real model, local inference)');
    // Kick off initial inference
    if (state.image) await runVision();
    await runText();
    recomputeDerived();
    renderAll();
  } catch (err) {
    console.error('CLIP load failed:', err);
    setModelStatus('is-error', 'Model failed to load — check network');
  }
}

// ---------- Image loading ----------
function loadImageFromElement(img) {
  state.image = img;
  state.imageEmbedding = null;
  renderPrelude();
  if (!state.modelLoading) runVision().then(() => {
    recomputeDerived();
    renderAll();
  });
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${url}`));
    img.src = url;
  });
}

async function loadImageWithFallback(urls) {
  for (const url of urls) {
    try {
      return await loadImageFromUrl(url);
    } catch (err) {
      console.warn(err.message, '— trying next fallback');
    }
  }
  throw new Error('All sample URLs failed');
}

async function pickSample(key) {
  const s = SAMPLES.find((x) => x.key === key);
  if (!s) return;
  document.querySelectorAll('#sample-grid .sample-thumb').forEach((b) =>
    b.classList.toggle('is-active', b.dataset.sample === key));
  const cap = document.getElementById('prelude-caption');
  if (cap) cap.textContent = `Loading "${s.label}"\u2026`;
  try {
    const img = await loadImageWithFallback(s.urls);
    loadImageFromElement(img);
    if (cap) cap.textContent = `"${s.label}" loaded. CLIP's view of it is below.`;
  } catch (err) {
    console.error(err);
    if (cap) cap.textContent = `Could not load "${s.label}". Drop a photo into the upload zone.`;
  }
}

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => loadImageFromElement(img);
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

// ---------- Inference ----------
async function runVision() {
  if (!state.visionModel || !state.processor || !state.image) return;
  try {
    const t0 = performance.now();
    // Draw the image into a canvas (224x224 is CLIP's input size; the
    // processor will resize / center-crop internally but we feed a
    // reasonable-sized canvas for consistent behavior).
    const w = state.image.naturalWidth || state.image.width;
    const h = state.image.naturalHeight || state.image.height;
    const side = Math.min(w, h);
    const off = document.createElement('canvas');
    off.width = off.height = 480;
    const ctx = off.getContext('2d');
    ctx.drawImage(state.image, (w - side) / 2, (h - side) / 2, side, side, 0, 0, 480, 480);
    const rawImg = await RawImage.fromCanvas(off);
    const vision_inputs = await state.processor(rawImg);
    const { image_embeds } = await state.visionModel(vision_inputs);
    // Normalize
    const dim = image_embeds.dims[image_embeds.dims.length - 1];
    const arr = new Float32Array(image_embeds.data);
    l2normalize(arr, dim);
    state.imageEmbedding = arr;
    state.lastEmbedImageMs = performance.now() - t0;
  } catch (err) {
    console.error('Vision inference failed:', err);
  }
}

async function runText() {
  if (!state.textModel || !state.tokenizer || !state.labels.length) {
    state.textEmbeddings = [];
    return;
  }
  try {
    const prompts = state.labels.map((l) => TEMPLATES[state.templateKey](l));
    const text_inputs = state.tokenizer(prompts, { padding: true, truncation: true });
    const { text_embeds } = await state.textModel(text_inputs);
    const dim = text_embeds.dims[text_embeds.dims.length - 1];
    const data = new Float32Array(text_embeds.data);
    const embs = [];
    for (let i = 0; i < state.labels.length; i++) {
      const vec = data.slice(i * dim, (i + 1) * dim);
      l2normalize(vec, dim);
      embs.push(vec);
    }
    state.textEmbeddings = embs;
  } catch (err) {
    console.error('Text inference failed:', err);
    state.textEmbeddings = [];
  }
}

function l2normalize(vec, dim) {
  let n = 0;
  for (let i = 0; i < dim; i++) n += vec[i] * vec[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < dim; i++) vec[i] /= n;
}

function cosineSim(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function softmax(scores, tau) {
  if (!scores.length) return [];
  const scaled = scores.map((s) => s / tau);
  const m = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function recomputeDerived() {
  if (!state.imageEmbedding || state.textEmbeddings.length !== state.labels.length) {
    state.similarities = [];
    state.probabilities = [];
    return;
  }
  state.similarities = state.textEmbeddings.map((t) => cosineSim(state.imageEmbedding, t));
  state.probabilities = softmax(state.similarities, state.tau);
}

// ---------- Rendering ----------
function renderPrelude() {
  const canvas = document.getElementById('preludeCanvas');
  if (!canvas) return;
  const size = 480;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = size * dpr; canvas.height = size * dpr;
  canvas.style.width = size + 'px'; canvas.style.height = size + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!state.image) {
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#c4beb1';
    ctx.font = '14px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Pick a sample or upload a photo', size / 2, size / 2);
    return;
  }
  const w = state.image.naturalWidth || state.image.width;
  const h = state.image.naturalHeight || state.image.height;
  const side = Math.min(w, h);
  ctx.drawImage(state.image, (w - side) / 2, (h - side) / 2, side, side, 0, 0, size, size);

  const ms = document.getElementById('inference-ms');
  if (ms) ms.textContent = state.lastEmbedImageMs > 0 ? `${state.lastEmbedImageMs.toFixed(0)} ms` : '—';
}

function renderEmbeddings() {
  const imgPre = document.getElementById('image-embedding-preview');
  if (imgPre) {
    if (!state.imageEmbedding) {
      imgPre.textContent = state.modelLoading
        ? 'Waiting for model\u2026'
        : 'Waiting for image\u2026';
    } else {
      imgPre.innerHTML = embeddingHtml(state.imageEmbedding, 'image');
    }
  }
  const txtPre = document.getElementById('text-embedding-preview');
  if (txtPre) {
    if (!state.textEmbeddings.length) {
      txtPre.textContent = state.modelLoading
        ? 'Waiting for model\u2026'
        : 'Waiting for labels\u2026';
    } else {
      const label = TEMPLATES[state.templateKey](state.labels[0]);
      txtPre.innerHTML = `<div style="margin-bottom: 0.4rem; font-family: var(--sans); font-size: 0.82rem; color: var(--muted);">"${escapeHtml(label)}"</div>` +
        embeddingHtml(state.textEmbeddings[0], 'text');
    }
  }
}

function embeddingHtml(vec, kind) {
  const n = 12;
  const dim = vec.length;
  const shown = [];
  for (let i = 0; i < Math.min(n, dim); i++) shown.push(vec[i]);
  // Also compute norm (should be ~1 after L2 normalize)
  let norm = 0; for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  const cells = shown.map((v) => {
    const cls = v >= 0 ? 'vector-cell vector-cell--pos' : 'vector-cell vector-cell--neg';
    return `<span class="${cls}">${v.toFixed(3)}</span>`;
  }).join('');
  return `<div style="font-family: var(--sans); font-size: 0.82rem; color: var(--muted); margin-bottom: 0.35rem;">${dim}-dim ${kind} embedding, ‖v‖ = ${norm.toFixed(3)} (unit-normalised)</div>` +
    `<div class="vector-row">${cells}<span class="vector-cell" style="background:#faf7ef;color:var(--muted)">… ${dim - n}</span></div>`;
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderLabelEditor() {
  const ed = document.getElementById('label-editor');
  if (!ed) return;
  ed.innerHTML = '';
  state.labels.forEach((lab, i) => {
    const sim = state.similarities[i];
    const prob = state.probabilities[i];
    const row = document.createElement('div');
    row.className = 'label-row';
    row.innerHTML = `
      <input type="text" value="${escapeHtml(lab)}" data-idx="${i}" />
      <span class="score">${sim != null ? sim.toFixed(3) : '—'}</span>
      <span class="score" style="color:var(--warm);">${prob != null ? (prob * 100).toFixed(1) + '%' : '—'}</span>
      <button class="remove-btn" data-remove="${i}" title="Remove">×</button>
    `;
    ed.appendChild(row);
  });
  ed.querySelectorAll('input[data-idx]').forEach((inp) => {
    inp.addEventListener('change', async () => {
      const i = parseInt(inp.dataset.idx, 10);
      state.labels[i] = inp.value.trim() || '(empty)';
      await runText();
      recomputeDerived();
      renderAll();
    });
  });
  ed.querySelectorAll('[data-remove]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const i = parseInt(btn.dataset.remove, 10);
      state.labels.splice(i, 1);
      await runText();
      recomputeDerived();
      renderAll();
    });
  });
}

function renderRawSimTable() {
  const wrap = document.getElementById('raw-sim-table');
  if (!wrap) return;
  if (!state.similarities.length) {
    wrap.innerHTML = '<p style="font-family: var(--sans); color: var(--muted); font-style: italic;">Waiting for model and labels&hellip;</p>';
    return;
  }
  const rows = state.labels.map((lab, i) => {
    const sim = state.similarities[i];
    return { lab, sim, prompt: TEMPLATES[state.templateKey](lab) };
  }).sort((a, b) => b.sim - a.sim);
  let html = '<table class="examples-table"><thead><tr><th>Rank</th><th>Label</th><th>Prompt sent to text encoder</th><th>Cosine similarity</th></tr></thead><tbody>';
  rows.forEach((r, rank) => {
    const style = rank === 0 ? ' style="background:rgba(217,98,43,0.05);"' : '';
    html += `<tr${style}><td>${rank + 1}</td><td><strong>${escapeHtml(r.lab)}</strong></td><td><code>"${escapeHtml(r.prompt)}"</code></td><td><strong>${r.sim.toFixed(4)}</strong></td></tr>`;
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

function renderBarChart() {
  const chart = document.getElementById('bar-chart');
  if (!chart) return;
  if (!state.probabilities.length) {
    chart.innerHTML = '<p style="font-family: var(--sans); color: var(--muted); font-style: italic;">Predictions will appear here once the model and labels are ready.</p>';
    document.getElementById('top-label').textContent = '—';
    document.getElementById('top-prob').textContent = '—';
    document.getElementById('entropy').textContent = '—';
    return;
  }
  const rows = state.labels.map((lab, i) => ({ lab, prob: state.probabilities[i] }))
    .sort((a, b) => b.prob - a.prob);
  chart.innerHTML = rows.map((r, rank) => {
    const isTop = rank === 0 ? ' is-top' : '';
    return `<div class="bar-row${isTop}">
      <div class="bar-label"><strong>${rank === 0 ? '★ ' : ''}</strong>${escapeHtml(r.lab)}</div>
      <div class="bar-value">${(r.prob * 100).toFixed(1)}%</div>
      <div class="bar-track"><div class="bar-fill" style="width: ${r.prob * 100}%"></div></div>
    </div>`;
  }).join('');

  document.getElementById('top-label').textContent = rows[0].lab;
  document.getElementById('top-prob').textContent = (rows[0].prob * 100).toFixed(1) + '%';
  // Entropy
  let H = 0;
  for (const p of state.probabilities) if (p > 1e-12) H -= p * Math.log(p);
  document.getElementById('entropy').textContent = H.toFixed(3);
}

function renderAll() {
  renderPrelude();
  renderEmbeddings();
  renderLabelEditor();
  renderRawSimTable();
  renderBarChart();
}

// ---------- Math ----------
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-encoders':
      '\\text{vision}: I \\mapsto v_I \\in \\mathbb{R}^{512}, \\quad \\text{text}: t \\mapsto v_t \\in \\mathbb{R}^{512}',
    'math-cosine':
      '\\mathrm{cos}(v_I, v_t) = \\frac{v_I \\cdot v_t}{\\|v_I\\|\\,\\|v_t\\|} = v_I \\cdot v_t \\quad (\\text{since both are unit-norm})',
    'math-softmax':
      'P(\\text{label}_i \\mid I) = \\frac{\\exp(v_I \\cdot v_{t_i} / \\tau)}{\\sum_j \\exp(v_I \\cdot v_{t_j} / \\tau)}'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

// ---------- Input wiring ----------
function wireControls() {
  // Samples
  const grid = document.getElementById('sample-grid');
  if (grid) {
    grid.innerHTML = SAMPLES.map((s) => `<button class="sample-thumb" data-sample="${s.key}">${s.label}</button>`).join('');
    grid.querySelectorAll('[data-sample]').forEach((b) => {
      b.addEventListener('click', () => pickSample(b.dataset.sample));
    });
  }

  // Upload
  const zone = document.getElementById('upload-zone');
  const input = document.getElementById('photo-input');
  input.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) handleFile(f);
  });
  ['dragenter', 'dragover'].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('drag-over'); }));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  // Webcam
  document.getElementById('webcam-btn').addEventListener('click', async (e) => {
    e.preventDefault(); e.stopPropagation();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement('video');
      video.srcObject = stream; video.playsInline = true;
      await video.play();
      setTimeout(() => {
        const off = document.createElement('canvas');
        off.width = video.videoWidth; off.height = video.videoHeight;
        off.getContext('2d').drawImage(video, 0, 0);
        stream.getTracks().forEach((t) => t.stop());
        loadImageFromElement(off);
      }, 300);
    } catch (err) {
      alert('Webcam access failed: ' + err.message);
    }
  });

  // Template selector
  document.querySelectorAll('#template-row [data-template]').forEach((b) => {
    b.addEventListener('click', async () => {
      document.querySelectorAll('#template-row [data-template]').forEach((bb) =>
        bb.classList.toggle('is-active', bb === b));
      state.templateKey = b.dataset.template;
      await runText();
      recomputeDerived();
      renderAll();
    });
  });

  // Label editor buttons
  document.getElementById('btn-add-label').addEventListener('click', async () => {
    state.labels.push('new label');
    await runText();
    recomputeDerived();
    renderAll();
  });
  document.getElementById('btn-reset-labels').addEventListener('click', async () => {
    state.labels = [...DEFAULT_LABELS];
    await runText();
    recomputeDerived();
    renderAll();
  });

  // Temperature slider
  const tau = document.getElementById('tau-slider');
  const tauVal = document.getElementById('tau-val');
  const invVal = document.getElementById('inv-tau-val');
  tau.addEventListener('input', () => {
    state.tau = parseFloat(tau.value);
    tauVal.textContent = state.tau.toFixed(3);
    invVal.textContent = (1 / state.tau).toFixed(0);
    recomputeDerived();
    renderLabelEditor();
    renderBarChart();
  });
}

// ---------- Boot ----------
async function init() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wireControls();
  renderAll();
  // Load default image first; model loads in parallel
  pickSample(SAMPLES[0].key);
  loadModels();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
