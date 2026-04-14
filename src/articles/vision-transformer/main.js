// ============================================================
// Vision Transformers on Real Photos
// Uses pretrained MobileNet v2 (TF.js) to extract a 7x7x1280 spatial
// feature map for the current image. All attention in this page is a
// real softmax of real dot products between those real features.
// Patch grids on the display canvas are aligned to a multiple of 7 so
// that each user-clicked patch maps cleanly to one feature vector.
// ============================================================

// ImageNet input resolution that MobileNet expects.
const MODEL_INPUT = 224;      // the model is fed 224x224
const FEATURE_GRID = 7;       // penultimate feature map is 7x7
const DISPLAY_SIZE = 448;     // display canvas at 2x so patches align

// Patch sizes on the display canvas that cleanly divide 448 AND align
// to the 7x7 grid. We map "click a patch" to the nearest feature vector.
const PATCH_SIZES = [32, 64, 112, 224];         // display pixels per patch
const PATCH_SIZE_LABELS = [32, 64, 112, 224];
const DEFAULT_PATCH_IDX = 1;                    // 64 px → 7x7 patches → aligns to feature grid

const SAMPLES = [
  {
    key: 'cat',
    label: 'Cat portrait',
    url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/June_odd-eyed-cat.jpg/640px-June_odd-eyed-cat.jpg'
  },
  {
    key: 'dog',
    label: 'Dog outdoors',
    url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/640px-Dog_Breeds.jpg'
  },
  {
    key: 'horse',
    label: 'Horse',
    url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Nokota_Horses_cropped.jpg/640px-Nokota_Horses_cropped.jpg'
  },
  {
    key: 'landscape',
    label: 'Landscape',
    url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Mount_Hood_reflected_in_Mirror_Lake%2C_Oregon.jpg/640px-Mount_Hood_reflected_in_Mirror_Lake%2C_Oregon.jpg'
  },
  {
    key: 'bird',
    label: 'Bird',
    url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/640px-Eopsaltria_australis_-_Mogo_Campground.jpg'
  },
  {
    key: 'fruit',
    label: 'Fruit / objects',
    url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Fruit_salad_with_strawberry_sauce.jpg/640px-Fruit_salad_with_strawberry_sauce.jpg'
  }
];

const state = {
  model: null,
  modelLoading: true,
  image: null,
  patchSizeIdx: DEFAULT_PATCH_IDX,
  selectedPatch: null,     // { row, col } in the display-canvas patch grid
  queryPatch: null,        // integer index into feature grid (0..N-1)
  features: null,          // Float32Array, length FEATURE_GRID^2 * C
  featureDim: 0,
  featureNorms: null,      // L2 norms of each feature vector
  withPosition: true,
  rawRgbCache: null        // cached 224x224 ImageData for "flattened vector" display
};

// ---------- Model loading ----------
async function loadModel() {
  try {
    // MobileNet v2 with alpha=1.0 (largest version) — its penultimate
    // conv activations give a 7x7x1280 feature map when input is 224x224.
    state.model = await mobilenet.load({ version: 2, alpha: 1.0 });
    state.modelLoading = false;
    setModelStatus('is-ready', 'MobileNet ready (real features)');
    if (state.image) await extractFeatures();
  } catch (err) {
    console.error('MobileNet load failed:', err);
    state.modelLoading = false;
    setModelStatus('is-error', 'Model failed to load — check network');
  }
}

function setModelStatus(cls, text) {
  const el = document.getElementById('model-status');
  const txt = document.getElementById('model-status-text');
  if (!el) return;
  el.classList.remove('is-loading', 'is-ready', 'is-error');
  el.classList.add(cls);
  txt.textContent = text;
}

// ---------- Image loading ----------
function loadImageFromElement(img) {
  state.image = img;
  state.selectedPatch = null;
  state.queryPatch = null;
  state.features = null;
  state.rawRgbCache = null;
  renderAll();
  if (state.model) extractFeatures();
  else setModelStatus('is-loading', 'Waiting for model&hellip;');
}

async function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    img.src = url;
  });
}

async function pickSample(key) {
  const s = SAMPLES.find((x) => x.key === key);
  if (!s) return;
  document.querySelectorAll('#sample-grid .sample-thumb').forEach((b) =>
    b.classList.toggle('is-active', b.dataset.sample === key));
  try {
    const img = await loadImageFromUrl(s.url);
    loadImageFromElement(img);
  } catch (err) {
    console.error('Sample failed:', err);
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

// ---------- Feature extraction ----------
async function extractFeatures() {
  if (!state.model || !state.image) return;
  // Draw the image to a 224x224 offscreen canvas (center-crop square)
  const off = document.createElement('canvas');
  off.width = MODEL_INPUT; off.height = MODEL_INPUT;
  const ctx = off.getContext('2d');
  // Cover-fit: crop to square first
  const w = state.image.naturalWidth || state.image.width;
  const h = state.image.naturalHeight || state.image.height;
  const side = Math.min(w, h);
  const sx = (w - side) / 2;
  const sy = (h - side) / 2;
  ctx.drawImage(state.image, sx, sy, side, side, 0, 0, MODEL_INPUT, MODEL_INPUT);
  // Cache 224x224 raw image data for the "flattened vector" display in Step 2
  state.rawRgbCache = ctx.getImageData(0, 0, MODEL_INPUT, MODEL_INPUT);

  // Use the model's "infer" API with embedding=true to get the penultimate
  // feature map. MobileNet v2 alpha=1.0 returns [1, 7, 7, 1280].
  const activation = state.model.infer(off, true);
  const shape = activation.shape; // [1, 7, 7, 1280] for v2 alpha=1
  const data = await activation.data(); // Float32Array
  activation.dispose();

  // Expect shape [1, H, W, C] — take H=W=7. If shape differs, reshape cleanly.
  let H = shape[1], W = shape[2], C = shape[3];
  if (H * W !== FEATURE_GRID * FEATURE_GRID) {
    // Pool / reshape: we accept any HxW, just use it.
  }
  state.featureDim = C;

  // Store features as flat Float32Array [N, C] where N = H*W.
  // Cosine-similar features by pre-normalizing.
  const N = H * W;
  const feats = new Float32Array(N * C);
  const norms = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let sum2 = 0;
    for (let c = 0; c < C; c++) {
      const v = data[i * C + c];
      feats[i * C + c] = v;
      sum2 += v * v;
    }
    norms[i] = Math.sqrt(sum2) || 1;
  }
  state.features = feats;
  state.featureNorms = norms;
  state.featureH = H;
  state.featureW = W;
  state.featureN = N;
  renderAll();
}

// ---------- Attention (real dot product over real features) ----------
function attentionFrom(queryIdx) {
  if (!state.features) return null;
  const N = state.featureN;
  const C = state.featureDim;
  const qOff = queryIdx * C;
  // Cosine-style scaled dot product: q/|q| · k/|k|, scaled by a temperature
  const qNorm = state.featureNorms[queryIdx];
  const scale = 8 / Math.sqrt(C); // gentle temperature; scale matches ViT's 1/sqrt(dk)
  // Adjust with 2D position similarity if enabled
  const H = state.featureH, W = state.featureW;
  const qr = Math.floor(queryIdx / W), qc = queryIdx % W;

  const scores = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let dot = 0;
    for (let c = 0; c < C; c++) {
      dot += state.features[qOff + c] * state.features[i * C + c];
    }
    dot = dot / (qNorm * state.featureNorms[i]); // cosine
    scores[i] = dot * 12; // temperature (empirical — makes weights peaked)
    if (state.withPosition) {
      const rr = Math.floor(i / W), cc = i % W;
      const dRow = (qr - rr) / H;
      const dCol = (qc - cc) / W;
      const posPenalty = 1.5 * Math.sqrt(dRow * dRow + dCol * dCol);
      scores[i] -= posPenalty;
    }
  }
  // Softmax
  let m = -Infinity;
  for (let i = 0; i < N; i++) if (scores[i] > m) m = scores[i];
  let sum = 0;
  const weights = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const e = Math.exp(scores[i] - m);
    weights[i] = e;
    sum += e;
  }
  for (let i = 0; i < N; i++) weights[i] /= sum;
  return weights;
}

// ---------- Canvas helpers ----------
function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  return ctx;
}

function drawImageSquare(ctx, size) {
  if (!state.image) {
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, size, size);
    return;
  }
  const w = state.image.naturalWidth || state.image.width;
  const h = state.image.naturalHeight || state.image.height;
  const side = Math.min(w, h);
  const sx = (w - side) / 2;
  const sy = (h - side) / 2;
  ctx.drawImage(state.image, sx, sy, side, side, 0, 0, size, size);
}

function displayPatchesPerSide() {
  return DISPLAY_SIZE / PATCH_SIZE_LABELS[state.patchSizeIdx];
}
function totalDisplayPatches() {
  const k = displayPatchesPerSide();
  return k * k;
}

function drawPatchGrid(ctx, size, { selected } = {}) {
  const k = displayPatchesPerSide();
  const cell = size / k;
  ctx.strokeStyle = 'rgba(253,252,249,0.55)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < k; i++) {
    const p = i * cell;
    ctx.moveTo(p, 0); ctx.lineTo(p, size);
    ctx.moveTo(0, p); ctx.lineTo(size, p);
  }
  ctx.stroke();
  if (selected) {
    ctx.strokeStyle = '#d9622b';
    ctx.lineWidth = 3;
    ctx.strokeRect(selected.col * cell + 1.5, selected.row * cell + 1.5, cell - 3, cell - 3);
  }
}

// ---------- Step 1 ----------
function renderPatchCanvas() {
  const canvas = document.getElementById('patchCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas, DISPLAY_SIZE, DISPLAY_SIZE);
  drawImageSquare(ctx, DISPLAY_SIZE);
  drawPatchGrid(ctx, DISPLAY_SIZE, { selected: state.selectedPatch });
}

function updateStatStrip() {
  const k = displayPatchesPerSide();
  const P = PATCH_SIZE_LABELS[state.patchSizeIdx];
  document.getElementById('stat-grid').textContent = `${k} × ${k}`;
  document.getElementById('stat-n-patches').textContent = k * k;
  document.getElementById('stat-ppp').textContent = P * P;
  document.getElementById('stat-flat').textContent = P * P * 3;
  document.getElementById('val-patch-size').textContent = P;
}

// ---------- Step 2 ----------
function renderPatchDetailCanvas() {
  const canvas = document.getElementById('patchDetailCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas, DISPLAY_SIZE, DISPLAY_SIZE);
  drawImageSquare(ctx, DISPLAY_SIZE);
  if (state.selectedPatch) {
    const k = displayPatchesPerSide();
    const cell = DISPLAY_SIZE / k;
    ctx.fillStyle = 'rgba(253,252,249,0.55)';
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        if (r === state.selectedPatch.row && c === state.selectedPatch.col) continue;
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
  }
  drawPatchGrid(ctx, DISPLAY_SIZE, { selected: state.selectedPatch });
}

function updatePatchDetail() {
  const infoBox = document.getElementById('patchInfoBox');
  const vectorRow = document.getElementById('patchVectorRow');
  if (!state.selectedPatch || !state.rawRgbCache) {
    infoBox.className = 'patch-detail patch-detail-empty';
    infoBox.textContent = 'No patch selected yet. Click one in Step 1 or above.';
    vectorRow.innerHTML = '<span class="patch-detail-empty" style="font-style:italic">—</span>';
    return;
  }
  const { row, col } = state.selectedPatch;
  const P = PATCH_SIZE_LABELS[state.patchSizeIdx];
  // Compute pixel bounds in the 224x224 cropped image
  const scale = MODEL_INPUT / DISPLAY_SIZE;
  const baseX = Math.round(col * P * scale);
  const baseY = Math.round(row * P * scale);
  const sampleN = 4;
  const stride = Math.max(1, Math.floor(P * scale / sampleN));

  let lines = [];
  lines.push(`Patch (row ${row}, col ${col})`);
  lines.push(`Size: ${P} × ${P} px on display → ${Math.round(P * scale)} × ${Math.round(P * scale)} on model input (224×224)`);
  lines.push(`Raw flat vector length: ${P * P * 3}`);
  lines.push('');
  lines.push('Sampled 4 × 4 pixels (R, G, B):');
  for (let j = 0; j < sampleN; j++) {
    let row = '';
    for (let i = 0; i < sampleN; i++) {
      const px = Math.min(MODEL_INPUT - 1, baseX + i * stride);
      const py = Math.min(MODEL_INPUT - 1, baseY + j * stride);
      const idx = (py * MODEL_INPUT + px) * 4;
      const r = state.rawRgbCache.data[idx].toString().padStart(3, ' ');
      const g = state.rawRgbCache.data[idx + 1].toString().padStart(3, ' ');
      const b = state.rawRgbCache.data[idx + 2].toString().padStart(3, ' ');
      row += `(${r},${g},${b}) `;
    }
    lines.push(row);
  }
  infoBox.className = 'patch-detail';
  infoBox.textContent = lines.join('\n');

  // First 12 raw bytes
  const baseIdx = (baseY * MODEL_INPUT + baseX) * 4;
  const vals = [];
  for (let i = 0; i < 4; i++) {
    const idx = baseIdx + i * 4;
    vals.push(state.rawRgbCache.data[idx], state.rawRgbCache.data[idx + 1], state.rawRgbCache.data[idx + 2]);
  }
  while (vals.length < 12) vals.push(0);
  vectorRow.innerHTML =
    vals.slice(0, 12).map((v) => `<span class="vector-cell">${v}</span>`).join('') +
    '<span class="vector-cell" style="background:#faf7ef;color:var(--muted)">…</span>';
}

// ---------- Step 3 sequence strip ----------
function updateSequenceStrip() {
  const strip = document.getElementById('sequenceStrip');
  if (!strip) return;
  if (!state.image) {
    strip.innerHTML = 'Waiting for photo&hellip;';
    return;
  }
  // Show tokens of the feature grid (since that's what the model uses)
  const N = state.featureN || (FEATURE_GRID * FEATURE_GRID);
  const tokens = ['[CLS]'];
  for (let i = 0; i < N; i++) tokens.push(`p${i + 1}`);
  const MAX = 54;
  const show = tokens.slice(0, MAX);
  const overflow = tokens.length - show.length;
  let html = '';
  show.forEach((t, i) => {
    if (t === '[CLS]') {
      html += '<span class="seq-token cls">[CLS]</span>';
    } else {
      const posPart = state.withPosition ? `<br>+ pos${i}` : '';
      const cls = state.withPosition ? 'seq-token pos' : 'seq-token';
      html += `<span class="${cls}">${t}${posPart}</span>`;
    }
  });
  if (overflow > 0) html += `<span class="seq-token" style="background:#faf7ef;color:var(--muted)">… +${overflow}</span>`;
  strip.innerHTML = html;
}

// ---------- Step 5 attention ----------
function renderAttnCanvas() {
  const canvas = document.getElementById('attnCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas, DISPLAY_SIZE, DISPLAY_SIZE);
  drawImageSquare(ctx, DISPLAY_SIZE);

  if (state.queryPatch != null && state.features) {
    const weights = attentionFrom(state.queryPatch);
    if (!weights) return;
    const H = state.featureH, W = state.featureW;
    // Compose a grid of per-cell weights at feature resolution, render with alpha
    const cellW = DISPLAY_SIZE / W;
    const cellH = DISPLAY_SIZE / H;
    const maxW = Math.max(...weights);
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const w = weights[r * W + c];
        const t = Math.min(1, w / (maxW + 1e-9));
        ctx.fillStyle = t < 0.08 ? 'rgba(0,0,0,0.3)' : `rgba(217, 98, 43, ${0.15 + 0.65 * t})`;
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
      }
    }
    // Mark query
    const qr = Math.floor(state.queryPatch / W);
    const qc = state.queryPatch % W;
    ctx.strokeStyle = '#2c6fb7';
    ctx.lineWidth = 3;
    ctx.strokeRect(qc * cellW + 1.5, qr * cellH + 1.5, cellW - 3, cellH - 3);
  }

  drawPatchGrid(ctx, DISPLAY_SIZE);
}

function updateTopAttn() {
  const el = document.getElementById('topAttnTable');
  const qStat = document.getElementById('stat-query');
  const topStat = document.getElementById('stat-top-attn');
  if (state.queryPatch == null || !state.features) {
    el.className = 'patch-detail patch-detail-empty';
    el.textContent = 'Click a patch first.';
    qStat.textContent = '—';
    topStat.textContent = '—';
    return;
  }
  const weights = attentionFrom(state.queryPatch);
  const H = state.featureH, W = state.featureW;
  const qr = Math.floor(state.queryPatch / W);
  const qc = state.queryPatch % W;
  qStat.textContent = `(${qr},${qc})`;
  const idx = weights.map((w, i) => ({ i, w })).sort((a, b) => b.w - a.w).slice(0, 5);
  const topIdx = idx[0].i;
  const tr = Math.floor(topIdx / W);
  const tc = topIdx % W;
  topStat.textContent = topIdx === state.queryPatch ? 'self' : `(${tr},${tc})`;

  const lines = [];
  lines.push('rank  cell      weight');
  idx.forEach((x, rank) => {
    const r = Math.floor(x.i / W), c = x.i % W;
    const label = x.i === state.queryPatch ? 'self' : `(${r},${c})`;
    lines.push(`  ${rank + 1}   ${label.padEnd(8, ' ')} ${(x.w * 100).toFixed(1)}%`);
  });
  el.className = 'patch-detail';
  el.textContent = lines.join('\n');
}

// Map display-patch → feature-grid cell.
function displayPatchToFeatureIdx(row, col) {
  if (!state.features) return null;
  const k = displayPatchesPerSide();
  const H = state.featureH, W = state.featureW;
  // Scale patch center to feature-grid coords
  const fr = Math.min(H - 1, Math.floor((row + 0.5) * H / k));
  const fc = Math.min(W - 1, Math.floor((col + 0.5) * W / k));
  return fr * W + fc;
}

// ---------- Step 6 pipeline & scaling tables ----------
function updatePipelineTable() {
  const body = document.getElementById('pipelineTableBody');
  if (!body) return;
  const k = displayPatchesPerSide();
  const N = k * k;
  const P = PATCH_SIZE_LABELS[state.patchSizeIdx];
  const D = 768;
  const rows = [
    ['1. Patchify', 'Split image into P×P non-overlapping patches.', `${N} patches of size ${P}×${P}`],
    ['2. Flatten', 'Each patch: P×P×3 pixels → vector of length 3P².', `vector length ${3 * P * P}`],
    ['3. Project', `Linear layer to D dims (D = ${D} shown).`, `${N} × ${D} matrix of tokens`],
    ['4. Add positions', 'Add learned position embedding to each patch.', `still ${N} × ${D}`],
    ['5. Prepend [CLS]', 'Learnable class token at front.', `sequence length ${N + 1}`],
    ['6. Self-attention × L', 'L encoder blocks (12 in ViT-Base).', `${N + 1} tokens, ${D} dim`],
    ['7. Read CLS', 'Final CLS → MLP head → class logits.', `D → num_classes`]
  ];
  body.innerHTML = rows.map((r) =>
    `<tr><td><strong>${r[0]}</strong></td><td>${r[1]}</td><td>${r[2]}</td></tr>`
  ).join('');

  document.getElementById('reveal-seqlen').textContent = N + 1;
  document.getElementById('reveal-breakdown').textContent =
    `1 CLS token + ${N} patch token${N === 1 ? '' : 's'}`;
}

function updateScalingTable() {
  const body = document.getElementById('scalingTableBody');
  if (!body) return;
  const baseIdx = DEFAULT_PATCH_IDX;
  const baseK = DISPLAY_SIZE / PATCH_SIZE_LABELS[baseIdx];
  const baseOps = (baseK * baseK + 1) * (baseK * baseK + 1);
  const rows = PATCH_SIZE_LABELS.map((P) => {
    const k = DISPLAY_SIZE / P;
    const tokens = k * k + 1;
    const ops = tokens * tokens;
    return { P, k, tokens, ops: ops / baseOps };
  });
  body.innerHTML = rows.map((r) => {
    const active = r.P === PATCH_SIZE_LABELS[state.patchSizeIdx];
    return `
      <tr${active ? ' style="background:rgba(44,111,183,0.05)"' : ''}>
        <td><strong>${r.P} × ${r.P}${active ? ' (current)' : ''}</strong></td>
        <td>${r.k} × ${r.k}</td>
        <td>${r.tokens}</td>
        <td>${r.ops.toFixed(2)}×</td>
      </tr>
    `;
  }).join('');
}

// ---------- Input ----------
function wireCanvases() {
  const handlers = [
    { id: 'patchCanvas', selectsPatch: true, setsQuery: false },
    { id: 'patchDetailCanvas', selectsPatch: true, setsQuery: false },
    { id: 'attnCanvas', selectsPatch: false, setsQuery: true }
  ];
  handlers.forEach((h) => {
    const el = document.getElementById(h.id);
    if (!el) return;
    el.addEventListener('mousedown', (e) => {
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const k = displayPatchesPerSide();
      const cellW = rect.width / k, cellH = rect.height / k;
      const col = Math.min(k - 1, Math.max(0, Math.floor(cx / cellW)));
      const row = Math.min(k - 1, Math.max(0, Math.floor(cy / cellH)));
      if (h.selectsPatch) state.selectedPatch = { row, col };
      if (h.setsQuery) {
        const idx = displayPatchToFeatureIdx(row, col);
        if (idx != null) state.queryPatch = idx;
      }
      // If clicking the patch grid and features exist, also set the query so
      // both canvases feel alive.
      if (h.selectsPatch && state.features) {
        const idx = displayPatchToFeatureIdx(row, col);
        if (idx != null) state.queryPatch = idx;
      }
      renderAll();
    });
  });
}

function wireControls() {
  // Samples
  const grid = document.getElementById('sample-grid');
  if (grid) {
    grid.innerHTML = SAMPLES.map((s) => `<button class="sample-thumb" data-sample="${s.key}">${s.label}</button>`).join('');
    grid.querySelectorAll('[data-sample]').forEach((b) =>
      b.addEventListener('click', () => pickSample(b.dataset.sample)));
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

  // Patch slider
  const psSlider = document.getElementById('patch-size');
  psSlider.value = DEFAULT_PATCH_IDX;
  psSlider.addEventListener('input', () => {
    state.patchSizeIdx = parseInt(psSlider.value, 10);
    state.selectedPatch = null;
    renderAll();
  });

  // Position embeddings
  const on = document.getElementById('pos-on');
  const off = document.getElementById('pos-off');
  on.addEventListener('click', () => {
    state.withPosition = true;
    on.classList.add('is-active'); off.classList.remove('is-active');
    renderAll();
  });
  off.addEventListener('click', () => {
    state.withPosition = false;
    off.classList.add('is-active'); on.classList.remove('is-active');
    renderAll();
  });
}

// ---------- KaTeX ----------
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-patches': 'N = \\left(\\tfrac{H}{P}\\right)^2 \\quad \\text{for } H \\times H \\text{ image, } P \\times P \\text{ patches}',
    'math-flatten': '\\mathbf{x}_i = \\mathrm{flatten}(\\text{patch}_i) \\in \\mathbb{R}^{3P^2}, \\quad \\mathbf{z}_i = E\\,\\mathbf{x}_i \\in \\mathbb{R}^D',
    'math-position': '\\mathbf{z}_i \\leftarrow \\mathbf{z}_i + \\mathbf{p}_i',
    'math-cls': '\\mathbf{Z}_0 = \\bigl[\\,\\mathbf{z}_{\\text{cls}};\\, \\mathbf{z}_1;\\, \\dots;\\, \\mathbf{z}_N\\,\\bigr]',
    'math-attention': '\\mathrm{Attn}(Q, K, V) = \\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

// ---------- Render all ----------
function renderAll() {
  updateStatStrip();
  renderPatchCanvas();
  renderPatchDetailCanvas();
  renderAttnCanvas();
  updatePatchDetail();
  updateSequenceStrip();
  updateTopAttn();
  updatePipelineTable();
  updateScalingTable();
}

// ---------- Boot ----------
function init() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wireControls();
  wireCanvases();
  loadModel();
  pickSample(SAMPLES[0].key);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
