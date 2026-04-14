// ============================================================
// Image Segmentation with a Real Segmenter
// Loads TF.js DeepLab v3+ (Pascal VOC) and runs it on the user's photo.
// All metrics (mean IoU, per-class IoU, pixel accuracy) compare the
// user's painted mask to the model's real output.
// ============================================================

// Pascal VOC 21 classes
const PASCAL_CLASSES = [
  'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
  'dog', 'horse', 'motorbike', 'person', 'pottedplant',
  'sheep', 'sofa', 'train', 'tvmonitor'
];

// Pascal VOC colormap (standard)
function pascalColor(idx) {
  // Deterministic colormap generator from the official VOC devkit
  let r = 0, g = 0, b = 0;
  let id = idx;
  for (let i = 0; i < 8; i++) {
    r = r | (((id >> 0) & 1) << (7 - i));
    g = g | (((id >> 1) & 1) << (7 - i));
    b = b | (((id >> 2) & 1) << (7 - i));
    id = id >> 3;
  }
  return [r, g, b];
}

// Samples use picsum.photos with specific deterministic IDs.
// These CDN URLs are CORS-enabled and stable; each `/id/N` returns the
// same photo every time. We include fallback Wikimedia URLs in case the
// primary fails.
const SAMPLES = [
  {
    key: 'puppy',
    label: 'Puppy',
    urls: [
      'https://picsum.photos/id/237/640/400'
    ]
  },
  {
    key: 'dog',
    label: 'Corgi',
    urls: [
      'https://picsum.photos/id/1025/640/400'
    ]
  },
  {
    key: 'person',
    label: 'Person in scene',
    urls: [
      'https://picsum.photos/id/64/640/400'
    ]
  },
  {
    key: 'bike',
    label: 'Bicycle / urban',
    urls: [
      'https://picsum.photos/id/145/640/400'
    ]
  },
  {
    key: 'living-room',
    label: 'Living room',
    urls: [
      'https://picsum.photos/id/116/640/400'
    ]
  },
  {
    key: 'horse',
    label: 'Horse / field',
    urls: [
      'https://picsum.photos/id/173/640/400'
    ]
  }
];

const state = {
  model: null,
  modelLoading: true,
  // Working resolution (model runs in "pascal" native size internally).
  // We display the image scaled to a canvas of at most displayW x displayH.
  displayW: 640,
  displayH: 400,
  image: null,         // HTMLImageElement or Canvas
  imageW: 0,
  imageH: 0,
  // Segmentation output: Uint8Array of length W*H of class IDs (native resolution)
  segMap: null,
  segW: 0,
  segH: 0,
  // User mask painted on a canvas at the display resolution; stored as
  // Uint8Array of length displayW * displayH.
  userMask: null,
  currentLabel: 0,
  brushSize: 14,
  alpha: 0.55,
  view: 'image',
  // Region growing
  rgRegion: null,
  rgCount: 0,
  rgSeed: null,
  tau: 30,
  rgMode: 'seed',
  inferenceMs: 0,
  classesInImage: []  // sorted list of class IDs present
};

// ---------- Model loading ----------
async function loadModel() {
  try {
    state.model = await deeplab.load({ base: 'pascal', quantizationBytes: 2 });
    state.modelLoading = false;
    setModelStatus('is-ready', 'Segmenter ready');
    if (state.image) await runSegmentation();
  } catch (err) {
    console.error('DeepLab load failed:', err);
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
  state.imageW = img.naturalWidth || img.width;
  state.imageH = img.naturalHeight || img.height;
  const maxW = 640;
  if (state.imageW > maxW) {
    state.displayW = maxW;
    state.displayH = Math.round(state.imageH * maxW / state.imageW);
  } else {
    state.displayW = state.imageW;
    state.displayH = state.imageH;
  }
  state.userMask = new Uint8Array(state.displayW * state.displayH);
  state.segMap = null;
  state.rgRegion = null;
  state.rgSeed = null;
  state.rgCount = 0;
  renderAll();
  if (state.model) runSegmentation();
  else setModelStatus('is-loading', 'Waiting for segmenter&hellip;');
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(new Error(`Failed to load ${url}`));
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
  if (cap) cap.textContent = `Loading “${s.label}”\u2026`;
  try {
    const img = await loadImageWithFallback(s.urls);
    loadImageFromElement(img);
    if (cap) cap.textContent = `${s.label} — DeepLab output drawn over it.`;
  } catch (err) {
    console.error('Sample load failed:', err);
    if (cap) cap.textContent = `Could not load “${s.label}”. Drop a photo into the upload area above.`;
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

// ---------- Run segmentation ----------
async function runSegmentation() {
  if (!state.model || !state.image) return;
  const t0 = performance.now();
  // Draw the image to a canvas at display resolution so segmentation aligns
  const off = document.createElement('canvas');
  off.width = state.displayW;
  off.height = state.displayH;
  off.getContext('2d').drawImage(state.image, 0, 0, state.displayW, state.displayH);

  const result = await state.model.segment(off);
  // result = { legend: {className: color}, height, width, segmentationMap: Uint8ClampedArray (RGB), ... }
  // But we want class indices. The library also exposes rawSegmentationMap in newer versions.
  // Instead, the cleanest way is to decode the RGB map back to class indices using PASCAL colormap.
  const segMap = rgbMapToClassMap(result.segmentationMap, result.width, result.height);
  state.segMap = segMap;
  state.segW = result.width;
  state.segH = result.height;
  state.inferenceMs = performance.now() - t0;

  // Upsample segMap to (displayW, displayH) for per-pixel metrics vs user mask
  state.segMapDisplay = resizeLabelMap(segMap, state.segW, state.segH, state.displayW, state.displayH);

  // Count classes in image
  const classSet = new Set();
  for (let i = 0; i < state.segMapDisplay.length; i++) {
    classSet.add(state.segMapDisplay[i]);
  }
  state.classesInImage = [...classSet].sort((a, b) => a - b);

  renderAll();
}

function rgbMapToClassMap(rgbArr, w, h) {
  // Build reverse lookup: color_key -> class id
  const colorToId = new Map();
  for (let i = 0; i < 21; i++) {
    const [r, g, b] = pascalColor(i);
    colorToId.set((r << 16) | (g << 8) | b, i);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = rgbArr[i * 4];
    const g = rgbArr[i * 4 + 1];
    const b = rgbArr[i * 4 + 2];
    const key = (r << 16) | (g << 8) | b;
    out[i] = colorToId.has(key) ? colorToId.get(key) : 0;
  }
  return out;
}

function resizeLabelMap(src, sw, sh, dw, dh) {
  // Nearest-neighbour resize
  const out = new Uint8Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor(y * sh / dh));
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.floor(x * sw / dw));
      out[y * dw + x] = src[sy * sw + sx];
    }
  }
  return out;
}

// ---------- Canvas helpers ----------
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = state.displayW * dpr;
  canvas.height = state.displayH * dpr;
  canvas.style.width = state.displayW + 'px';
  canvas.style.height = state.displayH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  return ctx;
}

function drawImage(ctx) {
  if (!state.image) {
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, state.displayW, state.displayH);
    return;
  }
  ctx.drawImage(state.image, 0, 0, state.displayW, state.displayH);
}

function drawMaskOverlay(ctx, labelMap, alpha, skipBackground = true) {
  if (!labelMap) return;
  const buf = new Uint8ClampedArray(state.displayW * state.displayH * 4);
  for (let i = 0; i < labelMap.length; i++) {
    const c = labelMap[i];
    if (c === 0 && skipBackground) {
      buf[i * 4 + 3] = 0;
    } else {
      const [r, g, b] = pascalColor(c);
      buf[i * 4] = r; buf[i * 4 + 1] = g; buf[i * 4 + 2] = b;
      buf[i * 4 + 3] = Math.round(alpha * 255);
    }
  }
  const off = document.createElement('canvas');
  off.width = state.displayW; off.height = state.displayH;
  off.getContext('2d').putImageData(
    new ImageData(buf, state.displayW, state.displayH), 0, 0);
  ctx.drawImage(off, 0, 0);
}

// ---------- Prelude & Step 1 ----------
function renderPrelude() {
  const canvas = document.getElementById('preludeCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  drawImage(ctx);
  if (state.segMapDisplay) drawMaskOverlay(ctx, state.segMapDisplay, 0.55);

  document.getElementById('classes-found').textContent =
    state.classesInImage.length ? state.classesInImage.length : '—';
  document.getElementById('inference-ms').textContent =
    state.inferenceMs > 0 ? `${state.inferenceMs.toFixed(0)} ms` : '—';
  document.getElementById('num-pixels').textContent =
    (state.displayW * state.displayH).toLocaleString();

  const paletteRow = document.getElementById('palette-row');
  if (paletteRow) {
    if (state.classesInImage.length === 0) {
      paletteRow.innerHTML = '<span style="font-family:var(--sans); color:var(--muted); font-style:italic;">No classes detected yet.</span>';
    } else {
      paletteRow.innerHTML = state.classesInImage.map((c) => {
        const [r, g, b] = pascalColor(c);
        return `<span class="label-chip"><span class="swatch" style="background:rgb(${r},${g},${b})"></span>${PASCAL_CLASSES[c]} (id ${c})</span>`;
      }).join('');
    }
  }
}

function renderModelCanvas() {
  const canvas = document.getElementById('modelCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  if (state.view === 'image') {
    drawImage(ctx);
  } else if (state.view === 'overlay') {
    drawImage(ctx);
    if (state.segMapDisplay) drawMaskOverlay(ctx, state.segMapDisplay, state.alpha);
  } else {
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, state.displayW, state.displayH);
    if (state.segMapDisplay) drawMaskOverlay(ctx, state.segMapDisplay, 1, false);
  }
}

// ---------- Step 2 paint ----------
function renderReferenceCanvas() {
  const canvas = document.getElementById('referenceCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  drawImage(ctx);
  if (state.segMapDisplay) drawMaskOverlay(ctx, state.segMapDisplay, 0.3);
}

function renderPaintCanvas() {
  const canvas = document.getElementById('paintCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  drawImage(ctx);
  ctx.fillStyle = 'rgba(253,252,249,0.5)';
  ctx.fillRect(0, 0, state.displayW, state.displayH);
  if (state.userMask) drawMaskOverlay(ctx, state.userMask, 0.85);
}

function renderPaintToolbar() {
  const bar = document.getElementById('paint-toolbar');
  if (!bar) return;
  const classesHere = state.classesInImage.length ? state.classesInImage : [0];
  bar.innerHTML = classesHere.map((c) => {
    const [r, g, b] = pascalColor(c);
    const active = state.currentLabel === c ? ' is-active' : '';
    const bg = state.currentLabel === c ? `rgb(${r},${g},${b})` : 'white';
    const col = state.currentLabel === c ? 'white' : `rgb(${r},${g},${b})`;
    return `<button class="mode-button${active}" data-label="${c}" style="border-color:rgb(${r},${g},${b});color:${col};background:${bg}">${PASCAL_CLASSES[c]}</button>`;
  }).join('') +
  `<button class="mode-button" data-label="-1" style="border-color:var(--border);color:var(--muted)">Eraser</button>`;
  bar.querySelectorAll('[data-label]').forEach((b) => {
    b.addEventListener('click', () => {
      const v = parseInt(b.dataset.label, 10);
      state.currentLabel = v === -1 ? 0 : v;
      renderPaintToolbar();
    });
  });
}

function paintAt(x, y) {
  if (!state.userMask) return;
  const r = state.brushSize;
  const xi = Math.round(x);
  const yi = Math.round(y);
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy > r * r) continue;
      const nx = xi + dx, ny = yi + dy;
      if (nx < 0 || nx >= state.displayW || ny < 0 || ny >= state.displayH) continue;
      state.userMask[ny * state.displayW + nx] = state.currentLabel;
    }
  }
}

function updatePaintMetrics() {
  if (!state.userMask || !state.segMapDisplay) {
    document.getElementById('pix-painted').textContent = '0';
    document.getElementById('pixel-acc').textContent = '—';
    document.getElementById('mean-iou').textContent = '—';
    document.getElementById('per-class-iou').innerHTML = '';
    return;
  }
  const gt = state.segMapDisplay;
  const um = state.userMask;
  let painted = 0;
  let correct = 0;
  const perCls = {};
  for (let i = 0; i < um.length; i++) {
    const u = um[i];
    const g = gt[i];
    if (u !== 0) painted++;
    if (u === g) correct++;
    const toTrack = new Set([g, u].filter((x) => x > 0));
    toTrack.forEach((c) => {
      if (!perCls[c]) perCls[c] = { inter: 0, union: 0 };
      if (u === c && g === c) perCls[c].inter++;
      if (u === c || g === c) perCls[c].union++;
    });
  }
  document.getElementById('pix-painted').textContent = painted.toLocaleString();
  document.getElementById('pixel-acc').textContent = (correct / um.length).toFixed(3);

  const clsList = Object.keys(perCls).map(Number);
  let miou = 0;
  clsList.forEach((c) => {
    const r = perCls[c];
    miou += r.union > 0 ? r.inter / r.union : 0;
  });
  miou = clsList.length ? miou / clsList.length : 0;
  document.getElementById('mean-iou').textContent = miou.toFixed(3);

  const wrap = document.getElementById('per-class-iou');
  if (wrap) {
    if (clsList.length === 0) {
      wrap.innerHTML = '<p style="font-family:var(--sans); color:var(--muted); font-style:italic;">Start painting to see per-class IoU.</p>';
    } else {
      let html = '<table class="examples-table"><thead><tr><th>Class</th><th>Intersection</th><th>Union</th><th>IoU</th></tr></thead><tbody>';
      clsList.forEach((c) => {
        const r = perCls[c];
        const iou = r.union > 0 ? r.inter / r.union : 0;
        const [R, G, B] = pascalColor(c);
        html += `<tr><td><span class="label-chip" style="margin:0;padding:0.1rem 0.5rem;"><span class="swatch" style="background:rgb(${R},${G},${B})"></span>${PASCAL_CLASSES[c]}</span></td><td>${r.inter.toLocaleString()}</td><td>${r.union.toLocaleString()}</td><td><strong>${iou.toFixed(3)}</strong></td></tr>`;
      });
      html += '</tbody></table>';
      wrap.innerHTML = html;
    }
  }
}

// ---------- Step 3 region growing ----------
function runRegionGrow(sx, sy, tau, mode) {
  if (!state.image) return null;
  // Build an ImageData for the displayed image once, cache on state
  if (!state.cachedRgb) {
    const off = document.createElement('canvas');
    off.width = state.displayW; off.height = state.displayH;
    const oc = off.getContext('2d');
    oc.drawImage(state.image, 0, 0, state.displayW, state.displayH);
    state.cachedRgb = oc.getImageData(0, 0, state.displayW, state.displayH).data;
  }
  const rgb = state.cachedRgb;
  const W = state.displayW, H = state.displayH;
  const idx0 = (sy * W + sx) * 4;
  const seed = [rgb[idx0], rgb[idx0 + 1], rgb[idx0 + 2]];
  const visited = new Uint8Array(W * H);
  const region = new Uint8Array(W * H);
  const queue = [[sx, sy]];
  visited[sy * W + sx] = 1; region[sy * W + sx] = 1;
  let rSum = seed[0], gSum = seed[1], bSum = seed[2], cnt = 1;
  let steps = 0;
  const MAX = W * H;
  while (queue.length && steps < MAX) {
    const [x, y] = queue.shift(); steps++;
    const nb = [[1,0],[-1,0],[0,1],[0,-1]];
    for (const [dx, dy] of nb) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
      const i = ny * W + nx;
      if (visited[i]) continue;
      visited[i] = 1;
      const p = i * 4;
      const r = rgb[p], g = rgb[p + 1], b = rgb[p + 2];
      const ref = mode === 'seed' ? seed : [rSum / cnt, gSum / cnt, bSum / cnt];
      const d = Math.hypot(r - ref[0], g - ref[1], b - ref[2]);
      if (d <= tau) {
        region[i] = 1;
        queue.push([nx, ny]);
        if (mode === 'running') { rSum += r; gSum += g; bSum += b; cnt++; }
      }
    }
  }
  return { region, count: cnt, seed };
}

function renderRgCanvas() {
  const canvas = document.getElementById('rgCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  drawImage(ctx);
  if (state.rgRegion) {
    const buf = new Uint8ClampedArray(state.displayW * state.displayH * 4);
    for (let i = 0; i < state.rgRegion.length; i++) {
      if (state.rgRegion[i]) {
        buf[i * 4] = 217; buf[i * 4 + 1] = 98; buf[i * 4 + 2] = 43; buf[i * 4 + 3] = 180;
      }
    }
    const off = document.createElement('canvas');
    off.width = state.displayW; off.height = state.displayH;
    off.getContext('2d').putImageData(new ImageData(buf, state.displayW, state.displayH), 0, 0);
    ctx.drawImage(off, 0, 0);
    if (state.rgSeed) {
      ctx.fillStyle = '#2c6fb7';
      ctx.strokeStyle = 'white'; ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(state.rgSeed.x + 0.5, state.rgSeed.y + 0.5, 6, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
    }
  }
  // Stats
  const colorEl = document.getElementById('seed-color');
  const countEl = document.getElementById('region-size');
  const iouEl = document.getElementById('rg-iou');
  if (state.rgRegion && state.rgSeed) {
    const s = state.cachedRgb;
    const idx = (state.rgSeed.y * state.displayW + state.rgSeed.x) * 4;
    colorEl.textContent = `(${s[idx]},${s[idx+1]},${s[idx+2]})`;
    countEl.textContent = state.rgCount.toLocaleString();
    // IoU vs model output (for seed pixel's predicted class)
    if (state.segMapDisplay) {
      const seedClass = state.segMapDisplay[state.rgSeed.y * state.displayW + state.rgSeed.x];
      let inter = 0, union = 0;
      for (let i = 0; i < state.rgRegion.length; i++) {
        const a = state.rgRegion[i] === 1;
        const b = state.segMapDisplay[i] === seedClass && seedClass !== 0;
        if (a && b) inter++;
        if (a || b) union++;
      }
      iouEl.textContent = union > 0 ? (inter / union).toFixed(3) : '—';
    } else {
      iouEl.textContent = '—';
    }
  } else {
    colorEl.textContent = '—';
    countEl.textContent = '0';
    iouEl.textContent = '—';
  }
}

// ---------- Input wiring ----------
function wireTabs() {
  document.querySelectorAll('[data-view]').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      state.view = b.dataset.view;
      renderModelCanvas();
    });
  });
  const alphaSlider = document.getElementById('alpha-slider');
  const alphaVal = document.getElementById('alpha-val');
  alphaSlider.addEventListener('input', () => {
    state.alpha = parseFloat(alphaSlider.value);
    alphaVal.textContent = state.alpha.toFixed(2);
    renderModelCanvas();
    renderPrelude();
  });
}

function wireHover() {
  const canvas = document.getElementById('modelCanvas');
  const info = document.getElementById('hover-info');
  if (!canvas || !info) return;
  canvas.addEventListener('mousemove', (e) => {
    if (!state.segMapDisplay) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * state.displayW);
    const y = Math.floor((e.clientY - rect.top) / rect.height * state.displayH);
    const c = state.segMapDisplay[y * state.displayW + x];
    info.textContent = `(${x},${y}) → ${PASCAL_CLASSES[c]} (id ${c})`;
  });
  canvas.addEventListener('mouseleave', () => { info.textContent = '—'; });
}

function wirePainting() {
  const canvas = document.getElementById('paintCanvas');
  if (!canvas) return;
  let painting = false;
  const getXY = (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    return { x: cx / rect.width * state.displayW, y: cy / rect.height * state.displayH };
  };
  const onDown = (e) => {
    painting = true;
    const { x, y } = getXY(e);
    paintAt(x, y);
    renderPaintCanvas();
    updatePaintMetrics();
    if (e.touches) e.preventDefault();
  };
  const onMove = (e) => {
    if (!painting) return;
    const { x, y } = getXY(e);
    paintAt(x, y);
    renderPaintCanvas();
    updatePaintMetrics();
    if (e.touches) e.preventDefault();
  };
  const onUp = () => { painting = false; };
  canvas.addEventListener('mousedown', onDown);
  canvas.addEventListener('touchstart', onDown, { passive: false });
  window.addEventListener('mousemove', onMove);
  window.addEventListener('touchmove', onMove, { passive: false });
  window.addEventListener('mouseup', onUp);
  window.addEventListener('touchend', onUp);

  document.getElementById('btn-clear-mask').addEventListener('click', () => {
    state.userMask = new Uint8Array(state.displayW * state.displayH);
    renderPaintCanvas();
    updatePaintMetrics();
  });
  document.getElementById('btn-show-gt').addEventListener('click', () => {
    if (state.segMapDisplay) {
      state.userMask = new Uint8Array(state.segMapDisplay);
      renderPaintCanvas();
      updatePaintMetrics();
    }
  });

  const brushSlider = document.getElementById('brush-size');
  const brushVal = document.getElementById('brush-size-val');
  brushSlider.addEventListener('input', () => {
    state.brushSize = parseInt(brushSlider.value, 10);
    brushVal.textContent = state.brushSize;
  });
}

function wireRG() {
  const canvas = document.getElementById('rgCanvas');
  if (!canvas) return;
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * state.displayW);
    const y = Math.floor((e.clientY - rect.top) / rect.height * state.displayH);
    const res = runRegionGrow(x, y, state.tau, state.rgMode);
    if (res) {
      state.rgSeed = { x, y };
      state.rgRegion = res.region;
      state.rgCount = res.count;
      renderRgCanvas();
    }
  });
  const tauSlider = document.getElementById('tau-slider');
  const tauVal = document.getElementById('tau-val');
  tauSlider.addEventListener('input', () => {
    state.tau = parseInt(tauSlider.value, 10);
    tauVal.textContent = state.tau;
    if (state.rgSeed) {
      const res = runRegionGrow(state.rgSeed.x, state.rgSeed.y, state.tau, state.rgMode);
      state.rgRegion = res.region; state.rgCount = res.count;
      renderRgCanvas();
    }
  });
  const seedBtn = document.getElementById('mode-seed');
  const runBtn = document.getElementById('mode-running');
  const lbl = document.getElementById('rg-mode-label');
  seedBtn.addEventListener('click', () => {
    state.rgMode = 'seed';
    seedBtn.classList.add('is-active'); runBtn.classList.remove('is-active');
    lbl.textContent = 'seed pixel colour';
    if (state.rgSeed) {
      const res = runRegionGrow(state.rgSeed.x, state.rgSeed.y, state.tau, 'seed');
      state.rgRegion = res.region; state.rgCount = res.count;
      renderRgCanvas();
    }
  });
  runBtn.addEventListener('click', () => {
    state.rgMode = 'running';
    runBtn.classList.add('is-active'); seedBtn.classList.remove('is-active');
    lbl.textContent = 'running mean colour';
    if (state.rgSeed) {
      const res = runRegionGrow(state.rgSeed.x, state.rgSeed.y, state.tau, 'running');
      state.rgRegion = res.region; state.rgCount = res.count;
      renderRgCanvas();
    }
  });
}

function wireUpload() {
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
  const webBtn = document.getElementById('webcam-btn');
  webBtn.addEventListener('click', async (e) => {
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
}

function renderAll() {
  state.cachedRgb = null; // invalidate; rebuilt on next RG run
  renderPrelude();
  renderModelCanvas();
  renderReferenceCanvas();
  renderPaintCanvas();
  renderPaintToolbar();
  renderRgCanvas();
  updatePaintMetrics();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-function':
      'f \\colon \\{1, \\dots, W\\} \\times \\{1, \\dots, H\\} \\to \\{0, 1, \\dots, C - 1\\}',
    'math-metrics':
      '\\text{Pixel accuracy} = \\frac{\\#\\{p : \\hat f(p) = f(p)\\}}{W H} \\qquad ' +
      '\\mathrm{IoU}_c = \\frac{|\\hat M_c \\cap M_c|}{|\\hat M_c \\cup M_c|}',
    'math-losses':
      '\\mathcal L_{\\text{Dice}} = 1 - \\frac{2 \\sum_p \\hat p \\, y}{\\sum_p \\hat p^2 + \\sum_p y^2}, \\quad ' +
      '\\mathcal L_{\\text{IoU}} = 1 - \\frac{\\sum_p \\hat p \\, y}{\\sum_p \\hat p + y - \\hat p \\, y}'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

function init() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }

  const grid = document.getElementById('sample-grid');
  if (grid) {
    grid.innerHTML = SAMPLES.map((s) => `<button class="sample-thumb" data-sample="${s.key}">${s.label}</button>`).join('');
    grid.querySelectorAll('[data-sample]').forEach((b) =>
      b.addEventListener('click', () => pickSample(b.dataset.sample)));
  }
  wireUpload();
  wireTabs();
  wireHover();
  wirePainting();
  wireRG();
  loadModel();
  pickSample(SAMPLES[0].key);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
