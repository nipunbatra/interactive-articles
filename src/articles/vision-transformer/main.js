// ============================================================
// Vision Transformers, Step by Step
// Real patch extraction, real dot-product attention with softmax,
// computed live on a 64×64 image the user picks or draws.
// ============================================================

const IMG_SIZE = 64;                // 64×64 internal resolution
const DISPLAY_SIZE = 320;           // canvas CSS size
const PATCH_SIZES = [4, 8, 16, 32]; // slider maps to these
const PATCH_SIZE_LABELS = [4, 8, 16, 32];
const DEFAULT_PATCH_SIZE_INDEX = 2; // 16

// ---------- Image generators ----------
// Each generator fills a Uint8ClampedArray of length 4*IMG_SIZE*IMG_SIZE
// (RGBA) that we treat as our "image".
function makeBlankImage() {
  const data = new Uint8ClampedArray(IMG_SIZE * IMG_SIZE * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 30;
    data[i + 1] = 30;
    data[i + 2] = 38;
    data[i + 3] = 255;
  }
  return data;
}

function setPixel(data, x, y, r, g, b) {
  const idx = (y * IMG_SIZE + x) * 4;
  data[idx] = r;
  data[idx + 1] = g;
  data[idx + 2] = b;
  data[idx + 3] = 255;
}

function drawDisk(data, cx, cy, radius, r, g, b) {
  const r2 = radius * radius;
  for (let y = Math.max(0, cy - radius); y < Math.min(IMG_SIZE, cy + radius + 1); y++) {
    for (let x = Math.max(0, cx - radius); x < Math.min(IMG_SIZE, cx + radius + 1); x++) {
      const dx = x - cx, dy = y - cy;
      if (dx * dx + dy * dy <= r2) setPixel(data, x, y, r, g, b);
    }
  }
}

function drawLine(data, x0, y0, x1, y1, r, g, b, thick = 2) {
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)) * 2 + 1;
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    const px = Math.round(x0 + (x1 - x0) * t);
    const py = Math.round(y0 + (y1 - y0) * t);
    drawDisk(data, px, py, thick, r, g, b);
  }
}

// Digit "7"
function makeDigit7() {
  const d = makeBlankImage();
  for (let i = 0; i < d.length; i += 4) { d[i] = 30; d[i+1] = 30; d[i+2] = 38; d[i+3] = 255; }
  // horizontal bar
  drawLine(d, 14, 14, 50, 14, 240, 230, 210, 3);
  // slanted stroke
  drawLine(d, 50, 14, 24, 52, 240, 230, 210, 3);
  return d;
}

// Cat-like silhouette (body + triangular ears + eyes)
function makeCat() {
  const d = makeBlankImage();
  for (let i = 0; i < d.length; i += 4) { d[i] = 50; d[i+1] = 65; d[i+2] = 100; d[i+3] = 255; }
  // body (filled ellipse-ish)
  for (let y = 20; y < 60; y++) {
    for (let x = 10; x < 54; x++) {
      const rx = (x - 32) / 24;
      const ry = (y - 40) / 22;
      if (rx * rx + ry * ry <= 1) setPixel(d, x, y, 230, 200, 130);
    }
  }
  // ears (triangles)
  for (let y = 6; y < 22; y++) {
    const half = Math.max(0, 16 - y);
    for (let x = 14 - half; x <= 14 + half; x++) setPixel(d, x, y, 230, 200, 130);
    for (let x = 50 - half; x <= 50 + half; x++) setPixel(d, x, y, 230, 200, 130);
  }
  // eyes
  drawDisk(d, 24, 32, 2, 40, 40, 60);
  drawDisk(d, 40, 32, 2, 40, 40, 60);
  // nose
  drawDisk(d, 32, 40, 2, 200, 90, 90);
  return d;
}

// Concentric circles
function makeCircles() {
  const d = makeBlankImage();
  for (let i = 0; i < d.length; i += 4) { d[i] = 250; d[i+1] = 245; d[i+2] = 225; d[i+3] = 255; }
  const cx = 32, cy = 32;
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const r = Math.hypot(x - cx, y - cy);
      if (r > 28 && r < 30) setPixel(d, x, y, 44, 111, 183);
      else if (r > 20 && r < 22) setPixel(d, x, y, 217, 98, 43);
      else if (r > 12 && r < 14) setPixel(d, x, y, 30, 119, 112);
      else if (r < 6) setPixel(d, x, y, 26, 24, 21);
    }
  }
  return d;
}

// Checkerboard
function makeChecker() {
  const d = makeBlankImage();
  const tile = 8;
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const on = (Math.floor(x / tile) + Math.floor(y / tile)) % 2 === 0;
      setPixel(d, x, y, on ? 240 : 30, on ? 230 : 30, on ? 210 : 38);
    }
  }
  return d;
}

// Color gradient
function makeGradient() {
  const d = makeBlankImage();
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const r = Math.round(255 * x / IMG_SIZE);
      const g = Math.round(255 * y / IMG_SIZE);
      const b = Math.round(255 * (1 - x / IMG_SIZE) * (1 - y / IMG_SIZE));
      setPixel(d, x, y, r, g, b);
    }
  }
  return d;
}

const IMAGE_FACTORY = {
  digit7: makeDigit7,
  cat: makeCat,
  circles: makeCircles,
  checker: makeChecker,
  gradient: makeGradient
};

// ---------- State ----------
let state = {
  imageKey: 'digit7',
  imageData: null,       // Uint8ClampedArray (RGBA), IMG_SIZE×IMG_SIZE
  patchSize: 16,         // current patch edge
  selectedPatch: null,   // {row, col} or null
  queryPatch: null,      // for attention (independent from selected)
  withPosition: true,
  drawing: false,        // in "draw your own" mode
  drawActive: false      // user currently pressing/dragging to draw
};

function loadImage(key) {
  state.imageKey = key;
  if (key === 'custom') {
    state.drawing = true;
    // start with blank canvas if not already drawing
    if (!state.imageData || state.imageData.length !== IMG_SIZE * IMG_SIZE * 4) {
      state.imageData = makeBlankImage();
    }
  } else {
    state.drawing = false;
    state.imageData = IMAGE_FACTORY[key]();
  }
  state.selectedPatch = null;
  state.queryPatch = null;
  updateAllVisuals();
}

// ---------- Patch helpers ----------
function patchesPerSide() {
  return IMG_SIZE / state.patchSize;
}
function totalPatches() {
  const k = patchesPerSide();
  return k * k;
}

function getPatchPixels(row, col) {
  const P = state.patchSize;
  const pixels = new Uint8ClampedArray(P * P * 4);
  for (let j = 0; j < P; j++) {
    for (let i = 0; i < P; i++) {
      const sx = col * P + i;
      const sy = row * P + j;
      const src = (sy * IMG_SIZE + sx) * 4;
      const dst = (j * P + i) * 4;
      pixels[dst] = state.imageData[src];
      pixels[dst + 1] = state.imageData[src + 1];
      pixels[dst + 2] = state.imageData[src + 2];
      pixels[dst + 3] = state.imageData[src + 3];
    }
  }
  return pixels;
}

// Compute a small feature vector for every patch — this is our stand-in for
// the learned patch embedding + position embedding. It's hand-crafted, but
// it's a real function of the pixels so attention weights behave realistically.
function computeFeatures(withPosition) {
  const k = patchesPerSide();
  const feats = [];
  for (let r = 0; r < k; r++) {
    for (let c = 0; c < k; c++) {
      feats.push(patchFeature(r, c, withPosition));
    }
  }
  return feats;
}

function patchFeature(row, col, withPosition) {
  const P = state.patchSize;
  // Mean RGB, gradient magnitude, brightness variance
  let sumR = 0, sumG = 0, sumB = 0;
  let gradX = 0, gradY = 0;
  const baseX = col * P;
  const baseY = row * P;
  let count = 0;
  for (let j = 0; j < P; j++) {
    for (let i = 0; i < P; i++) {
      const idx = ((baseY + j) * IMG_SIZE + (baseX + i)) * 4;
      const r = state.imageData[idx];
      const g = state.imageData[idx + 1];
      const b = state.imageData[idx + 2];
      sumR += r; sumG += g; sumB += b; count++;

      // Simple edge: difference with neighbor
      if (i + 1 < P) {
        const idx2 = ((baseY + j) * IMG_SIZE + (baseX + i + 1)) * 4;
        gradX += Math.abs(state.imageData[idx2] - r) +
                 Math.abs(state.imageData[idx2 + 1] - g) +
                 Math.abs(state.imageData[idx2 + 2] - b);
      }
      if (j + 1 < P) {
        const idx2 = ((baseY + j + 1) * IMG_SIZE + (baseX + i)) * 4;
        gradY += Math.abs(state.imageData[idx2] - r) +
                 Math.abs(state.imageData[idx2 + 1] - g) +
                 Math.abs(state.imageData[idx2 + 2] - b);
      }
    }
  }
  const inv = 1 / count;
  const rMean = (sumR * inv) / 255;
  const gMean = (sumG * inv) / 255;
  const bMean = (sumB * inv) / 255;
  const gx = gradX / (count * 255);
  const gy = gradY / (count * 255);
  const k = patchesPerSide();
  const px = (col + 0.5) / k * 2 - 1;
  const py = (row + 0.5) / k * 2 - 1;
  if (withPosition) {
    return [rMean, gMean, bMean, gx, gy, px, py, px * py];
  }
  return [rMean, gMean, bMean, gx, gy, 0, 0, 0];
}

function softmax(scores) {
  const m = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function attentionFrom(queryIdx, withPosition) {
  const feats = computeFeatures(withPosition);
  const q = feats[queryIdx];
  // Temperature controls how peaked attention is; pick something that
  // reads well visually. The 1/sqrt(dk) scaling in real ViT is here as 3.
  const scale = 6;
  const scores = feats.map((f) => {
    let s = 0;
    for (let i = 0; i < f.length; i++) s += q[i] * f[i];
    return s * scale;
  });
  return softmax(scores);
}

// ---------- Canvas drawing ----------
function getCtx(canvas, w = DISPLAY_SIZE, h = DISPLAY_SIZE) {
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

function drawImageTo(ctx, w, h) {
  if (!state.imageData) return;
  // Draw via an offscreen ImageData, then scale up to the canvas.
  const off = document.createElement('canvas');
  off.width = IMG_SIZE;
  off.height = IMG_SIZE;
  const offCtx = off.getContext('2d');
  const imgData = new ImageData(state.imageData.slice(), IMG_SIZE, IMG_SIZE);
  offCtx.putImageData(imgData, 0, 0);
  ctx.drawImage(off, 0, 0, w, h);
}

function drawPatchGrid(ctx, w, h, { selected } = {}) {
  const k = patchesPerSide();
  const cell = w / k;
  ctx.strokeStyle = 'rgba(253, 252, 249, 0.55)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < k; i++) {
    const p = i * cell;
    ctx.moveTo(p, 0); ctx.lineTo(p, h);
    ctx.moveTo(0, p); ctx.lineTo(w, p);
  }
  ctx.stroke();

  if (selected) {
    ctx.strokeStyle = '#d9622b';
    ctx.lineWidth = 3;
    ctx.strokeRect(selected.col * cell + 1.5, selected.row * cell + 1.5,
                   cell - 3, cell - 3);
  }
}

function renderPatchCanvas() {
  const canvas = document.getElementById('patchCanvas');
  if (!canvas) return;
  const ctx = getCtx(canvas);
  drawImageTo(ctx, DISPLAY_SIZE, DISPLAY_SIZE);
  drawPatchGrid(ctx, DISPLAY_SIZE, DISPLAY_SIZE, { selected: state.selectedPatch });
}

function renderPatchDetailCanvas() {
  const canvas = document.getElementById('patchDetailCanvas');
  if (!canvas) return;
  const ctx = getCtx(canvas);
  drawImageTo(ctx, DISPLAY_SIZE, DISPLAY_SIZE);
  drawPatchGrid(ctx, DISPLAY_SIZE, DISPLAY_SIZE, { selected: state.selectedPatch });

  // If a patch is selected, shade the others slightly
  if (state.selectedPatch) {
    const k = patchesPerSide();
    const cell = DISPLAY_SIZE / k;
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(253, 252, 249, 0.55)';
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        if (r === state.selectedPatch.row && c === state.selectedPatch.col) continue;
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
    ctx.restore();
    drawPatchGrid(ctx, DISPLAY_SIZE, DISPLAY_SIZE, { selected: state.selectedPatch });
  }
}

function renderAttnCanvas() {
  const canvas = document.getElementById('attnCanvas');
  if (!canvas) return;
  const ctx = getCtx(canvas);
  drawImageTo(ctx, DISPLAY_SIZE, DISPLAY_SIZE);

  const k = patchesPerSide();
  const cell = DISPLAY_SIZE / k;

  if (state.queryPatch !== null) {
    const weights = attentionFrom(state.queryPatch, state.withPosition);
    const maxW = Math.max(...weights);
    // Overlay heatmap
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        const idx = r * k + c;
        const w = weights[idx];
        const t = Math.min(1, w / (maxW + 1e-9));
        // orange heat
        ctx.fillStyle = `rgba(217, 98, 43, ${0.15 + 0.65 * t})`;
        if (t < 0.08) ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
    // Mark query
    const qr = Math.floor(state.queryPatch / k);
    const qc = state.queryPatch % k;
    ctx.strokeStyle = '#2c6fb7';
    ctx.lineWidth = 3;
    ctx.strokeRect(qc * cell + 1.5, qr * cell + 1.5, cell - 3, cell - 3);
  }

  drawPatchGrid(ctx, DISPLAY_SIZE, DISPLAY_SIZE);
}

// ---------- Sidebars ----------
function updateStatStrip() {
  const k = patchesPerSide();
  const ppp = state.patchSize * state.patchSize;
  document.getElementById('stat-grid').textContent = `${k} × ${k}`;
  document.getElementById('stat-n-patches').textContent = k * k;
  document.getElementById('stat-ppp').textContent = ppp;
  document.getElementById('stat-flat').textContent = ppp * 3;
}

function updatePatchDetail() {
  const infoBox = document.getElementById('patchInfoBox');
  const vectorRow = document.getElementById('patchVectorRow');
  if (!state.selectedPatch) {
    infoBox.className = 'patch-detail patch-detail-empty';
    infoBox.textContent = 'No patch selected yet. Click one in Step 1 or above.';
    vectorRow.innerHTML = '<span class="patch-detail-empty" style="font-style:italic">—</span>';
    return;
  }
  const { row, col } = state.selectedPatch;
  const P = state.patchSize;
  const pixels = getPatchPixels(row, col);

  // Show 8×8 pixel preview (scaled) or the full P×P for small patches
  let lines = [];
  lines.push(`Patch (row ${row}, col ${col})`);
  lines.push(`Size: ${P} × ${P} × 3 channels = ${P * P * 3} values`);
  lines.push('');
  lines.push('Top-left 4 × 4 pixels (R, G, B):');
  const sampleN = Math.min(4, P);
  for (let j = 0; j < sampleN; j++) {
    let row = '';
    for (let i = 0; i < sampleN; i++) {
      const idx = (j * P + i) * 4;
      const r = pixels[idx].toString().padStart(3, ' ');
      const g = pixels[idx + 1].toString().padStart(3, ' ');
      const b = pixels[idx + 2].toString().padStart(3, ' ');
      row += `(${r},${g},${b}) `;
    }
    lines.push(row);
  }
  infoBox.className = 'patch-detail';
  infoBox.textContent = lines.join('\n');

  // Vector row: first 12 entries of the raster-flattened RGB vector
  const flatN = 12;
  let html = '';
  for (let i = 0; i < flatN && i < pixels.length; i++) {
    // Skip alpha: 0=R,1=G,2=B,3=A,4=R...
    // Take only RGB in raster order, so read 3 per pixel
  }
  // Build first 12 values as alternating R/G/B for the first 4 pixels
  const flat = [];
  for (let p = 0; p < 4 && p * P < P * P; p++) {
    flat.push(pixels[p * 4], pixels[p * 4 + 1], pixels[p * 4 + 2]);
  }
  while (flat.length < 12) flat.push(0);
  html = flat.slice(0, 12).map((v) => `<span class="vector-cell">${v}</span>`).join('');
  vectorRow.innerHTML = html + '<span class="vector-cell" style="background:#faf7ef;color:var(--muted)">…</span>';
}

function updateSequenceStrip() {
  const strip = document.getElementById('sequenceStrip');
  if (!strip) return;
  const k = patchesPerSide();
  const N = k * k;
  const tokens = ['[CLS]'];
  for (let i = 0; i < N; i++) tokens.push(`p${i + 1}`);

  // Cap display
  const MAX = 48;
  const show = tokens.slice(0, MAX);
  const overflow = tokens.length - show.length;

  let html = '';
  show.forEach((t, i) => {
    if (t === '[CLS]') {
      html += `<span class="seq-token cls">[CLS]</span>`;
    } else {
      const posPart = state.withPosition ? `<br>+ pos${i}` : '';
      const cls = state.withPosition ? 'seq-token pos' : 'seq-token';
      html += `<span class="${cls}">${t}${posPart}</span>`;
    }
  });
  if (overflow > 0) html += `<span class="seq-token" style="background:#faf7ef;color:var(--muted)">… +${overflow}</span>`;
  strip.innerHTML = html;
}

function updateTopAttn() {
  const el = document.getElementById('topAttnTable');
  const qStat = document.getElementById('stat-query');
  const topStat = document.getElementById('stat-top-attn');
  if (state.queryPatch === null) {
    el.className = 'patch-detail patch-detail-empty';
    el.textContent = 'Click a patch first.';
    qStat.textContent = '—';
    topStat.textContent = '—';
    return;
  }
  const k = patchesPerSide();
  const qr = Math.floor(state.queryPatch / k);
  const qc = state.queryPatch % k;
  const weights = attentionFrom(state.queryPatch, state.withPosition);
  const indexed = weights.map((w, i) => ({ i, w }));
  indexed.sort((a, b) => b.w - a.w);
  const top = indexed.slice(0, 5);

  qStat.textContent = `(${qr},${qc})`;
  const topPatch = top[0];
  const tr = Math.floor(topPatch.i / k);
  const tc = topPatch.i % k;
  topStat.textContent = topPatch.i === state.queryPatch ? 'self' : `(${tr},${tc})`;

  let lines = [];
  lines.push('Rank  Patch    Weight');
  top.forEach((x, rank) => {
    const r = Math.floor(x.i / k);
    const c = x.i % k;
    const label = x.i === state.queryPatch ? 'self' : `(${r},${c})`;
    lines.push(`  ${rank + 1}   ${label.padEnd(7, ' ')} ${(x.w * 100).toFixed(1)}%`);
  });
  el.className = 'patch-detail';
  el.textContent = lines.join('\n');
}

function updatePipelineTable() {
  const body = document.getElementById('pipelineTableBody');
  if (!body) return;
  const k = patchesPerSide();
  const N = k * k;
  const P = state.patchSize;
  const D = 384; // fixed embedding dim for illustration
  const rows = [
    ['1. Patchify', 'Split image into P×P non-overlapping patches.', `${N} patches of size ${P}×${P}`],
    ['2. Flatten', `Each patch: P×P×3 pixels → vector of length 3P².`, `vector length ${3 * P * P}`],
    ['3. Project', `Linear layer to D dims (D = ${D} shown here).`, `${N} × ${D} matrix of tokens`],
    ['4. Add positions', `Add learned position embedding of size D to each patch.`, `still ${N} × ${D}`],
    ['5. Prepend [CLS]', `Add one learnable CLS token vector at the front.`, `sequence length ${N + 1}`],
    ['6. Self-attention × L', `Run the sequence through L encoder blocks (12 in ViT-Base).`, `${N + 1} tokens, ${D}-dim each`],
    ['7. Read out CLS', `Take the final CLS vector and push through an MLP head.`, `D → num_classes logits`]
  ];
  body.innerHTML = rows.map((r) =>
    `<tr><td><strong>${r[0]}</strong></td><td>${r[1]}</td><td>${r[2]}</td></tr>`
  ).join('');
}

function updateScalingTable() {
  const body = document.getElementById('scalingTableBody');
  if (!body) return;
  // All relative to patch=16 for a 64×64 image
  const basePatches = Math.pow(IMG_SIZE / 16, 2);
  const baseOps = Math.pow(basePatches + 1, 2);
  const rows = PATCH_SIZES.map((P) => {
    const k = IMG_SIZE / P;
    const N = k * k;
    const tokens = N + 1;
    const ops = tokens * tokens;
    return {
      P, k, tokens, ops: ops / baseOps
    };
  });
  body.innerHTML = rows.map((r) => {
    const isActive = r.P === state.patchSize;
    return `
      <tr${isActive ? ' style="background:rgba(44,111,183,0.05)"' : ''}>
        <td><strong>${r.P} × ${r.P}${isActive ? ' (current)' : ''}</strong></td>
        <td>${r.k} × ${r.k}</td>
        <td>${r.tokens}</td>
        <td>${r.ops.toFixed(2)}×</td>
      </tr>
    `;
  }).join('');
}

function updateRevealCard() {
  const k = patchesPerSide();
  const N = k * k;
  document.getElementById('reveal-seqlen').textContent = N + 1;
  document.getElementById('reveal-breakdown').textContent =
    `1 CLS token + ${N} patch token${N === 1 ? '' : 's'}`;
}

// ---------- Input handling ----------
function patchFromCanvasXY(canvas, clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const mx = (clientX - rect.left) / rect.width * DISPLAY_SIZE;
  const my = (clientY - rect.top) / rect.height * DISPLAY_SIZE;
  const k = patchesPerSide();
  const cell = DISPLAY_SIZE / k;
  const col = Math.max(0, Math.min(k - 1, Math.floor(mx / cell)));
  const row = Math.max(0, Math.min(k - 1, Math.floor(my / cell)));
  return { row, col };
}

function imgXYFromCanvas(canvas, clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const mx = (clientX - rect.left) / rect.width * IMG_SIZE;
  const my = (clientY - rect.top) / rect.height * IMG_SIZE;
  return {
    x: Math.max(0, Math.min(IMG_SIZE - 1, Math.floor(mx))),
    y: Math.max(0, Math.min(IMG_SIZE - 1, Math.floor(my)))
  };
}

function paintAt(x, y, radius = 3) {
  const color = [240, 230, 210];
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const px = x + dx;
      const py = y + dy;
      if (px < 0 || px >= IMG_SIZE || py < 0 || py >= IMG_SIZE) continue;
      if (dx * dx + dy * dy > radius * radius) continue;
      setPixel(state.imageData, px, py, color[0], color[1], color[2]);
    }
  }
}

function wireCanvasInput() {
  const cvs = [
    { el: document.getElementById('patchCanvas'), selectsPatch: true, attnOnClick: true },
    { el: document.getElementById('patchDetailCanvas'), selectsPatch: true, attnOnClick: false },
    { el: document.getElementById('attnCanvas'), selectsPatch: false, attnOnClick: true }
  ];

  cvs.forEach((c) => {
    if (!c.el) return;
    const onDown = (e) => {
      const clientX = (e.touches ? e.touches[0].clientX : e.clientX);
      const clientY = (e.touches ? e.touches[0].clientY : e.clientY);
      if (state.drawing) {
        state.drawActive = true;
        const { x, y } = imgXYFromCanvas(c.el, clientX, clientY);
        paintAt(x, y);
        updateAllVisuals();
        if (e.touches) e.preventDefault();
        return;
      }
      const { row, col } = patchFromCanvasXY(c.el, clientX, clientY);
      const k = patchesPerSide();
      if (c.selectsPatch) state.selectedPatch = { row, col };
      if (c.attnOnClick) state.queryPatch = row * k + col;
      updateAllVisuals();
      if (e.touches) e.preventDefault();
    };
    const onMove = (e) => {
      if (!state.drawing || !state.drawActive) return;
      const clientX = (e.touches ? e.touches[0].clientX : e.clientX);
      const clientY = (e.touches ? e.touches[0].clientY : e.clientY);
      const { x, y } = imgXYFromCanvas(c.el, clientX, clientY);
      paintAt(x, y);
      updateAllVisuals();
      if (e.touches) e.preventDefault();
    };
    const onUp = () => { state.drawActive = false; };

    c.el.addEventListener('mousedown', onDown);
    c.el.addEventListener('touchstart', onDown, { passive: false });
    c.el.addEventListener('mousemove', onMove);
    c.el.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('mouseup', onUp);
    window.addEventListener('touchend', onUp);
  });
}

// ---------- Top-level update ----------
function updateAllVisuals() {
  updateStatStrip();
  renderPatchCanvas();
  renderPatchDetailCanvas();
  renderAttnCanvas();
  updatePatchDetail();
  updateSequenceStrip();
  updateTopAttn();
  updatePipelineTable();
  updateScalingTable();
  updateRevealCard();
}

// ---------- KaTeX ----------
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-patches':
      'N = \\left(\\tfrac{H}{P}\\right)^2 \\qquad (\\text{for } H \\times H \\text{ image, } P \\times P \\text{ patches})',
    'math-flatten':
      '\\mathbf{x}_i = \\mathrm{flatten}(\\text{patch}_i) \\in \\mathbb{R}^{3P^2}, \\quad \\mathbf{z}_i = E\\,\\mathbf{x}_i \\in \\mathbb{R}^D',
    'math-position':
      '\\mathbf{z}_i \\leftarrow \\mathbf{z}_i + \\mathbf{p}_i \\qquad \\text{for } i = 1, \\dots, N',
    'math-cls':
      '\\mathbf{Z}_0 = \\bigl[\\,\\mathbf{z}_{\\text{cls}};\\, \\mathbf{z}_1;\\, \\mathbf{z}_2;\\, \\dots;\\, \\mathbf{z}_N\\,\\bigr] \\in \\mathbb{R}^{(N+1) \\times D}',
    'math-attention':
      '\\mathrm{Attn}(Q, K, V) = \\mathrm{softmax}\\!\\left(\\frac{Q K^\\top}{\\sqrt{d_k}}\\right) V'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try {
      katex.render(blocks[id], el, { displayMode: true, throwOnError: false });
    } catch (_) { /* no-op */ }
  });
}

// ---------- Boot ----------
function init() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }

  // Image buttons
  const btnClear = document.getElementById('btn-clear-draw');
  document.querySelectorAll('#image-buttons [data-image]').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#image-buttons [data-image]').forEach(
        (bb) => bb.classList.remove('is-active')
      );
      b.classList.add('is-active');
      loadImage(b.dataset.image);
      if (b.dataset.image === 'custom') {
        btnClear.style.display = 'inline-flex';
      } else {
        btnClear.style.display = 'none';
      }
    });
  });
  if (btnClear) {
    btnClear.addEventListener('click', () => {
      state.imageData = makeBlankImage();
      updateAllVisuals();
    });
  }

  // Patch size slider
  const psSlider = document.getElementById('patch-size');
  const psVal = document.getElementById('val-patch-size');
  psSlider.value = DEFAULT_PATCH_SIZE_INDEX;
  state.patchSize = PATCH_SIZE_LABELS[DEFAULT_PATCH_SIZE_INDEX];
  psVal.textContent = state.patchSize;
  psSlider.addEventListener('input', () => {
    const idx = parseInt(psSlider.value, 10);
    state.patchSize = PATCH_SIZE_LABELS[idx];
    psVal.textContent = state.patchSize;
    // Reset selection, re-evaluate
    state.selectedPatch = null;
    state.queryPatch = null;
    updateAllVisuals();
  });

  // Position embedding toggles
  const posOn = document.getElementById('pos-on');
  const posOff = document.getElementById('pos-off');
  const updatePosBtns = () => {
    posOn.classList.toggle('is-active', state.withPosition);
    posOff.classList.toggle('is-active', !state.withPosition);
  };
  posOn.addEventListener('click', () => {
    state.withPosition = true;
    updatePosBtns();
    updateAllVisuals();
  });
  posOff.addEventListener('click', () => {
    state.withPosition = false;
    updatePosBtns();
    updateAllVisuals();
  });

  wireCanvasInput();
  loadImage('digit7');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
