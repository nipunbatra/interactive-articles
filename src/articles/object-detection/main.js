// ============================================================
// Object Detection with a Real Detector
// Runs TF.js + COCO-SSD live on the user's photo.
// All metrics (IoU, NMS, PR curve, AP) operate on the detector's
// real output — no synthetic boxes anywhere.
// ============================================================

// ---------- Sample photos ----------
// Use picsum.photos with specific IDs — stable URLs, always CORS-enabled.
// Upload is the primary path; samples just save a click for first-time viewers.
const SAMPLES = [
  {
    key: 'dog',
    label: 'Puppy',
    urls: ['https://picsum.photos/id/237/640/400']
  },
  {
    key: 'corgi',
    label: 'Corgi',
    urls: ['https://picsum.photos/id/1025/640/400']
  },
  {
    key: 'person',
    label: 'Person in scene',
    urls: ['https://picsum.photos/id/64/640/400']
  },
  {
    key: 'street',
    label: 'Street scene',
    urls: ['https://picsum.photos/id/145/640/400']
  },
  {
    key: 'room',
    label: 'Living room',
    urls: ['https://picsum.photos/id/116/640/400']
  },
  {
    key: 'kitchen',
    label: 'Still life',
    urls: ['https://picsum.photos/id/292/640/400']
  }
];

// ---------- State ----------
const state = {
  model: null,              // loaded cocoSsd model
  modelLoading: true,
  image: null,              // HTMLImageElement / HTMLCanvasElement of current photo
  imageBitmap: null,        // for re-drawing
  imageW: 0,
  imageH: 0,
  displayW: 720,
  displayH: 480,
  rawDetections: [],        // full raw detections from model.detect()
  confThreshold: 0.3,
  nmsThreshold: 0.5,
  nmsConfFloor: 0.2,
  gtBoxes: [],              // user-drawn GT boxes: [{x, y, w, h}]
  selectedDetIdx: -1,
  selectedGtIdx: -1,
  drawingGt: false,         // toggle: "draw GT" mode
  liveDraw: null,           // in-progress drag: {x0, y0, x1, y1}
  lastIoU: null,
  inferenceMs: 0
};

// ---------- TF.js model loading ----------
async function loadModel() {
  try {
    // cocoSsd global attached by the CDN script
    state.model = await cocoSsd.load({ base: 'mobilenet_v2' });
    state.modelLoading = false;
    setModelStatus('is-ready', 'Detector ready');
    if (state.image) await runDetection();
  } catch (err) {
    console.error('Model load failed:', err);
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
  state.imageBitmap = img;
  state.imageW = img.naturalWidth || img.width;
  state.imageH = img.naturalHeight || img.height;
  // Display dimensions cap at 720 wide
  const maxW = 720;
  if (state.imageW > maxW) {
    state.displayW = maxW;
    state.displayH = Math.round(state.imageH * maxW / state.imageW);
  } else {
    state.displayW = state.imageW;
    state.displayH = state.imageH;
  }
  // Reset selections / GT
  state.gtBoxes = [];
  state.selectedDetIdx = -1;
  state.rawDetections = [];
  renderAll();
  if (state.model) runDetection();
  else setModelStatus('is-loading', 'Waiting for detector&hellip;');
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
  document.querySelectorAll('#sample-grid .sample-thumb').forEach((b) => {
    b.classList.toggle('is-active', b.dataset.sample === key);
  });
  const cap = document.getElementById('prelude-caption');
  if (cap) cap.textContent = `Loading “${s.label}”\u2026`;
  try {
    const img = await loadImageWithFallback(s.urls);
    loadImageFromElement(img);
    if (cap) cap.textContent = `${s.label} — detector predictions will be drawn over it.`;
  } catch (err) {
    console.error('Sample load failed:', err);
    if (cap) cap.textContent = `Could not load “${s.label}”. Drop a photo into the upload area.`;
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

// ---------- Detection ----------
async function runDetection() {
  if (!state.model || !state.imageBitmap) return;
  const t0 = performance.now();
  // Ask for many candidates and no internal filtering so we can show
  // confidence sweep and user-driven NMS honestly.
  const dets = await state.model.detect(state.imageBitmap, 40, 0.05);
  state.rawDetections = dets.map((d, i) => ({
    idx: i,
    cls: d.class,
    score: d.score,
    x: d.bbox[0],
    y: d.bbox[1],
    w: d.bbox[2],
    h: d.bbox[3]
  }));
  state.inferenceMs = performance.now() - t0;
  renderAll();
}

// ---------- Geometry ----------
function iouOf(a, b) {
  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(a.x + a.w, b.x + b.w);
  const iy2 = Math.min(a.y + a.h, b.y + b.h);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  if (union <= 0) return 0;
  return inter / union;
}

function bestDetFor(gt, dets) {
  let bestI = -1, bestIoU = 0;
  for (let i = 0; i < dets.length; i++) {
    const u = iouOf(gt, dets[i]);
    if (u > bestIoU) { bestI = i; bestIoU = u; }
  }
  return { idx: bestI, iou: bestIoU };
}

// ---------- Canvas drawing primitives ----------
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = state.displayW * dpr;
  canvas.height = state.displayH * dpr;
  canvas.style.width = state.displayW + 'px';
  canvas.style.height = state.displayH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function drawImage(ctx) {
  if (!state.imageBitmap) {
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, state.displayW, state.displayH);
    return;
  }
  ctx.drawImage(state.imageBitmap, 0, 0, state.displayW, state.displayH);
}

function imgToDisp(box) {
  const sx = state.displayW / state.imageW;
  const sy = state.displayH / state.imageH;
  return { x: box.x * sx, y: box.y * sy, w: box.w * sx, h: box.h * sy };
}
function dispToImg(box) {
  const sx = state.imageW / state.displayW;
  const sy = state.imageH / state.displayH;
  return { x: box.x * sx, y: box.y * sy, w: box.w * sx, h: box.h * sy };
}

function drawBox(ctx, box, color, label = null, opacity = 1, dash = []) {
  const b = imgToDisp(box);
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.setLineDash(dash);
  ctx.strokeRect(b.x, b.y, b.w, b.h);
  ctx.setLineDash([]);
  if (label) {
    ctx.font = '600 12px system-ui, sans-serif';
    const metrics = ctx.measureText(label);
    const labelW = metrics.width + 10;
    const labelH = 18;
    const lx = b.x;
    const ly = Math.max(labelH, b.y) - labelH;
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, labelW, labelH);
    ctx.fillStyle = 'white';
    ctx.fillText(label, lx + 5, ly + 13);
  }
  ctx.restore();
}

function drawIntersection(ctx, a, b) {
  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(a.x + a.w, b.x + b.w);
  const iy2 = Math.min(a.y + a.h, b.y + b.h);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  if (iw <= 0 || ih <= 0) return;
  const d = imgToDisp({ x: ix1, y: iy1, w: iw, h: ih });
  ctx.fillStyle = 'rgba(138, 94, 182, 0.4)';
  ctx.fillRect(d.x, d.y, d.w, d.h);
}

function drawHandles(ctx, box, color) {
  const b = imgToDisp(box);
  const corners = [
    { x: b.x, y: b.y }, { x: b.x + b.w, y: b.y },
    { x: b.x, y: b.y + b.h }, { x: b.x + b.w, y: b.y + b.h }
  ];
  corners.forEach((c) => {
    ctx.fillStyle = color;
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

// ---------- Step 0 prelude canvas ----------
function renderPrelude() {
  const canvas = document.getElementById('preludeCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  drawImage(ctx);
  // Draw the raw detections with COCO-SSD's "default" threshold of 0.5
  const kept = state.rawDetections.filter((d) => d.score >= 0.5);
  kept.forEach((d) => {
    drawBox(ctx, d, '#d9622b',
      `${d.cls} ${(d.score * 100).toFixed(0)}%`, Math.min(1, 0.4 + d.score));
  });
  document.getElementById('raw-count').textContent =
    state.rawDetections.length > 0 ? state.rawDetections.length : '—';
  document.getElementById('inference-ms').textContent =
    state.inferenceMs > 0 ? `${state.inferenceMs.toFixed(0)} ms` : '—';
}

// ---------- Step 1 raw list canvas + table ----------
function renderRawList() {
  const canvas = document.getElementById('rawListCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  drawImage(ctx);
  state.rawDetections.forEach((d, i) => {
    const selected = i === state.selectedDetIdx;
    drawBox(ctx, d, selected ? '#2c6fb7' : '#d9622b',
      `${d.cls} ${(d.score * 100).toFixed(0)}%`, Math.min(1, 0.3 + d.score * 0.7));
  });

  const tbody = document.querySelector('#rawTable tbody');
  if (!tbody) return;
  if (state.rawDetections.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--muted);">No detections yet — load a photo.</td></tr>';
    return;
  }
  tbody.innerHTML = state.rawDetections
    .slice()
    .sort((a, b) => b.score - a.score)
    .map((d) => {
      const cls = d.idx === state.selectedDetIdx ? ' class="is-selected"' : '';
      return `<tr${cls} data-idx="${d.idx}">
        <td>${d.idx + 1}</td>
        <td>${d.cls}</td>
        <td>${(d.score * 100).toFixed(1)}%</td>
        <td>${Math.round(d.x)},${Math.round(d.y)},${Math.round(d.w)},${Math.round(d.h)}</td>
      </tr>`;
    }).join('');
  tbody.querySelectorAll('tr[data-idx]').forEach((row) => {
    row.addEventListener('click', () => {
      state.selectedDetIdx = parseInt(row.dataset.idx, 10);
      renderRawList();
    });
  });
}

// ---------- Step 2 IoU canvas ----------
function renderIoU() {
  const canvas = document.getElementById('iouCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  drawImage(ctx);

  // Track IoUs for the stats
  const perGt = state.gtBoxes.map((gt) => bestDetFor(gt, state.rawDetections));
  state.lastIoU = perGt.length ? perGt[perGt.length - 1].iou : null;

  // Draw each GT with its best-match det + intersection
  state.gtBoxes.forEach((gt, i) => {
    drawBox(ctx, gt, '#1e7770', `GT ${i + 1}`, 1);
    drawHandles(ctx, gt, '#1e7770');
    const { idx, iou } = perGt[i];
    if (idx >= 0 && iou > 0) {
      const d = state.rawDetections[idx];
      drawBox(ctx, d, '#d9622b',
        `${d.cls} ${(d.score * 100).toFixed(0)}%  IoU ${iou.toFixed(2)}`, 0.95);
      drawIntersection(ctx, gt, d);
    }
  });

  // In-progress drawing
  if (state.liveDraw) {
    const { x0, y0, x1, y1 } = state.liveDraw;
    const box = {
      x: Math.min(x0, x1), y: Math.min(y0, y1),
      w: Math.abs(x1 - x0), h: Math.abs(y1 - y0)
    };
    drawBox(ctx, box, '#1e7770', 'GT', 0.7, [6, 4]);
  }

  document.getElementById('gt-count').textContent = state.gtBoxes.length;
  if (perGt.length) {
    const mean = perGt.reduce((acc, p) => acc + p.iou, 0) / perGt.length;
    document.getElementById('mean-iou').textContent = mean.toFixed(3);
  } else {
    document.getElementById('mean-iou').textContent = '—';
  }
  document.getElementById('last-iou').textContent =
    state.lastIoU != null ? state.lastIoU.toFixed(3) : '—';
}

// ---------- Step 3 threshold canvas + metrics ----------
function renderThresh() {
  const canvas = document.getElementById('threshCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  drawImage(ctx);

  const total = state.rawDetections.length;
  let showing = 0;
  state.rawDetections.forEach((d) => {
    const above = d.score >= state.confThreshold;
    if (above) showing++;
    drawBox(ctx, d, above ? '#d9622b' : '#9a917f',
      above ? `${d.cls} ${(d.score * 100).toFixed(0)}%` : null,
      above ? Math.min(1, 0.5 + d.score * 0.5) : 0.3);
  });
  state.gtBoxes.forEach((gt, i) => drawBox(ctx, gt, '#1e7770', `GT ${i + 1}`, 1, [6, 4]));

  document.getElementById('showing-count').textContent = showing;
  document.getElementById('total-count').textContent = total;

  const { tp, fp, fn } = countAtThreshold(state.rawDetections, state.gtBoxes, state.confThreshold, 0.5);
  document.getElementById('tp').textContent = tp;
  document.getElementById('fp').textContent = fp;
  document.getElementById('fn').textContent = fn;
  const P = tp + fp > 0 ? tp / (tp + fp) : null;
  const R = tp + fn > 0 ? tp / (tp + fn) : null;
  const F1 = (P != null && R != null && P + R > 0) ? 2 * P * R / (P + R) : null;
  document.getElementById('precision').textContent = P == null ? '—' : P.toFixed(3);
  document.getElementById('recall').textContent = R == null ? '—' : R.toFixed(3);
  document.getElementById('f1').textContent = F1 == null ? '—' : F1.toFixed(3);
}

function countAtThreshold(dets, gts, conf, iouT) {
  const kept = dets.filter((d) => d.score >= conf).slice().sort((a, b) => b.score - a.score);
  const gtUsed = new Array(gts.length).fill(false);
  let tp = 0, fp = 0;
  kept.forEach((d) => {
    let bestI = -1, bestIoU = 0;
    for (let i = 0; i < gts.length; i++) {
      if (gtUsed[i]) continue;
      const u = iouOf(d, gts[i]);
      if (u > bestIoU) { bestI = i; bestIoU = u; }
    }
    if (bestI >= 0 && bestIoU >= iouT) {
      tp++;
      gtUsed[bestI] = true;
    } else {
      fp++;
    }
  });
  const fn = gtUsed.filter((u) => !u).length;
  return { tp, fp, fn };
}

// ---------- Step 4 NMS ----------
function runNMS(dets, iouT, confFloor) {
  const filtered = dets.filter((d) => d.score >= confFloor)
    .slice().sort((a, b) => b.score - a.score);
  const alive = filtered.map((d) => ({ ...d, alive: true }));
  const kept = [];
  const suppressed = [];
  const trace = [];
  let step = 1;
  while (true) {
    const head = alive.find((a) => a.alive);
    if (!head) break;
    kept.push(head);
    trace.push({ step, action: 'keep', cls: head.cls, score: head.score, ref: null });
    head.alive = false;
    for (const cand of alive) {
      if (!cand.alive) continue;
      const u = iouOf(head, cand);
      if (u >= iouT) {
        cand.alive = false;
        suppressed.push(cand);
        trace.push({ step, action: `suppress (IoU ${u.toFixed(2)})`, cls: cand.cls, score: cand.score, ref: head });
      }
    }
    step++;
  }
  return { kept, suppressed, trace, pre: filtered.length };
}

function renderNMS() {
  const canvas = document.getElementById('nmsCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  drawImage(ctx);
  const { kept, suppressed, trace, pre } = runNMS(state.rawDetections, state.nmsThreshold, state.nmsConfFloor);
  suppressed.forEach((d) => drawBox(ctx, d, '#9a917f', null, 0.3));
  kept.forEach((d) => drawBox(ctx, d, '#d9622b',
    `${d.cls} ${(d.score * 100).toFixed(0)}%`, 0.95));

  document.getElementById('nms-in').textContent = pre;
  document.getElementById('nms-out').textContent = kept.length;
  document.getElementById('nms-sup').textContent = suppressed.length;

  const tbody = document.querySelector('#nms-trace-table tbody');
  if (tbody) {
    if (!trace.length) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--muted);">No detections above the floor.</td></tr>';
    } else {
      tbody.innerHTML = trace.map((t) => {
        const refStr = t.ref ? `${t.ref.cls} ${(t.ref.score * 100).toFixed(0)}%` : '—';
        return `<tr><td>${t.step}</td><td>${t.action}</td><td>${t.cls}</td><td>${(t.score * 100).toFixed(1)}%</td><td>${refStr}</td></tr>`;
      }).join('');
    }
  }
}

// ---------- Step 6 PR curve & AP ----------
function computePRCurve() {
  if (!state.gtBoxes.length || !state.rawDetections.length) return { pts: [], ap: 0 };
  const sorted = state.rawDetections.slice().sort((a, b) => b.score - a.score);
  const gtUsed = new Array(state.gtBoxes.length).fill(false);
  const pts = [];
  let tp = 0, fp = 0;
  sorted.forEach((d) => {
    let bestI = -1, bestIoU = 0;
    for (let i = 0; i < state.gtBoxes.length; i++) {
      if (gtUsed[i]) continue;
      const u = iouOf(d, state.gtBoxes[i]);
      if (u > bestIoU) { bestI = i; bestIoU = u; }
    }
    if (bestI >= 0 && bestIoU >= 0.5) { tp++; gtUsed[bestI] = true; }
    else fp++;
    const P = tp / (tp + fp);
    const R = tp / state.gtBoxes.length;
    pts.push({ P, R, s: d.score });
  });
  // 11-point interpolated AP
  let ap = 0;
  for (let r = 0; r <= 1.0001; r += 0.1) {
    let maxP = 0;
    for (const p of pts) if (p.R >= r - 1e-9 && p.P > maxP) maxP = p.P;
    ap += maxP / 11;
  }
  return { pts, ap };
}

function renderPR() {
  const canvas = document.getElementById('prCanvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = 640 * dpr; canvas.height = 320 * dpr;
  canvas.style.width = '640px'; canvas.style.height = '320px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, 640, 320);

  const margin = { top: 30, right: 30, bottom: 40, left: 60 };
  const pw = 640 - margin.left - margin.right;
  const ph = 320 - margin.top - margin.bottom;

  ctx.strokeStyle = '#f0ebe1'; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i <= 5; i++) {
    const x = margin.left + (pw / 5) * i;
    ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + ph);
    const y = margin.top + (ph / 5) * i;
    ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + pw, y);
  }
  ctx.stroke();
  ctx.strokeStyle = '#c4beb1'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + ph);
  ctx.lineTo(margin.left + pw, margin.top + ph);
  ctx.stroke();

  ctx.fillStyle = '#9a917f'; ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Recall', margin.left + pw / 2, margin.top + ph + 24);
  ctx.save(); ctx.translate(margin.left - 42, margin.top + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('Precision', 0, 0); ctx.restore();
  ctx.textAlign = 'right';
  ctx.fillText('1', margin.left - 6, margin.top + 4);
  ctx.fillText('0', margin.left - 6, margin.top + ph + 4);
  ctx.textAlign = 'center';
  ctx.fillText('0', margin.left, margin.top + ph + 14);
  ctx.fillText('1', margin.left + pw, margin.top + ph + 14);

  const { pts, ap } = computePRCurve();
  if (pts.length) {
    ctx.fillStyle = 'rgba(44, 111, 183, 0.15)';
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = margin.left + p.R * pw;
      const y = margin.top + (1 - p.P) * ph;
      if (i === 0) ctx.moveTo(x, margin.top + ph);
      ctx.lineTo(x, y);
    });
    ctx.lineTo(margin.left + pts[pts.length - 1].R * pw, margin.top + ph);
    ctx.closePath(); ctx.fill();

    ctx.strokeStyle = '#2c6fb7'; ctx.lineWidth = 2.5;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = margin.left + p.R * pw;
      const y = margin.top + (1 - p.P) * ph;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = '#d9622b';
    pts.forEach((p) => {
      const x = margin.left + p.R * pw;
      const y = margin.top + (1 - p.P) * ph;
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    });
    document.getElementById('apValue').textContent = ap.toFixed(3);
  } else {
    ctx.fillStyle = '#9a917f'; ctx.font = '13px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Draw ground-truth boxes in Step 2 to populate this curve.',
                 margin.left + pw / 2, margin.top + ph / 2);
    document.getElementById('apValue').textContent = '—';
  }
}

// ---------- Input: IoU canvas drawing / dragging ----------
function wireIoUInput() {
  const canvas = document.getElementById('iouCanvas');
  if (!canvas) return;
  let drawing = false;
  let start = null;
  let dragMode = null; // { kind: 'move'|'nw'|'ne'|'sw'|'se', idx }
  let dragStart = null;

  function getXYImg(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    const disp = { x: cx / rect.width * state.displayW, y: cy / rect.height * state.displayH };
    // Convert display to image coords
    const sx = state.imageW / state.displayW;
    const sy = state.imageH / state.displayH;
    return { x: disp.x * sx, y: disp.y * sy, dispX: disp.x, dispY: disp.y };
  }

  function hitGt(x, y) {
    // Check corners then insides (in display coords for tolerance)
    const tol = 10 * state.imageW / state.displayW;
    for (let i = state.gtBoxes.length - 1; i >= 0; i--) {
      const g = state.gtBoxes[i];
      const corners = [
        { kind: 'nw', x: g.x, y: g.y },
        { kind: 'ne', x: g.x + g.w, y: g.y },
        { kind: 'sw', x: g.x, y: g.y + g.h },
        { kind: 'se', x: g.x + g.w, y: g.y + g.h }
      ];
      for (const c of corners) {
        if (Math.abs(x - c.x) < tol && Math.abs(y - c.y) < tol) return { kind: c.kind, idx: i };
      }
      if (x > g.x && x < g.x + g.w && y > g.y && y < g.y + g.h) {
        return { kind: 'move', idx: i };
      }
    }
    return null;
  }

  const onDown = (e) => {
    if (!state.imageBitmap) return;
    const p = getXYImg(e);
    if (state.drawingGt) {
      drawing = true;
      start = p;
      state.liveDraw = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
      if (e.touches) e.preventDefault();
      return;
    }
    const hit = hitGt(p.x, p.y);
    if (hit) {
      dragMode = hit;
      dragStart = { p, box: { ...state.gtBoxes[hit.idx] } };
      if (e.touches) e.preventDefault();
    }
  };
  const onMove = (e) => {
    if (!state.imageBitmap) return;
    const p = getXYImg(e);
    if (drawing) {
      state.liveDraw = { x0: start.x, y0: start.y, x1: p.x, y1: p.y };
      renderIoU();
      if (e.touches) e.preventDefault();
      return;
    }
    if (dragMode) {
      const g = state.gtBoxes[dragMode.idx];
      if (dragMode.kind === 'move') {
        g.x = dragStart.box.x + (p.x - dragStart.p.x);
        g.y = dragStart.box.y + (p.y - dragStart.p.y);
      } else {
        let x1 = dragStart.box.x, y1 = dragStart.box.y;
        let x2 = x1 + dragStart.box.w, y2 = y1 + dragStart.box.h;
        if (dragMode.kind === 'nw') { x1 = p.x; y1 = p.y; }
        if (dragMode.kind === 'ne') { x2 = p.x; y1 = p.y; }
        if (dragMode.kind === 'sw') { x1 = p.x; y2 = p.y; }
        if (dragMode.kind === 'se') { x2 = p.x; y2 = p.y; }
        g.x = Math.min(x1, x2);
        g.y = Math.min(y1, y2);
        g.w = Math.max(10, Math.abs(x2 - x1));
        g.h = Math.max(10, Math.abs(y2 - y1));
      }
      renderAll();
      if (e.touches) e.preventDefault();
    }
  };
  const onUp = () => {
    if (drawing && state.liveDraw) {
      const { x0, y0, x1, y1 } = state.liveDraw;
      const w = Math.abs(x1 - x0), h = Math.abs(y1 - y0);
      if (w > 10 && h > 10) {
        state.gtBoxes.push({ x: Math.min(x0, x1), y: Math.min(y0, y1), w, h });
      }
      state.liveDraw = null;
      state.drawingGt = false;
      const btn = document.getElementById('btn-draw-gt');
      if (btn) btn.textContent = 'Draw ground-truth box';
      renderAll();
    }
    drawing = false;
    dragMode = null;
  };

  canvas.addEventListener('mousedown', onDown);
  canvas.addEventListener('touchstart', onDown, { passive: false });
  window.addEventListener('mousemove', onMove);
  window.addEventListener('touchmove', onMove, { passive: false });
  window.addEventListener('mouseup', onUp);
  window.addEventListener('touchend', onUp);
}

// ---------- Render orchestrator ----------
function renderAll() {
  renderPrelude();
  renderRawList();
  renderIoU();
  renderThresh();
  renderNMS();
  renderPR();
}

// ---------- Math ----------
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-row': '\\text{detection} = (x_1, y_1, x_2, y_2,\\; \\text{class},\\; \\text{score})',
    'math-iou': '\\mathrm{IoU}(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}',
    'math-prf':
      '\\text{Precision} = \\frac{TP}{TP + FP}, \\quad \\text{Recall} = \\frac{TP}{TP + FN}, \\quad F_1 = \\frac{2PR}{P + R}'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

// ---------- Boot ----------
function init() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }

  // Sample grid
  const grid = document.getElementById('sample-grid');
  if (grid) {
    grid.innerHTML = SAMPLES.map((s) => `<button class="sample-thumb" data-sample="${s.key}">${s.label}</button>`).join('');
    grid.querySelectorAll('[data-sample]').forEach((b) => {
      b.addEventListener('click', () => pickSample(b.dataset.sample));
    });
  }

  // Upload
  const uploadZone = document.getElementById('upload-zone');
  const photoInput = document.getElementById('photo-input');
  photoInput.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) handleFile(f);
  });
  ['dragenter', 'dragover'].forEach((ev) =>
    uploadZone.addEventListener(ev, (e) => {
      e.preventDefault(); uploadZone.classList.add('drag-over');
    }));
  ['dragleave', 'drop'].forEach((ev) =>
    uploadZone.addEventListener(ev, (e) => {
      e.preventDefault(); uploadZone.classList.remove('drag-over');
    }));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  // Webcam
  const webcamBtn = document.getElementById('webcam-btn');
  webcamBtn.addEventListener('click', async (e) => {
    e.preventDefault(); e.stopPropagation();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement('video');
      video.srcObject = stream; video.playsInline = true;
      await video.play();
      // Capture a single frame after 300ms
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

  // Draw GT button
  const drawBtn = document.getElementById('btn-draw-gt');
  drawBtn.addEventListener('click', () => {
    state.drawingGt = !state.drawingGt;
    drawBtn.textContent = state.drawingGt ? 'Click-drag on the image &hellip; (click to cancel)' : 'Draw ground-truth box';
  });
  document.getElementById('btn-clear-gt').addEventListener('click', () => {
    state.gtBoxes = [];
    renderAll();
  });

  // Sliders
  const confSlider = document.getElementById('conf-slider');
  const confVal = document.getElementById('conf-val');
  confSlider.addEventListener('input', () => {
    state.confThreshold = parseFloat(confSlider.value);
    confVal.textContent = state.confThreshold.toFixed(2);
    renderThresh();
  });
  const nmsSlider = document.getElementById('nms-slider');
  const nmsVal = document.getElementById('nms-val');
  nmsSlider.addEventListener('input', () => {
    state.nmsThreshold = parseFloat(nmsSlider.value);
    nmsVal.textContent = state.nmsThreshold.toFixed(2);
    renderNMS();
  });
  const nmsConfSlider = document.getElementById('nms-conf-slider');
  const nmsConfVal = document.getElementById('nms-conf-val');
  nmsConfSlider.addEventListener('input', () => {
    state.nmsConfFloor = parseFloat(nmsConfSlider.value);
    nmsConfVal.textContent = state.nmsConfFloor.toFixed(2);
    renderNMS();
  });

  wireIoUInput();
  loadModel();
  // Start with the first sample
  pickSample(SAMPLES[0].key);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
