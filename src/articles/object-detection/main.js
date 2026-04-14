// ============================================================
// Object Detection, from Boxes Up
// Live IoU, confidence threshold sweep, NMS, and PR-curve AP,
// all on synthetic scenes with real ground-truth boxes.
// ============================================================

const CANVAS_W = 640;
const CANVAS_H = 400;

// ---------- Scenes ----------
// Each scene has: background draw function, list of ground-truth boxes,
// and a deterministic RNG seed for generating fake detections.
const SCENES = {
  cat: {
    label: 'Single cat',
    bg: drawCatBackground,
    gts: [{ x: 230, y: 130, w: 180, h: 200, cls: 'cat' }],
    seed: 11,
    noise: 0.9
  },
  pedestrians: {
    label: 'Crowded pedestrians',
    bg: drawPedestriansBackground,
    gts: [
      { x: 90,  y: 120, w: 80, h: 220, cls: 'person' },
      { x: 170, y: 115, w: 78, h: 225, cls: 'person' },
      { x: 250, y: 120, w: 80, h: 220, cls: 'person' },
      { x: 330, y: 122, w: 78, h: 218, cls: 'person' },
      { x: 420, y: 120, w: 82, h: 225, cls: 'person' }
    ],
    seed: 42,
    noise: 1.2
  },
  cars: {
    label: 'Cars on a road',
    bg: drawCarsBackground,
    gts: [
      { x: 60,  y: 200, w: 160, h: 90, cls: 'car' },
      { x: 240, y: 210, w: 180, h: 100, cls: 'car' },
      { x: 440, y: 220, w: 160, h: 90, cls: 'car' }
    ],
    seed: 7,
    noise: 1.0
  },
  fruit: {
    label: 'Overlapping fruit',
    bg: drawFruitBackground,
    gts: [
      { x: 120, y: 160, w: 130, h: 130, cls: 'apple' },
      { x: 210, y: 120, w: 140, h: 140, cls: 'orange' },
      { x: 320, y: 170, w: 130, h: 130, cls: 'apple' },
      { x: 420, y: 140, w: 140, h: 140, cls: 'orange' }
    ],
    seed: 99,
    noise: 1.3
  },
  dogs: {
    label: 'Two dogs, one leash',
    bg: drawDogsBackground,
    gts: [
      { x: 110, y: 180, w: 180, h: 180, cls: 'dog' },
      { x: 320, y: 170, w: 200, h: 190, cls: 'dog' }
    ],
    seed: 23,
    noise: 1.0
  }
};

// ---------- Simple seeded RNG ----------
function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function generateDetections(scene) {
  // Produce 3–6 detections per ground-truth, each with jittered box and a
  // confidence that reflects IoU with the truth. Also inject a few
  // low-confidence false positives.
  const rng = mulberry32(scene.seed);
  const out = [];
  scene.gts.forEach((g) => {
    const nCand = 4 + Math.floor(rng() * 3);
    for (let k = 0; k < nCand; k++) {
      const jitter = (rng() - 0.5) * 2 * scene.noise;
      const sx = (rng() - 0.5) * 0.25 * g.w * scene.noise;
      const sy = (rng() - 0.5) * 0.25 * g.h * scene.noise;
      const sw = g.w * (1 + (rng() - 0.5) * 0.35 * scene.noise);
      const sh = g.h * (1 + (rng() - 0.5) * 0.35 * scene.noise);
      const box = {
        x: g.x + sx,
        y: g.y + sy,
        w: sw,
        h: sh,
        cls: g.cls
      };
      const iou = iouOf(box, g);
      const conf = Math.max(0.02, Math.min(0.99, iou * 0.85 + rng() * 0.15));
      out.push({ ...box, score: conf });
    }
  });
  // Add some false positives
  const nFp = 2 + Math.floor(rng() * 3);
  for (let k = 0; k < nFp; k++) {
    const w = 40 + rng() * 120;
    const h = 40 + rng() * 120;
    const x = 10 + rng() * (CANVAS_W - w - 20);
    const y = 10 + rng() * (CANVAS_H - h - 20);
    out.push({ x, y, w, h, cls: 'noise', score: 0.1 + rng() * 0.35 });
  }
  // Sort descending by score
  out.sort((a, b) => b.score - a.score);
  return out;
}

// ---------- Box math ----------
function clip(b) {
  return {
    x1: Math.max(0, b.x),
    y1: Math.max(0, b.y),
    x2: Math.min(CANVAS_W, b.x + b.w),
    y2: Math.min(CANVAS_H, b.y + b.h)
  };
}
function area(b) {
  const c = clip(b);
  return Math.max(0, c.x2 - c.x1) * Math.max(0, c.y2 - c.y1);
}
function iouOf(a, b) {
  const ca = clip(a), cb = clip(b);
  const ix1 = Math.max(ca.x1, cb.x1);
  const iy1 = Math.max(ca.y1, cb.y1);
  const ix2 = Math.min(ca.x2, cb.x2);
  const iy2 = Math.min(ca.y2, cb.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const aa = (ca.x2 - ca.x1) * (ca.y2 - ca.y1);
  const ab = (cb.x2 - cb.x1) * (cb.y2 - cb.y1);
  const union = aa + ab - inter;
  if (union <= 0) return 0;
  return inter / union;
}

// ---------- State ----------
let state = {
  sceneKey: 'cat',
  conf: 0.3,
  nmsIoU: 0.5,
  nmsMode: 'overlay',
  // Interactive boxes for Step 1
  truthBox: { x: 180, y: 110, w: 220, h: 220 },
  predBox:  { x: 260, y: 170, w: 200, h: 180 },
  dragging: null, // { kind: 'truth'|'pred', mode: 'move'|'resize' }
  dragStart: null
};

function loadScene(key) {
  state.sceneKey = key;
  const sc = SCENES[key];
  // Position the interactive boxes to reasonable defaults based on the GT
  const g = sc.gts[0];
  state.truthBox = { x: g.x, y: g.y, w: g.w, h: g.h };
  state.predBox = {
    x: g.x + 40, y: g.y + 20,
    w: Math.max(60, g.w * 0.85), h: Math.max(60, g.h * 0.85)
  };
  updateAll();
}

// ---------- Canvas setup ----------
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = CANVAS_W * dpr;
  canvas.height = CANVAS_H * dpr;
  canvas.style.width = CANVAS_W + 'px';
  canvas.style.height = CANVAS_H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// ---------- Scene backgrounds ----------
function fillBg(ctx, color) {
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
}

function drawCatBackground(ctx) {
  fillBg(ctx, '#3a4560');
  // Sun-like gradient
  const grd = ctx.createRadialGradient(420, 100, 10, 420, 100, 200);
  grd.addColorStop(0, 'rgba(255, 220, 150, 0.8)');
  grd.addColorStop(1, 'rgba(255, 220, 150, 0)');
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
  // Floor
  ctx.fillStyle = '#5a4b3b';
  ctx.fillRect(0, 320, CANVAS_W, 80);
  // Cat silhouette
  ctx.fillStyle = '#1a1815';
  ctx.beginPath();
  ctx.ellipse(320, 240, 80, 80, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(320, 170, 50, 50, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(280, 140); ctx.lineTo(290, 110); ctx.lineTo(300, 135);
  ctx.moveTo(350, 135); ctx.lineTo(345, 110); ctx.lineTo(360, 140);
  ctx.fill();
  // Tail
  ctx.strokeStyle = '#1a1815';
  ctx.lineWidth = 14;
  ctx.beginPath();
  ctx.moveTo(400, 260);
  ctx.quadraticCurveTo(450, 200, 410, 160);
  ctx.stroke();
}

function drawPedestriansBackground(ctx) {
  fillBg(ctx, '#5c6b7a');
  ctx.fillStyle = '#8fa3b8';
  ctx.fillRect(0, 310, CANVAS_W, 90);
  const cols = ['#d9622b', '#1e7770', '#2c6fb7', '#a3428a', '#c49a2e'];
  const xs = [130, 210, 290, 370, 460];
  xs.forEach((x, i) => {
    ctx.fillStyle = cols[i];
    ctx.beginPath();
    ctx.ellipse(x, 155, 18, 22, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillRect(x - 22, 175, 44, 110);
    ctx.fillStyle = '#2a2418';
    ctx.fillRect(x - 14, 285, 10, 50);
    ctx.fillRect(x + 4,  285, 10, 50);
  });
}

function drawCarsBackground(ctx) {
  fillBg(ctx, '#2a3a4a');
  // Road
  ctx.fillStyle = '#3a434c';
  ctx.fillRect(0, 200, CANVAS_W, 200);
  // Lane markings
  ctx.strokeStyle = '#e0d8c6';
  ctx.lineWidth = 3;
  ctx.setLineDash([25, 25]);
  ctx.beginPath(); ctx.moveTo(0, 300); ctx.lineTo(CANVAS_W, 300); ctx.stroke();
  ctx.setLineDash([]);
  // Cars
  const carsData = [
    { x: 140, y: 245, w: 160, col: '#d9622b' },
    { x: 330, y: 260, w: 180, col: '#1e7770' },
    { x: 520, y: 265, w: 160, col: '#8a5eb6' }
  ];
  carsData.forEach((c) => {
    ctx.fillStyle = c.col;
    ctx.fillRect(c.x - c.w / 2, c.y, c.w, 50);
    ctx.fillRect(c.x - c.w / 2 + 20, c.y - 25, c.w - 40, 30);
    ctx.fillStyle = '#1a1815';
    ctx.beginPath(); ctx.arc(c.x - c.w / 2 + 30, c.y + 55, 12, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(c.x + c.w / 2 - 30, c.y + 55, 12, 0, Math.PI * 2); ctx.fill();
  });
}

function drawFruitBackground(ctx) {
  fillBg(ctx, '#e8dcc4');
  const fruits = [
    { x: 185, y: 225, r: 60, col: '#c53a2b', type: 'apple' },
    { x: 280, y: 190, r: 65, col: '#e89234', type: 'orange' },
    { x: 385, y: 235, r: 60, col: '#a82d20', type: 'apple' },
    { x: 490, y: 210, r: 65, col: '#eda54a', type: 'orange' }
  ];
  fruits.forEach((f) => {
    ctx.fillStyle = f.col;
    ctx.beginPath(); ctx.arc(f.x, f.y, f.r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 2;
    ctx.stroke();
    if (f.type === 'apple') {
      ctx.strokeStyle = '#3a2a1a';
      ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(f.x, f.y - f.r); ctx.lineTo(f.x + 8, f.y - f.r - 14); ctx.stroke();
    }
  });
}

function drawDogsBackground(ctx) {
  fillBg(ctx, '#a5c77a');
  ctx.fillStyle = '#7fa85a';
  ctx.fillRect(0, 300, CANVAS_W, 100);
  // Two dog silhouettes
  const dogs = [
    { x: 200, y: 250, col: '#7c4a2a' },
    { x: 420, y: 240, col: '#c49a2e' }
  ];
  dogs.forEach((d) => {
    ctx.fillStyle = d.col;
    ctx.beginPath(); ctx.ellipse(d.x, d.y, 75, 45, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(d.x + 60, d.y - 30, 30, 30, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillRect(d.x - 40, d.y + 30, 15, 60);
    ctx.fillRect(d.x + 30, d.y + 30, 15, 60);
  });
  // Leash
  ctx.strokeStyle = '#1a1815';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(260, 210); ctx.quadraticCurveTo(310, 300, 480, 220); ctx.stroke();
}

// ---------- Box drawing ----------
function drawBox(ctx, b, color, label = null, opacity = 1, dash = []) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.globalAlpha = opacity;
  ctx.setLineDash(dash);
  ctx.strokeRect(b.x, b.y, b.w, b.h);
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
  if (label) {
    ctx.font = '600 12px system-ui, sans-serif';
    const tw = ctx.measureText(label).width + 10;
    ctx.fillStyle = color;
    ctx.globalAlpha = opacity;
    ctx.fillRect(b.x, b.y - 18, tw, 18);
    ctx.globalAlpha = 1;
    ctx.fillStyle = 'white';
    ctx.fillText(label, b.x + 5, b.y - 5);
  }
}

function drawHandles(ctx, b, color) {
  const handles = [
    { x: b.x, y: b.y }, { x: b.x + b.w, y: b.y },
    { x: b.x, y: b.y + b.h }, { x: b.x + b.w, y: b.y + b.h },
    { x: b.x + b.w / 2, y: b.y + b.h / 2 }
  ];
  handles.forEach((h, i) => {
    ctx.fillStyle = i === 4 ? 'white' : color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(h.x, h.y, i === 4 ? 5 : 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

// ---------- Step 1 & 2: interactive IoU canvas ----------
function renderIoUCanvas() {
  const canvas = document.getElementById('iouCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  const sc = SCENES[state.sceneKey];
  sc.bg(ctx);

  const a = state.truthBox, b = state.predBox;
  // Intersection shading
  const ca = clip(a), cb = clip(b);
  const ix1 = Math.max(ca.x1, cb.x1);
  const iy1 = Math.max(ca.y1, cb.y1);
  const ix2 = Math.min(ca.x2, cb.x2);
  const iy2 = Math.min(ca.y2, cb.y2);
  if (ix2 > ix1 && iy2 > iy1) {
    ctx.fillStyle = 'rgba(138, 94, 182, 0.35)';
    ctx.fillRect(ix1, iy1, ix2 - ix1, iy2 - iy1);
  }

  drawBox(ctx, a, getComputedStyle(document.documentElement).getPropertyValue('--truth').trim() || '#1e7770', 'truth');
  drawBox(ctx, b, getComputedStyle(document.documentElement).getPropertyValue('--pred').trim() || '#d9622b', 'pred');
  drawHandles(ctx, a, '#1e7770');
  drawHandles(ctx, b, '#d9622b');

  // Update stats
  const aA = area(a), aB = area(b);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const iou = iouOf(a, b);
  const union = aA + aB - inter;
  document.getElementById('area-truth').textContent = Math.round(aA).toLocaleString();
  document.getElementById('area-pred').textContent = Math.round(aB).toLocaleString();
  document.getElementById('area-inter').textContent = Math.round(inter).toLocaleString();
  document.getElementById('iou-value').textContent = iou.toFixed(3);
  document.getElementById('iouExplain').textContent =
    `IoU = intersection / union = ${Math.round(inter)} / ${Math.round(union)} = ${iou.toFixed(3)}`;
  const verdict = document.getElementById('iouVerdict');
  let msg;
  if (iou >= 0.9) msg = 'Near-perfect match. A detector with this IoU is essentially identifying the object exactly.';
  else if (iou >= 0.7) msg = 'Good detection. This is the range strict benchmarks care about (AP@0.75).';
  else if (iou >= 0.5) msg = 'Acceptable by the classic "is it a true positive?" threshold of 0.5 — but not great.';
  else if (iou >= 0.3) msg = 'Partial overlap. Most benchmarks would call this a miss.';
  else if (iou > 0) msg = 'Barely touching. Clearly a wrong box.';
  else msg = 'No overlap at all. IoU = 0.';
  verdict.textContent = msg;
}

// ---------- Step 3 raw detections ----------
function renderRawCanvas() {
  const canvas = document.getElementById('rawCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  const sc = SCENES[state.sceneKey];
  sc.bg(ctx);

  const dets = generateDetections(sc);
  // Draw GTs
  sc.gts.forEach((g) => drawBox(ctx, g, '#1e7770', null, 1, [8, 6]));

  let showing = 0;
  dets.forEach((d) => {
    const above = d.score >= state.conf;
    const color = above ? '#d9622b' : '#9a917f';
    const alpha = above ? Math.max(0.35, d.score) : 0.35;
    drawBox(ctx, d, color, above ? `${d.cls} ${(d.score * 100).toFixed(0)}%` : null, alpha);
    if (above) showing++;
  });

  document.getElementById('showing-count').textContent = showing;
  document.getElementById('total-count').textContent = dets.length;

  // Compute TP/FP/FN at the current threshold
  const { tp, fp, fn } = countAtThreshold(dets, sc.gts, state.conf, 0.5);
  document.getElementById('tp').textContent = tp;
  document.getElementById('fp').textContent = fp;
  document.getElementById('fn').textContent = fn;
  const P = tp + fp > 0 ? tp / (tp + fp) : 0;
  const R = tp + fn > 0 ? tp / (tp + fn) : 0;
  const F1 = (P + R > 0) ? 2 * P * R / (P + R) : 0;
  document.getElementById('precision').textContent = P.toFixed(3);
  document.getElementById('recall').textContent = R.toFixed(3);
  document.getElementById('f1').textContent = F1.toFixed(3);
}

function countAtThreshold(dets, gts, conf, iouT) {
  // Filter and sort by score desc, then greedily match to un-used GTs
  const kept = dets.filter((d) => d.score >= conf).slice().sort((a, b) => b.score - a.score);
  const gtUsed = new Array(gts.length).fill(false);
  let tp = 0, fp = 0;
  kept.forEach((d) => {
    let bestI = -1, bestIoU = 0;
    for (let i = 0; i < gts.length; i++) {
      if (gtUsed[i]) continue;
      if (gts[i].cls !== d.cls && d.cls !== 'noise' && gts[i].cls !== 'noise') {
        // class mismatch — skip (but in our tiny demo class is largely same within scene)
      }
      const iou = iouOf(d, gts[i]);
      if (iou > bestIoU) { bestIoU = iou; bestI = i; }
    }
    if (bestI >= 0 && bestIoU >= iouT && d.cls !== 'noise') {
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
function runNMS(dets, iouT) {
  const filtered = dets.filter((d) => d.score >= state.conf).slice().sort((a, b) => b.score - a.score);
  const kept = [];
  const suppressed = [];
  const trace = [];
  const active = filtered.map((d, i) => ({ ...d, idx: i, alive: true }));
  while (true) {
    const head = active.find((a) => a.alive);
    if (!head) break;
    kept.push(head);
    trace.push(`keep #${head.idx + 1} (conf ${(head.score * 100).toFixed(0)}%)`);
    head.alive = false;
    for (const cand of active) {
      if (!cand.alive) continue;
      const iou = iouOf(head, cand);
      if (iou >= iouT) {
        cand.alive = false;
        suppressed.push(cand);
        trace.push(`  suppress #${cand.idx + 1} (IoU ${iou.toFixed(2)} ≥ ${iouT.toFixed(2)})`);
      }
    }
  }
  return { kept, suppressed, trace };
}

function renderNMSCanvas() {
  const canvas = document.getElementById('nmsCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  const sc = SCENES[state.sceneKey];
  sc.bg(ctx);

  const dets = generateDetections(sc);
  sc.gts.forEach((g) => drawBox(ctx, g, '#1e7770', null, 1, [8, 6]));

  const { kept, suppressed, trace } = runNMS(dets, state.nmsIoU);
  if (state.nmsMode === 'overlay') {
    suppressed.forEach((d) => drawBox(ctx, d, '#9a917f', null, 0.35));
  }
  kept.forEach((d) => drawBox(ctx, d, '#d9622b', `${d.cls} ${(d.score * 100).toFixed(0)}%`, 0.95));

  document.getElementById('nms-in').textContent = dets.filter((d) => d.score >= state.conf).length;
  document.getElementById('nms-out').textContent = kept.length;
  document.getElementById('nms-sup').textContent = suppressed.length;

  const traceEl = document.getElementById('nms-trace');
  if (traceEl) traceEl.textContent = trace.join('\n') || '(no detections above confidence)';
}

// ---------- Step 5 anchor sketch ----------
function renderAnchorCanvas() {
  const canvas = document.getElementById('anchorCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  const sc = SCENES[state.sceneKey];
  sc.bg(ctx);
  sc.gts.forEach((g) => drawBox(ctx, g, '#1e7770', null, 1, [8, 6]));

  const cols = 8, rows = 5;
  const padX = 40, padY = 40;
  const dx = (CANVAS_W - 2 * padX) / (cols - 1);
  const dy = (CANVAS_H - 2 * padY) / (rows - 1);
  ctx.globalAlpha = 0.9;
  // Sample anchors at 3 aspect ratios
  const aspects = [
    { w: 60, h: 80 },
    { w: 100, h: 60 },
    { w: 80, h: 80 }
  ];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cx = padX + c * dx;
      const cy = padY + r * dy;
      ctx.fillStyle = 'rgba(44, 111, 183, 0.6)';
      ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.fill();
      if ((r + c) % 5 === 0) {
        aspects.forEach((a, i) => {
          ctx.strokeStyle = `rgba(44, 111, 183, ${0.35 - i * 0.08})`;
          ctx.lineWidth = 1;
          ctx.strokeRect(cx - a.w / 2, cy - a.h / 2, a.w, a.h);
        });
      }
    }
  }
  ctx.globalAlpha = 1;
}

// ---------- Step 6 PR curve and AP ----------
function computeAP(dets, gts, iouT = 0.5) {
  // Sort by confidence desc
  const sorted = dets.slice().sort((a, b) => b.score - a.score);
  const gtUsed = new Array(gts.length).fill(false);
  const pts = [];
  let tp = 0, fp = 0;
  sorted.forEach((d) => {
    let bestI = -1, bestIoU = 0;
    for (let i = 0; i < gts.length; i++) {
      if (gtUsed[i]) continue;
      const iou = iouOf(d, gts[i]);
      if (iou > bestIoU) { bestIoU = iou; bestI = i; }
    }
    if (bestI >= 0 && bestIoU >= iouT && d.cls !== 'noise') {
      tp++;
      gtUsed[bestI] = true;
    } else {
      fp++;
    }
    const P = tp / (tp + fp);
    const R = tp / gts.length;
    pts.push({ P, R });
  });
  // 11-point interpolation AP
  let ap = 0;
  for (let r = 0; r <= 1.0001; r += 0.1) {
    let maxP = 0;
    for (const p of pts) {
      if (p.R >= r - 1e-9 && p.P > maxP) maxP = p.P;
    }
    ap += maxP / 11;
  }
  return { ap, pts };
}

function renderPRCanvas() {
  const canvas = document.getElementById('prCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas);
  ctx.clearRect(0, 0, CANVAS_W, 320);

  const margin = { top: 30, right: 30, bottom: 40, left: 60 };
  const pw = CANVAS_W - margin.left - margin.right;
  const ph = 320 - margin.top - margin.bottom;

  // Axes
  ctx.strokeStyle = '#c4beb1';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + ph);
  ctx.lineTo(margin.left + pw, margin.top + ph);
  ctx.stroke();

  ctx.strokeStyle = '#f0ebe1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i <= 5; i++) {
    const x = margin.left + (pw / 5) * i;
    ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + ph);
    const y = margin.top + (ph / 5) * i;
    ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + pw, y);
  }
  ctx.stroke();

  ctx.fillStyle = '#9a917f';
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Recall', margin.left + pw / 2, margin.top + ph + 24);
  ctx.save();
  ctx.translate(margin.left - 42, margin.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Precision', 0, 0);
  ctx.restore();
  ctx.textAlign = 'right';
  ctx.fillText('1', margin.left - 6, margin.top + 4);
  ctx.fillText('0', margin.left - 6, margin.top + ph + 4);
  ctx.textAlign = 'center';
  ctx.fillText('0', margin.left, margin.top + ph + 14);
  ctx.fillText('1', margin.left + pw, margin.top + ph + 14);

  const sc = SCENES[state.sceneKey];
  const dets = generateDetections(sc);
  const { ap, pts } = computeAP(dets, sc.gts);

  // Curve
  ctx.strokeStyle = '#2c6fb7';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = margin.left + p.R * pw;
    const y = margin.top + (1 - p.P) * ph;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill under
  ctx.fillStyle = 'rgba(44, 111, 183, 0.15)';
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = margin.left + p.R * pw;
    const y = margin.top + (1 - p.P) * ph;
    if (i === 0) ctx.moveTo(x, margin.top + ph);
    ctx.lineTo(x, y);
  });
  ctx.lineTo(margin.left + (pts.length ? pts[pts.length - 1].R : 0) * pw, margin.top + ph);
  ctx.closePath();
  ctx.fill();

  // Points
  ctx.fillStyle = '#d9622b';
  pts.forEach((p) => {
    const x = margin.left + p.R * pw;
    const y = margin.top + (1 - p.P) * ph;
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
  });

  document.getElementById('apValue').textContent = ap.toFixed(3);
}

// ---------- Interactive box dragging ----------
function wireIoUDrag() {
  const canvas = document.getElementById('iouCanvas');
  if (!canvas) return;
  const getXY = (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    return {
      x: cx / rect.width * CANVAS_W,
      y: cy / rect.height * CANVAS_H
    };
  };

  function hitTest(x, y) {
    // Return { kind, mode } where kind = 'truth'|'pred', mode = 'move'|corner
    const boxes = [
      { key: 'predBox', b: state.predBox },   // pred on top first
      { key: 'truthBox', b: state.truthBox }
    ];
    for (const { key, b } of boxes) {
      const corners = [
        { x: b.x, y: b.y, mode: 'nw' },
        { x: b.x + b.w, y: b.y, mode: 'ne' },
        { x: b.x, y: b.y + b.h, mode: 'sw' },
        { x: b.x + b.w, y: b.y + b.h, mode: 'se' }
      ];
      for (const c of corners) {
        if (Math.hypot(x - c.x, y - c.y) < 12) return { key, mode: c.mode };
      }
      if (x > b.x && x < b.x + b.w && y > b.y && y < b.y + b.h) {
        return { key, mode: 'move' };
      }
    }
    return null;
  }

  function onDown(e) {
    const { x, y } = getXY(e);
    const hit = hitTest(x, y);
    if (hit) {
      state.dragging = hit;
      state.dragStart = { x, y, box: { ...state[hit.key] } };
      if (e.touches) e.preventDefault();
    }
  }
  function onMove(e) {
    if (!state.dragging) return;
    const { x, y } = getXY(e);
    const d = state.dragging;
    const s = state.dragStart;
    const b = { ...s.box };
    if (d.mode === 'move') {
      b.x = s.box.x + (x - s.x);
      b.y = s.box.y + (y - s.y);
    } else {
      // Corner resize
      let x1 = b.x, y1 = b.y, x2 = b.x + b.w, y2 = b.y + b.h;
      if (d.mode === 'nw') { x1 = x; y1 = y; }
      if (d.mode === 'ne') { x2 = x; y1 = y; }
      if (d.mode === 'sw') { x1 = x; y2 = y; }
      if (d.mode === 'se') { x2 = x; y2 = y; }
      const bx = Math.min(x1, x2);
      const by = Math.min(y1, y2);
      const bw = Math.max(20, Math.abs(x2 - x1));
      const bh = Math.max(20, Math.abs(y2 - y1));
      b.x = bx; b.y = by; b.w = bw; b.h = bh;
    }
    state[d.key] = b;
    renderIoUCanvas();
    if (e.touches) e.preventDefault();
  }
  function onUp() { state.dragging = null; }

  canvas.addEventListener('mousedown', onDown);
  canvas.addEventListener('touchstart', onDown, { passive: false });
  window.addEventListener('mousemove', onMove);
  window.addEventListener('touchmove', onMove, { passive: false });
  window.addEventListener('mouseup', onUp);
  window.addEventListener('touchend', onUp);
}

// ---------- Top-level update ----------
function updateAll() {
  renderIoUCanvas();
  renderRawCanvas();
  renderNMSCanvas();
  renderAnchorCanvas();
  renderPRCanvas();
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

  document.querySelectorAll('#scene-buttons [data-scene]').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#scene-buttons [data-scene]').forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      loadScene(b.dataset.scene);
    });
  });

  const confSlider = document.getElementById('conf-slider');
  const confVal = document.getElementById('conf-val');
  confSlider.addEventListener('input', () => {
    state.conf = parseFloat(confSlider.value);
    confVal.textContent = state.conf.toFixed(2);
    renderRawCanvas();
    renderNMSCanvas();
  });

  const nmsSlider = document.getElementById('nms-slider');
  const nmsVal = document.getElementById('nms-val');
  nmsSlider.addEventListener('input', () => {
    state.nmsIoU = parseFloat(nmsSlider.value);
    nmsVal.textContent = state.nmsIoU.toFixed(2);
    renderNMSCanvas();
  });

  const overlayBtn = document.getElementById('nms-overlay');
  const hideBtn = document.getElementById('nms-hide');
  const modeLbl = document.getElementById('nms-mode');
  overlayBtn.addEventListener('click', () => {
    state.nmsMode = 'overlay';
    overlayBtn.classList.add('is-active');
    hideBtn.classList.remove('is-active');
    modeLbl.textContent = 'overlay';
    renderNMSCanvas();
  });
  hideBtn.addEventListener('click', () => {
    state.nmsMode = 'hide';
    hideBtn.classList.add('is-active');
    overlayBtn.classList.remove('is-active');
    modeLbl.textContent = 'hide';
    renderNMSCanvas();
  });

  wireIoUDrag();
  loadScene('cat');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
