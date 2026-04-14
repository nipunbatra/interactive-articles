// ============================================================
// Image Segmentation, Pixel by Pixel
// - Procedural RGB scenes + procedurally-aligned ground-truth masks
// - Paintable user mask with live mean-IoU and pixel accuracy
// - BFS region growing from a user-clicked seed
// All per-pixel computations run in plain JS over Uint8 arrays.
// ============================================================

const IMG_W = 160;  // internal resolution
const IMG_H = 100;
const DISPLAY_W = 640;
const DISPLAY_H = 400;
const HALF_W = 420;
const HALF_H = 280;

// Class IDs are shared across scenes (but each scene uses a subset)
const CLASSES = {
  0: { name: 'background', color: '#2a2418' },
  1: { name: 'sky',        color: '#6aa0d8' },
  2: { name: 'road',       color: '#4a4a55' },
  3: { name: 'grass',      color: '#7fa85a' },
  4: { name: 'person',     color: '#d9622b' },
  5: { name: 'car',        color: '#2c6fb7' },
  6: { name: 'cat',        color: '#c49a2e' },
  7: { name: 'building',   color: '#8a5eb6' },
  8: { name: 'counter',    color: '#a98559' },
  9: { name: 'apple',      color: '#c53a2b' },
 10: { name: 'orange',     color: '#e89234' },
 11: { name: 'bowl',       color: '#8b6a3c' },
 12: { name: 'cup',        color: '#1e7770' },
 13: { name: 'cloud',      color: '#f0ebe1' }
};

// Convert hex → rgb
function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

// ---------- Scenes ----------
// Each scene defines a pure function (x, y) → { rgb, classId, instanceId }.
// We then rasterize both the RGB image and the label maps.

const SCENES = {
  roadScene: {
    label: 'Road scene',
    caption: 'Sky above, road below, two cars ahead, a pedestrian on the side.',
    classes: [1, 2, 4, 5, 7, 0],
    pixelFn: (x, y) => {
      // Normalized coords
      const nx = x / IMG_W, ny = y / IMG_H;
      // Sky
      if (ny < 0.4) {
        return { rgb: [106 + (ny * 80) | 0, 160 + (ny * 60) | 0, 216 + (ny * 30) | 0], cls: 1, inst: 0 };
      }
      // Building (right side horizon)
      if (nx > 0.75 && ny < 0.6) {
        return { rgb: [138, 94, 182], cls: 7, inst: 0 };
      }
      // Pedestrian (left)
      const pxN = 0.15, pyN = 0.55;
      const dxp = (nx - pxN) * 5, dyp = (ny - pyN) * 3;
      if (dxp * dxp + dyp * dyp < 0.35 && ny > 0.45 && ny < 0.78) {
        return { rgb: [217, 98, 43], cls: 4, inst: 1 };
      }
      // Cars (two)
      const c1 = { cx: 0.4, cy: 0.7, wx: 0.14, hy: 0.08, col: [44, 111, 183], inst: 2 };
      const c2 = { cx: 0.65, cy: 0.66, wx: 0.12, hy: 0.07, col: [176, 48, 48], inst: 3 };
      for (const car of [c1, c2]) {
        if (Math.abs(nx - car.cx) < car.wx && Math.abs(ny - car.cy) < car.hy) {
          return { rgb: car.col, cls: 5, inst: car.inst };
        }
      }
      // Road
      const roadGray = 74 - (ny - 0.6) * 20;
      return { rgb: [roadGray, roadGray, roadGray + 5], cls: 2, inst: 0 };
    }
  },

  twoCats: {
    label: 'Two cats on grass',
    caption: 'A grass field with two cats (different individuals, same class).',
    classes: [3, 6, 1],
    pixelFn: (x, y) => {
      const nx = x / IMG_W, ny = y / IMG_H;
      // Sky at top
      if (ny < 0.25) return { rgb: [150, 180, 210], cls: 1, inst: 0 };
      // Cat 1 (left)
      const dx1 = (nx - 0.3) * 4, dy1 = (ny - 0.65) * 3;
      if (dx1 * dx1 + dy1 * dy1 < 0.45) {
        return { rgb: [196, 154, 46], cls: 6, inst: 1 };
      }
      // Cat 2 (right)
      const dx2 = (nx - 0.72) * 4, dy2 = (ny - 0.68) * 3;
      if (dx2 * dx2 + dy2 * dy2 < 0.45) {
        return { rgb: [210, 175, 70], cls: 6, inst: 2 };
      }
      // Grass
      const jitter = (Math.sin(x * 0.9 + y * 1.2) + Math.cos(x * 1.7 - y * 0.8)) * 8;
      return { rgb: [127 + jitter, 168 + jitter, 90], cls: 3, inst: 0 };
    }
  },

  kitchen: {
    label: 'Kitchen counter',
    caption: 'A counter with a bowl, two apples, and a cup.',
    classes: [8, 11, 9, 12, 0],
    pixelFn: (x, y) => {
      const nx = x / IMG_W, ny = y / IMG_H;
      // Counter
      if (ny > 0.55) return { rgb: [169, 133, 89], cls: 8, inst: 0 };
      // Bowl (big, back)
      const dxb = (nx - 0.4) * 3.3, dyb = (ny - 0.48) * 3;
      if (dxb * dxb + dyb * dyb < 0.45) {
        // Apples inside bowl
        const da1 = (nx - 0.34) * 10, da1y = (ny - 0.48) * 10;
        if (da1 * da1 + da1y * da1y < 1.5) {
          return { rgb: [197, 58, 43], cls: 9, inst: 2 };
        }
        const da2 = (nx - 0.45) * 10, da2y = (ny - 0.49) * 10;
        if (da2 * da2 + da2y * da2y < 1.5) {
          return { rgb: [168, 45, 32], cls: 9, inst: 3 };
        }
        return { rgb: [139, 106, 60], cls: 11, inst: 1 };
      }
      // Cup (right)
      const dxc = (nx - 0.72) * 7, dyc = (ny - 0.42) * 4;
      if (Math.abs(dxc) < 1 && Math.abs(dyc) < 1) {
        return { rgb: [30, 119, 112], cls: 12, inst: 4 };
      }
      // Wall / background
      return { rgb: [42, 36, 24], cls: 0, inst: 0 };
    }
  },

  sky: {
    label: 'Sky & horizon',
    caption: 'Wide sky with clouds, thin horizon of ground.',
    classes: [1, 13, 3],
    pixelFn: (x, y) => {
      const nx = x / IMG_W, ny = y / IMG_H;
      // Ground band
      if (ny > 0.85) return { rgb: [127, 168, 90], cls: 3, inst: 0 };
      // Clouds
      const c1 = (nx - 0.3) ** 2 * 7 + (ny - 0.35) ** 2 * 20;
      const c2 = (nx - 0.7) ** 2 * 10 + (ny - 0.25) ** 2 * 25;
      const c3 = (nx - 0.5) ** 2 * 14 + (ny - 0.55) ** 2 * 16;
      if (c1 < 1 || c2 < 1 || c3 < 1) {
        return { rgb: [240, 235, 225], cls: 13, inst: 0 };
      }
      // Sky gradient
      const r = 106 + ny * 80;
      const g = 160 + ny * 50;
      const b = 216;
      return { rgb: [r | 0, g | 0, b], cls: 1, inst: 0 };
    }
  },

  fruitBowl: {
    label: 'Fruit bowl',
    caption: 'Four fruits (two apples, two oranges) on a bowl, on a counter.',
    classes: [8, 11, 9, 10, 0],
    pixelFn: (x, y) => {
      const nx = x / IMG_W, ny = y / IMG_H;
      if (ny > 0.7) return { rgb: [169, 133, 89], cls: 8, inst: 0 };
      // Bowl: big ellipse
      const dxb = (nx - 0.5) * 2.6, dyb = (ny - 0.55) * 3.5;
      if (dxb * dxb + dyb * dyb < 0.8) {
        // Fruits arranged
        const fruits = [
          { cx: 0.36, cy: 0.45, r: 0.1, col: [197, 58, 43], cls: 9, inst: 2 },
          { cx: 0.55, cy: 0.42, r: 0.1, col: [232, 146, 52], cls: 10, inst: 3 },
          { cx: 0.72, cy: 0.47, r: 0.1, col: [168, 45, 32], cls: 9, inst: 4 },
          { cx: 0.46, cy: 0.55, r: 0.09, col: [237, 165, 74], cls: 10, inst: 5 }
        ];
        for (const f of fruits) {
          const dd = Math.hypot((nx - f.cx) * IMG_W / IMG_H, ny - f.cy);
          if (dd < f.r) return { rgb: f.col, cls: f.cls, inst: f.inst };
        }
        return { rgb: [139, 106, 60], cls: 11, inst: 1 };
      }
      return { rgb: [232, 220, 196], cls: 0, inst: 0 };
    }
  }
};

// ---------- Rasterize ----------
function buildScene(key) {
  const sc = SCENES[key];
  const rgb = new Uint8ClampedArray(IMG_W * IMG_H * 4);
  const cls = new Uint8Array(IMG_W * IMG_H);
  const inst = new Uint8Array(IMG_W * IMG_H);
  for (let y = 0; y < IMG_H; y++) {
    for (let x = 0; x < IMG_W; x++) {
      const { rgb: pix, cls: c, inst: i } = sc.pixelFn(x, y);
      const idx = (y * IMG_W + x) * 4;
      rgb[idx] = pix[0];
      rgb[idx + 1] = pix[1];
      rgb[idx + 2] = pix[2];
      rgb[idx + 3] = 255;
      cls[y * IMG_W + x] = c;
      inst[y * IMG_W + x] = i;
    }
  }
  return { rgb, cls, inst };
}

// ---------- State ----------
let state = {
  sceneKey: 'roadScene',
  scene: null,
  view: 'image',
  currentLabel: 1,
  brushSize: 10,
  userMask: null,    // Uint8Array, 0 = unlabeled, else class id
  tau: 30,
  rgMode: 'seed',     // 'seed' | 'running'
  seed: null,
  region: null,
  regionCount: 0,
  steps: 0
};

function loadScene(key) {
  state.sceneKey = key;
  state.scene = buildScene(key);
  state.userMask = new Uint8Array(IMG_W * IMG_H); // all 0 = unlabeled
  state.seed = null;
  state.region = null;
  state.regionCount = 0;
  state.steps = 0;
  const sc = SCENES[key];
  document.getElementById('scene-caption').textContent = sc.caption;
  renderAll();
  renderPalette();
  renderPaintToolbar();
}

// ---------- Canvas helpers ----------
function getCtx(canvas, w, h) {
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

function blitRgb(ctx, w, h) {
  const off = document.createElement('canvas');
  off.width = IMG_W; off.height = IMG_H;
  const octx = off.getContext('2d');
  const imageData = new ImageData(state.scene.rgb.slice(), IMG_W, IMG_H);
  octx.putImageData(imageData, 0, 0);
  ctx.drawImage(off, 0, 0, w, h);
}

function blitMask(ctx, w, h, maskArr, alpha = 0.55, instance = false) {
  const buf = new Uint8ClampedArray(IMG_W * IMG_H * 4);
  for (let i = 0; i < IMG_W * IMG_H; i++) {
    const label = maskArr[i];
    const idx = i * 4;
    if (label === 0 && !instance) {
      buf[idx + 3] = 0;
    } else if (label === 0 && instance) {
      buf[idx + 3] = 0;
    } else {
      let color;
      if (instance) {
        // Pseudo-random color per instance id
        const hue = (label * 67) % 360;
        color = hslToRgb(hue / 360, 0.6, 0.55);
      } else {
        color = hexToRgb(CLASSES[label].color);
      }
      buf[idx] = color[0];
      buf[idx + 1] = color[1];
      buf[idx + 2] = color[2];
      buf[idx + 3] = Math.round(alpha * 255);
    }
  }
  const off = document.createElement('canvas');
  off.width = IMG_W; off.height = IMG_H;
  const octx = off.getContext('2d');
  octx.putImageData(new ImageData(buf, IMG_W, IMG_H), 0, 0);
  ctx.drawImage(off, 0, 0, w, h);
}

function hslToRgb(h, s, l) {
  let r, g, b;
  if (s === 0) { r = g = b = l; }
  else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// ---------- Step 1 view canvas ----------
function renderViewCanvas() {
  const canvas = document.getElementById('viewCanvas');
  if (!canvas) return;
  const ctx = getCtx(canvas, DISPLAY_W, DISPLAY_H);
  if (state.view === 'image') {
    blitRgb(ctx, DISPLAY_W, DISPLAY_H);
  } else if (state.view === 'semantic') {
    blitMask(ctx, DISPLAY_W, DISPLAY_H, state.scene.cls, 1.0);
  } else if (state.view === 'instance') {
    blitRgb(ctx, DISPLAY_W, DISPLAY_H);
    blitMask(ctx, DISPLAY_W, DISPLAY_H, state.scene.inst, 0.8, true);
  } else if (state.view === 'panoptic') {
    // Panoptic: semantic colors, but with a thin boundary between instances
    blitMask(ctx, DISPLAY_W, DISPLAY_H, state.scene.cls, 1.0);
    drawInstanceBoundaries(ctx, DISPLAY_W, DISPLAY_H);
  }
  const captionEl = document.getElementById('view-caption');
  const capTexts = {
    image: 'Original RGB image.',
    semantic: 'Semantic: every pixel labeled by class. Same-class instances merge.',
    instance: 'Instance: each object (car 1, car 2, cat 1, cat 2) gets a unique mask color.',
    panoptic: 'Panoptic: semantic colors + instance boundaries (thin white lines separate instances of the same class).'
  };
  captionEl.textContent = capTexts[state.view];
}

function drawInstanceBoundaries(ctx, w, h) {
  // Compute 4-neighborhood differences and paint a 1-px line where instance changes but class is same
  const off = document.createElement('canvas');
  off.width = IMG_W; off.height = IMG_H;
  const octx = off.getContext('2d');
  const buf = new Uint8ClampedArray(IMG_W * IMG_H * 4);
  for (let y = 0; y < IMG_H; y++) {
    for (let x = 0; x < IMG_W; x++) {
      const i = y * IMG_W + x;
      let boundary = false;
      const nbs = [
        { dx: 1, dy: 0 }, { dx: -1, dy: 0 },
        { dx: 0, dy: 1 }, { dx: 0, dy: -1 }
      ];
      for (const n of nbs) {
        const xx = x + n.dx, yy = y + n.dy;
        if (xx < 0 || xx >= IMG_W || yy < 0 || yy >= IMG_H) continue;
        const j = yy * IMG_W + xx;
        if (state.scene.inst[i] !== state.scene.inst[j] &&
            state.scene.cls[i] === state.scene.cls[j] &&
            state.scene.inst[i] !== 0 && state.scene.inst[j] !== 0) {
          boundary = true;
          break;
        }
      }
      if (boundary) {
        buf[i * 4] = 255; buf[i * 4 + 1] = 255; buf[i * 4 + 2] = 255; buf[i * 4 + 3] = 255;
      }
    }
  }
  octx.putImageData(new ImageData(buf, IMG_W, IMG_H), 0, 0);
  ctx.drawImage(off, 0, 0, w, h);
}

function renderPalette() {
  const row = document.getElementById('palette-row');
  if (!row) return;
  const classesInScene = [...new Set(Array.from(state.scene.cls))].filter((c) => c > 0).sort((a, b) => a - b);
  row.innerHTML = classesInScene.map((c) => {
    const cl = CLASSES[c];
    return `<span class="label-chip"><span class="swatch" style="background:${cl.color}"></span>${cl.name} (id ${c})</span>`;
  }).join('');
}

// ---------- Step 2 painting ----------
function renderReferenceCanvas() {
  const canvas = document.getElementById('referenceCanvas');
  if (!canvas) return;
  const ctx = getCtx(canvas, HALF_W, HALF_H);
  blitRgb(ctx, HALF_W, HALF_H);
  blitMask(ctx, HALF_W, HALF_H, state.scene.cls, 0.35);
}

function renderPaintCanvas() {
  const canvas = document.getElementById('paintCanvas');
  if (!canvas) return;
  const ctx = getCtx(canvas, HALF_W, HALF_H);
  // Dim the underlying image so the mask is visible
  blitRgb(ctx, HALF_W, HALF_H);
  ctx.fillStyle = 'rgba(253,252,249,0.6)';
  ctx.fillRect(0, 0, HALF_W, HALF_H);
  blitMask(ctx, HALF_W, HALF_H, state.userMask, 0.85);
}

function renderPaintToolbar() {
  const bar = document.getElementById('paint-toolbar');
  if (!bar) return;
  const classesInScene = [...new Set(Array.from(state.scene.cls))].filter((c) => c > 0).sort((a, b) => a - b);
  bar.innerHTML =
    `<button class="mode-button" data-label="0">Eraser</button>` +
    classesInScene.map((c) => {
      const cl = CLASSES[c];
      const active = state.currentLabel === c ? ' is-active' : '';
      return `<button class="mode-button${active}" data-label="${c}" style="border-color:${cl.color};color:${state.currentLabel === c ? 'white' : cl.color};background:${state.currentLabel === c ? cl.color : 'white'}">${cl.name}</button>`;
    }).join('');
  bar.querySelectorAll('[data-label]').forEach((b) => {
    b.addEventListener('click', () => {
      state.currentLabel = parseInt(b.dataset.label, 10);
      renderPaintToolbar();
    });
  });
}

function paintAt(x, y) {
  const r = state.brushSize;
  const xi = Math.round(x * IMG_W / HALF_W);
  const yi = Math.round(y * IMG_H / HALF_H);
  const rr = Math.max(1, Math.round(r * IMG_W / HALF_W));
  for (let dy = -rr; dy <= rr; dy++) {
    for (let dx = -rr; dx <= rr; dx++) {
      if (dx * dx + dy * dy > rr * rr) continue;
      const nx = xi + dx, ny = yi + dy;
      if (nx < 0 || nx >= IMG_W || ny < 0 || ny >= IMG_H) continue;
      state.userMask[ny * IMG_W + nx] = state.currentLabel;
    }
  }
}

function updatePaintMetrics() {
  // Pixel accuracy: fraction of painted pixels whose label matches GT
  // (unlabeled pixels count against us for "painted" but we'll report both)
  const gt = state.scene.cls;
  const um = state.userMask;
  let painted = 0;
  let correct = 0;
  let total = 0;
  const perCls = {}; // cls -> { inter, union }
  for (let i = 0; i < um.length; i++) {
    total++;
    const u = um[i];
    const g = gt[i];
    if (u !== 0) painted++;
    if (u === g) correct++;
    // For IoU, we compare user's class to GT class per class
    const classesToTrack = new Set([g, u].filter((x) => x > 0));
    classesToTrack.forEach((c) => {
      if (!perCls[c]) perCls[c] = { inter: 0, union: 0 };
      const uBool = u === c;
      const gBool = g === c;
      if (uBool && gBool) perCls[c].inter++;
      if (uBool || gBool) perCls[c].union++;
    });
  }
  document.getElementById('pix-painted').textContent = painted.toLocaleString();
  document.getElementById('pixel-acc').textContent = (correct / total).toFixed(3);
  // Mean IoU across tracked classes (ignore background)
  const clsList = Object.keys(perCls).map(Number).filter((c) => c > 0);
  let miou = 0;
  clsList.forEach((c) => {
    const r = perCls[c];
    miou += r.union > 0 ? r.inter / r.union : 0;
  });
  miou = clsList.length ? miou / clsList.length : 0;
  document.getElementById('mean-iou').textContent = miou.toFixed(3);

  // Per-class IoU table
  const tableWrap = document.getElementById('per-class-iou');
  if (tableWrap) {
    let html = '<table class="examples-table"><thead><tr><th>Class</th><th>Intersection</th><th>Union</th><th>IoU</th></tr></thead><tbody>';
    clsList.forEach((c) => {
      const r = perCls[c];
      const iou = r.union > 0 ? r.inter / r.union : 0;
      html += `<tr><td><span class="label-chip" style="margin:0;padding:0.1rem 0.5rem;"><span class="swatch" style="background:${CLASSES[c].color}"></span>${CLASSES[c].name}</span></td><td>${r.inter.toLocaleString()}</td><td>${r.union.toLocaleString()}</td><td><strong>${iou.toFixed(3)}</strong></td></tr>`;
    });
    html += '</tbody></table>';
    tableWrap.innerHTML = html;
  }

  document.getElementById('num-pixels').textContent = (DISPLAY_W * DISPLAY_H).toLocaleString();
}

// ---------- Step 3 region growing ----------
function runRegionGrow(seedX, seedY, tau, mode) {
  const visited = new Uint8Array(IMG_W * IMG_H);
  const region = new Uint8Array(IMG_W * IMG_H);
  const queue = [[seedX, seedY]];
  visited[seedY * IMG_W + seedX] = 1;
  const idx0 = (seedY * IMG_W + seedX) * 4;
  const seedColor = [state.scene.rgb[idx0], state.scene.rgb[idx0 + 1], state.scene.rgb[idx0 + 2]];
  let rSum = seedColor[0], gSum = seedColor[1], bSum = seedColor[2];
  let count = 1;
  region[seedY * IMG_W + seedX] = 1;
  let steps = 0;
  const MAX_STEPS = IMG_W * IMG_H;
  while (queue.length && steps < MAX_STEPS) {
    const [x, y] = queue.shift();
    steps++;
    const nbs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    for (const [dx, dy] of nbs) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= IMG_W || ny < 0 || ny >= IMG_H) continue;
      const idx = ny * IMG_W + nx;
      if (visited[idx]) continue;
      visited[idx] = 1;
      const pi = idx * 4;
      const r = state.scene.rgb[pi], g = state.scene.rgb[pi + 1], b = state.scene.rgb[pi + 2];
      let ref;
      if (mode === 'seed') ref = seedColor;
      else ref = [rSum / count, gSum / count, bSum / count];
      const diff = Math.sqrt(
        (r - ref[0]) ** 2 + (g - ref[1]) ** 2 + (b - ref[2]) ** 2
      );
      if (diff <= tau) {
        region[idx] = 1;
        queue.push([nx, ny]);
        if (mode === 'running') {
          rSum += r; gSum += g; bSum += b; count++;
        }
      }
    }
  }
  return { region, count, steps, seedColor };
}

function renderRgCanvas() {
  const canvas = document.getElementById('rgCanvas');
  if (!canvas) return;
  const ctx = getCtx(canvas, DISPLAY_W, DISPLAY_H);
  blitRgb(ctx, DISPLAY_W, DISPLAY_H);

  if (state.region) {
    // Dim image
    ctx.fillStyle = 'rgba(253,252,249,0.35)';
    ctx.fillRect(0, 0, DISPLAY_W, DISPLAY_H);
    // Overlay region as orange
    const buf = new Uint8ClampedArray(IMG_W * IMG_H * 4);
    for (let i = 0; i < IMG_W * IMG_H; i++) {
      if (state.region[i]) {
        buf[i * 4] = 217; buf[i * 4 + 1] = 98; buf[i * 4 + 2] = 43; buf[i * 4 + 3] = 200;
      }
    }
    const off = document.createElement('canvas');
    off.width = IMG_W; off.height = IMG_H;
    const octx = off.getContext('2d');
    octx.putImageData(new ImageData(buf, IMG_W, IMG_H), 0, 0);
    ctx.drawImage(off, 0, 0, DISPLAY_W, DISPLAY_H);

    // Draw seed
    const scale = DISPLAY_W / IMG_W;
    const sx = (state.seed.x + 0.5) * scale;
    const sy = (state.seed.y + 0.5) * scale;
    ctx.fillStyle = '#2c6fb7';
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function updateRgStats() {
  const colorEl = document.getElementById('seed-color');
  const countEl = document.getElementById('region-size');
  const stepsEl = document.getElementById('steps-run');
  if (state.region) {
    const [r, g, b] = state.region && state.seed ? (() => {
      const idx = (state.seed.y * IMG_W + state.seed.x) * 4;
      return [state.scene.rgb[idx], state.scene.rgb[idx + 1], state.scene.rgb[idx + 2]];
    })() : [0, 0, 0];
    colorEl.textContent = `(${r},${g},${b})`;
    countEl.textContent = state.regionCount.toLocaleString();
    stepsEl.textContent = state.steps.toLocaleString();
  } else {
    colorEl.textContent = '—';
    countEl.textContent = '0';
    stepsEl.textContent = '0';
  }
}

// ---------- Input wiring ----------
function wireViewTabs() {
  document.querySelectorAll('[data-view]').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      state.view = b.dataset.view;
      renderViewCanvas();
    });
  });
}

function wireScenes() {
  document.querySelectorAll('#scene-buttons [data-scene]').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#scene-buttons [data-scene]').forEach((bb) => bb.classList.remove('is-active'));
      b.classList.add('is-active');
      loadScene(b.dataset.scene);
    });
  });
}

function wirePainting() {
  const canvas = document.getElementById('paintCanvas');
  if (!canvas) return;
  let painting = false;
  const getXY = (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    return {
      x: cx / rect.width * HALF_W,
      y: cy / rect.height * HALF_H
    };
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
    state.userMask = new Uint8Array(IMG_W * IMG_H);
    renderPaintCanvas();
    updatePaintMetrics();
  });
  document.getElementById('btn-show-gt').addEventListener('click', () => {
    state.userMask = new Uint8Array(state.scene.cls);
    renderPaintCanvas();
    updatePaintMetrics();
  });

  const brushSlider = document.getElementById('brush-size');
  const brushVal = document.getElementById('brush-size-val');
  brushSlider.addEventListener('input', () => {
    state.brushSize = parseInt(brushSlider.value, 10);
    brushVal.textContent = state.brushSize;
  });
}

function wireRegionGrow() {
  const canvas = document.getElementById('rgCanvas');
  if (!canvas) return;
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width * IMG_W;
    const my = (e.clientY - rect.top) / rect.height * IMG_H;
    state.seed = { x: Math.max(0, Math.min(IMG_W - 1, Math.floor(mx))),
                   y: Math.max(0, Math.min(IMG_H - 1, Math.floor(my))) };
    const res = runRegionGrow(state.seed.x, state.seed.y, state.tau, state.rgMode);
    state.region = res.region;
    state.regionCount = res.count;
    state.steps = res.steps;
    renderRgCanvas();
    updateRgStats();
  });

  const tauSlider = document.getElementById('tau-slider');
  const tauVal = document.getElementById('tau-val');
  tauSlider.addEventListener('input', () => {
    state.tau = parseInt(tauSlider.value, 10);
    tauVal.textContent = state.tau;
    if (state.seed) {
      const res = runRegionGrow(state.seed.x, state.seed.y, state.tau, state.rgMode);
      state.region = res.region;
      state.regionCount = res.count;
      state.steps = res.steps;
      renderRgCanvas();
      updateRgStats();
    }
  });

  const seedBtn = document.getElementById('mode-seed');
  const runningBtn = document.getElementById('mode-running');
  const label = document.getElementById('rg-mode-label');
  seedBtn.addEventListener('click', () => {
    state.rgMode = 'seed';
    seedBtn.classList.add('is-active');
    runningBtn.classList.remove('is-active');
    label.textContent = 'seed pixel colour';
    if (state.seed) {
      const res = runRegionGrow(state.seed.x, state.seed.y, state.tau, 'seed');
      state.region = res.region; state.regionCount = res.count; state.steps = res.steps;
      renderRgCanvas(); updateRgStats();
    }
  });
  runningBtn.addEventListener('click', () => {
    state.rgMode = 'running';
    runningBtn.classList.add('is-active');
    seedBtn.classList.remove('is-active');
    label.textContent = 'running mean colour';
    if (state.seed) {
      const res = runRegionGrow(state.seed.x, state.seed.y, state.tau, 'running');
      state.region = res.region; state.regionCount = res.count; state.steps = res.steps;
      renderRgCanvas(); updateRgStats();
    }
  });
}

// ---------- Full render ----------
function renderAll() {
  renderViewCanvas();
  renderReferenceCanvas();
  renderPaintCanvas();
  updatePaintMetrics();
  renderRgCanvas();
  updateRgStats();
}

// ---------- Math ----------
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
  wireScenes();
  wireViewTabs();
  wirePainting();
  wireRegionGrow();
  loadScene('roadScene');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
