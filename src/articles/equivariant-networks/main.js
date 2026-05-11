// ============================================================
// Equivariant Networks demo
// Synthesise an airplane silhouette; rotate by theta; compute the
// "feature vector" produced by three networks:
//  - vanilla CNN: single 3x3 filter; responds to fixed orientation
//  - augmented CNN: average response over a small set of orientations
//  - C4 group conv: concatenate responses over 4 rotated copies
// Then plot feature-distance from theta=0 across theta.
// ============================================================

const SIZE = 64;
const STATE = { theta: 0, shape: null };

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  return ctx;
}
function randn() { const u = Math.random() || 1e-12, v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

function makeShape() {
  // An airplane silhouette: fuselage horizontal, wings perpendicular
  const arr = new Float32Array(SIZE * SIZE);
  const cx = SIZE / 2, cy = SIZE / 2;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const dx = x - cx, dy = y - cy;
      // Body: ellipse along x
      const body = (dx * dx) / 200 + (dy * dy) / 10 < 1 ? 1 : 0;
      // Wings: vertical strip in the middle
      const wings = Math.abs(dx) < 4 && Math.abs(dy) < 18 ? 1 : 0;
      // Tail
      const tail = dx < -10 && Math.abs(dy) < 6 ? 1 : 0;
      arr[y * SIZE + x] = Math.max(body, wings, tail) > 0 ? 1 : 0;
    }
  }
  return arr;
}

function rotateImage(arr, theta) {
  const out = new Float32Array(SIZE * SIZE);
  const cx = SIZE / 2, cy = SIZE / 2;
  const ct = Math.cos(-theta), st = Math.sin(-theta);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const sx = ct * (x - cx) - st * (y - cy) + cx;
      const sy = st * (x - cx) + ct * (y - cy) + cy;
      const ix = Math.round(sx), iy = Math.round(sy);
      if (ix >= 0 && ix < SIZE && iy >= 0 && iy < SIZE) out[y * SIZE + x] = arr[iy * SIZE + ix];
    }
  }
  return out;
}

// Filters: a fixed 3x3 "horizontal-line" detector
const FILTER_H = [-1, -1, -1, 2, 2, 2, -1, -1, -1];
function conv3(img, kernel) {
  const out = new Float32Array(SIZE * SIZE);
  for (let y = 1; y < SIZE - 1; y++) {
    for (let x = 1; x < SIZE - 1; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          s += img[(y + dy) * SIZE + (x + dx)] * kernel[(dy + 1) * 3 + (dx + 1)];
        }
      }
      out[y * SIZE + x] = Math.max(0, s);
    }
  }
  return out;
}

function globalPool(arr) {
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

function vanillaCNNFeature(img) {
  // Single fixed filter, global avg pool -> scalar feature
  return globalPool(conv3(img, FILTER_H));
}

function augmentedCNNFeature(img) {
  // Average over 4 input rotations (90 degree increments) — proxy for training-with-augmentation behaviour
  let s = 0;
  for (let k = 0; k < 4; k++) {
    s += globalPool(conv3(rotateImage(img, k * Math.PI / 2), FILTER_H));
  }
  return s / 4;
}

function c4GroupFeature(img) {
  // Concatenate responses to 4 rotated copies of the filter (group conv at the first layer)
  // Then sum-pool over the group axis: this gives an *invariant* feature.
  const filters = [];
  for (let k = 0; k < 4; k++) {
    // 90-degree rotated filter
    const rotated = new Array(9);
    for (let i = 0; i < 9; i++) {
      const r = Math.floor(i / 3), c = i % 3;
      let nr, nc;
      if (k === 0) { nr = r; nc = c; }
      else if (k === 1) { nr = c; nc = 2 - r; }
      else if (k === 2) { nr = 2 - r; nc = 2 - c; }
      else { nr = 2 - c; nc = r; }
      rotated[nr * 3 + nc] = FILTER_H[i];
    }
    filters.push(rotated);
  }
  let s = 0;
  for (const f of filters) s += globalPool(conv3(img, f));
  return s / 4;
}

function drawScalarImage(canvas, arr) {
  const ctx = setupCanvas(canvas, 160, 160);
  const cell = 160 / SIZE;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = arr[y * SIZE + x];
      const t = Math.max(0, Math.min(1, v));
      const c = Math.round(253 - 200 * t);
      ctx.fillStyle = `rgb(${c}, ${c}, ${c})`;
      ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, 160, 160);
}

function drawFeatureMap(canvas, arr) {
  const ctx = setupCanvas(canvas, 160, 160);
  const cell = 160 / SIZE;
  let maxV = 0; for (const v of arr) maxV = Math.max(maxV, v);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = arr[y * SIZE + x] / Math.max(maxV, 1e-6);
      const t = Math.max(0, Math.min(1, v));
      ctx.fillStyle = `rgba(217, 98, 43, ${0.1 + 0.8 * t})`;
      ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, 160, 160);
}

function renderInputs() {
  if (!STATE.shape) STATE.shape = makeShape();
  const rotated = rotateImage(STATE.shape, STATE.theta * Math.PI / 180);
  drawScalarImage(document.getElementById('eq-input'), rotated);
  drawFeatureMap(document.getElementById('eq-vanilla'), conv3(rotated, FILTER_H));
  // Aug: just show the same feature map for simplicity
  drawFeatureMap(document.getElementById('eq-aug'), conv3(rotated, FILTER_H));
  // C4: sum over 4 rotated filters' feature maps
  const c4Map = new Float32Array(SIZE * SIZE);
  for (let k = 0; k < 4; k++) {
    const rotatedFilter = new Array(9);
    for (let i = 0; i < 9; i++) {
      const r = Math.floor(i / 3), c = i % 3;
      let nr, nc;
      if (k === 0) { nr = r; nc = c; }
      else if (k === 1) { nr = c; nc = 2 - r; }
      else if (k === 2) { nr = 2 - r; nc = 2 - c; }
      else { nr = 2 - c; nc = r; }
      rotatedFilter[nr * 3 + nc] = FILTER_H[i];
    }
    const fm = conv3(rotated, rotatedFilter);
    for (let i = 0; i < c4Map.length; i++) c4Map[i] += fm[i];
  }
  drawFeatureMap(document.getElementById('eq-c4'), c4Map);
}

function renderCurve() {
  const canvas = document.getElementById('eq-curve');
  if (!canvas) return;
  const W = 880, H = 280;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 14, t: 18, b: 32 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const base = STATE.shape;
  const refV = vanillaCNNFeature(base);
  const refA = augmentedCNNFeature(base);
  const refC = c4GroupFeature(base);
  const N = 60;
  const vals = { v: [], a: [], c: [] };
  for (let i = 0; i <= N; i++) {
    const t = (i / N) * 360;
    const r = rotateImage(base, t * Math.PI / 180);
    vals.v.push(Math.abs(vanillaCNNFeature(r) - refV));
    vals.a.push(Math.abs(augmentedCNNFeature(r) - refA));
    vals.c.push(Math.abs(c4GroupFeature(r) - refC));
  }
  const all = vals.v.concat(vals.a).concat(vals.c);
  const hi = Math.max(...all) * 1.1 || 0.1;
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = hi * (1 - i / 4);
    const y = m.t + i / 4 * py;
    ctx.fillText(v.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let v = 0; v <= 360; v += 60) {
    const x = m.l + (v / 360) * px;
    ctx.fillText(`${v}°`, x, m.t + py + 16);
  }
  function plot(arr, color, dashed) {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [4, 4] : []);
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = m.l + (i / N) * px;
      const y = m.t + (1 - v / hi) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  plot(vals.v, '#d9622b', false);
  plot(vals.a, '#9b59b6', true);
  plot(vals.c, '#1e7770', false);
  // Current theta marker
  const cx = m.l + (STATE.theta / 360) * px;
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(cx, m.t); ctx.lineTo(cx, m.t + py); ctx.stroke();
  ctx.setLineDash([]);
  // Legend
  ctx.font = '11px Manrope';
  ctx.fillStyle = '#3b342b';
  ctx.textAlign = 'left';
  ctx.fillText('vanilla CNN', m.l + 8, m.t + 14);
  ctx.fillStyle = '#9c3f15';
  ctx.fillText('augmented CNN (dashed)', m.l + 100, m.t + 14);
  ctx.fillStyle = '#1e7770';
  ctx.fillText('C4 group conv', m.l + 260, m.t + 14);
}

function refresh() {
  renderInputs(); renderCurve();
}

function wire() {
  STATE.shape = makeShape();
  document.getElementById('eq-theta').addEventListener('input', (e) => {
    STATE.theta = parseInt(e.target.value, 10);
    document.getElementById('eq-theta-val').textContent = `${STATE.theta}°`;
    refresh();
  });
  document.getElementById('eq-newshape').addEventListener('click', () => { STATE.shape = makeShape(); refresh(); });
  refresh();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-equiv':
      'f(T_g \\, x) = T_g^\'\\, f(x)\\quad\\text{(equivariant)},\\qquad f(T_g \\, x) = f(x)\\quad\\text{(invariant)}',
    'math-gcnn':
      '\\bigl[\\,(f \\star \\psi)\\,\\bigr](g) = \\sum_{h \\in G} f(h)\\,\\psi(g^{-1} h)\\quad\\text{for any group element }g \\in G'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}
function boot() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wire();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
