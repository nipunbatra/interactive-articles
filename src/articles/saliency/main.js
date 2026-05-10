// ============================================================
// Saliency: side-by-side gradient / SmoothGrad / IG / Grad-CAM on
// a hand-crafted tiny CNN.
// CNN: input 32x32 (single channel), 3x3 conv with 4 hand-picked
// kernels (blob, edge, ring, sym), ReLU, global-avg-pool over 4
// channels, linear head W: 4 -> 1 producing class logit.
// All gradients computed analytically.
// ============================================================

const SIZE = 32;

const KERNELS = [
  // blob
  [ 0.5, 1, 0.5, 1, 1.5, 1, 0.5, 1, 0.5 ],
  // edge horizontal
  [ -1, -1, -1, 0, 0, 0, 1, 1, 1 ],
  // ring (positive surround, negative center)
  [ 0.5, 0.5, 0.5, 0.5, -2, 0.5, 0.5, 0.5, 0.5 ],
  // sym diagonal
  [ -1, 0, 1, 0, 0, 0, 1, 0, -1 ]
];
const HEAD = [1.0, 0.4, 0.8, -0.2]; // weights from 4 channels to logit
const BIAS = -1.0;

const STATE = { img: null, kind: 'hot' };

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

function makeImage(kind) {
  const arr = new Float32Array(SIZE * SIZE);
  if (kind === 'hot') {
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const dx = x - 22, dy = y - 14;
      const d2 = dx * dx + dy * dy;
      arr[y * SIZE + x] = 0.15 + 0.85 * Math.exp(-d2 / 18) + 0.04 * randn();
    }
  } else if (kind === 'two') {
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const d1 = (x - 9) * (x - 9) + (y - 9) * (y - 9);
      const d2 = (x - 22) * (x - 22) + (y - 22) * (y - 22);
      const v = 0.20 + 0.85 * Math.exp(-d1 / 14) + 0.45 * Math.exp(-d2 / 14) + 0.04 * randn();
      arr[y * SIZE + x] = v;
    }
  } else { // striped
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const v = 0.4 + 0.4 * Math.sin(x * 0.7) + 0.04 * randn();
      arr[y * SIZE + x] = v;
    }
  }
  // Clamp 0..1
  for (let i = 0; i < arr.length; i++) arr[i] = Math.max(0, Math.min(1, arr[i]));
  return arr;
}

function relu(x) { return Math.max(0, x); }
function dRelu(x) { return x > 0 ? 1 : 0; }

function conv3(img, kernel) {
  // valid -> SIZE x SIZE with reflection padding
  const out = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = Math.max(0, Math.min(SIZE - 1, x + dx));
          const yy = Math.max(0, Math.min(SIZE - 1, y + dy));
          s += img[yy * SIZE + xx] * kernel[(dy + 1) * 3 + (dx + 1)];
        }
      }
      out[y * SIZE + x] = s;
    }
  }
  return out;
}

function forward(img) {
  // Returns: pre, activations, gap, logit
  const N = SIZE * SIZE;
  const channels = KERNELS.map((k) => conv3(img, k));
  const acts = channels.map((c) => {
    const a = new Float32Array(N);
    for (let i = 0; i < N; i++) a[i] = relu(c[i]);
    return a;
  });
  // Global average pool
  const gap = acts.map((a) => {
    let s = 0;
    for (let i = 0; i < N; i++) s += a[i];
    return s / N;
  });
  let logit = BIAS;
  for (let c = 0; c < KERNELS.length; c++) logit += HEAD[c] * gap[c];
  return { pre: channels, acts, gap, logit };
}

function backwardPixel(img) {
  // dlogit/dimg via analytic backprop through 1 conv layer
  // logit = bias + sum_c HEAD[c] * (1/N) * sum_p relu(pre_c[p])
  // dlogit/d pre_c[p] = HEAD[c] * (1/N) * dRelu(pre_c[p])
  // dlogit/d img[q] = sum_{c, p, kernel offset (dx,dy)} HEAD[c]*(1/N)*dRelu(pre_c[p])*kernel_c[dx,dy] * 1{p+offset = q}
  const N = SIZE * SIZE;
  const fwd = forward(img);
  const grad = new Float32Array(N);
  for (let c = 0; c < KERNELS.length; c++) {
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const p = y * SIZE + x;
        const dr = HEAD[c] * (1 / N) * dRelu(fwd.pre[c][p]);
        if (dr === 0) continue;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const xx = Math.max(0, Math.min(SIZE - 1, x + dx));
            const yy = Math.max(0, Math.min(SIZE - 1, y + dy));
            grad[yy * SIZE + xx] += dr * KERNELS[c][(dy + 1) * 3 + (dx + 1)];
          }
        }
      }
    }
  }
  return { grad, logit: fwd.logit, fwd };
}

function smoothGrad(img, n = 20, sigma = 0.15) {
  const N = SIZE * SIZE;
  const acc = new Float32Array(N);
  for (let t = 0; t < n; t++) {
    const noisy = new Float32Array(N);
    for (let i = 0; i < N; i++) noisy[i] = img[i] + sigma * randn();
    const { grad } = backwardPixel(noisy);
    for (let i = 0; i < N; i++) acc[i] += grad[i];
  }
  for (let i = 0; i < N; i++) acc[i] /= n;
  return acc;
}

function integratedGradients(img, baseline, steps = 30) {
  const N = SIZE * SIZE;
  if (!baseline) baseline = new Float32Array(N); // zeros
  const acc = new Float32Array(N);
  for (let s = 1; s <= steps; s++) {
    const alpha = s / steps;
    const interp = new Float32Array(N);
    for (let i = 0; i < N; i++) interp[i] = baseline[i] + alpha * (img[i] - baseline[i]);
    const { grad } = backwardPixel(interp);
    for (let i = 0; i < N; i++) acc[i] += grad[i];
  }
  // IG[i] = (img[i] - baseline[i]) * mean grad
  for (let i = 0; i < N; i++) acc[i] = (img[i] - baseline[i]) * acc[i] / steps;
  return acc;
}

function gradCAM(img) {
  // For our 1-layer CNN: grad-cam over the 4 channels at the conv layer.
  // weight per channel = HEAD[c] (since GAP gives constant grad).
  // Map = relu(sum_c weight_c * activation_c[p]).
  const fwd = forward(img);
  const N = SIZE * SIZE;
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let c = 0; c < KERNELS.length; c++) s += HEAD[c] * fwd.acts[c][i];
    out[i] = Math.max(0, s);
  }
  return out;
}

// ---------- Render helpers ----------
function drawScalar(canvas, arr, opts = {}) {
  const ctx = setupCanvas(canvas, 220, 220);
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = Math.max(1e-6, hi - lo);
  const cell = 220 / SIZE;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = arr[y * SIZE + x];
      const t = (v - lo) / range;
      let r, g, b;
      if (opts.diverge) {
        // red negative, blue positive
        const t0 = (v / Math.max(Math.abs(lo), Math.abs(hi), 1e-6) + 1) / 2;
        r = Math.round(217 + (44 - 217) * t0);
        g = Math.round(98 + (111 - 98) * t0);
        b = Math.round(43 + (183 - 43) * t0);
      } else if (opts.gray) {
        const c = Math.round(255 * t);
        r = c; g = c; b = c;
      } else {
        // heat: light to dark teal
        r = Math.round(253 - 200 * t);
        g = Math.round(252 - 130 * t);
        b = Math.round(249 - 100 * t);
      }
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, 220, 220);
}

// ---------- Render ----------
function renderAll() {
  STATE.img = makeImage(STATE.kind);
  drawScalar(document.getElementById('sa-input'), STATE.img, { gray: true });
  const { fwd, grad } = backwardPixel(STATE.img);
  const sg = smoothGrad(STATE.img);
  const ig = integratedGradients(STATE.img);
  const cam = gradCAM(STATE.img);
  drawScalar(document.getElementById('sa-grad'), grad, { diverge: true });
  drawScalar(document.getElementById('sa-smooth'), sg, { diverge: true });
  drawScalar(document.getElementById('sa-ig'), ig, { diverge: true });
  drawScalar(document.getElementById('sa-cam'), cam);
  // Ground truth = same as input but binarized
  const gt = new Float32Array(STATE.img.length);
  for (let i = 0; i < STATE.img.length; i++) gt[i] = STATE.img[i] > 0.7 ? 1 : 0;
  drawScalar(document.getElementById('sa-gt'), gt);

  document.getElementById('sa-score').textContent = `class logit = ${fwd.logit.toFixed(3)} (sigmoid p = ${(1 / (1 + Math.exp(-fwd.logit))).toFixed(2)})`;
}

function wire() {
  document.getElementById('sa-img').addEventListener('change', (e) => {
    STATE.kind = e.target.value;
    renderAll();
  });
  renderAll();
}

function boot() { wire(); }
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
