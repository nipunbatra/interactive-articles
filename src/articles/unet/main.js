// ============================================================
// U-Net — section by section.
// We synthesise an input image, run a 4-level U-Net forward pass with
// hand-crafted convolution kernels (so the output is interpretable),
// and let the user toggle each skip connection.
// ============================================================

const SIZE = 64;
const POST_COLOR = '#1e7770';
const LIK_COLOR = '#d9622b';

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  return ctx;
}

// ---------- Synthesise input ----------
function makeImage() {
  const N = SIZE;
  const img = new Float32Array(N * N * 3);
  // background noise
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const k = (y * N + x) * 3;
      img[k] = 0.05 + 0.05 * Math.random();
      img[k + 1] = 0.05 + 0.05 * Math.random();
      img[k + 2] = 0.05 + 0.05 * Math.random();
    }
  }
  // ellipse 1 (red-ish)
  const e1 = {
    cx: 18 + Math.random() * 14, cy: 18 + Math.random() * 14,
    rx: 10 + Math.random() * 4, ry: 14 + Math.random() * 4
  };
  // ellipse 2 (blue-ish)
  const e2 = {
    cx: 38 + Math.random() * 14, cy: 32 + Math.random() * 14,
    rx: 12 + Math.random() * 4, ry: 8 + Math.random() * 4
  };
  const target = new Float32Array(N * N); // 0/1/2 labels
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const k = (y * N + x) * 3;
      const t = y * N + x;
      const d1 = ((x - e1.cx) ** 2) / (e1.rx ** 2) + ((y - e1.cy) ** 2) / (e1.ry ** 2);
      const d2 = ((x - e2.cx) ** 2) / (e2.rx ** 2) + ((y - e2.cy) ** 2) / (e2.ry ** 2);
      if (d1 < 1) {
        img[k] = 0.85 + 0.05 * Math.random();
        img[k + 1] = 0.25 + 0.05 * Math.random();
        img[k + 2] = 0.20 + 0.05 * Math.random();
        target[t] = 1;
      } else if (d2 < 1) {
        img[k] = 0.20 + 0.05 * Math.random();
        img[k + 1] = 0.40 + 0.05 * Math.random();
        img[k + 2] = 0.85 + 0.05 * Math.random();
        target[t] = 2;
      }
    }
  }
  return { img, target };
}

// ---------- Naive ops on Float32Array images (single-channel) ----------
function rgbToY(rgb, N) {
  // Convert HxWx3 -> HxW (luminance)
  const out = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) {
    const r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    out[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return out;
}

function downsample2x(img, N) {
  const M = N / 2;
  const out = new Float32Array(M * M);
  for (let y = 0; y < M; y++) {
    for (let x = 0; x < M; x++) {
      let s = 0;
      for (let dy = 0; dy < 2; dy++)
        for (let dx = 0; dx < 2; dx++)
          s += img[(2 * y + dy) * N + (2 * x + dx)];
      out[y * M + x] = s / 4;
    }
  }
  return out;
}

function upsample2x(img, M) {
  const N = M * 2;
  const out = new Float32Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      out[y * N + x] = img[Math.floor(y / 2) * M + Math.floor(x / 2)];
    }
  }
  return out;
}

function conv3(img, N, kernel) {
  const out = new Float32Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = Math.max(0, Math.min(N - 1, y + dy));
          const xx = Math.max(0, Math.min(N - 1, x + dx));
          s += img[yy * N + xx] * kernel[(dy + 1) * 3 + (dx + 1)];
        }
      }
      out[y * N + x] = s;
    }
  }
  return out;
}

function relu(img) {
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = Math.max(0, img[i]);
  return out;
}

function sigmoid(img) {
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = 1 / (1 + Math.exp(-img[i]));
  return out;
}

function add(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function subAbs(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.abs(a[i] - b[i]);
  return out;
}

// Hand-tuned kernels: edge detector and blob detector
const KERNEL_EDGE = [
  -1, -1, -1,
  -1,  8, -1,
  -1, -1, -1
];
const KERNEL_BLOB = [
   1, 1, 1,
   1, 1, 1,
   1, 1, 1
].map((v) => v / 9);

// ---------- Forward through "U-Net" ----------
const STATE = {
  image: null,
  feats: {},
  skips: { 1: true, 2: true, 3: true }
};

function forwardUnet() {
  if (!STATE.image) STATE.image = makeImage();
  const N = SIZE;
  const x0 = rgbToY(STATE.image.img, N);
  // Encoder: edge-then-downsample.
  const enc1 = relu(conv3(x0, N, KERNEL_EDGE)); // 64x64
  const e1d = downsample2x(enc1, N);            // 32x32
  const enc2 = relu(conv3(e1d, N / 2, KERNEL_EDGE)); // 32x32
  const e2d = downsample2x(enc2, N / 2);        // 16x16
  const enc3 = relu(conv3(e2d, N / 4, KERNEL_BLOB));
  const e3d = downsample2x(enc3, N / 4);        // 8x8
  const bot = relu(conv3(e3d, N / 8, KERNEL_BLOB));
  // Decoder: upsample, skip-add (concatenation in spirit), edge-detect
  const up3 = upsample2x(bot, N / 8);            // 16x16
  const dec3in = STATE.skips[3] ? add(up3, scaleArr(enc3, 0.7)) : up3;
  const dec3 = relu(conv3(dec3in, N / 4, KERNEL_BLOB));
  const up2 = upsample2x(dec3, N / 4);           // 32x32
  const dec2in = STATE.skips[2] ? add(up2, scaleArr(enc2, 0.7)) : up2;
  const dec2 = relu(conv3(dec2in, N / 2, KERNEL_EDGE));
  const up1 = upsample2x(dec2, N / 2);           // 64x64
  const dec1in = STATE.skips[1] ? add(up1, scaleArr(enc1, 0.7)) : up1;
  const dec1 = relu(conv3(dec1in, N, KERNEL_EDGE));
  // Output: turn into a 2-class (foreground vs background) probability via sigmoid
  // Bias toward bright regions of input
  const xLum = x0;
  const logits = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) logits[i] = (dec1[i] + 0.6 * xLum[i] - 0.18) * 6;
  const out = sigmoid(logits);
  // Multi-class: derive ellipse 1 vs ellipse 2 from RGB of input as a colour cue
  // We make the prediction a 3-class: bg/red/blue based on input colour gated by the
  // unet's foreground mask
  const N2 = N * N;
  const pred = new Float32Array(N2 * 3);
  for (let i = 0; i < N2; i++) {
    const fg = out[i];
    const r = STATE.image.img[i * 3];
    const g = STATE.image.img[i * 3 + 1];
    const b = STATE.image.img[i * 3 + 2];
    // bg
    pred[i * 3] = (1 - fg);
    // red ellipse
    pred[i * 3 + 1] = fg * Math.max(0, r - g - 0.05);
    // blue ellipse
    pred[i * 3 + 2] = fg * Math.max(0, b - g - 0.05);
    // normalise to [0,1] each
    const sum = pred[i * 3] + pred[i * 3 + 1] + pred[i * 3 + 2] + 1e-6;
    pred[i * 3] /= sum; pred[i * 3 + 1] /= sum; pred[i * 3 + 2] /= sum;
  }
  STATE.feats = { x0, enc1, e1d, enc2, e2d, enc3, e3d, bot, up3, dec3, up2, dec2, up1, dec1, out, pred };
  return STATE.feats;
}

function scaleArr(a, k) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * k;
  return out;
}

// ---------- Render helpers ----------
function drawSingleChannel(canvas, arr, N) {
  const ctx = setupCanvas(canvas, 128, 128);
  // Find min/max
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = Math.max(1e-6, hi - lo);
  const cell = 128 / N;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const t = (arr[y * N + x] - lo) / range;
      const r = Math.round(253 - 70 * t);
      const g = Math.round(252 - 100 * t);
      const b = Math.round(249 - 130 * t);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, 128, 128);
}

function drawRGB(canvas, rgb, N, sizePx = 128) {
  const ctx = setupCanvas(canvas, sizePx, sizePx);
  const cell = sizePx / N;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = (y * N + x) * 3;
      const r = Math.round(255 * Math.min(1, Math.max(0, rgb[i])));
      const g = Math.round(255 * Math.min(1, Math.max(0, rgb[i + 1])));
      const b = Math.round(255 * Math.min(1, Math.max(0, rgb[i + 2])));
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, sizePx, sizePx);
}

function drawTargetMask(canvas, target, N, sizePx = 128) {
  const ctx = setupCanvas(canvas, sizePx, sizePx);
  const cell = sizePx / N;
  const colors = [[253, 252, 249], [217, 98, 43], [44, 111, 183]];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const c = target[y * N + x];
      const [r, g, b] = colors[c | 0];
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
    }
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, sizePx, sizePx);
}

function drawCompareDiff(canvas, predRGB, target, N, sizePx = 256) {
  const ctx = setupCanvas(canvas, sizePx, sizePx);
  const cell = sizePx / N;
  const colors = [[253, 252, 249], [217, 98, 43], [44, 111, 183]];
  let sse = 0, count = 0;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = y * N + x;
      const [tr, tg, tb] = colors[target[i] | 0];
      const trN = tr / 255, tgN = tg / 255, tbN = tb / 255;
      const pr = predRGB[i * 3], pg = predRGB[i * 3 + 1], pb = predRGB[i * 3 + 2];
      const eR = (pr - trN), eG = (pg - tgN), eB = (pb - tbN);
      const e = (Math.abs(eR) + Math.abs(eG) + Math.abs(eB)) / 3;
      sse += e * e; count++;
      const t = Math.min(1, e * 1.6);
      ctx.fillStyle = `rgba(217, 98, 43, ${0.05 + 0.85 * t})`;
      ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
    }
  }
  const rmse = Math.sqrt(sse / count);
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, sizePx, sizePx);
  return rmse;
}

// ---------- Step 1: Architecture diagram ----------
function renderArchCanvas() {
  const canvas = document.getElementById('arch-canvas');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  // Layout: 4 down + bottleneck + 4 up
  const levels = 5; // input + 3 enc + bottleneck (we draw 5 down boxes for input + enc1..enc3 + bot)
  const xs = [80, 200, 320, 440, 560, 680, 800];
  const sizes = [64, 32, 16, 8, 4]; // down spatial
  const channelsDown = [3, 16, 32, 64, 128];
  // Top half (encoder) on left, bottom (decoder) mirroring
  const baseY = 60;
  const drawBox = (cx, cy, w, h, color, label) => {
    ctx.fillStyle = color;
    ctx.fillRect(cx - w / 2, cy - h / 2, w, h);
    ctx.strokeStyle = '#1a1815';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(cx - w / 2, cy - h / 2, w, h);
    if (label) {
      ctx.fillStyle = '#1a1815';
      ctx.font = '10.5px IBM Plex Mono';
      ctx.textAlign = 'center';
      ctx.fillText(label.size, cx, cy - h / 2 - 6);
      ctx.fillStyle = '#9a917f';
      ctx.fillText(label.ch, cx, cy + h / 2 + 14);
    }
  };
  // Encoder boxes
  for (let i = 0; i < 5; i++) {
    const w = 12 + (5 - i) * 14;
    const h = 12 + (5 - i) * 14;
    const cy = baseY + (5 - i) * 16;
    const cx = xs[i];
    drawBox(cx, cy, w, h, 'rgba(44,111,183,0.25)', { size: `${sizes[i]}×${sizes[i]}`, ch: `${channelsDown[i]}` });
  }
  // Decoder boxes (mirror)
  for (let i = 0; i < 4; i++) {
    const idx = 4 - 1 - i;
    const w = 12 + (5 - idx) * 14;
    const h = 12 + (5 - idx) * 14;
    const cy = baseY + (5 - idx) * 16;
    const cx = xs[5 + i];
    drawBox(cx, cy, w, h, 'rgba(30,119,112,0.25)', { size: `${sizes[idx]}×${sizes[idx]}`, ch: `${channelsDown[idx]}` });
  }
  // Bottleneck arrow text
  ctx.fillStyle = '#1a1815';
  ctx.font = '13px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('encoder ↓', xs[2], H - 30);
  ctx.fillText('bottleneck', xs[4], H - 30);
  ctx.fillText('decoder ↑', xs[5] + (xs[6] - xs[5]) / 2, H - 30);
  // Skip arrows
  ctx.strokeStyle = '#d9622b';
  ctx.lineWidth = 1.6;
  ctx.setLineDash([4, 4]);
  for (let i = 1; i <= 3; i++) {
    const fromX = xs[i];
    const toX = xs[8 - i];
    const y = baseY + (5 - i) * 16 + (12 + (5 - i) * 14) / 2 - 4;
    ctx.beginPath();
    ctx.moveTo(fromX + 28, y - 24);
    ctx.bezierCurveTo(fromX + 60, y - 100, toX - 60, y - 100, toX - 28, y - 24);
    ctx.stroke();
    ctx.fillStyle = '#9c3f15';
    ctx.font = '11px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText(`skip ${4 - i}`, (fromX + toX) / 2, y - 80);
  }
  ctx.setLineDash([]);
}

// ---------- Step 2: Forward feature thumbnails ----------
function renderFeatures() {
  const F = forwardUnet();
  drawRGB(document.getElementById('feat-input'), STATE.image.img, SIZE);
  drawSingleChannel(document.getElementById('feat-enc1'), F.e1d, SIZE / 2);
  drawSingleChannel(document.getElementById('feat-enc2'), F.e2d, SIZE / 4);
  drawSingleChannel(document.getElementById('feat-enc3'), F.e3d, SIZE / 8);
  drawSingleChannel(document.getElementById('feat-bot'), F.bot, SIZE / 8);
  drawSingleChannel(document.getElementById('feat-dec3'), F.dec3, SIZE / 4);
  drawSingleChannel(document.getElementById('feat-dec2'), F.dec2, SIZE / 2);
  drawSingleChannel(document.getElementById('feat-dec1'), F.dec1, SIZE);
  drawRGB(document.getElementById('feat-out'), F.pred, SIZE);
  drawTargetMask(document.getElementById('feat-tgt'), STATE.image.target, SIZE);
  document.getElementById('bottleneck-res').textContent = '4×4';
}

// ---------- Step 3: Compare with vs without skips ----------
function renderCompare() {
  const F = forwardUnet();
  drawRGB(document.getElementById('compare-pred'), F.pred, SIZE, 256);
  drawTargetMask(document.getElementById('compare-tgt'), STATE.image.target, SIZE, 256);
  const rmse = drawCompareDiff(document.getElementById('compare-diff'), F.pred, STATE.image.target, SIZE, 256);
  document.getElementById('rmse-label').textContent = `RMSE = ${rmse.toFixed(3)}`;
}

// ---------- Wire ----------
function wireUNet() {
  document.getElementById('re-roll').addEventListener('click', () => {
    STATE.image = makeImage();
    renderFeatures();
    renderCompare();
  });
  ['skip-1', 'skip-2', 'skip-3'].forEach((id) => {
    const i = parseInt(id.slice(-1), 10);
    document.getElementById(id).addEventListener('change', (e) => {
      STATE.skips[i] = e.target.checked;
      renderFeatures();
      renderCompare();
    });
  });
}

// ---------- Math + boot ----------
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-cat':
      'h_{\\text{dec}}^{(\\ell)} = \\mathrm{Conv}\\!\\left(\\bigl[\\mathrm{Up}(h_{\\text{dec}}^{(\\ell+1)});\\;\\;h_{\\text{enc}}^{(\\ell)}\\bigr]\\right)'
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
  renderArchCanvas();
  STATE.image = makeImage();
  renderFeatures();
  renderCompare();
  wireUNet();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
