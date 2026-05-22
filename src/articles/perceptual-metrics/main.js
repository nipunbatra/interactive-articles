// ============================================================
// Perceptual quality metrics: live PSNR / SSIM / perceptual-proxy
// on a hand-crafted reference image, distorted on demand.
//
// All math is single-channel grayscale in [0, 1]. The reference
// image is a 64x64 synthetic test card with a smooth gradient,
// a filled circle, a 4x4 checker patch, and a striped band — so
// that distortions of different *characters* (energy vs structure
// vs scale) are visually distinguishable.
//
// "Perceptual proxy" is a multi-scale Sobel-gradient L2 distance.
// It is not LPIPS; the article is explicit about that. It exists
// to show the *flavour* of feature-space distance without needing
// to ship a pretrained CNN in the browser.
// ============================================================

const SIZE = 64;
const L_RANGE = 1.0; // dynamic range

// ---------------- Reference image ----------------
function makeReference() {
  const a = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      // smooth diagonal gradient base
      let v = 0.20 + 0.55 * ((x + y) / (2 * SIZE - 2));
      // filled circle in upper-left quadrant
      const cx = 18, cy = 18, r = 9;
      if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r) v = 0.92;
      // 4x4 checker patch lower-left
      if (x >= 6 && x < 26 && y >= 38 && y < 58) {
        const bx = Math.floor((x - 6) / 5);
        const by = Math.floor((y - 38) / 5);
        v = ((bx + by) % 2) === 0 ? 0.10 : 0.85;
      }
      // horizontal striped band right side
      if (x >= 38 && x < 60 && y >= 10 && y < 54) {
        const s = Math.floor((y - 10) / 2);
        v = (s % 2) === 0 ? 0.30 : 0.78;
      }
      a[y * SIZE + x] = Math.max(0, Math.min(1, v));
    }
  }
  return a;
}

// ---------------- Random ----------------
let seed = 0xC0FFEE;
function rand() {
  // mulberry32 — deterministic so repeated runs match
  seed = (seed + 0x6D2B79F5) | 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function randn() {
  const u = Math.max(rand(), 1e-12), v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------------- Distortions ----------------
function addNoise(img, sigma) {
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) {
    out[i] = clamp01(img[i] + sigma * randn());
  }
  return out;
}

function gaussianBlur(img, sigma) {
  if (sigma <= 1e-3) return new Float32Array(img);
  // separable 1D Gaussian with radius = ceil(3 sigma)
  const r = Math.max(1, Math.ceil(3 * sigma));
  const k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + r] = v;
    s += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= s;

  // horizontal pass
  const tmp = new Float32Array(img.length);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      let v = 0;
      for (let i = -r; i <= r; i++) {
        const xx = Math.max(0, Math.min(SIZE - 1, x + i));
        v += img[y * SIZE + xx] * k[i + r];
      }
      tmp[y * SIZE + x] = v;
    }
  }
  // vertical pass
  const out = new Float32Array(img.length);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      let v = 0;
      for (let i = -r; i <= r; i++) {
        const yy = Math.max(0, Math.min(SIZE - 1, y + i));
        v += tmp[yy * SIZE + x] * k[i + r];
      }
      out[y * SIZE + x] = clamp01(v);
    }
  }
  return out;
}

function brightnessShift(img, delta) {
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = clamp01(img[i] + delta);
  return out;
}

function contrastScale(img, alpha) {
  // y = 0.5 + alpha * (x - 0.5) — keeps mid-gray fixed
  const out = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) out[i] = clamp01(0.5 + alpha * (img[i] - 0.5));
  return out;
}

function jpegBlock(img, block) {
  // Block-mean quantisation: a cartoon stand-in for JPEG DCT loss.
  // Real JPEG keeps low-frequency DCT terms; we keep only the DC term.
  const b = Math.max(1, Math.round(block));
  const out = new Float32Array(img.length);
  for (let by = 0; by < SIZE; by += b) {
    for (let bx = 0; bx < SIZE; bx += b) {
      let s = 0, n = 0;
      for (let yy = by; yy < Math.min(SIZE, by + b); yy++) {
        for (let xx = bx; xx < Math.min(SIZE, bx + b); xx++) {
          s += img[yy * SIZE + xx]; n++;
        }
      }
      const m = s / n;
      for (let yy = by; yy < Math.min(SIZE, by + b); yy++) {
        for (let xx = bx; xx < Math.min(SIZE, bx + b); xx++) {
          out[yy * SIZE + xx] = m;
        }
      }
    }
  }
  return out;
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

// ---------------- Metrics ----------------
function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s / a.length;
}

function psnr(a, b) {
  const m = mse(a, b);
  if (m < 1e-12) return 99;
  return 10 * Math.log10((L_RANGE * L_RANGE) / m);
}

// SSIM with sliding 8x8 box-filter window (computed via integral images
// for moments — O(N), so this is cheap even for 64x64).
function ssim(a, b, win = 8) {
  const C1 = (0.01 * L_RANGE) ** 2;
  const C2 = (0.03 * L_RANGE) ** 2;
  const N = SIZE * SIZE;

  // Pad with reflection so windows at the borders still get full count.
  const pad = Math.floor(win / 2);
  const padSize = SIZE + 2 * pad;
  const padded = (arr) => {
    const p = new Float32Array(padSize * padSize);
    for (let y = 0; y < padSize; y++) {
      const sy = Math.max(0, Math.min(SIZE - 1, y - pad));
      for (let x = 0; x < padSize; x++) {
        const sx = Math.max(0, Math.min(SIZE - 1, x - pad));
        p[y * padSize + x] = arr[sy * SIZE + sx];
      }
    }
    return p;
  };
  const ap = padded(a), bp = padded(b);

  // Integral images for: a, b, a^2, b^2, a*b
  function integral(arr, fn) {
    const out = new Float64Array((padSize + 1) * (padSize + 1));
    for (let y = 0; y < padSize; y++) {
      let row = 0;
      for (let x = 0; x < padSize; x++) {
        row += fn(arr[y * padSize + x]);
        out[(y + 1) * (padSize + 1) + (x + 1)] = out[y * (padSize + 1) + (x + 1)] + row;
      }
    }
    return out;
  }
  const Ia = integral(ap, (v) => v);
  const Ib = integral(bp, (v) => v);
  const Iaa = integral(ap, (v) => v * v);
  const Ibb = integral(bp, (v) => v * v);
  // For Iab we need ap*bp pointwise. Make a fused array.
  const ab = new Float32Array(padSize * padSize);
  for (let i = 0; i < ab.length; i++) ab[i] = ap[i] * bp[i];
  const Iab = integral(ab, (v) => v);

  const w = win, w2 = w * w;
  function box(I, y0, x0) {
    // sum over [y0, y0+w) x [x0, x0+w)
    return I[(y0 + w) * (padSize + 1) + (x0 + w)]
         - I[y0 * (padSize + 1) + (x0 + w)]
         - I[(y0 + w) * (padSize + 1) + x0]
         + I[y0 * (padSize + 1) + x0];
  }

  let sum = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const sA = box(Ia, y, x);
      const sB = box(Ib, y, x);
      const sAA = box(Iaa, y, x);
      const sBB = box(Ibb, y, x);
      const sAB = box(Iab, y, x);
      const muA = sA / w2;
      const muB = sB / w2;
      const sigA2 = Math.max(0, sAA / w2 - muA * muA);
      const sigB2 = Math.max(0, sBB / w2 - muB * muB);
      const sigAB = sAB / w2 - muA * muB;
      const num = (2 * muA * muB + C1) * (2 * sigAB + C2);
      const den = (muA * muA + muB * muB + C1) * (sigA2 + sigB2 + C2);
      sum += num / den;
    }
  }
  return sum / N;
}

// Multi-scale gradient-feature distance: stand-in for LPIPS.
// At each of three scales (full, /2, /4) compute Sobel gx, gy on
// each image; L2-normalise per-channel; accumulate per-pixel L2
// distance. Average over pixels and scales.
function perceptualProxy(a, b) {
  const scales = [1, 2, 4];
  let acc = 0;
  for (const s of scales) {
    const ad = downsample(a, s);
    const bd = downsample(b, s);
    const sz = SIZE / s;
    const ag = sobel(ad, sz);
    const bg = sobel(bd, sz);
    // Normalise each gradient field by its RMS so the metric isn't dominated by absolute brightness.
    const an = normalise(ag);
    const bn = normalise(bg);
    let d = 0;
    for (let i = 0; i < an.length; i++) {
      const dx = an[i].x - bn[i].x;
      const dy = an[i].y - bn[i].y;
      d += dx * dx + dy * dy;
    }
    acc += d / an.length;
  }
  return acc / scales.length;
}

function downsample(img, factor) {
  if (factor === 1) return img;
  const sz = Math.floor(SIZE / factor);
  const out = new Float32Array(sz * sz);
  for (let y = 0; y < sz; y++) {
    for (let x = 0; x < sz; x++) {
      let s = 0;
      for (let dy = 0; dy < factor; dy++) {
        for (let dx = 0; dx < factor; dx++) {
          s += img[(y * factor + dy) * SIZE + (x * factor + dx)];
        }
      }
      out[y * sz + x] = s / (factor * factor);
    }
  }
  return out;
}

function sobel(img, sz) {
  const out = new Array(sz * sz);
  const Kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const Ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 0; y < sz; y++) {
    for (let x = 0; x < sz; x++) {
      let gx = 0, gy = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = Math.max(0, Math.min(sz - 1, x + dx));
          const yy = Math.max(0, Math.min(sz - 1, y + dy));
          const v = img[yy * sz + xx];
          const ki = (dy + 1) * 3 + (dx + 1);
          gx += v * Kx[ki];
          gy += v * Ky[ki];
        }
      }
      out[y * sz + x] = { x: gx, y: gy };
    }
  }
  return out;
}

function normalise(g) {
  let s = 0;
  for (const v of g) s += v.x * v.x + v.y * v.y;
  const rms = Math.sqrt(s / g.length) + 1e-6;
  return g.map((v) => ({ x: v.x / rms, y: v.y / rms }));
}

// ---------------- Render ----------------
function drawImageScalar(canvas, arr, size = SIZE) {
  const dpr = window.devicePixelRatio || 1;
  const cssSize = canvas.dataset.size ? parseInt(canvas.dataset.size, 10) : canvas.clientWidth || 256;
  canvas.width = cssSize * dpr;
  canvas.height = cssSize * dpr;
  canvas.style.width = '';
  canvas.style.height = '';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  const cell = cssSize / size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = arr[y * size + x];
      const c = Math.round(255 * Math.max(0, Math.min(1, v)));
      ctx.fillStyle = `rgb(${c}, ${c}, ${c})`;
      ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
    }
  }
}

// ---------------- State and wiring ----------------
const STATE = {
  ref: null,
  mode: 'noise',
  strength: 35  // 0..100
};

const MODE_LABELS = {
  noise: 'noise std σ',
  blur: 'blur σ (pixels)',
  shift: 'brightness shift δ',
  contrast: 'contrast factor α',
  jpeg: 'block size (pixels)'
};

function paramFromStrength(mode, s) {
  // s in [0, 100]
  const t = s / 100;
  switch (mode) {
    case 'noise':    return t * 0.30;            // 0..0.30
    case 'blur':     return t * 3.0;             // 0..3.0 px sigma
    case 'shift':    return t * 0.40 - 0.0;      // 0..0.40
    case 'contrast': return 1 - t * 0.85;        // 1.0 down to ~0.15
    case 'jpeg':     return 1 + Math.round(t * 15); // block 1..16
  }
  return 0;
}

function applyDistortion(img, mode, param) {
  // Reset noise seed before each render so the noise pattern is
  // stable for a given strength — comparisons across reloads stay sane.
  seed = 0xC0FFEE;
  switch (mode) {
    case 'noise':    return addNoise(img, param);
    case 'blur':     return gaussianBlur(img, param);
    case 'shift':    return brightnessShift(img, param);
    case 'contrast': return contrastScale(img, param);
    case 'jpeg':     return jpegBlock(img, param);
  }
  return img;
}

function fmt(v, digits = 3) {
  if (!isFinite(v)) return '∞';
  return v.toFixed(digits);
}

function setMetricBadge(card, level) {
  card.classList.remove('is-good', 'is-warn', 'is-bad');
  if (level) card.classList.add(level);
}

function metricLevels(mseVal, psnrVal, ssimVal) {
  const psnrLevel = psnrVal >= 35 ? 'is-good' : psnrVal >= 25 ? 'is-warn' : 'is-bad';
  const ssimLevel = ssimVal >= 0.9 ? 'is-good' : ssimVal >= 0.6 ? 'is-warn' : 'is-bad';
  return { psnrLevel, ssimLevel };
}

function renderPlayground() {
  const param = paramFromStrength(STATE.mode, STATE.strength);
  const distorted = applyDistortion(STATE.ref, STATE.mode, param);

  drawImageScalar(document.getElementById('pm-ref'), STATE.ref);
  drawImageScalar(document.getElementById('pm-dist'), distorted);

  const mseVal = mse(STATE.ref, distorted);
  const psnrVal = psnr(STATE.ref, distorted);
  const ssimVal = ssim(STATE.ref, distorted);
  const lpipsVal = perceptualProxy(STATE.ref, distorted);

  document.getElementById('pm-mse').textContent = fmt(mseVal, 4);
  document.getElementById('pm-psnr').textContent = `${fmt(psnrVal, 2)} dB`;
  document.getElementById('pm-ssim').textContent = fmt(ssimVal, 3);
  document.getElementById('pm-lpips').textContent = fmt(lpipsVal, 3);

  const cards = document.querySelectorAll('#pm-metrics .metric-card');
  const { psnrLevel, ssimLevel } = metricLevels(mseVal, psnrVal, ssimVal);
  setMetricBadge(cards[1], psnrLevel);
  setMetricBadge(cards[2], ssimLevel);

  // Update sub-label
  const label = MODE_LABELS[STATE.mode];
  document.getElementById('pm-ctrl-label').textContent = label;
  document.getElementById('pm-ctrl-num').textContent = STATE.mode === 'jpeg' ? `${param}` : fmt(param, 2);
  document.getElementById('pm-dist-sub').textContent = `${label} = ${STATE.mode === 'jpeg' ? param : fmt(param, 2)}`;
}

// ---------------- Equal-PSNR gallery ----------------
// Calibrate one strength per distortion type so PSNR sits near a target.
function calibrateForTargetPSNR(ref, distortFn, lo, hi, target = 28.0, iters = 14) {
  // Monotone-ish in strength → binary search.
  let l = lo, h = hi;
  let best = null;
  for (let k = 0; k < iters; k++) {
    const mid = (l + h) / 2;
    const d = distortFn(ref, mid);
    const p = psnr(ref, d);
    if (best === null || Math.abs(p - target) < Math.abs(best.p - target)) {
      best = { mid, p, d };
    }
    if (p > target) {
      // distortion too weak (too high PSNR) → increase strength
      l = mid;
    } else {
      h = mid;
    }
  }
  return best;
}

function blendImages(a, b, alpha) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (1 - alpha) * a[i] + alpha * b[i];
  return out;
}

function buildGallery(ref) {
  const target = 28.0;
  // The JPEG-like block distortion at this image size is discrete in
  // block-size (block=1 is identity, block=2 already lands near 22 dB).
  // To hit an arbitrary PSNR we alpha-blend the fully block-quantised
  // image with the reference — still recognisably "blocky", but the
  // strength is continuous.
  const fullyBlocky = jpegBlock(ref, 4);

  const items = [
    { id: 0, label: 'reference', dist: new Float32Array(ref) },
    {
      id: 1, label: 'brightness shift',
      dist: calibrateForTargetPSNR(ref, (r, m) => brightnessShift(r, m), 0.0, 0.5, target).d
    },
    {
      id: 2, label: 'contrast scale',
      dist: calibrateForTargetPSNR(ref, (r, m) => contrastScale(r, 1 - m), 0.0, 0.8, target).d
    },
    {
      id: 3, label: 'Gaussian noise',
      dist: (() => {
        seed = 0xBADF00D;
        return calibrateForTargetPSNR(ref, (r, m) => { seed = 0xBADF00D; return addNoise(r, m); }, 0.001, 0.25, target).d;
      })()
    },
    {
      id: 4, label: 'Gaussian blur',
      dist: calibrateForTargetPSNR(ref, (r, m) => gaussianBlur(r, m), 0.05, 3.0, target).d
    },
    {
      id: 5, label: 'JPEG block',
      dist: calibrateForTargetPSNR(ref, (r, alpha) => blendImages(r, fullyBlocky, alpha), 0.0, 1.0, target).d
    }
  ];

  for (const it of items) {
    const canvas = document.getElementById(`pm-card-${it.id}`);
    canvas.dataset.size = '128';
    drawImageScalar(canvas, it.dist);
    const p = psnr(ref, it.dist);
    const s = ssim(ref, it.dist);
    const lp = perceptualProxy(ref, it.dist);
    const stats = document.getElementById(`pm-stats-${it.id}`);
    stats.innerHTML = it.id === 0
      ? '<strong>identity</strong>'
      : `<span>PSNR <strong>${fmt(p, 1)} dB</strong></span>
         <span>SSIM <strong>${fmt(s, 3)}</strong></span>
         <span>perc <strong>${fmt(lp, 3)}</strong></span>`;
  }
}

// ---------------- Wiring ----------------
function wire() {
  STATE.ref = makeReference();
  // Make reference / distorted canvases sized.
  document.getElementById('pm-ref').dataset.size = '256';
  document.getElementById('pm-dist').dataset.size = '256';

  document.getElementById('pm-mode').addEventListener('change', (e) => {
    STATE.mode = e.target.value;
    renderPlayground();
  });
  document.getElementById('pm-strength').addEventListener('input', (e) => {
    STATE.strength = parseFloat(e.target.value);
    renderPlayground();
  });

  renderPlayground();
  buildGallery(STATE.ref);
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-mse':
      '\\mathrm{MSE}(x, y) = \\frac{1}{N}\\sum_{i=1}^{N} (x_i - y_i)^2',
    'math-psnr':
      '\\mathrm{PSNR}(x, y) = 10 \\log_{10}\\!\\left( \\frac{L^2}{\\mathrm{MSE}(x, y)} \\right)',
    'math-ssim-l':
      '\\text{luminance: } l(x, y) = \\frac{2\\mu_x \\mu_y + C_1}{\\mu_x^2 + \\mu_y^2 + C_1}',
    'math-ssim-c':
      '\\text{contrast: } c(x, y) = \\frac{2\\sigma_x \\sigma_y + C_2}{\\sigma_x^2 + \\sigma_y^2 + C_2}',
    'math-ssim-s':
      '\\text{structure: } s(x, y) = \\frac{\\sigma_{xy} + C_3}{\\sigma_x \\sigma_y + C_3}',
    'math-ssim':
      '\\mathrm{SSIM}(x, y) = \\frac{(2\\mu_x \\mu_y + C_1)(2\\sigma_{xy} + C_2)}{(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}',
    'math-msssim':
      '\\mathrm{MS\\text{-}SSIM}(x, y) = [l_M(x, y)]^{\\alpha} \\prod_{j=1}^{M} [c_j(x, y)]^{\\beta_j} [s_j(x, y)]^{\\gamma_j}',
    'math-lpips':
      '\\mathrm{LPIPS}(x, y) = \\sum_{\\ell} \\frac{1}{H_\\ell W_\\ell} \\sum_{h, w} \\| w_\\ell \\odot (\\hat{\\phi}_{hw}^{(\\ell)}(x) - \\hat{\\phi}_{hw}^{(\\ell)}(y)) \\|_2^2',
    'math-fid':
      '\\mathrm{FID}(P_r, P_g) = \\| \\mu_r - \\mu_g \\|_2^2 + \\mathrm{Tr}\\!\\left( \\Sigma_r + \\Sigma_g - 2(\\Sigma_r \\Sigma_g)^{1/2} \\right)'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

function boot() {
  wire();
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
