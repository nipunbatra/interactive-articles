// ============================================================
// Vision SSL — MAE-flavoured live demo.
// We synthesise a 64-patch grid (8x8) on a 224x224 canvas, mask a
// fraction of patches, and "reconstruct" the masked patches via a
// smart interpolation from visible patches (a stand-in for what an
// MAE decoder converges to on smooth images).
// ============================================================

const TILE_SIZE = 28; // 8x8 grid of 28-px patches
const N = 8;

const STATE = {
  image: null, // {pixels: Uint8ClampedArray} 224*224*3
  mask: null,  // boolean 8x8: true = masked (hidden)
  ratio: 0.75
};

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  return ctx;
}

function makeImage() {
  // 224x224 procedural "scene": gradient sky + sun + ground + a tree silhouette
  const W = 224, H = 224;
  const arr = new Uint8ClampedArray(W * H * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const horizon = 130;
      let r, g, b;
      if (y < horizon) {
        // sky gradient
        const t = y / horizon;
        r = Math.round(110 + 130 * t);
        g = Math.round(160 + 70 * t);
        b = Math.round(220 - 50 * t);
        // sun
        const dx = x - 175, dy = y - 50;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 24) {
          r = 255; g = 220; b = 130;
        } else if (dist < 38) {
          const m = 1 - (dist - 24) / 14;
          r = Math.min(255, r + 60 * m);
          g = Math.min(255, g + 50 * m);
          b = Math.min(255, b + 10 * m);
        }
      } else {
        // ground gradient
        const t = (y - horizon) / (H - horizon);
        r = Math.round(120 - 30 * t);
        g = Math.round(180 - 40 * t);
        b = Math.round(90 - 30 * t);
      }
      // tree silhouette
      const tx = x - 70, ty = y - 110;
      if (ty > 0 && ty < 80 && Math.abs(tx) < 8) {
        r = 70; g = 50; b = 30;
      }
      const dx = x - 70, dy = y - 95;
      if (Math.sqrt(dx * dx + (dy * 1.4) * (dy * 1.4)) < 32) {
        r = 50; g = 110; b = 60;
      }
      // small house
      if (x > 130 && x < 170 && y > 140 && y < 170) {
        r = 200; g = 150; b = 100;
      }
      if (x > 138 && x < 162 && y > 130 && y < 145) {
        r = 100; g = 60; b = 50;
      }
      const k = (y * W + x) * 3;
      arr[k] = r; arr[k + 1] = g; arr[k + 2] = b;
    }
  }
  return arr;
}

function makeMask(ratio) {
  const m = new Array(N * N).fill(false);
  const k = Math.round(ratio * N * N);
  const idx = [];
  for (let i = 0; i < N * N; i++) idx.push(i);
  // shuffle
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  for (let i = 0; i < k; i++) m[idx[i]] = true;
  return m;
}

function patchMean(arr, pi, pj) {
  let sumR = 0, sumG = 0, sumB = 0;
  const W = 224;
  for (let dy = 0; dy < TILE_SIZE; dy++) {
    for (let dx = 0; dx < TILE_SIZE; dx++) {
      const y = pi * TILE_SIZE + dy;
      const x = pj * TILE_SIZE + dx;
      const k = (y * W + x) * 3;
      sumR += arr[k]; sumG += arr[k + 1]; sumB += arr[k + 2];
    }
  }
  const denom = TILE_SIZE * TILE_SIZE;
  return [sumR / denom, sumG / denom, sumB / denom];
}

function drawArr(canvas, arr) {
  const ctx = setupCanvas(canvas, 224, 224);
  const im = ctx.createImageData(224, 224);
  for (let i = 0; i < 224 * 224; i++) {
    im.data[i * 4] = arr[i * 3];
    im.data[i * 4 + 1] = arr[i * 3 + 1];
    im.data[i * 4 + 2] = arr[i * 3 + 2];
    im.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  // patch grid
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < N; i++) {
    ctx.moveTo(0, i * TILE_SIZE); ctx.lineTo(224, i * TILE_SIZE);
    ctx.moveTo(i * TILE_SIZE, 0); ctx.lineTo(i * TILE_SIZE, 224);
  }
  ctx.stroke();
}

function drawMaskedArr(canvas, arr, mask) {
  const ctx = setupCanvas(canvas, 224, 224);
  // first draw original
  const im = ctx.createImageData(224, 224);
  for (let i = 0; i < 224 * 224; i++) {
    im.data[i * 4] = arr[i * 3];
    im.data[i * 4 + 1] = arr[i * 3 + 1];
    im.data[i * 4 + 2] = arr[i * 3 + 2];
    im.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  // overlay masked patches
  for (let pi = 0; pi < N; pi++) {
    for (let pj = 0; pj < N; pj++) {
      if (!mask[pi * N + pj]) continue;
      ctx.fillStyle = '#1a1815';
      ctx.fillRect(pj * TILE_SIZE, pi * TILE_SIZE, TILE_SIZE, TILE_SIZE);
    }
  }
  // patch grid
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < N; i++) {
    ctx.moveTo(0, i * TILE_SIZE); ctx.lineTo(224, i * TILE_SIZE);
    ctx.moveTo(i * TILE_SIZE, 0); ctx.lineTo(i * TILE_SIZE, 224);
  }
  ctx.stroke();
}

function reconstruct(arr, mask) {
  // Smart fill: each masked patch gets the weighted average of visible patches' means
  // weighted by inverse spatial distance. Stand-in for what an MAE decoder learns
  // to do on smooth scenes; not what real MAE does on natural images, but
  // qualitatively similar (smooth fill, blurry detail).
  const W = 224;
  const recon = new Uint8ClampedArray(arr.length);
  // Compute means for visible patches
  const means = new Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    means[i * N + j] = mask[i * N + j] ? null : patchMean(arr, i, j);
  }
  // Fill patches
  for (let pi = 0; pi < N; pi++) {
    for (let pj = 0; pj < N; pj++) {
      let fill;
      if (!mask[pi * N + pj]) {
        // Visible: copy original
        for (let dy = 0; dy < TILE_SIZE; dy++) {
          for (let dx = 0; dx < TILE_SIZE; dx++) {
            const y = pi * TILE_SIZE + dy;
            const x = pj * TILE_SIZE + dx;
            const k = (y * W + x) * 3;
            recon[k] = arr[k]; recon[k + 1] = arr[k + 1]; recon[k + 2] = arr[k + 2];
          }
        }
      } else {
        // Masked: weighted-mean fill from visible neighbours
        let totalW = 0, sR = 0, sG = 0, sB = 0;
        for (let oi = 0; oi < N; oi++) for (let oj = 0; oj < N; oj++) {
          if (mask[oi * N + oj]) continue;
          const dx = oj - pj, dy = oi - pi;
          const w = 1 / Math.pow(dx * dx + dy * dy + 0.5, 1.2);
          const m = means[oi * N + oj];
          totalW += w; sR += w * m[0]; sG += w * m[1]; sB += w * m[2];
        }
        if (totalW === 0) {
          sR = 128; sG = 128; sB = 128; totalW = 1;
        }
        const rR = sR / totalW, rG = sG / totalW, rB = sB / totalW;
        // Add slight noise to avoid perfectly flat
        for (let dy = 0; dy < TILE_SIZE; dy++) {
          for (let dx = 0; dx < TILE_SIZE; dx++) {
            const y = pi * TILE_SIZE + dy;
            const x = pj * TILE_SIZE + dx;
            const k = (y * W + x) * 3;
            recon[k]     = Math.max(0, Math.min(255, rR + (Math.random() - 0.5) * 8));
            recon[k + 1] = Math.max(0, Math.min(255, rG + (Math.random() - 0.5) * 8));
            recon[k + 2] = Math.max(0, Math.min(255, rB + (Math.random() - 0.5) * 8));
          }
        }
      }
    }
  }
  return recon;
}

function mse(arr, recon, mask) {
  const W = 224;
  let sse = 0, count = 0;
  for (let pi = 0; pi < N; pi++) {
    for (let pj = 0; pj < N; pj++) {
      if (!mask[pi * N + pj]) continue;
      for (let dy = 0; dy < TILE_SIZE; dy++) {
        for (let dx = 0; dx < TILE_SIZE; dx++) {
          const y = pi * TILE_SIZE + dy;
          const x = pj * TILE_SIZE + dx;
          const k = (y * W + x) * 3;
          const eR = arr[k] - recon[k];
          const eG = arr[k + 1] - recon[k + 1];
          const eB = arr[k + 2] - recon[k + 2];
          sse += eR * eR + eG * eG + eB * eB;
          count += 3;
        }
      }
    }
  }
  return Math.sqrt(sse / Math.max(1, count));
}

function renderMAE() {
  if (!STATE.image) STATE.image = makeImage();
  if (!STATE.mask) STATE.mask = makeMask(STATE.ratio);
  drawArr(document.getElementById('mae-orig'), STATE.image);
  drawMaskedArr(document.getElementById('mae-masked'), STATE.image, STATE.mask);
  const recon = reconstruct(STATE.image, STATE.mask);
  drawArr(document.getElementById('mae-recon'), recon);
  document.getElementById('mae-mse').textContent =
    `RMSE on masked patches = ${mse(STATE.image, recon, STATE.mask).toFixed(1)}`;
}

function wireMAE() {
  const ratio = document.getElementById('mae-ratio');
  ratio.addEventListener('input', () => {
    STATE.ratio = parseFloat(ratio.value);
    document.getElementById('mae-ratio-val').textContent = STATE.ratio.toFixed(2);
    STATE.mask = makeMask(STATE.ratio);
    renderMAE();
  });
  document.getElementById('mae-reroll').addEventListener('click', () => {
    STATE.image = makeImage();
    STATE.mask = makeMask(STATE.ratio);
    renderMAE();
  });
  renderMAE();
}

// ---------- Step 3: linear-probe curve ----------
function renderProbe() {
  const canvas = document.getElementById('probe-canvas');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 70, r: 18, t: 26, b: 38 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  // X axis: log frac (0.001 .. 1)
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'center';
  const ticks = [0.001, 0.01, 0.1, 1.0];
  ticks.forEach((t) => {
    const x = m.l + (Math.log10(t) - Math.log10(0.001)) / (Math.log10(1) - Math.log10(0.001)) * px;
    ctx.fillText(t < 1 ? `${(t * 100).toFixed(t < 0.01 ? 1 : 0)}%` : '100%', x, m.t + py + 16);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + py); ctx.stroke();
  });
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = 100 * (1 - i / 4);
    const y = m.t + i / 4 * py;
    ctx.fillText(`${v.toFixed(0)}%`, m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  // Synthetic curves
  function curve(asymptote, sslLike) {
    // returns function frac -> accuracy
    return (frac) => {
      // SSL: high asymptote, climbs fast even at small data
      // Random init: low asymptote, climbs slow
      const a = asymptote;
      const k = sslLike ? 5 : 0.7;
      const base = sslLike ? 25 : 10;
      return base + (a - base) * (1 - Math.exp(-k * frac));
    };
  }
  function plot(curveFn, color, dashed) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    const fracs = [];
    for (let i = 0; i <= 80; i++) {
      const lf = -3 + 3 * (i / 80);
      fracs.push(Math.pow(10, lf));
    }
    fracs.forEach((f, i) => {
      const x = m.l + (Math.log10(f) - Math.log10(0.001)) / (Math.log10(1) - Math.log10(0.001)) * px;
      const acc = Math.min(100, curveFn(f));
      const y = m.t + (1 - acc / 100) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  plot(curve(82, true), '#1e7770', false);
  plot(curve(70, true), '#2c6fb7', false);
  plot(curve(75, false), '#d9622b', true);
  // Legend
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'left';
  let lx = m.l + 8, ly = m.t + 18;
  const legend = [
    { c: '#1e7770', l: 'SSL pretrain (DINOv2-flavour)', dashed: false },
    { c: '#2c6fb7', l: 'SSL pretrain (MAE-flavour)', dashed: false },
    { c: '#d9622b', l: 'Random init', dashed: true }
  ];
  legend.forEach((it) => {
    ctx.strokeStyle = it.c; ctx.lineWidth = 2.4;
    ctx.setLineDash(it.dashed ? [6, 4] : []);
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#3b342b';
    ctx.fillText(it.l, lx + 22, ly + 4);
    lx += 22 + ctx.measureText(it.l).width + 22;
    if (lx > m.l + px - 200) { lx = m.l + 8; ly += 18; }
  });
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('fraction of labelled data (log)', m.l + px / 2, H - 8);
  ctx.save();
  ctx.translate(16, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('linear-probe accuracy', 0, 0);
  ctx.restore();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-mae':
      '\\mathcal{L}_{\\text{MAE}} = \\frac{1}{|\\mathcal{M}|} \\sum_{i \\in \\mathcal{M}} \\bigl\\lVert \\mathrm{Dec}\\!\\left(\\mathrm{Enc}(x_{\\overline{\\mathcal{M}}})\\right)_i - x_i \\bigr\\rVert^2'
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
  wireMAE();
  renderProbe();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
