// ============================================================
// Vision SSL — live interactives.
// Plain (non-module) script so it runs under file:// as well as http://.
//   1. Contrastive / SimCLR lab: augment -> embed -> similarity matrix -> InfoNCE
//   2. MAE mask-and-reconstruct demo
//   3. Linear-probe efficiency curve
// Nothing here is a mock: every point in the scatter and every cell in the
// similarity matrix is computed from the actual augmented pixels drawn on the
// canvases (a structure-tensor "encoder" that recovers grating orientation).
// ============================================================
(function () {
  'use strict';

  // ---------- tiny deterministic RNG (mulberry32) ----------
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function setupCanvas(canvas, w, h, smooth) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = smooth !== false;
    return ctx;
  }

  function hsl(h, s, l) {
    h = ((h % 360) + 360) % 360; s = Math.max(0, Math.min(1, s)); l = Math.max(0, Math.min(1, l));
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const hp = h / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let r = 0, g = 0, b = 0;
    if (hp < 1) { r = c; g = x; } else if (hp < 2) { r = x; g = c; }
    else if (hp < 3) { g = c; b = x; } else if (hp < 4) { g = x; b = c; }
    else if (hp < 5) { r = x; b = c; } else { r = c; b = x; }
    const m = l - c / 2;
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
  }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function lerpRGB(c1, c2, t) {
    return `rgb(${Math.round(lerp(c1[0], c2[0], t))},${Math.round(lerp(c1[1], c2[1], t))},${Math.round(lerp(c1[2], c2[2], t))})`;
  }

  // ============================================================
  // 1. CONTRASTIVE / SimCLR LAB
  // ============================================================
  const SIM = {
    V: 104,                 // view pixel size
    N: 4,                   // images in the batch (=> 2N views)
    bases: [],              // {theta, freq, hue}
    controls: { aug: 0.35, color: 0.65, temp: 0.20 },
    seed: 7,
    views: []               // built each render
  };

  function initBases() {
    const angles = [0, 45, 90, 135];        // grating orientations (deg)
    const hues = [18, 150, 205, 320];       // base colours (nuisance the encoder ignores)
    const freqs = [4.5, 5.5, 4.0, 6.0];
    SIM.bases = angles.map((d, i) => ({ theta: d * Math.PI / 180, freq: freqs[i], hue: hues[i] }));
  }

  // Render one augmented view of a base image and return {rgba, emb, aug}
  function renderView(base, ctrl, seed) {
    const V = SIM.V;
    const r = rng(seed);
    // --- augmentation parameters (random crop / colour-jitter / brightness / noise) ---
    const zoom = 1 + ctrl.aug * (0.3 + 2.2 * r());        // crop / zoom-in
    const phase = r() * Math.PI * 2;                       // random translation of stripes
    const ox = (r() - 0.5) * ctrl.aug * 0.9;
    const oy = (r() - 0.5) * ctrl.aug * 0.9;
    const hueShift = (r() - 0.5) * ctrl.color * 320;       // colour jitter (degrees)
    const bright = 1 + (r() - 0.5) * (0.25 + ctrl.aug * 0.6);
    const noiseAmp = ctrl.aug * (40 + 110 * r());          // per-view pixel noise
    const ct = Math.cos(base.theta), st = Math.sin(base.theta);
    const c1 = hsl(base.hue + hueShift, 0.55, 0.34);
    const c2 = hsl(base.hue + hueShift + 24, 0.68, 0.74);
    const rgba = new Uint8ClampedArray(V * V * 4);
    for (let y = 0; y < V; y++) {
      for (let x = 0; x < V; x++) {
        // normalized centred coords, apply zoom (crop) + translation
        const u = (x / V - 0.5) / zoom + ox;
        const w = (y / V - 0.5) / zoom + oy;
        const proj = u * ct + w * st;
        const s = 0.5 + 0.5 * Math.sin(2 * Math.PI * base.freq * proj + phase);
        let R = lerp(c1[0], c2[0], s) * bright + (r() - 0.5) * noiseAmp;
        let G = lerp(c1[1], c2[1], s) * bright + (r() - 0.5) * noiseAmp;
        let B = lerp(c1[2], c2[2], s) * bright + (r() - 0.5) * noiseAmp;
        const k = (y * V + x) * 4;
        rgba[k] = R; rgba[k + 1] = G; rgba[k + 2] = B; rgba[k + 3] = 255;
      }
    }
    return { rgba, emb: embed(rgba, V, V) };
  }

  // The "encoder": a structure tensor on luminance. Returns a 2-D vector whose
  // DIRECTION encodes the dominant grating orientation (double-angle) and whose
  // MAGNITUDE is orientation coherence in ~[0,1]. Colour-blind by construction,
  // so hue jitter barely moves it; heavy crop + noise erode coherence.
  function embed(rgba, W, H) {
    let Jxx = 0, Jyy = 0, Jxy = 0;
    const lum = (k) => 0.299 * rgba[k] + 0.587 * rgba[k + 1] + 0.114 * rgba[k + 2];
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const gx = lum((y * W + x + 1) * 4) - lum((y * W + x - 1) * 4);
        const gy = lum(((y + 1) * W + x) * 4) - lum(((y - 1) * W + x) * 4);
        Jxx += gx * gx; Jyy += gy * gy; Jxy += gx * gy;
      }
    }
    const denom = Jxx + Jyy + 1e-6;
    return [(Jxx - Jyy) / denom, (2 * Jxy) / denom];
  }

  function cosSim(a, b) {
    const d = a[0] * b[0] + a[1] * b[1];
    const na = Math.hypot(a[0], a[1]) + 1e-9, nb = Math.hypot(b[0], b[1]) + 1e-9;
    return d / (na * nb);
  }

  function buildBatch() {
    const c = SIM.controls;
    const views = [];
    for (let i = 0; i < SIM.N; i++) {
      views.push(renderView(SIM.bases[i], c, SIM.seed * 1013 + i * 7919 + 101));
      views.push(renderView(SIM.bases[i], c, SIM.seed * 1013 + i * 7919 + 613));
    }
    return views;
  }

  function infoNCE(views, i, j, tau) {
    const pos = Math.exp(cosSim(views[i].emb, views[j].emb) / tau);
    let den = 0;
    for (let k = 0; k < views.length; k++) {
      if (k === i) continue;
      den += Math.exp(cosSim(views[i].emb, views[k].emb) / tau);
    }
    return -Math.log(pos / den);
  }

  function viewToOffscreen(view) {
    const off = document.createElement('canvas');
    off.width = SIM.V; off.height = SIM.V;
    const octx = off.getContext('2d');
    const im = octx.createImageData(SIM.V, SIM.V);
    im.data.set(view.rgba);
    octx.putImageData(im, 0, 0);
    return off;
  }

  function drawViewInto(canvasId, view) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = setupCanvas(canvas, 112, 112, true);
    ctx.drawImage(viewToOffscreen(view), 0, 0, 112, 112);
    ctx.strokeStyle = 'rgba(0,0,0,0.10)';
    ctx.strokeRect(0.5, 0.5, 111, 111);
  }

  // Scatter of every view's embedding on the orientation circle.
  function drawScatter(views) {
    const canvas = document.getElementById('sim-scatter');
    if (!canvas) return;
    const S = 300;
    const ctx = setupCanvas(canvas, S, S, true);
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = '#fbfaf6'; ctx.fillRect(0, 0, S, S);
    const cx = S / 2, cy = S / 2, R = S / 2 - 34;
    const map = (v) => [cx + v[0] * R, cy - v[1] * R];

    // guide circle + cross-hairs
    ctx.strokeStyle = '#e2d8c6'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
    ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
    ctx.fillStyle = '#9a917f'; ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('coherence = 1', cx, cy - R - 14);
    ctx.fillText('low coherence', cx, cy + 12);

    // ideal orientation anchors
    for (let i = 0; i < SIM.N; i++) {
      const th = 2 * SIM.bases[i].theta;
      const p = map([Math.cos(th), Math.sin(th)]);
      const col = hsl(SIM.bases[i].hue, 0.5, 0.55);
      ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},0.55)`;
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p[0], p[1]); ctx.stroke();
      ctx.setLineDash([]);
    }

    // connect the two anchor views (the positive pair)
    const a0 = map(views[0].emb), a1 = map(views[1].emb);
    ctx.strokeStyle = 'rgba(217,98,43,0.85)'; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(a0[0], a0[1]); ctx.lineTo(a1[0], a1[1]); ctx.stroke();

    // all points
    for (let k = 0; k < views.length; k++) {
      const base = SIM.bases[Math.floor(k / 2)];
      const p = map(views[k].emb);
      const col = hsl(base.hue, 0.55, 0.45);
      const isAnchor = k < 2;
      ctx.beginPath();
      ctx.arc(p[0], p[1], isAnchor ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = isAnchor ? '#d9622b' : '#fff';
      ctx.stroke();
    }
    // label anchor points (offset vertically so they stay legible even when the
    // positive pair is nearly coincident — which is exactly the good case)
    ctx.fillStyle = '#c0432a'; ctx.font = '700 11px Manrope, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('view A', a0[0], a0[1] - 14);
    ctx.fillText('view B', a1[0], a1[1] + 22);
  }

  // 2N x 2N cosine-similarity heatmap with thumbnails along the header row/col.
  function drawMatrix(views) {
    const canvas = document.getElementById('sim-matrix');
    if (!canvas) return;
    const M = views.length;                 // 2N
    const cell = 40, pad = 6;
    const dim = cell * (M + 1) + pad * 2;
    const ctx = setupCanvas(canvas, dim, dim, true);
    ctx.clearRect(0, 0, dim, dim);
    ctx.fillStyle = '#fbfaf6'; ctx.fillRect(0, 0, dim, dim);
    const x0 = pad + cell, y0 = pad + cell;

    // thumbnails header (row & column)
    for (let k = 0; k < M; k++) {
      const off = viewToOffscreen(views[k]);
      const base = SIM.bases[Math.floor(k / 2)];
      const col = hsl(base.hue, 0.5, 0.5);
      const cxp = x0 + k * cell, cyp = y0 + k * cell;
      ctx.drawImage(off, cxp + 3, pad + 3, cell - 6, cell - 6);          // top strip
      ctx.drawImage(off, pad + 3, cyp + 3, cell - 6, cell - 6);          // left strip
      ctx.strokeStyle = `rgb(${col[0]},${col[1]},${col[2]})`; ctx.lineWidth = 2;
      ctx.strokeRect(cxp + 3, pad + 3, cell - 6, cell - 6);
      ctx.strokeRect(pad + 3, cyp + 3, cell - 6, cell - 6);
    }

    // heatmap cells
    const lo = [246, 240, 226], hi = [26, 74, 122];
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < M; j++) {
        const s = cosSim(views[i].emb, views[j].emb);
        const t = Math.max(0, Math.min(1, (s + 0.3) / 1.3));
        const px = x0 + j * cell, py = y0 + i * cell;
        ctx.fillStyle = lerpRGB(lo, hi, t);
        ctx.fillRect(px, py, cell, cell);
        ctx.fillStyle = t > 0.55 ? 'rgba(255,255,255,0.92)' : 'rgba(47,42,34,0.72)';
        ctx.font = '600 11px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(s.toFixed(2), px + cell / 2, py + cell / 2);
      }
    }
    // positive-pair highlight (block diagonal 2x2)
    ctx.lineWidth = 2.5; ctx.strokeStyle = '#d9622b';
    for (let m = 0; m < SIM.N; m++) {
      const a = 2 * m;
      ctx.strokeRect(x0 + (a + 1) * cell, y0 + a * cell, cell, cell);
      ctx.strokeRect(x0 + a * cell, y0 + (a + 1) * cell, cell, cell);
    }
    // grid
    ctx.strokeStyle = 'rgba(47,42,34,0.10)'; ctx.lineWidth = 1;
    for (let i = 0; i <= M; i++) {
      ctx.beginPath(); ctx.moveTo(x0 + i * cell, y0); ctx.lineTo(x0 + i * cell, y0 + M * cell); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0, y0 + i * cell); ctx.lineTo(x0 + M * cell, y0 + i * cell); ctx.stroke();
    }
  }

  function renderSim() {
    const views = buildBatch();
    SIM.views = views;
    // anchor original (no augmentation) + its two views
    drawViewInto('sim-orig', renderView(SIM.bases[0], { aug: 0, color: 0 }, 999));
    drawViewInto('sim-viewA', views[0]);
    drawViewInto('sim-viewB', views[1]);
    drawScatter(views);
    drawMatrix(views);

    // stats for the anchor (i=0, positive j=1)
    const posSim = cosSim(views[0].emb, views[1].emb);
    let negSum = 0, negN = 0, maxNeg = -2;
    for (let k = 2; k < views.length; k++) { const s = cosSim(views[0].emb, views[k].emb); negSum += s; negN++; maxNeg = Math.max(maxNeg, s); }
    const meanNeg = negSum / Math.max(1, negN);
    const loss = infoNCE(views, 0, 1, SIM.controls.temp);
    const nearestIsPos = posSim > maxNeg;

    setText('sim-pos', posSim.toFixed(2));
    setText('sim-neg', meanNeg.toFixed(2));
    setText('sim-loss', loss.toFixed(2));
    const nn = document.getElementById('sim-nn');
    if (nn) {
      nn.textContent = nearestIsPos ? 'yes ✓' : 'no ✗';
      nn.style.color = nearestIsPos ? '#1e7770' : '#c0432a';
    }
  }

  function setText(id, t) { const el = document.getElementById(id); if (el) el.textContent = t; }

  function wireSim() {
    const bind = (id, key, valId, dp) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('input', () => {
        SIM.controls[key] = parseFloat(el.value);
        const v = document.getElementById(valId);
        if (v) v.textContent = SIM.controls[key].toFixed(dp);
        renderSim();
      });
    };
    bind('sim-aug', 'aug', 'sim-aug-val', 2);
    bind('sim-color', 'color', 'sim-color-val', 2);
    bind('sim-temp', 'temp', 'sim-temp-val', 2);
    const btn = document.getElementById('sim-reroll');
    if (btn) btn.addEventListener('click', () => { SIM.seed = (SIM.seed * 6364136 + 1) % 2147483647; renderSim(); });
  }

  // ============================================================
  // 2. MAE mask-and-reconstruct demo
  // ============================================================
  const MAE = { TILE: 28, N: 8, W: 224, image: null, mask: null, ratio: 0.75 };

  function maeMakeImage() {
    const W = MAE.W, H = MAE.W;
    const arr = new Uint8ClampedArray(W * H * 3);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const horizon = 130; let r, g, b;
        if (y < horizon) {
          const t = y / horizon;
          r = 110 + 130 * t; g = 160 + 70 * t; b = 220 - 50 * t;
          const dx = x - 175, dy = y - 50, dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 24) { r = 255; g = 220; b = 130; }
          else if (dist < 38) { const m = 1 - (dist - 24) / 14; r = Math.min(255, r + 60 * m); g = Math.min(255, g + 50 * m); b = Math.min(255, b + 10 * m); }
        } else {
          const t = (y - horizon) / (H - horizon);
          r = 120 - 30 * t; g = 180 - 40 * t; b = 90 - 30 * t;
        }
        const tx = x - 70, ty = y - 110;
        if (ty > 0 && ty < 80 && Math.abs(tx) < 8) { r = 70; g = 50; b = 30; }
        const dx = x - 70, dy = y - 95;
        if (Math.sqrt(dx * dx + (dy * 1.4) * (dy * 1.4)) < 32) { r = 50; g = 110; b = 60; }
        if (x > 130 && x < 170 && y > 140 && y < 170) { r = 200; g = 150; b = 100; }
        if (x > 138 && x < 162 && y > 130 && y < 145) { r = 100; g = 60; b = 50; }
        const k = (y * W + x) * 3; arr[k] = r; arr[k + 1] = g; arr[k + 2] = b;
      }
    }
    return arr;
  }

  function maeMakeMask(ratio) {
    const N = MAE.N, m = new Array(N * N).fill(false), idx = [];
    for (let i = 0; i < N * N; i++) idx.push(i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1));[idx[i], idx[j]] = [idx[j], idx[i]]; }
    const k = Math.round(ratio * N * N);
    for (let i = 0; i < k; i++) m[idx[i]] = true;
    return m;
  }

  function maePatchMean(arr, pi, pj) {
    const W = MAE.W, T = MAE.TILE; let sR = 0, sG = 0, sB = 0;
    for (let dy = 0; dy < T; dy++) for (let dx = 0; dx < T; dx++) {
      const k = ((pi * T + dy) * W + (pj * T + dx)) * 3; sR += arr[k]; sG += arr[k + 1]; sB += arr[k + 2];
    }
    const d = T * T; return [sR / d, sG / d, sB / d];
  }

  function maeGrid(ctx) {
    const T = MAE.TILE, N = MAE.N;
    ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 1; i < N; i++) { ctx.moveTo(0, i * T); ctx.lineTo(224, i * T); ctx.moveTo(i * T, 0); ctx.lineTo(i * T, 224); }
    ctx.stroke();
  }

  function maeDraw(canvasId, arr, mask) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = setupCanvas(canvas, 224, 224, false);
    const im = ctx.createImageData(224, 224);
    for (let i = 0; i < 224 * 224; i++) { im.data[i * 4] = arr[i * 3]; im.data[i * 4 + 1] = arr[i * 3 + 1]; im.data[i * 4 + 2] = arr[i * 3 + 2]; im.data[i * 4 + 3] = 255; }
    ctx.putImageData(im, 0, 0);
    if (mask) {
      const T = MAE.TILE, N = MAE.N;
      for (let pi = 0; pi < N; pi++) for (let pj = 0; pj < N; pj++) {
        if (!mask[pi * N + pj]) continue;
        ctx.fillStyle = '#141210'; ctx.fillRect(pj * T, pi * T, T, T);
      }
    }
    maeGrid(ctx);
  }

  function maeReconstruct(arr, mask) {
    const W = MAE.W, T = MAE.TILE, N = MAE.N;
    const recon = new Uint8ClampedArray(arr.length);
    const means = new Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) means[i * N + j] = mask[i * N + j] ? null : maePatchMean(arr, i, j);
    for (let pi = 0; pi < N; pi++) for (let pj = 0; pj < N; pj++) {
      if (!mask[pi * N + pj]) {
        for (let dy = 0; dy < T; dy++) for (let dx = 0; dx < T; dx++) {
          const k = ((pi * T + dy) * W + (pj * T + dx)) * 3; recon[k] = arr[k]; recon[k + 1] = arr[k + 1]; recon[k + 2] = arr[k + 2];
        }
      } else {
        let tw = 0, sR = 0, sG = 0, sB = 0;
        for (let oi = 0; oi < N; oi++) for (let oj = 0; oj < N; oj++) {
          if (mask[oi * N + oj]) continue;
          const dx = oj - pj, dy = oi - pi, wgt = 1 / Math.pow(dx * dx + dy * dy + 0.5, 1.2), mn = means[oi * N + oj];
          tw += wgt; sR += wgt * mn[0]; sG += wgt * mn[1]; sB += wgt * mn[2];
        }
        if (tw === 0) { sR = 128; sG = 128; sB = 128; tw = 1; }
        const rR = sR / tw, rG = sG / tw, rB = sB / tw;
        for (let dy = 0; dy < T; dy++) for (let dx = 0; dx < T; dx++) {
          const k = ((pi * T + dy) * W + (pj * T + dx)) * 3;
          recon[k] = rR + (Math.random() - 0.5) * 8; recon[k + 1] = rG + (Math.random() - 0.5) * 8; recon[k + 2] = rB + (Math.random() - 0.5) * 8;
        }
      }
    }
    return recon;
  }

  function maeRMSE(arr, recon, mask) {
    const W = MAE.W, T = MAE.TILE, N = MAE.N; let sse = 0, c = 0;
    for (let pi = 0; pi < N; pi++) for (let pj = 0; pj < N; pj++) {
      if (!mask[pi * N + pj]) continue;
      for (let dy = 0; dy < T; dy++) for (let dx = 0; dx < T; dx++) {
        const k = ((pi * T + dy) * W + (pj * T + dx)) * 3;
        const eR = arr[k] - recon[k], eG = arr[k + 1] - recon[k + 1], eB = arr[k + 2] - recon[k + 2];
        sse += eR * eR + eG * eG + eB * eB; c += 3;
      }
    }
    return Math.sqrt(sse / Math.max(1, c));
  }

  function renderMAE() {
    if (!MAE.image) MAE.image = maeMakeImage();
    if (!MAE.mask) MAE.mask = maeMakeMask(MAE.ratio);
    maeDraw('mae-orig', MAE.image, null);
    maeDraw('mae-masked', MAE.image, MAE.mask);
    const recon = maeReconstruct(MAE.image, MAE.mask);
    maeDraw('mae-recon', recon, null);
    const visible = Math.round((1 - MAE.ratio) * MAE.N * MAE.N);
    setText('mae-mse', `${visible}/${MAE.N * MAE.N} patches visible · RMSE on masked = ${maeRMSE(MAE.image, recon, MAE.mask).toFixed(1)}`);
  }

  function wireMAE() {
    const ratio = document.getElementById('mae-ratio');
    if (ratio) ratio.addEventListener('input', () => {
      MAE.ratio = parseFloat(ratio.value);
      setText('mae-ratio-val', MAE.ratio.toFixed(2));
      MAE.mask = maeMakeMask(MAE.ratio);
      renderMAE();
    });
    const btn = document.getElementById('mae-reroll');
    if (btn) btn.addEventListener('click', () => { MAE.image = maeMakeImage(); MAE.mask = maeMakeMask(MAE.ratio); renderMAE(); });
    renderMAE();
  }

  // ============================================================
  // 3. Linear-probe efficiency curve
  // ============================================================
  function renderProbe() {
    const canvas = document.getElementById('probe-canvas');
    if (!canvas) return;
    const W = 880, H = 320;
    const ctx = setupCanvas(canvas, W, H, true);
    ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
    const m = { l: 70, r: 18, t: 26, b: 40 };
    const px = W - m.l - m.r, py = H - m.t - m.b;
    ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
    const X = (f) => m.l + (Math.log10(f) + 3) / 3 * px;
    ctx.fillStyle = '#9a917f'; ctx.font = '11px "IBM Plex Mono", monospace'; ctx.textAlign = 'center';
    [0.001, 0.01, 0.1, 1.0].forEach((t) => {
      const x = X(t);
      ctx.fillText(t < 1 ? `${(t * 100).toFixed(t < 0.01 ? 1 : 0)}%` : '100%', x, m.t + py + 18);
      ctx.strokeStyle = '#f0ebe1'; ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + py); ctx.stroke();
    });
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const v = 100 * (1 - i / 4), y = m.t + i / 4 * py;
      ctx.fillText(`${v.toFixed(0)}%`, m.l - 6, y + 3);
      ctx.strokeStyle = '#f0ebe1'; ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
    }
    const curve = (asym, ssl) => (frac) => { const k = ssl ? 5 : 0.7, base = ssl ? 25 : 10; return base + (asym - base) * (1 - Math.exp(-k * frac)); };
    const plot = (fn, color, dashed) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2.6; ctx.setLineDash(dashed ? [6, 4] : []); ctx.beginPath();
      for (let i = 0; i <= 80; i++) {
        const f = Math.pow(10, -3 + 3 * (i / 80));
        const x = X(f), y = m.t + (1 - Math.min(100, fn(f)) / 100) * py;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.setLineDash([]);
    };
    plot(curve(82, true), '#1e7770', false);
    plot(curve(70, true), '#2c6fb7', false);
    plot(curve(75, false), '#d9622b', true);
    ctx.font = '12px Manrope, sans-serif'; ctx.textAlign = 'left';
    let lx = m.l + 10, ly = m.t + 16;
    [{ c: '#1e7770', l: 'SSL pretrain (DINOv2-flavour)', d: false }, { c: '#2c6fb7', l: 'SSL pretrain (MAE-flavour)', d: false }, { c: '#d9622b', l: 'Random init', d: true }].forEach((it) => {
      ctx.strokeStyle = it.c; ctx.lineWidth = 2.6; ctx.setLineDash(it.d ? [6, 4] : []);
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#3b342b'; ctx.fillText(it.l, lx + 24, ly + 4);
      lx += 24 + ctx.measureText(it.l).width + 22; if (lx > m.l + px - 200) { lx = m.l + 10; ly += 18; }
    });
    ctx.fillStyle = '#6e665b'; ctx.textAlign = 'center';
    ctx.fillText('fraction of labelled data (log scale)', m.l + px / 2, H - 8);
    ctx.save(); ctx.translate(16, m.t + py / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('linear-probe accuracy', 0, 0); ctx.restore();
  }

  // ============================================================
  function boot() {
    initBases();
    wireSim();
    renderSim();
    wireMAE();
    renderProbe();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
