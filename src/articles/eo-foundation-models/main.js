// ============================================================
// EO Foundation Models — multi-band MAE on a synthetic tile.
// We synthesise a 64x64 6-band tile, mask patches by band, and "reconstruct"
// via cross-band weighted-mean + spatial fill (stand-in for an MAE decoder).
// ============================================================

const SIZE = 64;
const BANDS = 6;
const TILE_SIZE = 8; // 8x8 patch grid
const N_PATCH = SIZE / TILE_SIZE;
const BAND_NAMES = ['B2 (blue, 490nm)', 'B3 (green, 560nm)', 'B4 (red, 665nm)', 'B8 (NIR, 842nm)', 'B11 (SWIR1, 1610nm)', 'B12 (SWIR2, 2190nm)'];

const STATE = { tile: null, mask: null, ratio: 0.5, mode: 'random' };

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

function makeTile() {
  // 6 bands with realistic correlations:
  // - vegetation (NDVI > 0): high B8, moderate B3, low B4
  // - water: low B8/B11/B12, moderate B2/B3
  // - soil/urban: high B11/B12, moderate B4
  // We synthesise three regions on the tile.
  const tile = new Array(BANDS).fill(0).map(() => new Float32Array(SIZE * SIZE));
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      // Region: top-left = vegetation, bottom-left = water, right = urban
      let region;
      if (x > SIZE * 0.55) region = 'urban';
      else if (y > SIZE * 0.6) region = 'water';
      else region = 'veg';
      const base = {
        veg:   [0.05, 0.10, 0.04, 0.45, 0.20, 0.10],
        water: [0.08, 0.06, 0.04, 0.02, 0.02, 0.01],
        urban: [0.18, 0.20, 0.22, 0.30, 0.40, 0.35]
      }[region];
      for (let b = 0; b < BANDS; b++) {
        const noise = 0.025 * randn();
        const tex = 0.04 * Math.sin(0.4 * x + 0.3 * y + b) * Math.cos(0.2 * x);
        tile[b][y * SIZE + x] = Math.max(0, Math.min(1, base[b] + noise + tex));
      }
    }
  }
  return tile;
}

function makeMask(ratio, mode) {
  const NP = N_PATCH * N_PATCH;
  const mask = new Array(BANDS).fill(0).map(() => new Array(NP).fill(false));
  if (mode === 'all-bands') {
    // Pick one set of patches and mask the same ones across all bands
    const target = Math.round(ratio * NP);
    const idx = []; for (let i = 0; i < NP; i++) idx.push(i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let k = 0; k < target; k++) {
      for (let b = 0; b < BANDS; b++) mask[b][idx[k]] = true;
    }
  } else if (mode === 'time') {
    // Pretend bands are 3 timesteps × 2 bands; mask whole timesteps (groups of 2 bands)
    for (let g = 0; g < 3; g++) {
      if (Math.random() < ratio) {
        for (let b = g * 2; b < g * 2 + 2; b++) for (let i = 0; i < NP; i++) mask[b][i] = true;
      }
    }
  } else {
    for (let b = 0; b < BANDS; b++) {
      const target = Math.round(ratio * NP);
      const idx = []; for (let i = 0; i < NP; i++) idx.push(i);
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      for (let k = 0; k < target; k++) mask[b][idx[k]] = true;
    }
  }
  return mask;
}

// Per-patch mean for each band
function patchMean(band, pi, pj) {
  let s = 0;
  for (let dy = 0; dy < TILE_SIZE; dy++) {
    for (let dx = 0; dx < TILE_SIZE; dx++) {
      const y = pi * TILE_SIZE + dy;
      const x = pj * TILE_SIZE + dx;
      s += band[y * SIZE + x];
    }
  }
  return s / (TILE_SIZE * TILE_SIZE);
}

function reconstruct() {
  // For each band and each masked patch, compute fill = α * spatial_mean(visible patches in band) + (1-α) * cross-band-correlated estimate.
  const recon = new Array(BANDS).fill(0).map(() => new Float32Array(SIZE * SIZE));
  // Compute means per (band, patch)
  const means = new Array(BANDS).fill(0).map(() => new Array(N_PATCH * N_PATCH).fill(0));
  for (let b = 0; b < BANDS; b++) {
    for (let pi = 0; pi < N_PATCH; pi++) {
      for (let pj = 0; pj < N_PATCH; pj++) {
        means[b][pi * N_PATCH + pj] = patchMean(STATE.tile[b], pi, pj);
      }
    }
  }
  for (let b = 0; b < BANDS; b++) {
    for (let pi = 0; pi < N_PATCH; pi++) {
      for (let pj = 0; pj < N_PATCH; pj++) {
        const k = pi * N_PATCH + pj;
        if (!STATE.mask[b][k]) {
          // Visible — copy
          for (let dy = 0; dy < TILE_SIZE; dy++) {
            for (let dx = 0; dx < TILE_SIZE; dx++) {
              const y = pi * TILE_SIZE + dy;
              const x = pj * TILE_SIZE + dx;
              recon[b][y * SIZE + x] = STATE.tile[b][y * SIZE + x];
            }
          }
        } else {
          // Mask: combine spatial neighbour mean (in same band) with cross-band coloc estimate
          let totalW = 0, sSpatial = 0;
          for (let oi = 0; oi < N_PATCH; oi++) for (let oj = 0; oj < N_PATCH; oj++) {
            if (STATE.mask[b][oi * N_PATCH + oj]) continue;
            const dx = oj - pj, dy = oi - pi;
            const w = 1 / Math.pow(dx * dx + dy * dy + 0.4, 1.1);
            totalW += w; sSpatial += w * means[b][oi * N_PATCH + oj];
          }
          const spatialEst = totalW > 0 ? sSpatial / totalW : 0.2;
          // Cross-band: average all visible bands at this patch, scaled by their global mean ratio
          let cross = 0, cnt = 0;
          for (let bb = 0; bb < BANDS; bb++) if (bb !== b && !STATE.mask[bb][k]) {
            // global mean ratio band b vs band bb
            let gm = 0, gv = 0;
            for (let i = 0; i < N_PATCH * N_PATCH; i++) { gm += means[b][i]; gv += means[bb][i]; }
            const ratio = gm / Math.max(0.001, gv);
            cross += ratio * means[bb][k];
            cnt++;
          }
          const crossEst = cnt > 0 ? cross / cnt : spatialEst;
          const fill = 0.55 * spatialEst + 0.45 * crossEst;
          for (let dy = 0; dy < TILE_SIZE; dy++) {
            for (let dx = 0; dx < TILE_SIZE; dx++) {
              const y = pi * TILE_SIZE + dy;
              const x = pj * TILE_SIZE + dx;
              recon[b][y * SIZE + x] = Math.max(0, Math.min(1, fill + 0.01 * randn()));
            }
          }
        }
      }
    }
  }
  return recon;
}

function bandRMSE(orig, rec, mask) {
  let sse = 0, count = 0;
  for (let pi = 0; pi < N_PATCH; pi++) {
    for (let pj = 0; pj < N_PATCH; pj++) {
      if (!mask[pi * N_PATCH + pj]) continue;
      for (let dy = 0; dy < TILE_SIZE; dy++) {
        for (let dx = 0; dx < TILE_SIZE; dx++) {
          const y = pi * TILE_SIZE + dy;
          const x = pj * TILE_SIZE + dx;
          const e = orig[y * SIZE + x] - rec[y * SIZE + x];
          sse += e * e; count++;
        }
      }
    }
  }
  return Math.sqrt(sse / Math.max(1, count));
}

function bandToColour(b, v) {
  const t = Math.max(0, Math.min(1, v * 2.5));
  const palettes = {
    0: [Math.round(255 * t * 0.5), Math.round(255 * t * 0.7), Math.round(255 * (0.3 + t * 0.7))],
    1: [Math.round(255 * t * 0.4), Math.round(255 * (0.3 + t * 0.7)), Math.round(255 * t * 0.4)],
    2: [Math.round(255 * (0.3 + t * 0.7)), Math.round(255 * t * 0.3), Math.round(255 * t * 0.3)],
    3: [Math.round(255 * t), Math.round(255 * t * 0.5), Math.round(255 * t * 0.3)],
    4: [Math.round(255 * (0.4 + t * 0.6)), Math.round(255 * (0.2 + t * 0.5)), Math.round(255 * t * 0.2)],
    5: [Math.round(255 * (0.5 + t * 0.5)), Math.round(255 * t * 0.4), Math.round(255 * t * 0.3)]
  };
  return palettes[b];
}

function drawBand(canvas, band, bandIdx, mask, label) {
  const px = 96;
  const ctx = setupCanvas(canvas, px, px);
  const cell = px / SIZE;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = band[y * SIZE + x];
      const [r, g, b] = bandToColour(bandIdx, v);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
    }
  }
  if (mask) {
    ctx.fillStyle = '#1a1815';
    for (let pi = 0; pi < N_PATCH; pi++) {
      for (let pj = 0; pj < N_PATCH; pj++) {
        if (!mask[pi * N_PATCH + pj]) continue;
        ctx.fillRect(pj * TILE_SIZE * cell, pi * TILE_SIZE * cell, TILE_SIZE * cell, TILE_SIZE * cell);
      }
    }
  }
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, px, px);
}

function renderStrips() {
  const origStrip = document.getElementById('eo-bands-orig');
  const maskStrip = document.getElementById('eo-bands-masked');
  const reconStrip = document.getElementById('eo-bands-recon');
  if (!origStrip) return;
  if (!STATE.tile) STATE.tile = makeTile();
  if (!STATE.mask) STATE.mask = makeMask(STATE.ratio, STATE.mode);
  const recon = reconstruct();
  function panel(strip, hasMask, isRecon) {
    strip.innerHTML = '';
    for (let b = 0; b < BANDS; b++) {
      const cell = document.createElement('div');
      cell.className = 'eo-band-cell';
      const lbl = document.createElement('div');
      lbl.className = 'eo-band-label';
      lbl.textContent = BAND_NAMES[b].split(' ')[0];
      const c = document.createElement('canvas');
      c.width = 96; c.height = 96;
      cell.appendChild(lbl);
      cell.appendChild(c);
      if (isRecon) {
        const rmse = bandRMSE(STATE.tile[b], recon[b], STATE.mask[b]);
        const r = document.createElement('div');
        r.className = 'eo-band-meta';
        r.textContent = `RMSE ${rmse.toFixed(3)}`;
        cell.appendChild(r);
      }
      strip.appendChild(cell);
      const data = isRecon ? recon[b] : STATE.tile[b];
      drawBand(c, data, b, hasMask ? STATE.mask[b] : null);
    }
  }
  panel(origStrip, false, false);
  panel(maskStrip, true, false);
  panel(reconStrip, false, true);
}

function wireEO() {
  document.getElementById('eo-ratio').addEventListener('input', (e) => {
    STATE.ratio = parseFloat(e.target.value);
    document.getElementById('eo-ratio-val').textContent = STATE.ratio.toFixed(2);
    STATE.mask = makeMask(STATE.ratio, STATE.mode);
    renderStrips();
  });
  document.getElementById('eo-bands').addEventListener('change', (e) => {
    STATE.mode = e.target.value;
    STATE.mask = makeMask(STATE.ratio, STATE.mode);
    renderStrips();
  });
  document.getElementById('eo-reroll').addEventListener('click', () => {
    STATE.tile = makeTile();
    STATE.mask = makeMask(STATE.ratio, STATE.mode);
    renderStrips();
  });
  renderStrips();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-eomae':
      '\\mathcal{L}_{\\text{EO-MAE}} = \\sum_{(b, t, p) \\in \\mathcal{M}} \\bigl\\lVert \\hat x_{b, t, p} - x_{b, t, p} \\bigr\\rVert^2 + \\lambda \\,\\mathcal{L}_{\\text{cross-band}} + \\mu\\,\\mathcal{L}_{\\text{temporal}}'
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
  wireEO();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
