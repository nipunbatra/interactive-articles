// ============================================================
// Optical Flow, Frame by Frame
// Real Lucas-Kanade + dense flow computed in pure JS on a real
// HTML5 video element. Frames are grabbed into a working buffer
// at reduced resolution for speed.
// ============================================================

const WORK_W = 320;     // internal resolution for flow math
const WORK_H = 180;
const DISPLAY_W = 640;  // canvas display resolution (2× working)
const DISPLAY_H = 360;

// Known-stable CORS-friendly public video samples (Google Cloud bucket,
// widely used in video demos and docs). Webm / mp4 both work; the Google
// sample bucket serves with permissive CORS headers.
const SAMPLES = [
  {
    key: 'big-buck',
    label: 'Big Buck Bunny (action)',
    url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'
  },
  {
    key: 'joyrides',
    label: 'Joyrides (cars)',
    url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4'
  },
  {
    key: 'blazes',
    label: 'Blazes (flames)',
    url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4'
  },
  {
    key: 'escapes',
    label: 'Escapes (running)',
    url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4'
  }
];

// ---------- State ----------
const state = {
  video: null,
  videoReady: false,
  playing: false,
  prevGray: null,    // Float32Array, WORK_W * WORK_H
  currGray: null,
  prevImage: null,   // ImageData, display-res, for prev panel
  currImage: null,
  prevFrameTime: 0,
  frameCount: 0,
  fps: 30,
  // LK params
  windowSize: 15,
  maxFeatures: 60,
  minQuality: 0.02,
  // Dense params
  gridStep: 8,
  magThreshold: 0.5,
  // Flow results
  lkPoints: [],      // {x, y, u, v, strength}
  denseFlow: null,   // Float32Array, WORK_W*WORK_H*2 (u, v)
  flowMs: 0
};

// ---------- Video sample loading ----------
function loadSample(url) {
  const video = state.video;
  if (!video) return;
  video.crossOrigin = 'anonymous';
  video.src = url;
  video.load();
  video.addEventListener('loadedmetadata', onMeta, { once: true });
  video.addEventListener('error', () => {
    console.error('Video failed to load:', url);
    const cap = document.getElementById('prelude-caption');
    if (cap) cap.textContent = 'Could not load this sample. Drop a local file above.';
  }, { once: true });
}

function onMeta() {
  const video = state.video;
  state.videoReady = true;
  state.prevGray = null;
  state.currGray = null;
  // Seek to a moderate start time so the first frame isn't a black title card
  try {
    video.currentTime = Math.min(5, (video.duration || 10) * 0.1);
  } catch (_) { /* no-op */ }
  video.addEventListener('seeked', () => {
    grabFrame();
    updateStats();
  }, { once: true });
}

function handleFile(file) {
  const video = state.video;
  if (!video) return;
  const url = URL.createObjectURL(file);
  video.src = url;
  video.load();
  video.addEventListener('loadedmetadata', onMeta, { once: true });
}

// ---------- Grab current video frame into working buffers ----------
function grabFrame() {
  const video = state.video;
  if (!video || !state.videoReady || !video.videoWidth) return;

  // Create / reuse offscreen canvas at working resolution
  if (!state.workCanvas) {
    state.workCanvas = document.createElement('canvas');
    state.workCanvas.width = WORK_W; state.workCanvas.height = WORK_H;
  }
  if (!state.dispCanvas) {
    state.dispCanvas = document.createElement('canvas');
    state.dispCanvas.width = DISPLAY_W; state.dispCanvas.height = DISPLAY_H;
  }
  // Work: downscaled for flow math
  const wctx = state.workCanvas.getContext('2d');
  wctx.drawImage(video, 0, 0, WORK_W, WORK_H);
  const workData = wctx.getImageData(0, 0, WORK_W, WORK_H);

  // Display
  const dctx = state.dispCanvas.getContext('2d');
  dctx.drawImage(video, 0, 0, DISPLAY_W, DISPLAY_H);
  const dispData = dctx.getImageData(0, 0, DISPLAY_W, DISPLAY_H);

  // Swap prev/curr
  state.prevGray = state.currGray;
  state.prevImage = state.currImage;
  state.currGray = rgbToGray(workData.data, WORK_W, WORK_H);
  state.currImage = dispData;

  // Compute flow (only if we have a prev frame)
  if (state.prevGray) {
    const t0 = performance.now();
    computeLK();
    computeDenseFlow();
    state.flowMs = performance.now() - t0;
  } else {
    state.lkPoints = [];
    state.denseFlow = null;
  }

  renderAll();
  updateStats();
}

function rgbToGray(rgba, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = rgba[i * 4], G = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    g[i] = 0.299 * r + 0.587 * G + 0.114 * b;
  }
  return g;
}

// ---------- Spatial gradients ----------
function computeGradients(gray, w, h) {
  const Ix = new Float32Array(w * h);
  const Iy = new Float32Array(w * h);
  // Sobel
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const tl = gray[i - w - 1], t = gray[i - w], tr = gray[i - w + 1];
      const l  = gray[i - 1],     r = gray[i + 1];
      const bl = gray[i + w - 1], b = gray[i + w], br = gray[i + w + 1];
      Ix[i] = (tr - tl + 2 * (r - l) + br - bl) / 8;
      Iy[i] = (bl - tl + 2 * (b - t) + br - tr) / 8;
    }
  }
  return { Ix, Iy };
}

// ---------- Shi-Tomasi corner detection ----------
// For each pixel, compute min eigenvalue of [[Ixx, Ixy],[Ixy, Iyy]]
// summed over a small window; pick local maxima.
function detectCorners(gray, w, h, nMax, minQuality, winSize = 5) {
  const { Ix, Iy } = computeGradients(gray, w, h);
  const half = Math.floor(winSize / 2);
  const score = new Float32Array(w * h);
  let maxScore = 0;
  for (let y = half + 1; y < h - half - 1; y++) {
    for (let x = half + 1; x < w - half - 1; x++) {
      let Sxx = 0, Syy = 0, Sxy = 0;
      for (let dy = -half; dy <= half; dy++) {
        for (let dx = -half; dx <= half; dx++) {
          const i = (y + dy) * w + (x + dx);
          Sxx += Ix[i] * Ix[i];
          Syy += Iy[i] * Iy[i];
          Sxy += Ix[i] * Iy[i];
        }
      }
      // min eigenvalue of [[Sxx, Sxy],[Sxy, Syy]]
      const trace = Sxx + Syy;
      const det = Sxx * Syy - Sxy * Sxy;
      const disc = Math.max(0, trace * trace / 4 - det);
      const lambdaMin = trace / 2 - Math.sqrt(disc);
      score[y * w + x] = lambdaMin;
      if (lambdaMin > maxScore) maxScore = lambdaMin;
    }
  }
  // Threshold & non-max suppression (3×3)
  const minScore = maxScore * minQuality;
  const candidates = [];
  for (let y = half + 2; y < h - half - 2; y++) {
    for (let x = half + 2; x < w - half - 2; x++) {
      const i = y * w + x;
      const s = score[i];
      if (s < minScore) continue;
      // 3x3 local max
      let isMax = true;
      for (let dy = -1; dy <= 1 && isMax; dy++) {
        for (let dx = -1; dx <= 1 && isMax; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (score[(y + dy) * w + (x + dx)] > s) isMax = false;
        }
      }
      if (isMax) candidates.push({ x, y, s });
    }
  }
  // Sort by score descending, enforce minimum distance
  candidates.sort((a, b) => b.s - a.s);
  const selected = [];
  const minDist = 6;
  for (const c of candidates) {
    if (selected.length >= nMax) break;
    let tooClose = false;
    for (const p of selected) {
      if (Math.abs(c.x - p.x) < minDist && Math.abs(c.y - p.y) < minDist) {
        tooClose = true; break;
      }
    }
    if (!tooClose) selected.push(c);
  }
  return { corners: selected, Ix, Iy };
}

// ---------- Lucas-Kanade ----------
function solveLK(x, y, Ix, Iy, It, w, h, half) {
  let Sxx = 0, Syy = 0, Sxy = 0, Sxt = 0, Syt = 0;
  const xmin = Math.max(1, x - half), xmax = Math.min(w - 2, x + half);
  const ymin = Math.max(1, y - half), ymax = Math.min(h - 2, y + half);
  for (let yy = ymin; yy <= ymax; yy++) {
    for (let xx = xmin; xx <= xmax; xx++) {
      const i = yy * w + xx;
      const ix = Ix[i], iy = Iy[i], it = It[i];
      Sxx += ix * ix; Syy += iy * iy; Sxy += ix * iy;
      Sxt += ix * it; Syt += iy * it;
    }
  }
  const det = Sxx * Syy - Sxy * Sxy;
  if (Math.abs(det) < 1e-6) return { u: 0, v: 0, valid: false };
  const invDet = 1 / det;
  // A^T A [u v] = -A^T b  →  [u v] = (A^T A)^-1 (-A^T b)
  const u = invDet * (-Syy * Sxt + Sxy * Syt);
  const v = invDet * ( Sxy * Sxt - Sxx * Syt);
  return { u, v, valid: true };
}

function computeLK() {
  if (!state.prevGray || !state.currGray) return;
  const w = WORK_W, h = WORK_H;
  const It = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) It[i] = state.currGray[i] - state.prevGray[i];
  const { corners, Ix, Iy } = detectCorners(
    state.currGray, w, h, state.maxFeatures, state.minQuality, 5);
  const half = Math.floor(state.windowSize / 2);
  const points = [];
  for (const c of corners) {
    const { u, v, valid } = solveLK(c.x, c.y, Ix, Iy, It, w, h, half);
    if (!valid) continue;
    // Clamp insane values
    if (!isFinite(u) || !isFinite(v)) continue;
    if (Math.abs(u) > 15 || Math.abs(v) > 15) continue;
    points.push({ x: c.x, y: c.y, u, v, s: c.s });
  }
  state.lkPoints = points;
}

// ---------- Dense flow (LK on a regular grid) ----------
function computeDenseFlow() {
  if (!state.prevGray || !state.currGray) return;
  const w = WORK_W, h = WORK_H;
  const It = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) It[i] = state.currGray[i] - state.prevGray[i];
  const { Ix, Iy } = computeGradients(state.currGray, w, h);
  const half = Math.floor(state.windowSize / 2);
  const step = state.gridStep;
  // Store flow at every grid cell, then bilinear-upsample to full resolution
  const gw = Math.floor(w / step);
  const gh = Math.floor(h / step);
  const gridFlow = new Float32Array(gw * gh * 2);
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      const x = gx * step + Math.floor(step / 2);
      const y = gy * step + Math.floor(step / 2);
      const { u, v } = solveLK(x, y, Ix, Iy, It, w, h, half);
      const iu = isFinite(u) ? Math.max(-8, Math.min(8, u)) : 0;
      const iv = isFinite(v) ? Math.max(-8, Math.min(8, v)) : 0;
      gridFlow[(gy * gw + gx) * 2]     = iu;
      gridFlow[(gy * gw + gx) * 2 + 1] = iv;
    }
  }
  state.denseFlow = gridFlow;
  state.denseGw = gw;
  state.denseGh = gh;
  state.denseStep = step;
}

// ---------- HSV to RGB ----------
function hsvToRgb(h, s, v) {
  h = ((h % 1) + 1) % 1;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r, g, b;
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// ---------- Canvas rendering ----------
function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function drawFrame(ctx, w, h, imageData) {
  if (!imageData) {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);
    return;
  }
  const off = document.createElement('canvas');
  off.width = imageData.width; off.height = imageData.height;
  off.getContext('2d').putImageData(imageData, 0, 0);
  ctx.drawImage(off, 0, 0, w, h);
}

function renderPreludeCanvas() {
  const canvas = document.getElementById('preludeCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas, DISPLAY_W, DISPLAY_H);
  drawFrame(ctx, DISPLAY_W, DISPLAY_H, state.currImage);
}

function renderPrevCurr() {
  const prev = document.getElementById('prevCanvas');
  const curr = document.getElementById('currCanvas');
  if (prev) {
    const ctx = setupCanvas(prev, 320, 180);
    drawFrame(ctx, 320, 180, state.prevImage);
    if (!state.prevImage) {
      ctx.fillStyle = '#c4beb1';
      ctx.font = '14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Step or play the video', 160, 90);
    }
  }
  if (curr) {
    const ctx = setupCanvas(curr, 320, 180);
    drawFrame(ctx, 320, 180, state.currImage);
  }
}

function renderDiff() {
  const canvas = document.getElementById('diffCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas, DISPLAY_W, DISPLAY_H);
  if (!state.prevGray || !state.currGray) {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, DISPLAY_W, DISPLAY_H);
    return;
  }
  const buf = new Uint8ClampedArray(WORK_W * WORK_H * 4);
  for (let i = 0; i < WORK_W * WORK_H; i++) {
    const d = state.currGray[i] - state.prevGray[i];
    // Map [-128..128] to color: red for positive, blue for negative, grey for 0
    const m = Math.min(255, Math.abs(d) * 3);
    if (d > 0) {
      buf[i * 4] = 128 + m / 2;
      buf[i * 4 + 1] = 128;
      buf[i * 4 + 2] = 128;
    } else {
      buf[i * 4] = 128;
      buf[i * 4 + 1] = 128;
      buf[i * 4 + 2] = 128 + m / 2;
    }
    buf[i * 4 + 3] = 255;
  }
  const off = document.createElement('canvas');
  off.width = WORK_W; off.height = WORK_H;
  off.getContext('2d').putImageData(new ImageData(buf, WORK_W, WORK_H), 0, 0);
  ctx.drawImage(off, 0, 0, DISPLAY_W, DISPLAY_H);
}

function renderLK() {
  const canvas = document.getElementById('lkCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas, DISPLAY_W, DISPLAY_H);
  drawFrame(ctx, DISPLAY_W, DISPLAY_H, state.currImage);
  // Scale factor from work to display
  const sx = DISPLAY_W / WORK_W;
  const sy = DISPLAY_H / WORK_H;
  const magAmp = 4;
  for (const p of state.lkPoints) {
    const px = p.x * sx, py = p.y * sy;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#2c6fb7';
    ctx.fill();
    // Arrow
    const tx = px + p.u * sx * magAmp;
    const ty = py + p.v * sy * magAmp;
    ctx.strokeStyle = '#d9622b';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    const ang = Math.atan2(ty - py, tx - px);
    const head = 4;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - head * Math.cos(ang - Math.PI / 6),
               ty - head * Math.sin(ang - Math.PI / 6));
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - head * Math.cos(ang + Math.PI / 6),
               ty - head * Math.sin(ang + Math.PI / 6));
    ctx.stroke();
  }

  // Stats
  const n = state.lkPoints.length;
  document.getElementById('lk-count').textContent = n;
  if (n === 0) {
    document.getElementById('lk-mean').textContent = '—';
    document.getElementById('lk-max').textContent = '—';
  } else {
    let sum = 0, mx = 0;
    for (const p of state.lkPoints) {
      const m = Math.hypot(p.u, p.v);
      sum += m;
      if (m > mx) mx = m;
    }
    document.getElementById('lk-mean').textContent = (sum / n).toFixed(2) + ' px';
    document.getElementById('lk-max').textContent = mx.toFixed(2) + ' px';
  }
}

function renderDense() {
  const canvas = document.getElementById('denseCanvas');
  if (!canvas) return;
  const ctx = setupCanvas(canvas, DISPLAY_W, DISPLAY_H);
  drawFrame(ctx, DISPLAY_W, DISPLAY_H, state.currImage);
  if (!state.denseFlow) return;
  // Build HSV-coloured overlay at grid resolution, upsample to display
  const gw = state.denseGw, gh = state.denseGh;
  const step = state.denseStep;
  const buf = new Uint8ClampedArray(gw * gh * 4);
  const maxMag = 5; // pixels; flow above this saturates
  for (let i = 0; i < gw * gh; i++) {
    const u = state.denseFlow[i * 2];
    const v = state.denseFlow[i * 2 + 1];
    const m = Math.hypot(u, v);
    if (m < state.magThreshold) {
      buf[i * 4 + 3] = 0;
      continue;
    }
    const angle = Math.atan2(-v, u); // -v so that +y downward gives hue magenta
    const hue = (angle / (2 * Math.PI) + 1) % 1;
    const val = Math.min(1, m / maxMag);
    const [r, g, b] = hsvToRgb(hue, 1, val);
    buf[i * 4] = r; buf[i * 4 + 1] = g; buf[i * 4 + 2] = b;
    buf[i * 4 + 3] = Math.round(val * 230);
  }
  const off = document.createElement('canvas');
  off.width = gw; off.height = gh;
  off.getContext('2d').putImageData(new ImageData(buf, gw, gh), 0, 0);
  // Scale to match the working image, then to display.
  // The grid spans WORK_W×WORK_H with `step` spacing, so effectively:
  //   grid covers [0, gw*step] × [0, gh*step] ≈ WORK image area.
  ctx.globalAlpha = 0.85;
  ctx.drawImage(off, 0, 0, DISPLAY_W, DISPLAY_H);
  ctx.globalAlpha = 1;
}

// ---------- Playback ----------
let rafId = null;
function playLoop() {
  if (!state.playing) return;
  const video = state.video;
  if (!video || video.paused || video.ended) {
    state.playing = false;
    document.getElementById('btn-play').textContent = 'Play';
    return;
  }
  // Grab a frame every ~2 frames of the video to keep compute manageable
  grabFrame();
  rafId = requestAnimationFrame(playLoop);
}

function togglePlay() {
  const video = state.video;
  if (!video || !state.videoReady) return;
  if (state.playing) {
    video.pause();
    state.playing = false;
    document.getElementById('btn-play').textContent = 'Play';
    if (rafId) cancelAnimationFrame(rafId);
  } else {
    video.play().then(() => {
      state.playing = true;
      document.getElementById('btn-play').textContent = 'Pause';
      playLoop();
    }).catch((err) => {
      console.warn('Play failed:', err);
    });
  }
}

function stepFrame(dir) {
  const video = state.video;
  if (!video || !state.videoReady) return;
  video.pause();
  state.playing = false;
  document.getElementById('btn-play').textContent = 'Play';
  const dt = 1 / state.fps;
  const target = Math.max(0, Math.min(video.duration - dt, video.currentTime + dir * dt));
  video.currentTime = target;
  video.addEventListener('seeked', () => {
    grabFrame();
  }, { once: true });
}

function resetVideo() {
  const video = state.video;
  if (!video) return;
  video.pause();
  state.playing = false;
  document.getElementById('btn-play').textContent = 'Play';
  video.currentTime = 0;
  state.prevGray = null;
  state.prevImage = null;
  state.lkPoints = [];
  state.denseFlow = null;
  video.addEventListener('seeked', () => { grabFrame(); }, { once: true });
}

// ---------- Stats ----------
function updateStats() {
  const video = state.video;
  const vs = document.getElementById('vid-size');
  const vf = document.getElementById('vid-fps');
  const ms = document.getElementById('flow-ms');
  const ps = document.getElementById('player-status');
  if (video && video.videoWidth) {
    vs.textContent = `${video.videoWidth}×${video.videoHeight}`;
    vf.textContent = `${state.fps} fps (assumed)`;
    if (ps) {
      const cur = video.currentTime.toFixed(2);
      const tot = (video.duration || 0).toFixed(2);
      ps.textContent = `t = ${cur}s / ${tot}s`;
    }
  }
  ms.textContent = state.flowMs > 0 ? `${state.flowMs.toFixed(0)} ms` : '—';
}

// ---------- Render-all ----------
function renderAll() {
  renderPreludeCanvas();
  renderPrevCurr();
  renderDiff();
  renderLK();
  renderDense();
}

// ---------- Math ----------
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-bc': 'I(x, y, t) = I(x + u, y + v, t + 1)',
    'math-ofce': 'I_x \\, u + I_y \\, v + I_t = 0',
    'math-lk':
      '\\sum_{(x, y) \\in W} \\begin{bmatrix} I_x \\\\ I_y \\end{bmatrix} \\begin{bmatrix} I_x & I_y \\end{bmatrix} \\begin{bmatrix} u \\\\ v \\end{bmatrix} = -\\sum_{(x, y) \\in W} \\begin{bmatrix} I_x \\\\ I_y \\end{bmatrix} I_t'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

// ---------- Input ----------
function wireControls() {
  const grid = document.getElementById('sample-grid');
  if (grid) {
    grid.innerHTML = SAMPLES.map((s) =>
      `<button class="sample-thumb" data-sample="${s.key}">${s.label}</button>`
    ).join('');
    grid.querySelectorAll('[data-sample]').forEach((b) => {
      b.addEventListener('click', () => {
        grid.querySelectorAll('[data-sample]').forEach((bb) =>
          bb.classList.toggle('is-active', bb === b));
        const s = SAMPLES.find((x) => x.key === b.dataset.sample);
        if (s) loadSample(s.url);
      });
    });
  }

  // Upload
  const zone = document.getElementById('upload-zone');
  const input = document.getElementById('video-input');
  input.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) handleFile(f);
  });
  ['dragenter', 'dragover'].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('drag-over'); }));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  // Player
  document.getElementById('btn-play').addEventListener('click', togglePlay);
  document.getElementById('btn-step-fwd').addEventListener('click', () => stepFrame(1));
  document.getElementById('btn-step-back').addEventListener('click', () => stepFrame(-1));
  document.getElementById('btn-reset').addEventListener('click', resetVideo);

  // Sliders
  const bind = (sliderId, valId, prop, parser = parseFloat) => {
    const s = document.getElementById(sliderId);
    const v = document.getElementById(valId);
    s.addEventListener('input', () => {
      state[prop] = parser(s.value);
      v.textContent = s.value;
      if (state.prevGray && state.currGray) {
        computeLK();
        computeDenseFlow();
        renderLK();
        renderDense();
      }
    });
  };
  bind('win-slider', 'win-val', 'windowSize', parseInt);
  bind('nfeat-slider', 'nfeat-val', 'maxFeatures', parseInt);
  bind('quality-slider', 'quality-val', 'minQuality', parseFloat);
  bind('grid-slider', 'grid-val', 'gridStep', parseInt);
  bind('mag-slider', 'mag-val', 'magThreshold', parseFloat);
}

// ---------- Boot ----------
function init() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  state.video = document.getElementById('video');
  wireControls();
  // Auto-load first sample
  if (SAMPLES[0]) {
    document.querySelectorAll('#sample-grid [data-sample]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.sample === SAMPLES[0].key));
    loadSample(SAMPLES[0].url);
  }
  // Video metadata
  state.video.addEventListener('play', () => { state.playing = true; });
  state.video.addEventListener('pause', () => { state.playing = false; });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
