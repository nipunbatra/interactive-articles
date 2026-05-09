// ============================================================
// Information Theory for ML, by Hand
// All quantities computed live in the browser.
// ============================================================

const SERIF = '13px Manrope';
const PRIOR_COLOR = '#2c6fb7';
const LIK_COLOR = '#d9622b';
const POST_COLOR = '#1e7770';
const TICK_COLOR = '#9a917f';

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function gaussianPdf(x, mu, s) {
  const z = (x - mu) / s;
  return Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * s);
}
function randn() {
  const u1 = Math.random() || 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ============================================================
// Step 1 — surprise curve
// ============================================================
function renderInfo() {
  const canvas = document.getElementById('info-canvas');
  if (!canvas) return;
  const W = 880, H = 240;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 18, t: 24, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  // Y-axis ticks
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = 0; v <= 8; v += 2) {
    const y = m.t + (1 - v / 8) * py;
    ctx.fillText(v.toFixed(0), m.l - 4, y + 4);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  // X-axis ticks
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    const x = m.l + v * px;
    ctx.fillText(v.toFixed(1), x, m.t + py + 16);
  }
  // Curve
  ctx.strokeStyle = LIK_COLOR;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i <= 240; i++) {
    const p = i / 240;
    const I = p === 0 ? 8 : Math.min(8, -Math.log2(p));
    const x = m.l + p * px;
    const y = m.t + (1 - I / 8) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = '#3b342b';
  ctx.font = SERIF;
  ctx.textAlign = 'center';
  ctx.fillText('information I(p) = −log₂ p   (bits)', m.l + px / 2, m.t - 8);
  ctx.fillText('p', m.l + px / 2, H - 4);
}

// ============================================================
// Step 2 — entropy of categorical (drag bars)
// ============================================================
const ENT = {
  K: 6,
  P: [0.30, 0.20, 0.18, 0.14, 0.10, 0.08],
  drag: null
};

function entropy(P) {
  let H = 0;
  for (const p of P) if (p > 0) H -= p * Math.log2(p);
  return H;
}

function renormalize(P) {
  const s = P.reduce((a, b) => a + b, 0) || 1;
  return P.map((v) => v / s);
}

function renderEntropy() {
  const canvas = document.getElementById('entropy-canvas');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 220, t: 30, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);

  // Bars
  const bw = px / ENT.K;
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    const y = m.t + (1 - v) * py;
    ctx.fillText(v.toFixed(1), m.l - 4, y + 4);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ENT.P.forEach((p, i) => {
    const x0 = m.l + i * bw + 8;
    const w = bw - 16;
    const h = p * py;
    ctx.fillStyle = `rgba(44,111,183,${0.3 + 0.5 * p})`;
    ctx.fillRect(x0, m.t + py - h, w, h);
    ctx.fillStyle = '#1a1815';
    ctx.font = '11px IBM Plex Mono';
    ctx.textAlign = 'center';
    ctx.fillText(p.toFixed(2), x0 + w / 2, m.t + py - h - 4);
    ctx.fillStyle = '#9a917f';
    ctx.font = SERIF;
    ctx.fillText(`x${i + 1}`, x0 + w / 2, m.t + py + 16);
  });

  // Side panel
  const sx = m.l + px + 24;
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 14px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('Entropy', sx, m.t + 14);
  ctx.font = '32px Manrope';
  ctx.fillStyle = PRIOR_COLOR;
  ctx.fillText(`${entropy(ENT.P).toFixed(3)} bits`, sx, m.t + 50);
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.fillText(`max H = log₂(${ENT.K}) = ${Math.log2(ENT.K).toFixed(2)} bits`, sx, m.t + 70);
  ctx.fillText(`min H = 0 (one-hot)`, sx, m.t + 88);
  ctx.fillText('Drag any bar top to reshape P.', sx, m.t + 116);
  ctx.fillText('Bars renormalise to sum to 1.', sx, m.t + 134);
}

function wireEntropy() {
  const canvas = document.getElementById('entropy-canvas');
  const m = { l: 60, r: 220, t: 30, b: 36 };
  function hit(mx, my) {
    const W = canvas.clientWidth, H = canvas.clientHeight;
    const px = W - m.l - m.r, py = H - m.t - m.b;
    const bw = px / ENT.K;
    if (mx < m.l || mx > m.l + px) return null;
    const idx = Math.floor((mx - m.l) / bw);
    if (idx < 0 || idx >= ENT.K) return null;
    return { idx, py };
  }
  function setBarFromY(idx, my) {
    const py = canvas.clientHeight - m.t - m.b;
    let v = 1 - (my - m.t) / py;
    v = Math.max(0.001, Math.min(1, v));
    ENT.P[idx] = v;
    ENT.P = renormalize(ENT.P);
    renderEntropy();
  }
  canvas.addEventListener('mousedown', (e) => {
    const r = canvas.getBoundingClientRect();
    const h = hit(e.clientX - r.left, e.clientY - r.top);
    if (h) ENT.drag = h.idx;
  });
  window.addEventListener('mousemove', (e) => {
    if (ENT.drag == null) return;
    const r = canvas.getBoundingClientRect();
    const my = e.clientY - r.top;
    setBarFromY(ENT.drag, my);
  });
  window.addEventListener('mouseup', () => { ENT.drag = null; });

  document.getElementById('ent-uniform').addEventListener('click', () => {
    ENT.P = new Array(ENT.K).fill(1 / ENT.K); renderEntropy();
  });
  document.getElementById('ent-onehot').addEventListener('click', () => {
    ENT.P = new Array(ENT.K).fill(1e-6); ENT.P[2] = 1; ENT.P = renormalize(ENT.P); renderEntropy();
  });
  document.getElementById('ent-randomize').addEventListener('click', () => {
    ENT.P = ENT.P.map(() => Math.random() + 0.05); ENT.P = renormalize(ENT.P); renderEntropy();
  });
}

// ============================================================
// Step 3 — cross-entropy = entropy + KL
// ============================================================
const CE = {
  K: 6,
  P: [0.30, 0.20, 0.18, 0.14, 0.10, 0.08],
  Q: [0.10, 0.10, 0.18, 0.30, 0.18, 0.14],
  drag: null
};

function crossEntropy(P, Q) {
  let H = 0;
  for (let i = 0; i < P.length; i++) {
    if (P[i] > 0) H -= P[i] * Math.log2(Math.max(Q[i], 1e-12));
  }
  return H;
}
function klBits(P, Q) {
  return crossEntropy(P, Q) - entropy(P);
}

function renderCE() {
  const canvas = document.getElementById('ce-canvas');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 270, t: 30, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  // Side-by-side bars: P in blue, Q in orange
  const bw = px / CE.K / 2;
  for (let i = 0; i < CE.K; i++) {
    const xP = m.l + i * 2 * bw + 6;
    const xQ = m.l + (i * 2 + 1) * bw + 6;
    const hP = CE.P[i] * py;
    const hQ = CE.Q[i] * py;
    ctx.fillStyle = 'rgba(44,111,183,0.7)';
    ctx.fillRect(xP, m.t + py - hP, bw - 8, hP);
    ctx.fillStyle = 'rgba(217,98,43,0.65)';
    ctx.fillRect(xQ, m.t + py - hQ, bw - 8, hQ);
    ctx.fillStyle = '#9a917f';
    ctx.font = SERIF;
    ctx.textAlign = 'center';
    ctx.fillText(`x${i + 1}`, m.l + i * 2 * bw + bw, m.t + py + 16);
  }

  // Side panel: split bar visualisation
  const sx = m.l + px + 24;
  const HP = entropy(CE.P);
  const KL = klBits(CE.P, CE.Q);
  const HPQ = HP + KL;
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 14px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('Decomposition (bits)', sx, m.t + 14);
  // Split bar
  const barW = 220;
  const blueW = HP / Math.max(0.1, HPQ) * barW;
  const orangeW = barW - blueW;
  const by = m.t + 32;
  ctx.fillStyle = 'rgba(44,111,183,0.85)';
  ctx.fillRect(sx, by, blueW, 26);
  ctx.fillStyle = 'rgba(217,98,43,0.85)';
  ctx.fillRect(sx + blueW, by, orangeW, 26);
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(sx, by, barW, 26);
  // labels
  ctx.font = '12px Manrope';
  ctx.fillStyle = '#1a1815';
  ctx.textAlign = 'left';
  ctx.fillText(`H(P, Q) = ${HPQ.toFixed(3)}`, sx, by + 50);
  ctx.fillStyle = PRIOR_COLOR;
  ctx.fillText(`H(P) = ${HP.toFixed(3)}`, sx, by + 70);
  ctx.fillStyle = LIK_COLOR;
  ctx.fillText(`KL(P‖Q) = ${KL.toFixed(3)}`, sx, by + 88);
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.fillText('Drag orange Q bars; KL drops to 0 when Q = P.', sx, by + 116);
}

function wireCE() {
  const canvas = document.getElementById('ce-canvas');
  const m = { l: 60, r: 270, t: 30, b: 36 };
  canvas.addEventListener('mousedown', (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const W = r.width, H = r.height;
    const px = W - m.l - m.r;
    if (mx < m.l || mx > m.l + px) return;
    const bw = px / CE.K / 2;
    const idx = Math.floor((mx - m.l) / bw);
    if (idx % 2 !== 1) return; // only Q bars are draggable
    const k = (idx - 1) / 2;
    if (k < 0 || k >= CE.K) return;
    CE.drag = k;
  });
  window.addEventListener('mousemove', (e) => {
    if (CE.drag == null) return;
    const r = canvas.getBoundingClientRect();
    const my = e.clientY - r.top;
    const py = r.height - m.t - m.b;
    let v = 1 - (my - m.t) / py;
    v = Math.max(0.001, Math.min(1, v));
    CE.Q[CE.drag] = v;
    CE.Q = renormalize(CE.Q);
    renderCE();
  });
  window.addEventListener('mouseup', () => { CE.drag = null; });
  document.getElementById('ce-match').addEventListener('click', () => {
    CE.Q = CE.P.slice(); renderCE();
  });
  document.getElementById('ce-uniform').addEventListener('click', () => {
    CE.Q = new Array(CE.K).fill(1 / CE.K); renderCE();
  });
  document.getElementById('ce-randomize').addEventListener('click', () => {
    CE.Q = CE.Q.map(() => Math.random() + 0.05); CE.Q = renormalize(CE.Q); renderCE();
  });
}

// ============================================================
// Step 4 — Forward vs reverse KL fitting Gaussian to bimodal
// ============================================================
const KLF = {
  mode: 'forward',
  // Bimodal target P: 0.55*N(-1.6, 0.5) + 0.45*N(1.5, 0.65)
  fit: { mu: 0, sigma: 1 }
};

function pBimodal(x) {
  return 0.55 * gaussianPdf(x, -1.6, 0.5) + 0.45 * gaussianPdf(x, 1.5, 0.65);
}

function fitKL(direction) {
  // Numerical minimisation over (mu, sigma) of forward or reverse KL
  // KL(P||Q) = ∫ p log(p/q) dx ; KL(Q||P) = ∫ q log(q/p) dx
  const xs = [];
  const N = 401;
  for (let i = 0; i < N; i++) xs.push(-5 + 10 * (i / (N - 1)));
  const dx = xs[1] - xs[0];
  const px = xs.map(pBimodal);
  function lossFor(mu, sigma) {
    let L = 0;
    for (let i = 0; i < N; i++) {
      const q = gaussianPdf(xs[i], mu, sigma);
      const p = px[i];
      if (direction === 'forward') {
        if (p > 1e-12 && q > 1e-12) L += p * Math.log(p / q) * dx;
      } else {
        if (q > 1e-12 && p > 1e-12) L += q * Math.log(q / p) * dx;
        else if (q > 1e-12) L += q * 30 * dx; // huge penalty if p=0 but q>0
      }
    }
    return L;
  }
  // Coordinate-descent grid search with refinement
  let best = { mu: 0, sigma: 1, val: Infinity };
  for (let mu = -3; mu <= 3.001; mu += 0.05) {
    for (let s = 0.2; s <= 2.5; s += 0.05) {
      const v = lossFor(mu, s);
      if (v < best.val) best = { mu, sigma: s, val: v };
    }
  }
  return best;
}

function renderKL() {
  const canvas = document.getElementById('kl-canvas');
  if (!canvas) return;
  const W = 880, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 18, t: 24, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  // axes
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'center';
  for (let v = -4; v <= 4; v++) {
    const x = m.l + (v + 5) / 10 * px;
    ctx.fillText(v, x, m.t + py + 16);
  }
  const xMin = -5, xMax = 5;
  // P
  const N = 320;
  let yMax = 0;
  for (let i = 0; i <= N; i++) {
    const x = xMin + (xMax - xMin) * (i / N);
    yMax = Math.max(yMax, pBimodal(x));
  }
  yMax *= 1.15;
  const xs = (x) => m.l + (x - xMin) / (xMax - xMin) * px;
  const ys = (y) => m.t + (1 - y / yMax) * py;
  // Fill P
  ctx.fillStyle = 'rgba(30,119,112,0.16)';
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const x = xMin + (xMax - xMin) * (i / N);
    const y = pBimodal(x);
    const sx = xs(x), sy = ys(y);
    if (i === 0) ctx.moveTo(sx, ys(0));
    ctx.lineTo(sx, sy);
  }
  ctx.lineTo(xs(xMax), ys(0));
  ctx.closePath();
  ctx.fill();
  // P stroke
  ctx.strokeStyle = POST_COLOR;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const x = xMin + (xMax - xMin) * (i / N);
    const sx = xs(x), sy = ys(pBimodal(x));
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  // Q
  ctx.strokeStyle = LIK_COLOR;
  ctx.lineWidth = 2.5;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const x = xMin + (xMax - xMin) * (i / N);
    const sx = xs(x), sy = ys(gaussianPdf(x, KLF.fit.mu, KLF.fit.sigma));
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // Update stats
  document.getElementById('kl-mu').textContent = KLF.fit.mu.toFixed(2);
  document.getElementById('kl-sigma').textContent = KLF.fit.sigma.toFixed(2);
  document.getElementById('kl-val').textContent = (KLF.fit.val ?? 0).toFixed(3);
  // Mark active button
  document.getElementById('kl-forward').classList.toggle('action-btn--ghost', KLF.mode !== 'forward');
  document.getElementById('kl-reverse').classList.toggle('action-btn--ghost', KLF.mode !== 'reverse');
}

function wireKL() {
  document.getElementById('kl-forward').addEventListener('click', () => {
    KLF.mode = 'forward';
    KLF.fit = fitKL('forward'); renderKL();
  });
  document.getElementById('kl-reverse').addEventListener('click', () => {
    KLF.mode = 'reverse';
    KLF.fit = fitKL('reverse'); renderKL();
  });
  KLF.fit = fitKL('forward'); renderKL();
}

// ============================================================
// Step 5 — Mutual information (2D Gaussian)
// ============================================================
const MIst = { rho: 0.7, noise: 0.3, samples: null };

function makeMISamples(n, rho, noise) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const x = randn();
    const y = rho * x + Math.sqrt(1 - rho * rho) * randn() * noise / 0.3;
    out[i] = { x, y };
  }
  return out;
}

function empiricalMI(samples, bins) {
  const xs = samples.map((s) => s.x);
  const ys = samples.map((s) => s.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const Pxy = new Array(bins).fill(0).map(() => new Array(bins).fill(0));
  const Px = new Array(bins).fill(0);
  const Py = new Array(bins).fill(0);
  samples.forEach((s) => {
    const ix = Math.min(bins - 1, Math.max(0, Math.floor((s.x - xMin) / (xMax - xMin + 1e-9) * bins)));
    const iy = Math.min(bins - 1, Math.max(0, Math.floor((s.y - yMin) / (yMax - yMin + 1e-9) * bins)));
    Pxy[ix][iy]++;
    Px[ix]++;
    Py[iy]++;
  });
  const N = samples.length;
  let mi = 0;
  for (let i = 0; i < bins; i++) {
    for (let j = 0; j < bins; j++) {
      if (Pxy[i][j] === 0) continue;
      const pxy = Pxy[i][j] / N;
      const px_ = Px[i] / N;
      const py_ = Py[j] / N;
      mi += pxy * Math.log2(pxy / (px_ * py_));
    }
  }
  return mi;
}

function renderMI() {
  const canvas = document.getElementById('mi-canvas');
  if (!canvas) return;
  const W = 880, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  // Left: scatter
  const sLeft = { l: 50, r: 460, t: 30, b: 40 };
  const px = W - sLeft.l - sLeft.r, py = H - sLeft.t - sLeft.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(sLeft.l, sLeft.t, px, py);
  // Compute samples
  if (!MIst.samples || MIst.samples.length !== 600) MIst.samples = makeMISamples(600, MIst.rho, MIst.noise);
  // Re-make samples on slider change
  MIst.samples = makeMISamples(600, MIst.rho, MIst.noise);
  const xs = MIst.samples.map((s) => s.x), ys = MIst.samples.map((s) => s.y);
  const xMin = -3, xMax = 3, yMin = -3, yMax = 3;
  const sx = (x) => sLeft.l + (x - xMin) / (xMax - xMin) * px;
  const sy = (y) => sLeft.t + (1 - (y - yMin) / (yMax - yMin)) * py;
  MIst.samples.forEach((p) => {
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(p.y), 1.8, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(44,111,183,0.55)';
    ctx.fill();
  });
  ctx.fillStyle = '#3b342b';
  ctx.font = SERIF;
  ctx.textAlign = 'center';
  ctx.fillText('joint scatter (X, Y)', sLeft.l + px / 2, sLeft.t - 8);
  ctx.fillText('X', sLeft.l + px / 2, H - 6);
  ctx.save();
  ctx.translate(14, sLeft.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Y', 0, 0);
  ctx.restore();

  // Right: stats
  const rx = sLeft.l + px + 36;
  const ry0 = sLeft.t + 4;
  const empirical = empiricalMI(MIst.samples, 18);
  const analytic = -0.5 * Math.log2(Math.max(1e-9, 1 - MIst.rho * MIst.rho));
  ctx.fillStyle = '#1a1815';
  ctx.font = 'bold 14px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('I(X; Y) (bits)', rx, ry0 + 12);
  // Bars
  const maxMI = 4;
  const barW = 280;
  const drawBar = (label, value, color, y) => {
    ctx.fillStyle = '#1a1815';
    ctx.font = '12px Manrope';
    ctx.textAlign = 'left';
    ctx.fillText(label, rx, y);
    ctx.strokeStyle = '#e2d8c6';
    ctx.strokeRect(rx, y + 6, barW, 22);
    ctx.fillStyle = color;
    const w = Math.max(0, Math.min(barW, (value / maxMI) * barW));
    ctx.fillRect(rx, y + 6, w, 22);
    ctx.fillStyle = '#1a1815';
    ctx.font = '12px IBM Plex Mono';
    ctx.fillText(value.toFixed(3), rx + barW + 8, y + 22);
  };
  drawBar(`empirical (18-bin histogram)`, empirical, 'rgba(44,111,183,0.7)', ry0 + 40);
  drawBar(`analytic Gaussian: −½ log₂(1−ρ²)`, analytic, 'rgba(30,119,112,0.7)', ry0 + 90);
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText(`ρ = ${MIst.rho.toFixed(2)}, noise = ${MIst.noise.toFixed(2)}`, rx, ry0 + 150);
  ctx.fillText(`I = 0 ⇔ ρ = 0 (independence). Slide ρ to 0.`, rx, ry0 + 168);
  ctx.fillText('Histogram MI lower-bounds the true MI;', rx, ry0 + 186);
  ctx.fillText('the gap shrinks as you take more samples.', rx, ry0 + 204);
}

function wireMI() {
  const rho = document.getElementById('mi-rho');
  const noise = document.getElementById('mi-n');
  rho.addEventListener('input', () => {
    MIst.rho = parseFloat(rho.value);
    document.getElementById('mi-rho-val').textContent = MIst.rho.toFixed(2);
    renderMI();
  });
  noise.addEventListener('input', () => {
    MIst.noise = parseFloat(noise.value);
    document.getElementById('mi-n-val').textContent = MIst.noise.toFixed(2);
    renderMI();
  });
}

// ============================================================
// Step 6 — InfoNCE bound vs analytic MI
// ============================================================
const NCE = { K: 16, signal: 2.0 };

function infoNCEBound(K, signal) {
  // For a Gaussian setup with paired correlation = rho determined by signal,
  // estimate the InfoNCE expectation analytically:
  //   InfoNCE = E[log K * exp(f(x,y+)) / (sum_k exp(f(x,y_k)))]
  // We synthesise: x ~ N(0, 1), y+ = rho*x + sqrt(1-rho^2) eps,  rho = tanh(signal/2)
  const rho = Math.tanh(signal / 2);
  // Analytic Gaussian MI:
  const I = -0.5 * Math.log2(Math.max(1e-9, 1 - rho * rho));
  // Estimate InfoNCE bound by Monte Carlo
  const T = 800;
  let bound = 0;
  for (let t = 0; t < T; t++) {
    const x = randn();
    const yp = rho * x + Math.sqrt(1 - rho * rho) * randn();
    let logSum = signal * x * yp; // f(x, y+)
    let maxL = signal * x * yp;
    const ks = [];
    for (let k = 0; k < K; k++) {
      const ynk = randn();
      ks.push(signal * x * ynk);
      maxL = Math.max(maxL, signal * x * ynk);
    }
    let s = Math.exp(signal * x * yp - maxL);
    for (let k = 0; k < K; k++) s += Math.exp(ks[k] - maxL);
    bound += (signal * x * yp - maxL - Math.log(s)) / Math.log(2);
  }
  bound /= T;
  // Lower bound on MI: log_2 K + InfoNCE_value
  const lower = Math.log2(K) + bound;
  return { I, lower, K, rho };
}

function renderNCE() {
  const canvas = document.getElementById('nce-canvas');
  if (!canvas) return;
  const W = 880, H = 280;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 12, t: 24, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  // X axis = signal (1..6), Y = MI (bits)
  const Ks = [4, 16, 64, 256];
  const colors = ['#2c6fb7', '#1e7770', '#9b59b6', '#d9622b'];
  // True MI curve
  const xs = (s) => m.l + (s - 0.1) / (6 - 0.1) * px;
  const ys = (mi) => m.t + (1 - mi / 6) * py;
  // Grid
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = 0; v <= 6; v++) {
    const y = ys(v);
    ctx.fillText(v.toFixed(0), m.l - 4, y + 4);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  for (let s = 0.1; s <= 6; s += 1) {
    const x = xs(s);
    ctx.fillText(s.toFixed(1), x, m.t + py + 16);
  }
  // Analytic MI curve
  ctx.strokeStyle = '#1a1815';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const sig = 0.1 + (6 - 0.1) * (i / 60);
    const rho = Math.tanh(sig / 2);
    const I = -0.5 * Math.log2(Math.max(1e-9, 1 - rho * rho));
    const x = xs(sig), y = ys(Math.min(6, I));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  // InfoNCE bounds
  Ks.forEach((K, ki) => {
    ctx.strokeStyle = colors[ki];
    ctx.lineWidth = 1.8;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    for (let i = 0; i <= 24; i++) {
      const sig = 0.1 + (6 - 0.1) * (i / 24);
      const r = infoNCEBound(K, sig);
      const I = Math.min(6, Math.max(0, r.lower));
      const x = xs(sig), y = ys(I);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  ctx.setLineDash([]);
  // Highlight current K and signal
  const cur = infoNCEBound(NCE.K, NCE.signal);
  const cx = xs(NCE.signal), cy = ys(Math.min(6, cur.lower));
  ctx.fillStyle = '#1a1815';
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  ctx.fill();
  // Legend
  ctx.fillStyle = '#3b342b';
  ctx.font = SERIF;
  ctx.textAlign = 'left';
  ctx.fillText('analytic MI', m.l + 8, m.t + 18);
  let lx = m.l + 110;
  Ks.forEach((K, ki) => {
    ctx.strokeStyle = colors[ki];
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(lx, m.t + 14); ctx.lineTo(lx + 18, m.t + 14);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#3b342b';
    ctx.fillText(`K=${K}`, lx + 22, m.t + 18);
    lx += 70;
  });
  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'right';
  ctx.fillText(`current: K=${NCE.K}, signal=${NCE.signal.toFixed(2)} → bound = ${cur.lower.toFixed(2)}, MI = ${cur.I.toFixed(2)}`, m.l + px - 8, m.t + 18);
  ctx.textAlign = 'center';
  ctx.fillText('signal', m.l + px / 2, H - 6);
}

function wireNCE() {
  const K = document.getElementById('nce-K');
  const s = document.getElementById('nce-s');
  K.addEventListener('input', () => {
    NCE.K = parseInt(K.value, 10);
    document.getElementById('nce-K-val').textContent = NCE.K;
    renderNCE();
  });
  s.addEventListener('input', () => {
    NCE.signal = parseFloat(s.value);
    document.getElementById('nce-s-val').textContent = NCE.signal.toFixed(1);
    renderNCE();
  });
}

// ============================================================
// Math + boot
// ============================================================
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-info': 'I(p) = -\\log_2 p \\quad \\text{bits}',
    'math-entropy': 'H(P) = \\mathbb{E}_{x \\sim P}[\\,-\\log_2 P(x)\\,] = -\\sum_x P(x)\\,\\log_2 P(x)',
    'math-ce':
      'H(P, Q) \\;=\\; -\\sum_x P(x) \\log Q(x) \\;=\\; \\underbrace{H(P)}_{\\text{entropy}} \\;+\\; \\underbrace{\\mathrm{KL}(P\\,\\|\\,Q)}_{\\text{wasted bits}}',
    'math-mi':
      'I(X; Y) = \\mathrm{KL}\\!\\left(p(x, y)\\,\\big\\|\\,p(x)p(y)\\right) = H(Y) - H(Y\\mid X)',
    'math-infonce':
      '\\mathcal{L}_{\\text{InfoNCE}} = -\\,\\mathbb{E}\\!\\left[\\log\\frac{\\exp f(x, y_+)}{\\sum_{k=0}^{K} \\exp f(x, y_k)}\\right] \\;\\Longrightarrow\\; I(X; Y) \\ge \\log K - \\mathcal{L}_{\\text{InfoNCE}}'
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
  renderInfo();
  wireEntropy(); renderEntropy();
  wireCE(); renderCE();
  wireKL();
  wireMI(); renderMI();
  wireNCE(); renderNCE();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
