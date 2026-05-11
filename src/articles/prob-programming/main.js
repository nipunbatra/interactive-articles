// ============================================================
// PPL article — coin-bias HMC and Bayesian regression HMC vs VI.
// ============================================================

const COIN = { k: 0, n: 0, samples: [], rhat: null, ess: null };

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
function randn() { const u = Math.random() || 1e-12, v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

function logGamma(z) {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
function betaPdf(x, a, b) {
  if (x <= 0 || x >= 1) return 0;
  const lp = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - (logGamma(a) + logGamma(b) - logGamma(a + b));
  return Math.exp(lp);
}

// Simple Metropolis HMC for the Beta-Binomial posterior in [0,1]
function runHMCCoin(nSamples = 500, nChains = 4) {
  const a = 1 + COIN.k, b = 1 + (COIN.n - COIN.k);
  function logTarget(theta) {
    if (theta <= 0 || theta >= 1) return -Infinity;
    return (a - 1) * Math.log(theta) + (b - 1) * Math.log(1 - theta);
  }
  const chains = [];
  for (let c = 0; c < nChains; c++) {
    const chain = [];
    let cur = 0.1 + 0.8 * Math.random();
    let accepted = 0;
    for (let i = 0; i < nSamples; i++) {
      const proposal = cur + 0.06 * randn();
      const logR = logTarget(proposal) - logTarget(cur);
      if (Math.log(Math.random() || 1e-12) < logR) {
        cur = proposal;
        accepted++;
      }
      chain.push(cur);
    }
    chains.push(chain);
  }
  return chains;
}

function rhatAndESS(chains) {
  const m = chains.length;
  const n = chains[0].length;
  const chainMeans = chains.map((c) => c.reduce((s, x) => s + x, 0) / n);
  const grandMean = chainMeans.reduce((s, x) => s + x, 0) / m;
  const B = n * chainMeans.reduce((s, x) => s + (x - grandMean) * (x - grandMean), 0) / (m - 1);
  const W = chains.reduce((s, c) => {
    const mean = c.reduce((s, x) => s + x, 0) / n;
    return s + c.reduce((s, x) => s + (x - mean) * (x - mean), 0) / (n - 1);
  }, 0) / m;
  const varHat = ((n - 1) / n) * W + B / n;
  const rhat = Math.sqrt(varHat / Math.max(1e-9, W));
  // Crude ESS: m*n / (1 + 2 * sum of autocorrelations up to lag 30)
  const total = m * n;
  let allSamples = [];
  chains.forEach((c) => { allSamples = allSamples.concat(c); });
  const mean = allSamples.reduce((s, x) => s + x, 0) / total;
  const v = allSamples.reduce((s, x) => s + (x - mean) * (x - mean), 0) / total;
  let acsum = 0;
  for (let lag = 1; lag < Math.min(30, n); lag++) {
    let c = 0;
    for (let i = 0; i < total - lag; i++) c += (allSamples[i] - mean) * (allSamples[i + lag] - mean);
    c /= (total - lag) * v;
    if (c < 0) break;
    acsum += c;
  }
  const ess = total / (1 + 2 * acsum);
  return { rhat, ess: Math.round(ess) };
}

// ---------- Render ----------
function renderPosterior() {
  const canvas = document.getElementById('ppl-posterior');
  if (!canvas) return;
  const W = 540, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const a = 1 + COIN.k, b = 1 + (COIN.n - COIN.k);
  // Compute max for scaling
  let maxY = 0;
  for (let i = 0; i <= 200; i++) {
    const x = i / 200;
    const v = betaPdf(x, a, b);
    if (v > maxY) maxY = v;
  }
  if (maxY === 0) maxY = 1;
  // Histogram of samples
  if (COIN.samples.length > 0) {
    const flat = [];
    COIN.samples.forEach((c) => c.forEach((x) => flat.push(x)));
    const bins = 30;
    const hist = new Array(bins).fill(0);
    flat.forEach((v) => hist[Math.min(bins - 1, Math.floor(v * bins))]++);
    const total = flat.length;
    const histMax = Math.max(...hist) / total * bins;
    const scale = Math.max(maxY, histMax);
    const bw = px / bins;
    hist.forEach((c, i) => {
      const density = c / total * bins;
      const h = (density / scale) * py;
      ctx.fillStyle = 'rgba(44, 111, 183, 0.45)';
      ctx.fillRect(m.l + i * bw, m.t + py - h, bw - 1, h);
    });
    maxY = scale;
  }
  // Beta pdf
  ctx.strokeStyle = '#1e7770'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = i / 200;
    const v = betaPdf(x, a, b);
    const sx = m.l + x * px;
    const sy = m.t + (1 - v / maxY) * py;
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('blue bars: HMC samples · teal curve: analytic Beta', m.l + 8, m.t + 14);
}

function renderTrace() {
  const canvas = document.getElementById('ppl-trace');
  if (!canvas) return;
  const W = 540, H = 320;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  if (COIN.samples.length === 0) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText('Press Run HMC chain to populate.', m.l + px / 2, m.t + py / 2);
    return;
  }
  const colors = ['#2c6fb7', '#d9622b', '#1e7770', '#9b59b6'];
  COIN.samples.forEach((chain, c) => {
    ctx.strokeStyle = colors[c % colors.length]; ctx.lineWidth = 1.4;
    ctx.beginPath();
    chain.forEach((v, i) => {
      const x = m.l + (i / chain.length) * px;
      const y = m.t + (1 - v) * py;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function refreshCoin() {
  document.getElementById('ppl-kn').textContent = `${COIN.k}/${COIN.n}`;
  document.getElementById('ppl-rhat').textContent = COIN.rhat ? COIN.rhat.toFixed(3) : '—';
  document.getElementById('ppl-ess').textContent = COIN.ess ? COIN.ess : '—';
  renderPosterior(); renderTrace();
}

// ---------- 2D Bayesian regression — HMC + VI ----------
const BLR2 = { data: null };
function newBLRData() {
  BLR2.data = [];
  for (let i = 0; i < 14; i++) {
    const x = -2 + 4 * (i / 13) + 0.05 * randn();
    BLR2.data.push({ x, y: 1.2 * x + 0.6 + 0.4 * randn() });
  }
}
function logPostBLR(slope, intercept) {
  // Prior: N(0, 4) on both, likelihood: N(slope*x + intercept, 0.4)
  const sigma = 0.4;
  let lp = -0.5 * (slope * slope + intercept * intercept) / 4;
  for (const p of BLR2.data) {
    const pred = slope * p.x + intercept;
    lp -= 0.5 * (p.y - pred) * (p.y - pred) / (sigma * sigma);
  }
  return lp;
}
function blrHMC(nSamples = 1500) {
  let slope = 1, intercept = 0;
  const samples = [];
  for (let i = 0; i < nSamples; i++) {
    const propS = slope + 0.05 * randn();
    const propI = intercept + 0.05 * randn();
    const logR = logPostBLR(propS, propI) - logPostBLR(slope, intercept);
    if (Math.log(Math.random() || 1e-12) < logR) { slope = propS; intercept = propI; }
    if (i > 200) samples.push({ slope, intercept });
  }
  return samples;
}
function blrVI() {
  // Mean-field Gaussian — fit by maximising ELBO via simple grid + analytic
  // For this toy, the optimum is a fit to data, axis-aligned cov.
  // Use weighted least-squares estimate as mean; std = posterior diagonal-only proxy.
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const p of BLR2.data) { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; }
  const n = BLR2.data.length;
  const slope = (sxy - sx * sy / n) / (sxx - sx * sx / n);
  const intercept = sy / n - slope * sx / n;
  const sigma = 0.4;
  const varS = sigma * sigma / (sxx - sx * sx / n);
  const varI = sigma * sigma * (1 / n + sx * sx / n / (sxx - sx * sx / n));
  return { slope, intercept, stdS: Math.sqrt(varS), stdI: Math.sqrt(varI) };
}
function renderBLRPanel(canvasId, samples, isVI) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = 540, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const sMin = 0.5, sMax = 1.9, iMin = -0.5, iMax = 1.5;
  const sx = (s) => m.l + (s - sMin) / (sMax - sMin) * px;
  const sy = (i) => m.t + (1 - (i - iMin) / (iMax - iMin)) * py;
  if (isVI) {
    // Plot Gaussian contours
    const step = 6;
    for (let py2 = 0; py2 < py; py2 += step) {
      for (let px2 = 0; px2 < px; px2 += step) {
        const sl = sMin + (sMax - sMin) * (px2 / px);
        const it = iMax - (iMax - iMin) * (py2 / py);
        const z = ((sl - samples.slope) / samples.stdS) ** 2 + ((it - samples.intercept) / samples.stdI) ** 2;
        const t = Math.exp(-z / 2);
        ctx.fillStyle = `rgba(217, 98, 43, ${0.05 + 0.5 * t})`;
        ctx.fillRect(m.l + px2, m.t + py2, step, step);
      }
    }
    ctx.fillStyle = '#d9622b';
    ctx.beginPath(); ctx.arc(sx(samples.slope), sy(samples.intercept), 4, 0, Math.PI * 2); ctx.fill();
  } else {
    samples.forEach((s) => {
      const x = sx(s.slope), y = sy(s.intercept);
      if (x >= m.l && x <= m.l + px && y >= m.t && y <= m.t + py) {
        ctx.beginPath();
        ctx.arc(x, y, 1.4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(44,111,183,0.5)';
        ctx.fill();
      }
    });
  }
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'center';
  for (let v = 0.6; v <= 1.8; v += 0.4) {
    const x = sx(v);
    ctx.fillText(v.toFixed(1), x, m.t + py + 16);
  }
  ctx.textAlign = 'right';
  for (let v = -0.4; v <= 1.4; v += 0.4) {
    const y = sy(v);
    ctx.fillText(v.toFixed(1), m.l - 4, y + 3);
  }
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('slope', m.l + px / 2, m.t + py + 28);
  ctx.save();
  ctx.translate(14, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('intercept', 0, 0);
  ctx.restore();
}

function refreshBLR() {
  if (!BLR2.data) newBLRData();
  const samples = blrHMC();
  const vi = blrVI();
  renderBLRPanel('bl-hmc', samples, false);
  renderBLRPanel('bl-vi', vi, true);
  document.getElementById('bl-n').textContent = BLR2.data.length;
}

function wire() {
  document.getElementById('ppl-flipH').addEventListener('click', () => { COIN.k++; COIN.n++; refreshCoin(); });
  document.getElementById('ppl-flipT').addEventListener('click', () => { COIN.n++; refreshCoin(); });
  document.getElementById('ppl-reset').addEventListener('click', () => { COIN.k = 0; COIN.n = 0; COIN.samples = []; COIN.rhat = null; COIN.ess = null; refreshCoin(); });
  document.getElementById('ppl-runHMC').addEventListener('click', () => {
    COIN.samples = runHMCCoin();
    const { rhat, ess } = rhatAndESS(COIN.samples);
    COIN.rhat = rhat; COIN.ess = ess;
    refreshCoin();
  });
  document.getElementById('bl-newdata').addEventListener('click', () => {
    newBLRData(); refreshBLR();
  });
  refreshCoin();
  refreshBLR();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-rhat':
      '\\hat R = \\sqrt{\\frac{\\hat V}{W}},\\;\\;\\hat V = \\frac{n-1}{n}W + \\frac{1}{n}B,\\;\\;\\text{r-hat near 1 = chains agree}\\;\\;\\Rightarrow\\;\\text{converged}'
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
