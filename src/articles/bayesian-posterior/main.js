// ============================================================
// The Posterior, Built Up
// All inference computed live in JS — Gaussian-Gaussian, Beta-Binomial,
// posterior-predictive vs MAP, grid + Metropolis MCMC + variational,
// Bayesian linear regression with uncertainty bands.
// ============================================================

// ---------- Math + plotting helpers ----------
const PRIOR_COLOR = '#2c6fb7';
const LIK_COLOR   = '#d9622b';
const POST_COLOR  = '#1e7770';
const ACCENT_DARK = '#1a1815';
const SOFT_GRID   = '#f0ebe1';
const TICK_COLOR  = '#9a917f';

function gaussianPdf(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * sigma);
}

function logGaussianPdf(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return -0.5 * z * z - Math.log(Math.sqrt(2 * Math.PI) * sigma);
}

function logGamma(z) {
  // Lanczos approximation
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function betaPdf(x, a, b) {
  if (x <= 0 || x >= 1) return 0;
  const lp = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)
           - (logGamma(a) + logGamma(b) - logGamma(a + b));
  return Math.exp(lp);
}

function randn() {
  const u1 = Math.random() || 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

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

function makePlot(canvas, opts) {
  const w = opts.w, h = opts.h;
  const ctx = setupCanvas(canvas, w, h);
  ctx.fillStyle = '#fdfcf9';
  ctx.fillRect(0, 0, w, h);
  const m = Object.assign({ l: 60, r: 16, t: 18, b: 36 }, opts.margin || {});
  const px = w - m.l - m.r;
  const py = h - m.t - m.b;
  const xMin = opts.xMin, xMax = opts.xMax;
  const yMin = opts.yMin, yMax = opts.yMax;
  const xs = (x) => m.l + (x - xMin) / (xMax - xMin) * px;
  const ys = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * py;
  // Border
  ctx.strokeStyle = '#e2d8c6';
  ctx.lineWidth = 1;
  ctx.strokeRect(m.l, m.t, px, py);
  // Grid + axis labels
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'center';
  const xticks = opts.xTicks || 6;
  for (let i = 0; i <= xticks; i++) {
    const v = xMin + (xMax - xMin) * (i / xticks);
    const x = xs(v);
    ctx.strokeStyle = SOFT_GRID;
    ctx.beginPath();
    ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + py);
    ctx.stroke();
    ctx.fillText(formatTick(v), x, m.t + py + 16);
  }
  ctx.textAlign = 'right';
  const yticks = opts.yTicks || 4;
  for (let i = 0; i <= yticks; i++) {
    const v = yMin + (yMax - yMin) * (i / yticks);
    const y = ys(v);
    ctx.strokeStyle = SOFT_GRID;
    ctx.beginPath();
    ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y);
    ctx.stroke();
    ctx.fillText(formatTick(v), m.l - 6, y + 4);
  }
  if (opts.xLabel) {
    ctx.fillStyle = '#6e665b';
    ctx.font = '12px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText(opts.xLabel, m.l + px / 2, h - 8);
  }
  if (opts.yLabel) {
    ctx.fillStyle = '#6e665b';
    ctx.font = '12px Manrope';
    ctx.save();
    ctx.translate(14, m.t + py / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(opts.yLabel, 0, 0);
    ctx.restore();
  }
  return { ctx, m, px, py, xs, ys, xMin, xMax, yMin, yMax, w, h };
}

function formatTick(v) {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10)  return v.toFixed(1);
  if (Math.abs(v) >= 1)   return v.toFixed(2);
  return v.toFixed(2);
}

function drawCurve(plot, fn, color, lineDash = []) {
  const { ctx, xs, ys, xMin, xMax, m, px } = plot;
  const N = 320;
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.4;
  ctx.setLineDash(lineDash);
  for (let i = 0; i <= N; i++) {
    const x = xMin + (xMax - xMin) * (i / N);
    const y = fn(x);
    const sx = xs(x), sy = ys(y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function fillBand(plot, lower, upper, color) {
  const { ctx, xs, ys, xMin, xMax } = plot;
  const N = 240;
  ctx.beginPath();
  ctx.fillStyle = color;
  for (let i = 0; i <= N; i++) {
    const x = xMin + (xMax - xMin) * (i / N);
    const sx = xs(x), sy = ys(upper(x));
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  for (let i = N; i >= 0; i--) {
    const x = xMin + (xMax - xMin) * (i / N);
    const sx = xs(x), sy = ys(lower(x));
    ctx.lineTo(sx, sy);
  }
  ctx.closePath();
  ctx.fill();
}

// ============================================================
// Step 1 — Gaussian-Gaussian conjugate
// ============================================================
const G1 = {
  mu0: 0, s0: 1, sig: 1,
  data: [],
  trueTheta: 1.5
};

function g1Posterior() {
  const tau0 = 1 / (G1.s0 * G1.s0);
  const tauD = G1.data.length / (G1.sig * G1.sig);
  const tauN = tau0 + tauD;
  const sN = Math.sqrt(1 / tauN);
  const dataMean = G1.data.length === 0 ? 0 : G1.data.reduce((a, b) => a + b, 0) / G1.data.length;
  const muN = (G1.mu0 * tau0 + dataMean * tauD) / tauN;
  return { mu: muN, s: sN, dataMean };
}

function renderG1() {
  const canvas = document.getElementById('g1-canvas');
  if (!canvas) return;
  const xMin = -4, xMax = 4;
  // Compute likelihood (treat as function of θ given data); normalize
  const post = g1Posterior();
  const yMaxRaw = Math.max(
    gaussianPdf(post.mu, post.mu, post.s),
    gaussianPdf(G1.mu0, G1.mu0, G1.s0)
  );
  const plot = makePlot(canvas, {
    w: 880, h: 380,
    xMin, xMax, yMin: 0, yMax: yMaxRaw * 1.15,
    xLabel: 'θ', yLabel: 'density'
  });

  // Draw filled prior in faint blue
  fillBand(plot, () => 0, (x) => gaussianPdf(x, G1.mu0, G1.s0),
    'rgba(44,111,183,0.12)');
  drawCurve(plot, (x) => gaussianPdf(x, G1.mu0, G1.s0), PRIOR_COLOR);

  if (G1.data.length > 0) {
    // Likelihood as function of θ (product of Gaussians) — proportional to a Gaussian in θ
    // log L(θ) = -0.5 sum (xi-θ)^2 / sig^2 ; this is Gaussian centred at mean, std sig/sqrt(n)
    const dataMean = post.dataMean;
    const likStd = G1.sig / Math.sqrt(G1.data.length);
    // For visibility, show normalized likelihood on θ
    drawCurve(plot, (x) => gaussianPdf(x, dataMean, likStd), LIK_COLOR, [6, 4]);
    // Posterior (filled)
    fillBand(plot, () => 0, (x) => gaussianPdf(x, post.mu, post.s),
      'rgba(30,119,112,0.18)');
    drawCurve(plot, (x) => gaussianPdf(x, post.mu, post.s), POST_COLOR);
  }

  // Draw data ticks at bottom of plot
  G1.data.forEach((d) => {
    const x = plot.xs(Math.max(xMin, Math.min(xMax, d)));
    const y = plot.m.t + plot.py;
    plot.ctx.strokeStyle = '#1a1815';
    plot.ctx.lineWidth = 2;
    plot.ctx.beginPath();
    plot.ctx.moveTo(x, y);
    plot.ctx.lineTo(x, y + 7);
    plot.ctx.stroke();
  });

  // True theta marker
  const tx = plot.xs(G1.trueTheta);
  plot.ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  plot.ctx.setLineDash([3, 3]);
  plot.ctx.beginPath();
  plot.ctx.moveTo(tx, plot.m.t);
  plot.ctx.lineTo(tx, plot.m.t + plot.py);
  plot.ctx.stroke();
  plot.ctx.setLineDash([]);
  plot.ctx.fillStyle = TICK_COLOR;
  plot.ctx.font = '11px Manrope';
  plot.ctx.textAlign = 'left';
  plot.ctx.fillText('true θ', tx + 4, plot.m.t + 14);

  // Legend
  drawLegend(plot, [
    { label: 'prior', color: PRIOR_COLOR, dash: false },
    { label: 'likelihood (∝)', color: LIK_COLOR, dash: true },
    { label: 'posterior', color: POST_COLOR, dash: false }
  ]);

  // Update stats
  document.getElementById('g1-n').textContent = G1.data.length;
  if (G1.data.length === 0) {
    document.getElementById('g1-postmu').textContent = G1.mu0.toFixed(2);
    document.getElementById('g1-posts').textContent = G1.s0.toFixed(2);
  } else {
    document.getElementById('g1-postmu').textContent = post.mu.toFixed(2);
    document.getElementById('g1-posts').textContent = post.s.toFixed(2);
  }
}

function drawLegend(plot, items) {
  const { ctx, m, px } = plot;
  ctx.font = '12px Manrope';
  ctx.textAlign = 'left';
  let x = m.l + 8;
  let y = m.t + 14;
  items.forEach((it) => {
    ctx.strokeStyle = it.color;
    ctx.lineWidth = 2.4;
    ctx.setLineDash(it.dash ? [6, 4] : []);
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x + 18, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#3b342b';
    ctx.fillText(it.label, x + 23, y + 4);
    x += 23 + ctx.measureText(it.label).width + 16;
  });
}

function wireG1() {
  const mu0 = document.getElementById('g1-mu0');
  const s0  = document.getElementById('g1-s0');
  const sig = document.getElementById('g1-sig');
  const onChange = () => {
    G1.mu0 = parseFloat(mu0.value);
    G1.s0  = parseFloat(s0.value);
    G1.sig = parseFloat(sig.value);
    document.getElementById('g1-mu0-val').textContent = G1.mu0.toFixed(1);
    document.getElementById('g1-s0-val').textContent  = G1.s0.toFixed(1);
    document.getElementById('g1-sig-val').textContent = G1.sig.toFixed(1);
    renderG1();
  };
  [mu0, s0, sig].forEach((el) => el.addEventListener('input', onChange));

  const canvas = document.getElementById('g1-canvas');
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const xMin = -4, xMax = 4;
    const m = { l: 60, r: 16 };
    const px = rect.width - m.l - m.r;
    const x = xMin + (cx - m.l) / px * (xMax - xMin);
    if (x >= xMin && x <= xMax) {
      G1.data.push(x);
      renderG1();
    }
  });

  document.getElementById('g1-add').addEventListener('click', () => {
    G1.data.push(G1.trueTheta + randn() * G1.sig);
    renderG1();
  });
  document.getElementById('g1-add5').addEventListener('click', () => {
    for (let i = 0; i < 5; i++) G1.data.push(G1.trueTheta + randn() * G1.sig);
    renderG1();
  });
  document.getElementById('g1-clear').addEventListener('click', () => {
    G1.data = [];
    renderG1();
  });
}

// ============================================================
// Step 2 — Posterior std vs n, three priors
// ============================================================
function renderG2() {
  const canvas = document.getElementById('g2-canvas');
  if (!canvas) return;
  const sig = 1; // fixed
  const priors = [
    { s0: 0.3, label: 'tight σ₀=0.3', color: '#1e7770' },
    { s0: 1.0, label: 'medium σ₀=1', color: '#2c6fb7' },
    { s0: 5.0, label: 'broad σ₀=5', color: '#d9622b' }
  ];
  const nMax = 100;
  const plot = makePlot(canvas, {
    w: 880, h: 320,
    xMin: 0, xMax: nMax, yMin: 0, yMax: 1.6,
    xLabel: 'sample size n', yLabel: 'posterior std σₙ'
  });
  priors.forEach((p) => {
    plot.ctx.strokeStyle = p.color;
    plot.ctx.lineWidth = 2.2;
    plot.ctx.beginPath();
    for (let n = 0; n <= nMax; n++) {
      const tau = 1 / (p.s0 * p.s0) + n / (sig * sig);
      const sN = Math.sqrt(1 / tau);
      const x = plot.xs(n), y = plot.ys(sN);
      if (n === 0) plot.ctx.moveTo(x, y); else plot.ctx.lineTo(x, y);
    }
    plot.ctx.stroke();
  });
  // Reference 1/sqrt(n)
  plot.ctx.strokeStyle = 'rgba(0,0,0,0.3)';
  plot.ctx.lineWidth = 1.2;
  plot.ctx.setLineDash([4, 4]);
  plot.ctx.beginPath();
  for (let n = 1; n <= nMax; n++) {
    const x = plot.xs(n), y = plot.ys(sig / Math.sqrt(n));
    if (n === 1) plot.ctx.moveTo(x, y); else plot.ctx.lineTo(x, y);
  }
  plot.ctx.stroke();
  plot.ctx.setLineDash([]);

  drawLegend(plot, [
    ...priors.map((p) => ({ label: p.label, color: p.color, dash: false })),
    { label: 'σ/√n (no prior)', color: 'rgba(0,0,0,0.3)', dash: true }
  ]);
}

// ============================================================
// Step 3 — Beta-Binomial coin
// ============================================================
const BB = {
  alpha: 1, beta: 1,
  k: 0, n: 0,
  trueBias: 0.65
};

function bbPosterior() {
  return { a: BB.alpha + BB.k, b: BB.beta + (BB.n - BB.k) };
}

function betaQuantile(a, b, p) {
  // Bisection on the CDF (numerical integration of pdf)
  const N = 512;
  const dx = 1 / N;
  let cdf = 0;
  let prev = 0;
  for (let i = 1; i <= N; i++) {
    const x = i * dx;
    const v = betaPdf(x, a, b);
    cdf += 0.5 * (prev + v) * dx;
    prev = v;
    if (cdf >= p) return x;
  }
  return 1;
}

function renderBB() {
  const canvas = document.getElementById('bb-canvas');
  if (!canvas) return;
  const { a: aP, b: bP } = bbPosterior();
  const yMax = Math.max(
    BB.n > 0 ? betaPdf((aP - 1) / Math.max(0.01, aP + bP - 2), aP, bP) : 0,
    betaPdf((BB.alpha - 1) / Math.max(0.01, BB.alpha + BB.beta - 2), BB.alpha, BB.beta) || 0,
    1.0
  );
  const plot = makePlot(canvas, {
    w: 880, h: 320,
    xMin: 0, xMax: 1, yMin: 0, yMax: Math.max(2, yMax * 1.15),
    xLabel: 'θ (probability of heads)', yLabel: 'density'
  });
  // Prior
  fillBand(plot, () => 0, (x) => betaPdf(x, BB.alpha, BB.beta), 'rgba(44,111,183,0.12)');
  drawCurve(plot, (x) => betaPdf(x, BB.alpha, BB.beta), PRIOR_COLOR);
  if (BB.n > 0) {
    fillBand(plot, () => 0, (x) => betaPdf(x, aP, bP), 'rgba(30,119,112,0.18)');
    drawCurve(plot, (x) => betaPdf(x, aP, bP), POST_COLOR);
  }
  // True bias marker
  const tx = plot.xs(BB.trueBias);
  plot.ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  plot.ctx.setLineDash([3, 3]);
  plot.ctx.beginPath();
  plot.ctx.moveTo(tx, plot.m.t); plot.ctx.lineTo(tx, plot.m.t + plot.py);
  plot.ctx.stroke();
  plot.ctx.setLineDash([]);
  plot.ctx.fillStyle = TICK_COLOR;
  plot.ctx.font = '11px Manrope';
  plot.ctx.textAlign = 'left';
  plot.ctx.fillText('true θ', tx + 4, plot.m.t + 14);

  drawLegend(plot, [
    { label: 'prior Beta(α, β)', color: PRIOR_COLOR, dash: false },
    { label: 'posterior Beta(α+k, β+n−k)', color: POST_COLOR, dash: false }
  ]);

  document.getElementById('bb-kn').textContent = `${BB.k}/${BB.n}`;
  if (BB.n === 0 && BB.alpha === 1 && BB.beta === 1) {
    document.getElementById('bb-mean').textContent = '0.50';
    document.getElementById('bb-ci').textContent = '[0.025, 0.975]';
  } else {
    const mean = aP / (aP + bP);
    const lo = betaQuantile(aP, bP, 0.025);
    const hi = betaQuantile(aP, bP, 0.975);
    document.getElementById('bb-mean').textContent = mean.toFixed(3);
    document.getElementById('bb-ci').textContent = `[${lo.toFixed(2)}, ${hi.toFixed(2)}]`;
  }
}

function wireBB() {
  const a = document.getElementById('bb-a');
  const b = document.getElementById('bb-b');
  const onPrior = () => {
    BB.alpha = parseFloat(a.value);
    BB.beta  = parseFloat(b.value);
    document.getElementById('bb-a-val').textContent = BB.alpha.toFixed(1);
    document.getElementById('bb-b-val').textContent = BB.beta.toFixed(1);
    renderBB();
  };
  [a, b].forEach((el) => el.addEventListener('input', onPrior));
  document.getElementById('bb-h').addEventListener('click', () => { BB.k++; BB.n++; renderBB(); });
  document.getElementById('bb-t').addEventListener('click', () => { BB.n++; renderBB(); });
  document.getElementById('bb-flip10').addEventListener('click', () => {
    for (let i = 0; i < 10; i++) {
      if (Math.random() < 0.5) BB.k++;
      BB.n++;
    }
    renderBB();
  });
  document.getElementById('bb-flip-bias').addEventListener('click', () => {
    for (let i = 0; i < 10; i++) {
      if (Math.random() < 0.8) BB.k++;
      BB.n++;
    }
    renderBB();
  });
  document.getElementById('bb-clear').addEventListener('click', () => {
    BB.k = 0; BB.n = 0; renderBB();
  });
}

// ============================================================
// Step 4 — Posterior predictive vs MAP plug-in
// ============================================================
const PP = { n: 3, sig: 1, mu0: 0, s0: 1, trueTheta: 1.5 };

function renderPP() {
  const canvas = document.getElementById('pp-canvas');
  if (!canvas) return;
  // Generate n samples deterministic-ish (use seeded random)
  const rng = mulberry32(42 + PP.n);
  const data = [];
  for (let i = 0; i < PP.n; i++) {
    const u1 = rng() || 1e-12;
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    data.push(PP.trueTheta + z * PP.sig);
  }
  const tau0 = 1 / (PP.s0 * PP.s0);
  const tauD = data.length / (PP.sig * PP.sig);
  const tauN = tau0 + tauD;
  const sN = Math.sqrt(1 / tauN);
  const dataMean = data.reduce((a, b) => a + b, 0) / data.length;
  const muN = (PP.mu0 * tau0 + dataMean * tauD) / tauN;
  // Plug-in MAP predictive: N(muN, sig)
  // Posterior predictive: N(muN, sqrt(sig^2 + sN^2))
  const mapStd = PP.sig;
  const ppStd  = Math.sqrt(PP.sig * PP.sig + sN * sN);

  const xMin = -4, xMax = 4;
  const yMax = Math.max(gaussianPdf(muN, muN, mapStd), gaussianPdf(muN, muN, ppStd)) * 1.15;
  const plot = makePlot(canvas, {
    w: 880, h: 360,
    xMin, xMax, yMin: 0, yMax,
    xLabel: 'x_new', yLabel: 'predictive density'
  });
  // MAP
  drawCurve(plot, (x) => gaussianPdf(x, muN, mapStd), LIK_COLOR, [6, 4]);
  // Posterior predictive
  fillBand(plot, () => 0, (x) => gaussianPdf(x, muN, ppStd), 'rgba(30,119,112,0.18)');
  drawCurve(plot, (x) => gaussianPdf(x, muN, ppStd), POST_COLOR);
  // Data ticks
  data.forEach((d) => {
    const x = plot.xs(Math.max(xMin, Math.min(xMax, d)));
    const y = plot.m.t + plot.py;
    plot.ctx.strokeStyle = '#1a1815';
    plot.ctx.lineWidth = 2;
    plot.ctx.beginPath();
    plot.ctx.moveTo(x, y); plot.ctx.lineTo(x, y + 7);
    plot.ctx.stroke();
  });

  drawLegend(plot, [
    { label: 'plug-in MAP predictive', color: LIK_COLOR, dash: true },
    { label: 'posterior predictive', color: POST_COLOR, dash: false }
  ]);

  // Annotation: width gap
  plot.ctx.fillStyle = '#6e665b';
  plot.ctx.font = '12px Manrope';
  plot.ctx.textAlign = 'right';
  plot.ctx.fillText(
    `MAP std = ${mapStd.toFixed(2)},   posterior-predictive std = ${ppStd.toFixed(2)}   (gap = ${(ppStd - mapStd).toFixed(2)})`,
    plot.m.l + plot.px - 8, plot.m.t + 16
  );
}

function wirePP() {
  const n = document.getElementById('pp-n');
  const sig = document.getElementById('pp-sig');
  const onChange = () => {
    PP.n = parseInt(n.value, 10);
    PP.sig = parseFloat(sig.value);
    document.getElementById('pp-n-val').textContent = PP.n;
    document.getElementById('pp-sig-val').textContent = PP.sig.toFixed(1);
    renderPP();
  };
  [n, sig].forEach((el) => el.addEventListener('input', onChange));
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ============================================================
// Step 5 — Non-conjugate posterior: grid + MCMC + VI
// True posterior is a fixed bimodal mixture: 0.6*N(-1.4, 0.5) + 0.4*N(1.6, 0.7)
// ============================================================
const NC = {
  running: false,
  raf: null,
  chain: [],
  current: 0,
  accepted: 0,
  proposed: 0,
  stepSize: 0.3,
  hist: null,         // Float32Array of bin counts
  binCount: 50,
  binMin: -4,
  binMax: 4
};

function ncTargetLogPdf(x) {
  const m1 = -1.4, s1 = 0.5, w1 = 0.6;
  const m2 = 1.6,  s2 = 0.7, w2 = 0.4;
  const c1 = w1 * gaussianPdf(x, m1, s1);
  const c2 = w2 * gaussianPdf(x, m2, s2);
  const p = c1 + c2;
  return p > 0 ? Math.log(p) : -1e9;
}
function ncTargetPdf(x) {
  return Math.exp(ncTargetLogPdf(x));
}

function ncReset() {
  NC.chain = [];
  NC.current = -1.4;
  NC.accepted = 0;
  NC.proposed = 0;
  NC.hist = new Float32Array(NC.binCount);
}

function ncFitVI() {
  // Closest single Gaussian by minimising forward KL on a grid would
  // require numerical optimisation; we cheat: fit a Gaussian to MCMC
  // samples we have. With 0 samples, fall back to grid mean/var.
  if (NC.chain.length > 50) {
    const mu = NC.chain.reduce((a, b) => a + b, 0) / NC.chain.length;
    const v = NC.chain.reduce((a, b) => a + (b - mu) * (b - mu), 0) / NC.chain.length;
    return { mu, sigma: Math.sqrt(v) };
  }
  // grid-based mode-find: approximate by computing weighted moments on grid
  const N = 400;
  let sumW = 0, sumX = 0, sumX2 = 0;
  for (let i = 0; i < N; i++) {
    const x = NC.binMin + (NC.binMax - NC.binMin) * (i / (N - 1));
    const p = ncTargetPdf(x);
    sumW += p; sumX += p * x; sumX2 += p * x * x;
  }
  const mu = sumX / sumW;
  const v = sumX2 / sumW - mu * mu;
  return { mu, sigma: Math.sqrt(Math.max(v, 0.05)) };
}

function ncStep(steps = 50) {
  for (let s = 0; s < steps; s++) {
    NC.proposed++;
    const proposal = NC.current + randn() * NC.stepSize;
    const logRatio = ncTargetLogPdf(proposal) - ncTargetLogPdf(NC.current);
    if (Math.log(Math.random() || 1e-12) < logRatio) {
      NC.current = proposal;
      NC.accepted++;
    }
    NC.chain.push(NC.current);
    if (NC.current >= NC.binMin && NC.current < NC.binMax) {
      const bin = Math.floor((NC.current - NC.binMin) / (NC.binMax - NC.binMin) * NC.binCount);
      NC.hist[bin]++;
    }
  }
}

function renderNC() {
  const canvas = document.getElementById('nc-canvas');
  if (!canvas) return;
  const xMin = NC.binMin, xMax = NC.binMax;
  const N = 400;
  let yMax = 0;
  for (let i = 0; i < N; i++) {
    const x = xMin + (xMax - xMin) * (i / (N - 1));
    yMax = Math.max(yMax, ncTargetPdf(x));
  }
  yMax *= 1.2;
  const plot = makePlot(canvas, {
    w: 880, h: 360,
    xMin, xMax, yMin: 0, yMax,
    xLabel: 'θ', yLabel: 'density'
  });
  // MCMC histogram (normalised)
  if (NC.chain.length > 0 && NC.hist) {
    const total = NC.chain.length;
    const binW = (NC.binMax - NC.binMin) / NC.binCount;
    const ctx = plot.ctx;
    ctx.fillStyle = 'rgba(44,111,183,0.35)';
    for (let i = 0; i < NC.binCount; i++) {
      const dens = NC.hist[i] / total / binW;
      const x0 = NC.binMin + i * binW;
      const x1 = x0 + binW;
      const xs0 = plot.xs(x0), xs1 = plot.xs(x1);
      const yt = plot.ys(dens), yb = plot.ys(0);
      ctx.fillRect(xs0, yt, xs1 - xs0 - 1, yb - yt);
    }
  }
  // True posterior
  drawCurve(plot, (x) => ncTargetPdf(x), POST_COLOR);
  // VI Gaussian fit
  const vi = ncFitVI();
  drawCurve(plot, (x) => gaussianPdf(x, vi.mu, vi.sigma), LIK_COLOR, [6, 4]);

  drawLegend(plot, [
    { label: 'true (grid)', color: POST_COLOR, dash: false },
    { label: 'MCMC samples', color: 'rgba(44,111,183,0.6)', dash: false },
    { label: 'VI Gaussian fit', color: LIK_COLOR, dash: true }
  ]);

  document.getElementById('nc-n').textContent = NC.chain.length;
  document.getElementById('nc-rate').textContent =
    NC.proposed === 0 ? '—' : `${(100 * NC.accepted / NC.proposed).toFixed(1)}%`;
}

function ncLoop() {
  if (!NC.running) return;
  ncStep(80);
  renderNC();
  NC.raf = requestAnimationFrame(ncLoop);
}

function wireNC() {
  ncReset();
  const tog = document.getElementById('nc-toggle');
  const reset = document.getElementById('nc-reset');
  const ss = document.getElementById('nc-stepsize');
  tog.addEventListener('click', () => {
    NC.running = !NC.running;
    tog.textContent = NC.running ? 'Pause MCMC' : 'Run MCMC';
    if (NC.running) ncLoop();
    else if (NC.raf) cancelAnimationFrame(NC.raf);
  });
  reset.addEventListener('click', () => {
    if (NC.raf) cancelAnimationFrame(NC.raf);
    ncReset();
    NC.running = false;
    tog.textContent = 'Run MCMC';
    renderNC();
  });
  ss.addEventListener('input', () => {
    NC.stepSize = parseFloat(ss.value);
    document.getElementById('nc-stepsize-val').textContent = NC.stepSize.toFixed(2);
  });
  renderNC();
}

// ============================================================
// Step 6 — Bayesian linear regression (basis = [1, x, sin(x), cos(x), sin(2x), cos(2x)])
// ============================================================
const BLR = {
  data: [],
  priorW: 1.0,
  noise: 0.4,
  basisDim: 7, // [1, x, sin(x), cos(x), sin(2x), cos(2x), x*0.5]
  xMin: -4, xMax: 4
};

function blrPhi(x) {
  return [1, 0.4 * x, Math.sin(x), Math.cos(x), Math.sin(2 * x), Math.cos(2 * x), 0.1 * x * x];
}

function matMul2(A, B) {
  const r = A.length, inn = A[0].length, c = B[0].length;
  const out = new Array(r);
  for (let i = 0; i < r; i++) {
    const row = new Array(c).fill(0);
    for (let k = 0; k < inn; k++) {
      const aik = A[i][k];
      const Bk = B[k];
      for (let j = 0; j < c; j++) row[j] += aik * Bk[j];
    }
    out[i] = row;
  }
  return out;
}
function matT(A) {
  const r = A.length, c = A[0].length;
  const out = new Array(c);
  for (let j = 0; j < c; j++) {
    out[j] = new Array(r);
    for (let i = 0; i < r; i++) out[j][i] = A[i][j];
  }
  return out;
}
function matInv(A) {
  // Gauss-Jordan for small symmetric matrix
  const n = A.length;
  const M = new Array(n);
  for (let i = 0; i < n; i++) {
    M[i] = new Array(2 * n).fill(0);
    for (let j = 0; j < n; j++) M[i][j] = A[i][j];
    M[i][n + i] = 1;
  }
  for (let i = 0; i < n; i++) {
    let pivot = M[i][i];
    let maxRow = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(pivot)) { pivot = M[r][i]; maxRow = r; }
    if (maxRow !== i) { const tmp = M[i]; M[i] = M[maxRow]; M[maxRow] = tmp; }
    if (Math.abs(pivot) < 1e-12) return null;
    for (let j = 0; j < 2 * n; j++) M[i][j] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const f = M[r][i];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[i][j];
    }
  }
  const inv = new Array(n);
  for (let i = 0; i < n; i++) inv[i] = M[i].slice(n);
  return inv;
}

function blrPosterior() {
  const D = BLR.basisDim;
  const tau = 1 / (BLR.priorW * BLR.priorW);
  const beta = 1 / (BLR.noise * BLR.noise);
  // Prior covariance Σ₀ = (1/τ) I; precision A0 = τ I
  const A = new Array(D);
  for (let i = 0; i < D; i++) {
    A[i] = new Array(D).fill(0);
    A[i][i] = tau;
  }
  // Build Φ (n × D), t (n)
  const n = BLR.data.length;
  const Phi = new Array(n);
  const t = new Array(n);
  for (let i = 0; i < n; i++) {
    Phi[i] = blrPhi(BLR.data[i].x);
    t[i] = BLR.data[i].y;
  }
  // A_n = A0 + β Φᵀ Φ
  if (n > 0) {
    const PhiT = matT(Phi);
    const PtP = matMul2(PhiT, Phi);
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) A[i][j] += beta * PtP[i][j];
  }
  const Sigma = matInv(A);
  if (!Sigma) return { mu: new Array(D).fill(0), Sigma: null };
  // m_n = β Σ_n Φᵀ t (prior mean = 0)
  const mu = new Array(D).fill(0);
  if (n > 0) {
    const PhiT = matT(Phi);
    const Ptt = new Array(D).fill(0);
    for (let i = 0; i < D; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += PhiT[i][j] * t[j];
      Ptt[i] = s;
    }
    for (let i = 0; i < D; i++) {
      let s = 0;
      for (let j = 0; j < D; j++) s += Sigma[i][j] * Ptt[j];
      mu[i] = beta * s;
    }
  }
  return { mu, Sigma };
}

function blrPredict(x, post) {
  const phi = blrPhi(x);
  const D = phi.length;
  let mean = 0;
  for (let i = 0; i < D; i++) mean += phi[i] * post.mu[i];
  // Variance: σ² + φᵀ Σ φ
  let qf = 0;
  if (post.Sigma) {
    const Sphi = new Array(D).fill(0);
    for (let i = 0; i < D; i++) {
      let s = 0;
      for (let j = 0; j < D; j++) s += post.Sigma[i][j] * phi[j];
      Sphi[i] = s;
    }
    for (let i = 0; i < D; i++) qf += phi[i] * Sphi[i];
  } else {
    qf = BLR.priorW * BLR.priorW * 4; // wide
  }
  const variance = BLR.noise * BLR.noise + qf;
  return { mean, std: Math.sqrt(variance) };
}

function renderBLR() {
  const canvas = document.getElementById('blr-canvas');
  if (!canvas) return;
  const post = blrPosterior();
  const xMin = BLR.xMin, xMax = BLR.xMax;
  const yMin = -3.5, yMax = 3.5;
  const plot = makePlot(canvas, {
    w: 880, h: 380,
    xMin, xMax, yMin, yMax,
    xLabel: 'x', yLabel: 'y'
  });
  // Uncertainty band
  const N = 240;
  const upper = new Array(N + 1);
  const lower = new Array(N + 1);
  for (let i = 0; i <= N; i++) {
    const x = xMin + (xMax - xMin) * (i / N);
    const p = blrPredict(x, post);
    upper[i] = p.mean + 2 * p.std;
    lower[i] = p.mean - 2 * p.std;
  }
  // Fill band
  const ctx = plot.ctx;
  ctx.beginPath();
  ctx.fillStyle = 'rgba(30,119,112,0.18)';
  for (let i = 0; i <= N; i++) {
    const x = xMin + (xMax - xMin) * (i / N);
    const sx = plot.xs(x), sy = plot.ys(Math.min(yMax, upper[i]));
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  for (let i = N; i >= 0; i--) {
    const x = xMin + (xMax - xMin) * (i / N);
    const sx = plot.xs(x), sy = plot.ys(Math.max(yMin, lower[i]));
    ctx.lineTo(sx, sy);
  }
  ctx.closePath();
  ctx.fill();
  // Mean curve
  drawCurve(plot, (x) => blrPredict(x, post).mean, PRIOR_COLOR);
  // Data points
  BLR.data.forEach((d) => {
    const sx = plot.xs(d.x), sy = plot.ys(d.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fillStyle = ACCENT_DARK;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
  document.getElementById('blr-n').textContent = BLR.data.length;
}

function wireBLR() {
  const canvas = document.getElementById('blr-canvas');
  const click = (e, isShift) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const m = { l: 60, r: 16, t: 18, b: 36 };
    const px = rect.width - m.l - m.r;
    const py = rect.height - m.t - m.b;
    const xMin = BLR.xMin, xMax = BLR.xMax, yMin = -3.5, yMax = 3.5;
    const x = xMin + (cx - m.l) / px * (xMax - xMin);
    const y = yMax - (cy - m.t) / py * (yMax - yMin);
    if (cx < m.l || cx > m.l + px || cy < m.t || cy > m.t + py) return;
    if (isShift) {
      // remove nearest
      let best = -1, bestD = 0.6 * 0.6;
      for (let i = 0; i < BLR.data.length; i++) {
        const dx = BLR.data[i].x - x;
        const dy = BLR.data[i].y - y;
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best >= 0) BLR.data.splice(best, 1);
    } else {
      BLR.data.push({ x, y });
    }
    renderBLR();
  };
  canvas.addEventListener('click', (e) => click(e, e.shiftKey));
  canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); click(e, true); });
  document.getElementById('blr-priorw').addEventListener('input', (e) => {
    BLR.priorW = parseFloat(e.target.value);
    document.getElementById('blr-priorw-val').textContent = BLR.priorW.toFixed(1);
    renderBLR();
  });
  document.getElementById('blr-noise').addEventListener('input', (e) => {
    BLR.noise = parseFloat(e.target.value);
    document.getElementById('blr-noise-val').textContent = BLR.noise.toFixed(2);
    renderBLR();
  });
  document.getElementById('blr-clear').addEventListener('click', () => {
    BLR.data = [];
    renderBLR();
  });
  document.getElementById('blr-seed').addEventListener('click', () => {
    BLR.data = [];
    for (let x = -3.5; x <= 3.5; x += 0.4) {
      if (x > -1.5 && x < 1.0) continue; // leave a hole
      const y = Math.sin(x) + 0.15 * randn();
      BLR.data.push({ x, y });
    }
    renderBLR();
  });
}

// ============================================================
// Static math
// ============================================================
function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-bayes':
      '\\underbrace{p(\\theta \\mid \\mathcal{D})}_{\\text{posterior}} \\;=\\; \\frac{\\;\\overbrace{p(\\mathcal{D} \\mid \\theta)}^{\\text{likelihood}}\\;\\overbrace{p(\\theta)}^{\\text{prior}}\\;}{p(\\mathcal{D})}',
    'math-gauss-gauss':
      '\\theta \\sim \\mathcal{N}(\\mu_0, \\sigma_0^2),\\ \\ x_i \\sim \\mathcal{N}(\\theta, \\sigma^2)\\ \\Longrightarrow\\ \\theta\\mid\\mathcal{D} \\sim \\mathcal{N}(\\mu_n, \\sigma_n^2)',
    'math-precision':
      '\\frac{1}{\\sigma_n^2} \\;=\\; \\underbrace{\\frac{1}{\\sigma_0^2}}_{\\text{prior precision}} \\;+\\; \\underbrace{\\frac{n}{\\sigma^2}}_{\\text{data precision}}, \\qquad \\mu_n = \\sigma_n^2\\!\\left(\\frac{\\mu_0}{\\sigma_0^2} + \\frac{\\sum x_i}{\\sigma^2}\\right)',
    'math-beta':
      '\\theta \\sim \\mathrm{Beta}(\\alpha, \\beta) \\;\\Longrightarrow\\; \\theta \\mid \\mathcal{D} \\sim \\mathrm{Beta}(\\alpha + k,\\;\\beta + n - k)',
    'math-blr':
      'p(y \\mid x, \\mathcal{D}) = \\mathcal{N}\\!\\bigl(\\,\\phi(x)^\\top \\boldsymbol{\\mu}_n,\\;\\sigma^2 + \\phi(x)^\\top \\Sigma_n\\, \\phi(x)\\,\\bigr)',
    'math-hier':
      '\\theta_g \\sim \\mathcal{N}(\\mu, \\tau^2),\\qquad y_{g,i} \\mid \\theta_g \\sim \\mathcal{N}(\\theta_g, \\sigma^2),\\qquad g = 1, \\dots, G'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

// ============================================================
// Boot
// ============================================================
function init() {
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  wireG1(); renderG1();
  renderG2();
  wireBB(); renderBB();
  wirePP(); renderPP();
  wireNC();
  wireBLR(); renderBLR();
  wireHP(); renderHP();
}

// ============================================================
// Step 6½ — Hierarchical / partial pooling
// 8 groups, each with a true theta_g drawn from N(0, 1.5). The g-th
// group has n_g observations from N(theta_g, sigma^2). We compare:
//   - no pooling (per-group MLE = sample mean)
//   - complete pooling (global sample mean)
//   - partial pooling (Bayesian posterior with prior theta_g ~ N(mu, tau^2))
// ============================================================
const HP = {
  groups: null,
  trueMu: 0,
  tau: 1.0,
  sigma: 1.0,
  withEmpty: false
};

function hpResample() {
  const G = 8;
  const groups = [];
  for (let g = 0; g < G; g++) {
    const trueTheta = randn() * 1.5;
    let n = 8 + Math.floor(Math.random() * 12);
    if (HP.withEmpty && g === 4) n = 2; // sparse group at index 4
    const obs = [];
    for (let i = 0; i < n; i++) obs.push(trueTheta + randn() * HP.sigma);
    groups.push({ trueTheta, obs });
  }
  HP.groups = groups;
}

function partialPoolEstimates() {
  // Two-stage approximation: estimate global mu = mean(group means),
  // then shrink each group estimate toward mu with weight w_g.
  const G = HP.groups.length;
  const ybar = HP.groups.map((g) => g.obs.reduce((a, b) => a + b, 0) / g.obs.length);
  const mu = ybar.reduce((a, b) => a + b, 0) / G;
  const ests = HP.groups.map((g, i) => {
    const n = g.obs.length;
    const w = (n / (HP.sigma * HP.sigma)) / (n / (HP.sigma * HP.sigma) + 1 / (HP.tau * HP.tau));
    return { mle: ybar[i], pool: w * ybar[i] + (1 - w) * mu, w };
  });
  return { mu, ests };
}

function renderHP() {
  if (!HP.groups) hpResample();
  const canvas = document.getElementById('hp-canvas');
  if (!canvas) return;
  const W = 880, H = 380;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 14, t: 18, b: 38 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const G = HP.groups.length;
  const all = [];
  HP.groups.forEach((g) => g.obs.forEach((o) => all.push(o)));
  let lo = Math.min(...all) - 0.6, hi = Math.max(...all) + 0.6;
  const sx = (gi) => m.l + (gi + 0.5) / G * px;
  const sy = (y) => m.t + (1 - (y - lo) / (hi - lo)) * py;
  // Y axis ticks
  ctx.fillStyle = TICK_COLOR;
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * (1 - i / 4);
    const y = m.t + i / 4 * py;
    ctx.fillText(v.toFixed(1), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  // X axis labels
  ctx.textAlign = 'center';
  for (let g = 0; g < G; g++) {
    const x = sx(g);
    ctx.fillStyle = '#9a917f';
    ctx.fillText(`grp ${g + 1}`, x, m.t + py + 16);
  }
  const { mu, ests } = partialPoolEstimates();
  // Global pool line
  ctx.strokeStyle = '#2c6fb7';
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(m.l, sy(mu)); ctx.lineTo(m.l + px, sy(mu)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#1a4f8a';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText(`global pool μ = ${mu.toFixed(2)}`, m.l + 6, sy(mu) - 4);
  // Per group: observations dots, MLE ×, posterior ●, true theta as ring
  HP.groups.forEach((g, gi) => {
    const x = sx(gi);
    g.obs.forEach((o) => {
      ctx.beginPath();
      ctx.arc(x + (Math.random() - 0.5) * 18, sy(o), 1.8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fill();
    });
    // True theta — small open ring
    ctx.strokeStyle = 'rgba(0,0,0,0.4)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(x, sy(g.trueTheta), 5, 0, Math.PI * 2);
    ctx.stroke();
    // MLE — orange ×
    const e = ests[gi];
    ctx.strokeStyle = '#d9622b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - 5, sy(e.mle) - 5); ctx.lineTo(x + 5, sy(e.mle) + 5);
    ctx.moveTo(x + 5, sy(e.mle) - 5); ctx.lineTo(x - 5, sy(e.mle) + 5);
    ctx.stroke();
    // Pool ● — teal disc
    ctx.fillStyle = '#1e7770';
    ctx.beginPath();
    ctx.arc(x, sy(e.pool), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1.4;
    ctx.stroke();
    // Shrinkage arrow MLE → pool
    if (Math.abs(e.mle - e.pool) > 0.04) {
      ctx.strokeStyle = 'rgba(30,119,112,0.55)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x, sy(e.mle));
      ctx.lineTo(x, sy(e.pool));
      ctx.stroke();
    }
    // Annotation: w_g (weight on data)
    ctx.fillStyle = '#6e665b';
    ctx.font = '10px IBM Plex Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`n=${g.obs.length}`, x, m.t + py + 28);
  });
  // Legend
  ctx.font = '11px Manrope';
  ctx.fillStyle = '#3b342b';
  ctx.textAlign = 'left';
  ctx.fillText('orange × = no-pool (MLE)', m.l + 8, m.t + 14);
  ctx.fillStyle = '#1e7770';
  ctx.fillText('teal ● = partial-pool posterior', m.l + 200, m.t + 14);
  ctx.fillStyle = '#1a4f8a';
  ctx.fillText('blue dashed = complete-pool', m.l + 410, m.t + 14);
  ctx.fillStyle = '#1a1815';
  ctx.fillText('open ring = true θ', m.l + 600, m.t + 14);
}

function wireHP() {
  document.getElementById('hp-tau').addEventListener('input', (e) => {
    HP.tau = parseFloat(e.target.value);
    document.getElementById('hp-tau-val').textContent = HP.tau.toFixed(2);
    renderHP();
  });
  document.getElementById('hp-sig').addEventListener('input', (e) => {
    HP.sigma = parseFloat(e.target.value);
    document.getElementById('hp-sig-val').textContent = HP.sigma.toFixed(2);
    hpResample();
    renderHP();
  });
  document.getElementById('hp-empty').addEventListener('change', (e) => {
    HP.withEmpty = e.target.checked;
    hpResample();
    renderHP();
  });
  document.getElementById('hp-resample').addEventListener('click', () => {
    hpResample();
    renderHP();
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
