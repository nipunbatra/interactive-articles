// GAN minimax dance · pedagogical interactive
// Seven SVG canvases, shared helpers

const NS = "http://www.w3.org/2000/svg";

// ── shared math helpers ──
function gauss(x, mu, sigma) {
  return Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
}

// scenarios for step 1 & step 5
const SCENARIOS = {
  bimodal: {
    desc: "Two symmetric Gaussians — think \"heights of adult men and women mixed together.\"",
    realPdf: x => 0.5 * gauss(x, -2, 0.5) + 0.5 * gauss(x, 2, 0.5),
    genPdf: (x, t) => {
      const alpha = Math.min(1, t / 800);
      const muShift = 2 * alpha;
      const sigma = 1.5 - 1.0 * alpha;
      return 0.5 * gauss(x, -muShift, sigma) + 0.5 * gauss(x, muShift, sigma);
    }
  },
  skewed: {
    desc: "A single skewed peak — think \"reaction times on a response test.\"",
    realPdf: x => 0.6 * gauss(x, -0.8, 0.5) + 0.4 * gauss(x, 0.5, 1.0),
    genPdf: (x, t) => {
      const alpha = Math.min(1, t / 800);
      const s = 1.5 - 1.0 * alpha;
      const mu1 = -0.8 * alpha, mu2 = 0.5 * alpha;
      return 0.6 * gauss(x, mu1, s) + 0.4 * gauss(x, mu2, s);
    }
  },
  trimodal: {
    desc: "Three modes — notoriously prone to mode collapse in GANs.",
    realPdf: x => (1/3) * (gauss(x, -3, 0.4) + gauss(x, 0, 0.4) + gauss(x, 3, 0.4)),
    genPdf: (x, t) => {
      const alpha = Math.min(1, t / 800);
      const s = 1.5 - 1.1 * alpha;
      const shift = 3 * alpha;
      return (1/3) * (gauss(x, -shift, s) + gauss(x, 0, s) + gauss(x, shift, s));
    }
  }
};
let currentScenario = "bimodal";

// ── SVG helpers ──
function el(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const k of Object.keys(attrs)) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

function gridAxes(svg, { W, H, margL = 50, margR = 20, margT = 24, margB = 34, xMin = -6, xMax = 6, ticks = [-6, -4, -2, 0, 2, 4, 6] }) {
  const plotW = W - margL - margR;
  const plotH = H - margT - margB;
  // background
  svg.appendChild(el("rect", { x: 0, y: 0, width: W, height: H, fill: "#fdfcf9" }));
  // axis lines
  svg.appendChild(el("line", { x1: margL, y1: H - margB, x2: W - margR, y2: H - margB, stroke: "#6e665b", "stroke-width": 1 }));
  svg.appendChild(el("line", { x1: margL, y1: margT, x2: margL, y2: H - margB, stroke: "#6e665b", "stroke-width": 1 }));
  // x ticks
  for (const xv of ticks) {
    const xp = margL + ((xv - xMin) / (xMax - xMin)) * plotW;
    svg.appendChild(el("line", { x1: xp, y1: H - margB, x2: xp, y2: H - margB + 4, stroke: "#6e665b" }));
    const t = el("text", { x: xp, y: H - margB + 18, "text-anchor": "middle", "font-family": "IBM Plex Mono, monospace", "font-size": 11, fill: "#6e665b" }, xv);
    svg.appendChild(t);
  }
  return {
    margL, margR, margT, margB, plotW, plotH, xMin, xMax,
    xScale: x => margL + ((x - xMin) / (xMax - xMin)) * plotW,
    yScaleP: (y, max = 0.6) => H - margB - (y / max) * plotH,
    yScaleD: y => H - margB - y * plotH
  };
}

function curve(svg, fn, { xMin, xMax, xScale, yScaleP, stroke, fill, strokeWidth = 2.2, dash, yMax = 0.6 }) {
  const step = (xMax - xMin) / 500;
  let d = "";
  let first = true;
  for (let x = xMin; x <= xMax; x += step) {
    const xp = xScale(x), yp = yScaleP(fn(x), yMax);
    d += (first ? `M ${xp} ${yp}` : ` L ${xp} ${yp}`);
    first = false;
  }
  const p = el("path", { d, fill: "none", stroke, "stroke-width": strokeWidth });
  if (dash) p.setAttribute("stroke-dasharray", dash);
  svg.appendChild(p);
  if (fill) {
    const fillD = d + ` L ${xScale(xMax)} ${yScaleP(0, yMax)} L ${xScale(xMin)} ${yScaleP(0, yMax)} Z`;
    const fp = el("path", { d: fillD, fill, "fill-opacity": 0.15 });
    svg.appendChild(fp);
  }
}

function curveD(svg, fn, { xMin, xMax, xScale, yScaleD }) {
  const step = (xMax - xMin) / 500;
  let d = "";
  let first = true;
  for (let x = xMin; x <= xMax; x += step) {
    const xp = xScale(x), yp = yScaleD(fn(x));
    d += (first ? `M ${xp} ${yp}` : ` L ${xp} ${yp}`);
    first = false;
  }
  const p = el("path", { d, fill: "none", stroke: "#4a6670", "stroke-width": 2, "stroke-dasharray": "6 4" });
  svg.appendChild(p);
}

function label(svg, text, x, y, color = "#6e665b", size = 11) {
  svg.appendChild(el("text", { x, y, "font-family": "Manrope, sans-serif", "font-size": size, fill: color }, text));
}

// ==========================================================================
// ── Step 1: target distribution ──
// ==========================================================================
function renderStep1() {
  const svg = document.getElementById("step1-plot");
  clear(svg);
  const ctx = gridAxes(svg, { W: 880, H: 240, margT: 20, margB: 36 });
  const fn = SCENARIOS[currentScenario].realPdf;
  curve(svg, fn, { ...ctx, stroke: "#7b9e89", fill: "#7b9e89", strokeWidth: 2.4 });
  label(svg, "x", 440, 234);
  label(svg, "p(x)", 12, 30);
}

document.querySelectorAll("#scenario-buttons .mode-button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#scenario-buttons .mode-button").forEach(b => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    currentScenario = btn.dataset.case;
    document.getElementById("scenario-desc").textContent = SCENARIOS[currentScenario].desc;
    renderStep1();
    renderStep3();
    renderStep4();
    renderStep5();
    renderStep6();
  });
});

// ==========================================================================
// ── Step 2: manual discriminator over a fixed "bad" generator ──
// ==========================================================================
const genBadPdf = x => gauss(x, 0, 1.5);
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function manualD(x, centre, slope) { return sigmoid((x - centre) * slope); }

function renderStep2() {
  const centre = parseFloat(document.getElementById("d-centre").value);
  const slope = parseFloat(document.getElementById("d-slope").value);
  document.getElementById("d-centre-val").textContent = centre.toFixed(1);
  document.getElementById("d-slope-val").textContent = slope.toFixed(1);

  const svg = document.getElementById("step2-plot");
  clear(svg);
  const ctx = gridAxes(svg, { W: 880, H: 280 });
  const realPdf = SCENARIOS.bimodal.realPdf;
  curve(svg, realPdf, { ...ctx, stroke: "#7b9e89", fill: "#7b9e89" });
  curve(svg, genBadPdf, { ...ctx, stroke: "#d9622b", fill: "#d9622b" });
  curveD(svg, x => manualD(x, centre, slope), ctx);

  label(svg, "real p(x)", 700, 35, "#7b9e89", 12);
  label(svg, "fake p_G(x)", 700, 52, "#d9622b", 12);
  label(svg, "D(x)  (0 to 1)", 700, 69, "#4a6670", 12);

  // Compute real-detect-accuracy (fraction of real mass where D > 0.5)
  let accReal = 0, accFake = 0, Lreal = 0, Lfake = 0;
  const dx = 0.02;
  for (let x = -10; x <= 10; x += dx) {
    const pr = realPdf(x), pg = genBadPdf(x);
    const d = manualD(x, centre, slope);
    if (d > 0.5) accReal += pr * dx;
    if (d < 0.5) accFake += pg * dx;
    if (pr > 1e-8) Lreal -= pr * Math.log(Math.max(d, 1e-8)) * dx;
    if (pg > 1e-8) Lfake -= pg * Math.log(Math.max(1 - d, 1e-8)) * dx;
  }
  document.getElementById("d-acc-real").textContent = (accReal * 100).toFixed(1) + "%";
  document.getElementById("d-acc-fake").textContent = (accFake * 100).toFixed(1) + "%";
  document.getElementById("d-loss-val").textContent = (Lreal + Lfake).toFixed(3);
}
document.getElementById("d-centre").addEventListener("input", renderStep2);
document.getElementById("d-slope").addEventListener("input", renderStep2);

// ==========================================================================
// ── Step 3: optimal discriminator ──
// ==========================================================================
function renderStep3() {
  const sigma = parseFloat(document.getElementById("g-sigma").value);
  document.getElementById("g-sigma-val").textContent = sigma.toFixed(2);

  const svg = document.getElementById("step3-plot");
  clear(svg);
  const ctx = gridAxes(svg, { W: 880, H: 300 });
  const realPdf = SCENARIOS[currentScenario].realPdf;
  // Fixed generator at origin with variable sigma
  const genPdf = x => 0.5 * gauss(x, -0.5, sigma) + 0.5 * gauss(x, 0.5, sigma);
  curve(svg, realPdf, { ...ctx, stroke: "#7b9e89", fill: "#7b9e89" });
  curve(svg, genPdf, { ...ctx, stroke: "#d9622b", fill: "#d9622b" });

  const Dstar = x => {
    const pr = realPdf(x), pg = genPdf(x);
    if (pr + pg < 1e-10) return 0.5;
    return pr / (pr + pg);
  };
  curveD(svg, Dstar, ctx);

  // horizontal 0.5 line
  svg.appendChild(el("line", {
    x1: ctx.margL, y1: ctx.yScaleD(0.5), x2: 880 - ctx.margR, y2: ctx.yScaleD(0.5),
    stroke: "#6e665b", "stroke-width": 0.6, "stroke-dasharray": "2 3", opacity: 0.5
  }));
  label(svg, "D = 0.5 (Nash)", 880 - ctx.margR - 110, ctx.yScaleD(0.5) - 4, "#6e665b", 10);
}
document.getElementById("g-sigma").addEventListener("input", renderStep3);

// ==========================================================================
// ── Step 4: G's gradient field · click to place sample ──
// ==========================================================================
let step4Sample = -1;  // initial sample position
function renderStep4() {
  const svg = document.getElementById("step4-plot");
  clear(svg);
  const ctx = gridAxes(svg, { W: 880, H: 300 });
  const realPdf = SCENARIOS[currentScenario].realPdf;
  const genPdf = x => gauss(x, 0, 1.5);
  const Dopt = x => {
    const pr = realPdf(x), pg = genPdf(x);
    return pr / (pr + pg + 1e-10);
  };

  curve(svg, realPdf, { ...ctx, stroke: "#7b9e89", fill: "#7b9e89" });
  curve(svg, genPdf, { ...ctx, stroke: "#d9622b", fill: "#d9622b" });
  curveD(svg, Dopt, ctx);

  // compute G's gradient: d/dx log D(x) = D'(x)/D(x)
  // numerically approximate
  function gGrad(x) {
    const h = 0.01;
    const numer = Math.log(Dopt(x + h) + 1e-10) - Math.log(Dopt(x - h) + 1e-10);
    return numer / (2 * h);
  }

  // Draw arrow at clicked sample
  const x = step4Sample;
  const xp = ctx.xScale(x);
  const baseY = ctx.yScaleP(genPdf(x));
  const grad = gGrad(x);
  const arrowLen = Math.max(-80, Math.min(80, grad * 30));
  // the sample dot
  svg.appendChild(el("circle", { cx: xp, cy: baseY, r: 6, fill: "#d9622b", stroke: "#1a1815", "stroke-width": 1 }));
  // the arrow
  svg.appendChild(el("line", { x1: xp, y1: baseY, x2: xp + arrowLen, y2: baseY, stroke: "#a8324b", "stroke-width": 2.5 }));
  // arrowhead
  const head = arrowLen > 0 ? -6 : 6;
  svg.appendChild(el("path", {
    d: `M ${xp + arrowLen} ${baseY} l ${head} -4 l 0 8 Z`,
    fill: "#a8324b"
  }));

  label(svg, `sample at x = ${x.toFixed(2)}   ·   D(x) = ${Dopt(x).toFixed(3)}   ·   gradient sign: ${grad > 0 ? "→ right" : "← left"}`, 50, 30, "#2f2a22", 12);
  label(svg, "real p(x)", 720, 50, "#7b9e89", 11);
  label(svg, "fake p_G(x)", 720, 67, "#d9622b", 11);
  label(svg, "D(x)", 720, 84, "#4a6670", 11);
}

document.getElementById("step4-plot").addEventListener("click", (e) => {
  const svg = document.getElementById("step4-plot");
  const pt = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
  const xMin = -6, xMax = 6, margL = 50, margR = 20, plotW = 880 - margL - margR;
  const x = xMin + (loc.x - margL) / plotW * (xMax - xMin);
  step4Sample = Math.max(xMin + 0.1, Math.min(xMax - 0.1, x));
  renderStep4();
});

// ==========================================================================
// ── Step 5: the full dance ──
// ==========================================================================
let step5Step = 0;
let step5Playing = false;
let step5Timer = null;

function pReal5(x) { return SCENARIOS[currentScenario].realPdf(x); }
function pGen5(x, t) { return SCENARIOS[currentScenario].genPdf(x, t); }
function D5(x, t) {
  const pr = pReal5(x), pg = pGen5(x, t);
  if (pr + pg < 1e-8) return 0.5;
  const opt = pr / (pr + pg);
  const noise = Math.sin(t / 80 + x * 0.5) * 0.08 * Math.exp(-t / 600);
  return Math.max(0.01, Math.min(0.99, opt + noise));
}

function renderStep5() {
  step5Step = parseInt(document.getElementById("step5-step").value);
  document.getElementById("step5-step-val").textContent = step5Step;

  const svg = document.getElementById("step5-plot");
  clear(svg);
  const ctx = gridAxes(svg, { W: 880, H: 320 });
  curve(svg, x => pReal5(x), { ...ctx, stroke: "#7b9e89", fill: "#7b9e89" });
  curve(svg, x => pGen5(x, step5Step), { ...ctx, stroke: "#d9622b", fill: "#d9622b" });
  curveD(svg, x => D5(x, step5Step), ctx);
  // 0.5 ref line
  svg.appendChild(el("line", {
    x1: ctx.margL, y1: ctx.yScaleD(0.5), x2: 880 - ctx.margR, y2: ctx.yScaleD(0.5),
    stroke: "#6e665b", "stroke-width": 0.6, "stroke-dasharray": "2 3", opacity: 0.4
  }));
  label(svg, `training step ${step5Step}`, 440, 22, "#2f2a22", 13);

  // losses panel
  const lossSvg = document.getElementById("step5-losses");
  clear(lossSvg);
  const lctx = gridAxes(lossSvg, { W: 880, H: 160, margT: 20, margB: 30, xMin: 0, xMax: 1000, ticks: [0, 250, 500, 750, 1000] });
  // synth loss curves
  const N = 120;
  const ds = [], gs = [];
  for (let i = 0; i <= N; i++) {
    const t = step5Step * i / N;
    const dl = 0.5 + 0.85 * (1 - Math.exp(-t / 400)) + 0.15 * Math.sin(t / 60) * Math.exp(-t / 800);
    const gl = 1.2 + 0.7 * Math.exp(-t / 300) + 0.20 * Math.sin(t / 50 + 1) * Math.exp(-t / 700);
    ds.push(dl); gs.push(gl);
  }
  const lplotW = 880 - lctx.margL - lctx.margR;
  const lplotH = 160 - lctx.margT - lctx.margB;
  const xS = i => lctx.margL + (i / N) * lplotW;
  const yS = v => 160 - lctx.margB - ((v - 0) / 3) * lplotH;
  const drawLoss = (arr, color) => {
    let d = "", first = true;
    arr.forEach((v, i) => {
      const xp = xS(i), yp = yS(v);
      d += (first ? `M ${xp} ${yp}` : ` L ${xp} ${yp}`);
      first = false;
    });
    lossSvg.appendChild(el("path", { d, fill: "none", stroke: color, "stroke-width": 2 }));
  };
  drawLoss(ds, "#4a6670");
  drawLoss(gs, "#a8324b");
  label(lossSvg, "D loss", 800, 35, "#4a6670", 11);
  label(lossSvg, "G loss", 800, 52, "#a8324b", 11);
  label(lossSvg, "training step →", 400, 150, "#6e665b", 11);
  label(lossSvg, "loss", 10, 30, "#6e665b", 11);
}

document.getElementById("step5-step").addEventListener("input", renderStep5);
document.getElementById("step5-reset").addEventListener("click", () => {
  step5Step = 0;
  document.getElementById("step5-step").value = 0;
  renderStep5();
});
document.getElementById("step5-play").addEventListener("click", () => {
  if (step5Playing) {
    clearInterval(step5Timer);
    step5Playing = false;
    document.getElementById("step5-play").textContent = "▶ Play";
    return;
  }
  step5Playing = true;
  document.getElementById("step5-play").textContent = "⏸ Pause";
  step5Step = 0;
  step5Timer = setInterval(() => {
    step5Step += 20;
    if (step5Step > 1000) {
      step5Step = 1000;
      clearInterval(step5Timer);
      step5Playing = false;
      document.getElementById("step5-play").textContent = "▶ Play";
    }
    document.getElementById("step5-step").value = step5Step;
    renderStep5();
  }, 80);
});

// ==========================================================================
// ── Step 6: mode collapse toggle ──
// ==========================================================================
let collapsed = false;
function renderStep6() {
  const svg = document.getElementById("step6-plot");
  clear(svg);
  const ctx = gridAxes(svg, { W: 880, H: 260 });
  const realPdf = SCENARIOS[currentScenario].realPdf;
  curve(svg, realPdf, { ...ctx, stroke: "#7b9e89", fill: "#7b9e89" });
  let genPdf;
  if (collapsed) {
    // collapsed to a single narrow mode near one of the real modes
    genPdf = x => gauss(x, 2, 0.3);
  } else {
    // healthy: matches real data
    genPdf = realPdf;
  }
  curve(svg, genPdf, { ...ctx, stroke: "#d9622b", fill: "#d9622b" });

  label(svg, "real data", 720, 40, "#7b9e89", 12);
  label(svg, "generator", 720, 58, "#d9622b", 12);
  label(svg, collapsed ? "G has collapsed onto ONE mode" : "G covers all real modes", 50, 30, collapsed ? "#a8324b" : "#7b9e89", 13);
}

document.getElementById("collapse-toggle").addEventListener("click", () => {
  collapsed = !collapsed;
  document.getElementById("collapse-label").textContent = collapsed ? "Currently: mode collapse" : "Currently: healthy trajectory";
  renderStep6();
});

// ── initial render ──
renderStep1();
renderStep2();
renderStep3();
renderStep4();
renderStep5();
renderStep6();
