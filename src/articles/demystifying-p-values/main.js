import { SCENARIOS } from './scenarios.js';
import { mean, meanGap, generateAllResplits, shuffle, renderMath, normalCdf, variance, standardError } from './math.js';
import { renderBubbles, setupCanvas } from './ui.js';

let state = {
  scenario: 'smartPills',
  bucketMixed: false,
  manualGaps: [],
  allCalculated: false,
  allResplits: [],
  observedGap: 0,
  animationProgress: 0,
  isDrawingMany: false,
  mcGaps: [],
  mcRunning: false
};

const elements = {
  scenarioButtons: document.querySelectorAll('[data-case]'),
  scenarioDesc: document.getElementById('scenario-desc'),
  groupABubbles: document.getElementById('groupA-bubbles'),
  groupBBubbles: document.getElementById('groupB-bubbles'),
  meanA: document.getElementById('meanA'),
  meanB: document.getElementById('meanB'),
  observedGap: document.getElementById('observedGap'),
  units: document.querySelectorAll('.unit'),

  step2: document.getElementById('step-2'),
  btnBucket: document.getElementById('btn-bucket'),
  bucketView: document.getElementById('bucket-view'),

  step3: document.getElementById('step-3'),
  btnDrawOne: document.getElementById('btn-draw-one'),
  btnDrawMany: document.getElementById('btn-draw-many'),
  manualResultBox: document.getElementById('manual-result-box'),
  manualCanvas: document.getElementById('manualCanvas'),

  step4: document.getElementById('step-4'),
  btnCalcAll: document.getElementById('btn-calc-all'),
  fullStats: document.getElementById('full-stats'),
  statObservedGap: document.getElementById('stat-observed-gap'),
  extremeCount: document.getElementById('extremeCount'),
  gapCanvas: document.getElementById('gapCanvas'),

  step5: document.getElementById('step-5'),
  finalPValue: document.getElementById('final-p-value'),
  pValueMath: document.getElementById('p-value-math'),
  pValueExplanation: document.getElementById('p-value-explanation'),

  parametricCanvas: document.getElementById('parametricCanvas'),
  btnRunMonteCarlo: document.getElementById('btn-run-monte-carlo'),
  monteCarloCanvas: document.getElementById('monteCarloCanvas'),
  mcCount: document.getElementById('mc-count'),
  mcPval: document.getElementById('mc-pval')
};

// --- Setup ---
function loadScenario(key) {
  state.scenario = key;
  const data = SCENARIOS[key];

  elements.scenarioButtons.forEach(btn => {
    btn.classList.toggle('is-active', btn.dataset.case === key);
  });
  elements.scenarioDesc.textContent = data.description;

  renderBubbles(elements.groupABubbles, data.groupA, 'group-a');
  renderBubbles(elements.groupBBubbles, data.groupB, 'group-b');

  const mA = mean(data.groupA);
  const mB = mean(data.groupB);
  state.observedGap = meanGap(data.groupA, data.groupB);

  elements.meanA.textContent = mA.toFixed(1);
  elements.meanB.textContent = mB.toFixed(1);
  elements.observedGap.textContent = state.observedGap.toFixed(1);

  elements.units.forEach(u => u.textContent = data.unit);

  state.bucketMixed = false;
  state.manualGaps = [];
  state.allCalculated = false;
  state.allResplits = [];
  state.animationProgress = 0;

  elements.bucketView.innerHTML = '<p class="bucket-empty-text">Bucket is empty. Click the button above to mix!</p>';
  elements.bucketView.classList.remove('filled');

  elements.manualResultBox.innerHTML = '<p>Click a button to draw random splits.</p>';
  elements.btnDrawMany.disabled = false;
  elements.btnDrawOne.disabled = false;
  elements.btnCalcAll.style.display = 'inline-flex';
  elements.btnCalcAll.disabled = false;
  elements.btnCalcAll.textContent = 'Calculate All 252 Splits';

  drawManualCanvas();

  elements.fullStats.classList.add('hidden');
  const ctx = elements.gapCanvas.getContext('2d');
  ctx.clearRect(0, 0, elements.gapCanvas.width, elements.gapCanvas.height);

  state.mcGaps = [];
  state.mcRunning = false;
  if (elements.mcCount) elements.mcCount.textContent = '0';
  if (elements.mcPval) elements.mcPval.textContent = '-';
  if (elements.btnRunMonteCarlo) {
    elements.btnRunMonteCarlo.disabled = false;
    elements.btnRunMonteCarlo.textContent = 'Run 1,000 Samples';
  }
  
  if (typeof drawParametric === 'function') drawParametric();
  if (typeof drawMonteCarloCanvas === 'function') drawMonteCarloCanvas();
}

// --- Step 2: Mix Bucket ---
elements.btnBucket.addEventListener('click', () => {
  const data = SCENARIOS[state.scenario];
  const allValues = [...data.groupA, ...data.groupB];

  elements.bucketView.innerHTML = '';
  elements.bucketView.classList.add('filled');
  renderBubbles(elements.bucketView, shuffle([...allValues]), 'neutral');

  state.bucketMixed = true;
});

// --- Step 3: Manual Draw ---
function drawSingleSplit() {
  const data = SCENARIOS[state.scenario];
  const allValues = shuffle([...data.groupA, ...data.groupB]);

  const newA = allValues.slice(0, 5);
  const newB = allValues.slice(5, 10);
  const gap = meanGap(newA, newB);

  state.manualGaps.push(gap);

  const isExtreme = gap >= state.observedGap - 1e-9;
  elements.manualResultBox.innerHTML = `
    <div class="flex-cols">
      <div><strong>A:</strong> [${newA.join(', ')}]</div>
      <div><strong>B:</strong> [${newB.join(', ')}]</div>
      <div><strong>Gap:</strong> <span style="color:${isExtreme ? 'var(--extreme)' : 'var(--accent)'};font-weight:bold;">${gap.toFixed(1)}</span>${isExtreme ? ' <span class="extreme-tag">as extreme!</span>' : ''}</div>
    </div>
  `;

  drawManualCanvas();
}

elements.btnDrawOne.addEventListener('click', () => {
  if (state.isDrawingMany) return;
  drawSingleSplit();
});

elements.btnDrawMany.addEventListener('click', async () => {
  if (state.isDrawingMany) return;
  state.isDrawingMany = true;
  elements.btnDrawMany.disabled = true;
  elements.btnDrawOne.disabled = true;

  elements.manualResultBox.innerHTML = `<p>Rapidly drawing 10 splits...</p>`;

  for (let i = 0; i < 10; i++) {
    drawSingleSplit();
    await new Promise(r => setTimeout(r, 100));
  }

  state.isDrawingMany = false;
  elements.btnDrawMany.disabled = false;
  elements.btnDrawOne.disabled = false;
});

function drawManualCanvas() {
  const canvas = elements.manualCanvas;
  const { ctx, w, h } = setupCanvas(canvas, 920, 220);

  ctx.clearRect(0, 0, w, h);

  const margin = { left: 50, right: 50, bottom: 40, top: 40 };
  const plotW = w - margin.left - margin.right;
  
  let maxAxis = Math.max(state.observedGap * 1.5, 10);
  if (state.manualGaps.length > 0) {
    maxAxis = Math.max(maxAxis, ...state.manualGaps) * 1.1; // Give 10% breathing room
  }

  // Background grid
  ctx.strokeStyle = '#f0ebe1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for(let i=0; i<=5; i++) {
    const x = margin.left + (plotW / 5) * i;
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, h - margin.bottom);
  }
  ctx.stroke();

  // Main axis
  ctx.beginPath();
  ctx.moveTo(margin.left, h - margin.bottom);
  ctx.lineTo(w - margin.right, h - margin.bottom);
  ctx.strokeStyle = '#c4beb1';
  ctx.lineWidth = 2;
  ctx.stroke();

  function getX(val) {
    return margin.left + (val / maxAxis) * plotW;
  }

  // Observed line
  const obsX = getX(state.observedGap);
  ctx.beginPath();
  ctx.moveTo(obsX, margin.top - 10);
  ctx.lineTo(obsX, h - margin.bottom);
  ctx.strokeStyle = 'rgba(217, 98, 43, 0.5)';
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#d9622b';
  ctx.font = '600 13px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Observed Gap', obsX, margin.top - 20);
  ctx.fillText(state.observedGap.toFixed(1), obsX, margin.top - 5);

  // Dots with jitter
  const dotYBase = h - margin.bottom - 15;
  state.manualGaps.forEach((gap, i) => {
    const x = getX(gap);
    const jitter = (Math.sin(i * 13.7) * 12);
    const y = dotYBase + jitter;
    const isExtreme = gap >= state.observedGap - 1e-9;

    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = isExtreme ? 'rgba(217, 98, 43, 0.7)' : 'rgba(44, 111, 183, 0.6)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // Axis Labels
  ctx.fillStyle = '#9a917f';
  ctx.textAlign = 'center';
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillText('0', margin.left, h - margin.bottom + 20);
  ctx.fillText(maxAxis.toFixed(0), w - margin.right, h - margin.bottom + 20);
  ctx.fillText('Mean Gap', w / 2, h - 5);

  // Count annotation
  if (state.manualGaps.length > 0) {
    const extremeCount = state.manualGaps.filter(g => g >= state.observedGap - 1e-9).length;
    ctx.fillStyle = '#6e665b';
    ctx.font = '500 12px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(
      `${state.manualGaps.length} draws, ${extremeCount} extreme`,
      w - margin.right, margin.top - 5
    );
  }
}

// --- Step 4 & 5: Calculate All ---
elements.btnCalcAll.addEventListener('click', () => {
  if (state.allCalculated) return;

  const data = SCENARIOS[state.scenario];
  const allValues = [...data.groupA, ...data.groupB];
  state.allResplits = generateAllResplits(allValues);

  elements.btnCalcAll.disabled = true;
  elements.btnCalcAll.textContent = 'Calculating...';

  state.animationProgress = 0;
  animateHistogram();
});

function drawHistogram() {
  const canvas = elements.gapCanvas;
  const { ctx, w, h } = setupCanvas(canvas, 920, 420);

  ctx.clearRect(0, 0, w, h);

  const margin = { top: 60, right: 50, bottom: 50, left: 60 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  // Bin data
  const bins = {};
  const visibleResplits = Math.floor(state.animationProgress * state.allResplits.length);
  const subset = state.allResplits.slice(0, visibleResplits);

  let maxGap = state.observedGap;
  subset.forEach(r => {
    const key = r.gap.toFixed(1);
    bins[key] = (bins[key] || 0) + 1;
    if (r.gap > maxGap) maxGap = r.gap;
  });

  const keys = Object.keys(bins).map(Number).sort((a, b) => a - b);
  const maxCount = Math.max(10, ...Object.values(bins));

  function getX(val) {
    return margin.left + (val / (maxGap * 1.2)) * plotW;
  }
  function getY(count) {
    return margin.top + plotH - (count / maxCount) * plotH;
  }

  // Grid lines
  ctx.strokeStyle = '#f0ebe1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (plotH / 4) * i;
    ctx.moveTo(margin.left, y);
    ctx.lineTo(w - margin.right, y);
  }
  ctx.stroke();

  // Axes
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top + plotH);
  ctx.lineTo(w - margin.right, margin.top + plotH);
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.strokeStyle = '#c4beb1';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Bars
  const barW = Math.max(6, plotW / (maxGap * 1.2 * 10) - 4);

  keys.forEach(k => {
    const count = bins[k.toFixed(1)];
    const x = getX(k);
    const y = getY(count);
    const isExtreme = k >= state.observedGap - 1e-9;

    ctx.fillStyle = isExtreme ? 'rgba(217, 98, 43, 0.85)' : 'rgba(44, 111, 183, 0.6)';

    const barH = margin.top + plotH - y;
    ctx.beginPath();
    ctx.roundRect(x - barW/2, y, barW, barH, [4, 4, 0, 0]);
    ctx.fill();

    // Count labels on bars when animation is complete
    if (state.animationProgress >= 1 && count > 0) {
      ctx.fillStyle = '#6e665b';
      ctx.font = '500 10px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(count, x, y - 5);
    }
  });

  // Observed Line
  const obsX = getX(state.observedGap);
  ctx.beginPath();
  ctx.moveTo(obsX, margin.top - 20);
  ctx.lineTo(obsX, margin.top + plotH);
  ctx.strokeStyle = '#d9622b';
  ctx.lineWidth = 2.5;
  ctx.setLineDash([6, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#d9622b';
  ctx.font = '600 14px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`Observed Gap: ${state.observedGap.toFixed(1)}`, obsX, margin.top - 30);

  // Y-axis labels
  ctx.fillStyle = '#9a917f';
  ctx.font = '12px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('0', margin.left - 10, margin.top + plotH + 4);
  ctx.fillText(maxCount.toString(), margin.left - 10, margin.top + 4);

  ctx.save();
  ctx.translate(margin.left - 35, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Number of Splits', 0, 0);
  ctx.restore();

  // X-axis label
  ctx.fillStyle = '#9a917f';
  ctx.font = '12px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Mean Gap', w / 2, h - 5);
}

function animateHistogram() {
  state.animationProgress += 0.025;
  if (state.animationProgress > 1) state.animationProgress = 1;

  drawHistogram();

  if (state.animationProgress < 1) {
    requestAnimationFrame(animateHistogram);
  } else {
    finishCalculation();
  }
}

function finishCalculation() {
  state.allCalculated = true;
  elements.btnCalcAll.style.display = 'none';
  elements.fullStats.classList.remove('hidden');

  const extremeSplits = state.allResplits.filter(r => r.gap >= state.observedGap - 1e-9);

  elements.statObservedGap.textContent = state.observedGap.toFixed(1);
  elements.extremeCount.textContent = `${extremeSplits.length}`;

  const pVal = extremeSplits.length / state.allResplits.length;
  const pValDisplay = pVal < 0.001 ? '< 0.001' : pVal.toFixed(3);
  elements.finalPValue.textContent = pValDisplay;
  elements.pValueMath.textContent = `p = ${extremeSplits.length} (extreme splits) / 252 (total splits) = ${pValDisplay}`;

  let summaryExact = document.getElementById('summary-exact-pval');
  if (summaryExact) summaryExact.textContent = pValDisplay;

  let explanation = "";
  if (pVal < 0.05) {
    explanation = `Only ${extremeSplits.length} out of 252 random splits produced a gap this large. That's just ${(pVal*100).toFixed(1)}% of the null world. It's very hard to explain this gap by chance alone, so we reject H\u2080 and conclude the treatment likely had a real effect.`;
  } else {
    explanation = `${extremeSplits.length} out of 252 random splits produced a gap this large or larger\u2014that's ${(pVal*100).toFixed(1)}% of the null world. A gap of ${state.observedGap.toFixed(1)} happens quite often by pure chance. We can't confidently say the treatment did anything.`;
  }
  elements.pValueExplanation.textContent = explanation;
}

// --- Bonus Section Visualizations ---
function drawParametric() {
  if (!elements.parametricCanvas) return;
  const { ctx, w, h } = setupCanvas(elements.parametricCanvas, 920, 280);
  ctx.clearRect(0, 0, w, h);

  // Compute standard deviation of the permutation null distribution
  const data = SCENARIOS[state.scenario];
  
  const varA = variance(data.groupA);
  const varB = variance(data.groupB);
  const sd = Math.sqrt(varA / data.groupA.length + varB / data.groupB.length);

  const margin = { top: 30, right: 50, bottom: 40, left: 50 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  const tStat = state.observedGap / sd;
  const paramSdEl = document.getElementById('param-sd');
  if (paramSdEl) paramSdEl.textContent = sd.toFixed(2);
  const paramObsEl = document.getElementById('param-obs');
  if (paramObsEl) paramObsEl.textContent = state.observedGap.toFixed(2);
  const paramTstatEl = document.getElementById('param-tstat');
  if (paramTstatEl) paramTstatEl.textContent = tStat.toFixed(2);
  // Calculate two-tailed p-value using normal distribution
  const paramPval = 2 * (1 - normalCdf(tStat));
  const pValStr = paramPval < 0.001 ? 'p < 0.001' : `p \\approx ${paramPval.toFixed(3)}`;
  const inlinePvalStr = paramPval < 0.001 ? '< 0.001' : `≈ ${paramPval.toFixed(3)}`;

  const paramConclusionEl = document.getElementById('param-conclusion-text');
  if (paramConclusionEl) {
    if (paramPval < 0.05) {
      paramConclusionEl.innerHTML = `Our gap is <strong>${tStat.toFixed(2)}</strong> times larger than the expected noise. It lands deep in the tail of the curve. To find the p-value, we don't count combinations; we just use calculus to find the area under the curve in those extreme tails. Because ${tStat.toFixed(2)} standard deviations is so far out, the area is tiny: <strong>p ${inlinePvalStr}</strong>.`;
    } else {
      paramConclusionEl.innerHTML = `Our gap is <strong>${tStat.toFixed(2)}</strong> times larger than the expected noise. Because ${tStat.toFixed(2)} is relatively close to zero, it lands in the fat middle of the curve. Using calculus to find the shaded area under the curve in the tails gives us a large probability: <strong>p ${inlinePvalStr}</strong>.`;
    }
  }

  const summaryParamPvalEl = document.getElementById('summary-param-pval');
  if (summaryParamPvalEl) summaryParamPvalEl.textContent = paramPval < 0.001 ? '< 0.001' : paramPval.toFixed(3);
  
  // Update groups info
  const paramGroupA = document.getElementById('param-group-a');
  if (paramGroupA) paramGroupA.textContent = '[' + data.groupA.join(', ') + ']';
  const paramMeanA = document.getElementById('param-mean-a');
  if (paramMeanA) paramMeanA.textContent = mean(data.groupA).toFixed(1);
  const paramVarA = document.getElementById('param-var-a');
  if (paramVarA) paramVarA.textContent = varA.toFixed(2);

  const paramGroupB = document.getElementById('param-group-b');
  if (paramGroupB) paramGroupB.textContent = '[' + data.groupB.join(', ') + ']';
  const paramMeanB = document.getElementById('param-mean-b');
  if (paramMeanB) paramMeanB.textContent = mean(data.groupB).toFixed(1);
  const paramVarB = document.getElementById('param-var-b');
  if (paramVarB) paramVarB.textContent = varB.toFixed(2);

  const sdMathEl = document.getElementById('math-sd-dynamic');
  if (sdMathEl && window.katex) {
    katex.render(`\\sigma \\approx \\sqrt{\\frac{\\text{Var}_A}{n_A} + \\frac{\\text{Var}_B}{n_B}} = \\sqrt{\\frac{${varA.toFixed(2)}}{5} + \\frac{${varB.toFixed(2)}}{5}} = ${sd.toFixed(2)}`, sdMathEl, { displayMode: true, throwOnError: false });
  }

  const tstatMathEl = document.getElementById('math-tstat-dynamic');
  if (tstatMathEl && window.katex) {
    katex.render(`t = \\frac{\\text{Gap}}{\\text{SD}} = \\frac{${state.observedGap.toFixed(2)}}{${sd.toFixed(2)}} = ${tStat.toFixed(2)}`, tstatMathEl, { displayMode: true, throwOnError: false });
  }

  const integralMathEl = document.getElementById('math-integral');
  if (integralMathEl && window.katex) {
    katex.render(`p = 2 \\times \\int_{${Math.abs(tStat).toFixed(2)}}^{\\infty} \\frac{1}{\\sqrt{2\\pi}} e^{-\\frac{x^2}{2}} dx \\approx ${paramPval.toFixed(3)}`, integralMathEl, { displayMode: true, throwOnError: false });
  }

  // Plot from -3.5 SD to +3.5 SD (or observed gap + breathing room)
  const maxAxis = Math.max(state.observedGap * 1.5, 3.5 * sd);

  function getX(val) {
    return margin.left + ((val + maxAxis) / (2 * maxAxis)) * plotW;
  }

  function normalPdf(x) {
    return (1 / (sd * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow(x / sd, 2));
  }

  const maxY = normalPdf(0);
  function getY(yVal) {
    return margin.top + plotH - (yVal / maxY) * plotH * 0.9;
  }

  // Draw X axis
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top + plotH);
  ctx.lineTo(w - margin.right, margin.top + plotH);
  ctx.strokeStyle = '#c4beb1';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Draw Bell Curve Path
  ctx.beginPath();
  for(let x = -maxAxis; x <= maxAxis; x += (maxAxis/100)) {
    const px = getX(x);
    const py = getY(normalPdf(x));
    if (x === -maxAxis) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = 'rgba(44, 111, 183, 0.8)';
  ctx.lineWidth = 3;
  ctx.stroke();

  // Shade extremes
  ctx.fillStyle = 'rgba(217, 98, 43, 0.3)';
  
  // Right tail
  ctx.beginPath();
  ctx.moveTo(getX(state.observedGap), margin.top + plotH);
  for(let x = state.observedGap; x <= maxAxis; x += (maxAxis/50)) {
    ctx.lineTo(getX(x), getY(normalPdf(x)));
  }
  ctx.lineTo(getX(maxAxis), margin.top + plotH);
  ctx.fill();
  
  // Left tail
  ctx.beginPath();
  ctx.moveTo(getX(-state.observedGap), margin.top + plotH);
  for(let x = -state.observedGap; x >= -maxAxis; x -= (maxAxis/50)) {
    ctx.lineTo(getX(x), getY(normalPdf(x)));
  }
  ctx.lineTo(getX(-maxAxis), margin.top + plotH);
  ctx.fill();

  // Annotations
  ctx.fillStyle = '#d9622b';
  ctx.font = '600 13px system-ui, sans-serif';
  ctx.textAlign = 'center';
  const rx = getX(state.observedGap);
  const lx = getX(-state.observedGap);
  ctx.fillText(`\u2265 ${state.observedGap.toFixed(1)}`, Math.min(rx + 15, w - 20), margin.top + plotH - 10);
  ctx.fillText(`\u2264 -${state.observedGap.toFixed(1)}`, Math.max(lx - 15, 20), margin.top + plotH - 10);
  
  // X-axis labels
  ctx.fillStyle = '#9a917f';
  ctx.font = '12px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('0', getX(0), margin.top + plotH + 20);
  ctx.fillText(`+${maxAxis.toFixed(0)}`, getX(maxAxis), margin.top + plotH + 20);
  ctx.fillText(`-${maxAxis.toFixed(0)}`, getX(-maxAxis), margin.top + plotH + 20);
  ctx.fillText('Mean Difference (A - B)', w / 2, h - 5);
}

function drawMonteCarloCanvas() {
  if (!elements.monteCarloCanvas) return;
  const canvas = elements.monteCarloCanvas;
  const { ctx, w, h } = setupCanvas(canvas, 920, 280);
  ctx.clearRect(0, 0, w, h);

  const margin = { top: 40, right: 50, bottom: 40, left: 60 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  // Bin data
  const bins = {};
  let maxGap = state.observedGap;
  state.mcGaps.forEach(gap => {
    const key = gap.toFixed(1);
    bins[key] = (bins[key] || 0) + 1;
    if (gap > maxGap) maxGap = gap;
  });

  const keys = Object.keys(bins).map(Number).sort((a, b) => a - b);
  const maxCount = Math.max(10, ...Object.values(bins));

  function getX(val) {
    return margin.left + (val / (maxGap * 1.2)) * plotW;
  }
  function getY(count) {
    return margin.top + plotH - (count / maxCount) * plotH;
  }

  // Grid lines
  ctx.strokeStyle = '#f0ebe1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (plotH / 4) * i;
    ctx.moveTo(margin.left, y);
    ctx.lineTo(w - margin.right, y);
  }
  ctx.stroke();

  // Axes
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top + plotH);
  ctx.lineTo(w - margin.right, margin.top + plotH);
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.strokeStyle = '#c4beb1';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Bars
  const barW = Math.max(4, plotW / (maxGap * 1.2 * 10) - 2);
  keys.forEach(k => {
    const count = bins[k.toFixed(1)];
    const x = getX(k);
    const y = getY(count);
    
    ctx.fillStyle = k >= state.observedGap - 1e-9 ? 'rgba(217, 98, 43, 0.85)' : 'rgba(44, 111, 183, 0.6)';
    const barH = margin.top + plotH - y;
    ctx.beginPath();
    ctx.roundRect(x - barW/2, y, barW, barH, [3, 3, 0, 0]);
    ctx.fill();
  });

  // Observed Line
  const obsX = getX(state.observedGap);
  ctx.beginPath();
  ctx.moveTo(obsX, margin.top - 15);
  ctx.lineTo(obsX, margin.top + plotH);
  ctx.strokeStyle = '#d9622b';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  ctx.stroke();
  ctx.setLineDash([]);
  
  ctx.fillStyle = '#d9622b';
  ctx.font = '600 13px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`\u2265 ${state.observedGap.toFixed(1)}`, obsX, margin.top - 20);

  // Labels
  ctx.fillStyle = '#9a917f';
  ctx.font = '12px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('0', margin.left - 10, margin.top + plotH + 4);
  ctx.fillText(maxCount.toString(), margin.left - 10, margin.top + 4);
  ctx.textAlign = 'center';
  ctx.fillText('Mean Gap', w / 2, h - 5);
}

if (elements.btnRunMonteCarlo) {
  elements.btnRunMonteCarlo.addEventListener('click', () => {
    if (state.mcRunning) return;
    state.mcRunning = true;
    state.mcGaps = [];
    elements.btnRunMonteCarlo.disabled = true;
    elements.btnRunMonteCarlo.textContent = 'Simulating...';
    
    const data = SCENARIOS[state.scenario];
    const allValues = [...data.groupA, ...data.groupB];
    const targetCount = 1000;
    const batchSize = 50;
    
    function mcTick() {
      if (state.mcGaps.length >= targetCount) {
        state.mcRunning = false;
        elements.btnRunMonteCarlo.textContent = 'Run Another 1,000';
        elements.btnRunMonteCarlo.disabled = false;
        return;
      }
      
      for (let i = 0; i < batchSize; i++) {
        const shuffled = shuffle([...allValues]);
        const newA = shuffled.slice(0, 5);
        const newB = shuffled.slice(5, 10);
        state.mcGaps.push(meanGap(newA, newB));
      }
      
      const extremeCount = state.mcGaps.filter(g => g >= state.observedGap - 1e-9).length;
      const pval = extremeCount / state.mcGaps.length;
      const pValDisplay = pval < 0.001 ? '< 0.001' : pval.toFixed(3);
      
      elements.mcCount.textContent = state.mcGaps.length.toLocaleString();
      elements.mcPval.textContent = pValDisplay;
      
      const summaryMcPvalEl = document.getElementById('summary-mc-pval');
      if (summaryMcPvalEl) summaryMcPvalEl.textContent = pValDisplay;
      
      drawMonteCarloCanvas();
      requestAnimationFrame(mcTick);
    }
    
    requestAnimationFrame(mcTick);
  });
}

// --- Init ---
elements.scenarioButtons.forEach(btn => {
  btn.addEventListener('click', () => loadScenario(btn.dataset.case));
});

function init() {
  if (window.katex) {
    renderMath();
  } else {
    var s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  loadScenario('smartPills');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}