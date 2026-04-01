const SCENARIOS = {
  smartPills: {
    name: "Smart Pills vs Placebo",
    groupA: [85, 88, 82, 89, 81], // mean 85.0
    groupB: [92, 95, 91, 90, 96], // mean 92.8
    unit: "pts",
    description: "A pharmaceutical company tests a new 'smart pill' designed to boost cognitive function. Five students take the pill, and five take a sugar pill. These are their scores on a subsequent memory test."
  },
  fertilizer: {
    name: "New Fertilizer vs Old",
    groupA: [12, 14, 13, 15, 14], // mean 13.6
    groupB: [13, 15, 14, 14, 15], // mean 14.2
    unit: "cm",
    description: "A botanist wants to see if a new experimental fertilizer makes plants grow taller than the standard mix. Five seedlings get the old mix, and five get the new one. Here are their heights after two weeks."
  },
  coffee: {
    name: "Coffee vs Decaf",
    groupA: [58, 62, 55, 60, 56], // mean 58.2
    groupB: [65, 71, 62, 68, 64], // mean 66.0
    unit: "wpm",
    description: "Does caffeine actually make you type faster? We asked ten people to transcribe a document. Five were secretly given decaf, while the other five were given strong espresso. Here are their typing speeds."
  }
};

let state = {
  scenario: 'smartPills',
  bucketMixed: false,
  manualGaps: [],
  allCalculated: false,
  allResplits: [],
  observedGap: 0,
  animationProgress: 0,
  isDrawingMany: false
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
  pValueExplanation: document.getElementById('p-value-explanation')
};

// --- KaTeX math rendering ---
function renderMath() {
  if (!window.katex) return;
  var blocks = {
    'math-choose':    ['\\binom{10}{5} = \\frac{10!}{5!\\;5!} = 252', true],
    'math-pval':      ['p = \\frac{\\text{\\# splits with gap} \\ge \\text{observed gap}}{\\text{total splits (252)}}', true],
    'math-bigchoose': ['\\binom{40}{20} = 137{,}846{,}528{,}640', true],
  };
  Object.keys(blocks).forEach(function (id) {
    var el = document.getElementById(id);
    if (!el) return;
    try {
      katex.render(blocks[id][0], el, { displayMode: blocks[id][1], throwOnError: false });
    } catch (_) {}
  });
}

// --- Math Helpers ---
function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function meanGap(a, b) {
  return Math.abs(mean(a) - mean(b));
}

function combinations(array, choose) {
  const results = [];
  function helper(start, combo) {
    if (combo.length === choose) {
      results.push([...combo]);
      return;
    }
    for (let i = start; i < array.length; i++) {
      combo.push(array[i]);
      helper(i + 1, combo);
      combo.pop();
    }
  }
  helper(0, []);
  return results;
}

function generateAllResplits(values) {
  const indexed = values.map((value, index) => ({ value, index }));
  const groupAs = combinations(indexed, values.length / 2);

  return groupAs.map(groupA => {
    const groupAIds = new Set(groupA.map(item => item.index));
    const groupB = indexed.filter(item => !groupAIds.has(item.index));
    const sampleA = groupA.map(item => item.value);
    const sampleB = groupB.map(item => item.value);
    return {
      sampleA,
      sampleB,
      gap: meanGap(sampleA, sampleB)
    };
  });
}

function shuffle(array) {
  let currentIndex = array.length, randomIndex;
  while (currentIndex !== 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
  }
  return array;
}

// --- Render Helpers ---
function renderBubbles(container, values, type = 'neutral') {
  container.innerHTML = '';
  values.forEach((v, i) => {
    const el = document.createElement('div');
    el.className = `bubble ${type}`;
    el.textContent = v;
    el.style.animationDelay = `${i * 0.04}s`;
    container.appendChild(el);
  });
}

function unlockStep(stepEl) {
  if (stepEl.classList.contains('disabled')) {
    stepEl.classList.remove('disabled');
    stepEl.classList.add('unlocked');
    setTimeout(() => stepEl.classList.remove('unlocked'), 1200);
  }
}

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

  // Reset downstream state
  state.bucketMixed = false;
  state.manualGaps = [];
  state.allCalculated = false;
  state.allResplits = [];
  state.animationProgress = 0;

  elements.bucketView.innerHTML = '<p class="bucket-empty-text">Bucket is empty. Click the button above to mix!</p>';
  elements.bucketView.classList.remove('filled');

  [elements.step2, elements.step3, elements.step4, elements.step5].forEach(el => {
    el.classList.add('disabled');
    el.classList.remove('unlocked');
  });

  // Unlock step 2 immediately
  unlockStep(elements.step2);

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
}

// --- Step 2: Mix Bucket ---
elements.btnBucket.addEventListener('click', () => {
  const data = SCENARIOS[state.scenario];
  const allValues = [...data.groupA, ...data.groupB];

  elements.bucketView.innerHTML = '';
  elements.bucketView.classList.add('filled');
  renderBubbles(elements.bucketView, shuffle([...allValues]), 'neutral');

  state.bucketMixed = true;
  unlockStep(elements.step3);
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

  if (state.manualGaps.length >= 1) {
    unlockStep(elements.step4);
  }
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

function setupCanvas(canvas, logicalWidth, logicalHeight) {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== logicalWidth * dpr) {
    canvas.width = logicalWidth * dpr;
    canvas.height = logicalHeight * dpr;
    canvas.style.width = logicalWidth + 'px';
    canvas.style.height = logicalHeight + 'px';
  }
  const ctx = canvas.getContext('2d');
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
  return { ctx, w: logicalWidth, h: logicalHeight };
}

function drawManualCanvas() {
  const canvas = elements.manualCanvas;
  const { ctx, w, h } = setupCanvas(canvas, 920, 220);

  ctx.clearRect(0, 0, w, h);

  const margin = { left: 50, right: 50, bottom: 40, top: 40 };
  const plotW = w - margin.left - margin.right;
  const maxAxis = Math.max(state.observedGap * 1.5, 10);

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

  // Step 5 Update
  unlockStep(elements.step5);

  const pVal = extremeSplits.length / state.allResplits.length;
  elements.finalPValue.textContent = pVal.toFixed(3);
  elements.pValueMath.textContent = `p = ${extremeSplits.length} (extreme splits) / 252 (total splits) = ${pVal.toFixed(3)}`;

  let explanation = "";
  if (pVal < 0.05) {
    explanation = `Only ${extremeSplits.length} out of 252 random splits produced a gap this large. That's just ${(pVal*100).toFixed(1)}% of the null world. It's very hard to explain this gap by chance alone, so we reject H\u2080 and conclude the treatment likely had a real effect.`;
  } else {
    explanation = `${extremeSplits.length} out of 252 random splits produced a gap this large or larger\u2014that's ${(pVal*100).toFixed(1)}% of the null world. A gap of ${state.observedGap.toFixed(1)} happens quite often by pure chance. We can't confidently say the treatment did anything.`;
  }
  elements.pValueExplanation.textContent = explanation;
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
