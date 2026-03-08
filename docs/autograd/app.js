const RULES = [
  {
    id: 'mul',
    label: 'Multiply',
    short: 'a × b',
    tex: 'f = a \\times b, \\qquad \\frac{\\partial f}{\\partial a} = b, \\qquad \\frac{\\partial f}{\\partial b} = a',
    summary: 'One input becomes the scale factor for the other input\'s gradient.',
    explanation: 'When two values are multiplied, each parent receives the upstream gradient scaled by the other parent. This is the most important rule for linear layers because weights and activations meet here.',
  },
  {
    id: 'add',
    label: 'Add',
    short: 'a + b',
    tex: 'f = a + b, \\qquad \\frac{\\partial f}{\\partial a} = 1, \\qquad \\frac{\\partial f}{\\partial b} = 1',
    summary: 'Addition just copies the upstream gradient to every parent.',
    explanation: 'An addition node does not scale anything. Each parent receives the same upstream message unchanged. This is where gradient accumulation becomes visible when many paths merge.',
  },
  {
    id: 'exp',
    label: 'Exponent',
    short: 'exp(a)',
    tex: 'f = e^a, \\qquad \\frac{\\partial f}{\\partial a} = e^a = f',
    summary: 'The derivative of an exponential is the exponential itself.',
    explanation: 'This is why the forward pass caches outputs. Once you already know e^a, the backward rule is immediate: multiply the upstream gradient by that stored output.',
  },
  {
    id: 'reciprocal',
    label: 'Reciprocal',
    short: '1 / a',
    tex: 'f = \\frac{1}{a}, \\qquad \\frac{\\partial f}{\\partial a} = -\\frac{1}{a^2}',
    summary: 'Reciprocals flip the sign and square the denominator.',
    explanation: 'The sigmoid can be built from a reciprocal after creating a denominator. The sign flip here is one of the moments where learners often lose the thread, so it helps to isolate it.',
  },
  {
    id: 'log',
    label: 'Log',
    short: 'log(a)',
    tex: 'f = \\log(a), \\qquad \\frac{\\partial f}{\\partial a} = \\frac{1}{a}',
    summary: 'The log turns multiplication into addition and its gradient into a reciprocal.',
    explanation: 'Cross-entropy style losses end with a logarithm. Once the upstream gradient reaches the log node, the only local work left is multiplying by 1/a.',
  },
];

const SINGLE_INPUT = { x: 1.6, w: 1.15, b: -0.8 };
const FORWARD_STAGE_END = 8;
const FINAL_STAGE = 17;

const GRAPH_NODES = [
  { id: 'x', label: 'x', sub: 'input', x: 120, y: 170, kind: 'input', forwardStep: 0, backwardStep: 17 },
  { id: 'w', label: 'w', sub: 'parameter', x: 120, y: 350, kind: 'parameter', forwardStep: 0, backwardStep: 17 },
  { id: 'mul', label: 'wx', sub: 'multiply', x: 330, y: 260, kind: 'op', forwardStep: 1, backwardStep: 16 },
  { id: 'b', label: 'b', sub: 'parameter', x: 330, y: 440, kind: 'parameter', forwardStep: 0, backwardStep: 15 },
  { id: 'z', label: 'z', sub: 'add', x: 540, y: 350, kind: 'op', forwardStep: 2, backwardStep: 15 },
  { id: 'neg', label: '-z', sub: 'negate', x: 750, y: 350, kind: 'op', forwardStep: 3, backwardStep: 14 },
  { id: 'exp', label: 'e^-z', sub: 'exp', x: 960, y: 350, kind: 'op', forwardStep: 4, backwardStep: 13 },
  { id: 'denom', label: '1 + exp', sub: 'add', x: 1170, y: 350, kind: 'op', forwardStep: 5, backwardStep: 12 },
  { id: 'p', label: 'p', sub: 'sigmoid', x: 1350, y: 350, kind: 'op', forwardStep: 6, backwardStep: 11 },
  { id: 'logp', label: 'log p', sub: 'log', x: 1350, y: 170, kind: 'op', forwardStep: 7, backwardStep: 10 },
  { id: 'L', label: 'L', sub: 'loss', x: 1350, y: 530, kind: 'loss', forwardStep: 8, backwardStep: 9 },
];

const GRAPH_EDGES = [
  { id: 'x-mul', from: 'x', to: 'mul' },
  { id: 'w-mul', from: 'w', to: 'mul' },
  { id: 'mul-z', from: 'mul', to: 'z' },
  { id: 'b-z', from: 'b', to: 'z' },
  { id: 'z-neg', from: 'z', to: 'neg' },
  { id: 'neg-exp', from: 'neg', to: 'exp' },
  { id: 'exp-denom', from: 'exp', to: 'denom' },
  { id: 'denom-p', from: 'denom', to: 'p' },
  { id: 'p-logp', from: 'p', to: 'logp' },
  { id: 'logp-L', from: 'logp', to: 'L' },
];

const GRAPH_STAGES = [
  {
    direction: 'Setup',
    focus: 'Inputs',
    activeNodes: ['x', 'w', 'b'],
    activeEdges: [],
    formula: 'z = wx + b',
    headline: (s) => 'Inputs are known before any work happens.',
    explanation: (s) => `The example arrives as <code>x = ${formatNumber(s.values.x)}</code>. The learnable parameters start at <code>w = ${formatNumber(s.values.w)}</code> and <code>b = ${formatNumber(s.values.b)}</code>. No gradients exist yet because the loss has not been built.`,
  },
  {
    direction: 'Forward',
    focus: 'Multiply',
    activeNodes: ['x', 'w', 'mul'],
    activeEdges: ['x-mul', 'w-mul'],
    formula: 'wx = w \\cdot x',
    headline: (s) => 'The first operation forms the weighted input.',
    explanation: (s) => `Multiplication creates <code>wx = ${formatNumber(s.values.mul)}</code>. Later, when gradients return, this node will send <code>x</code> back to <code>w</code> and <code>w</code> back to <code>x</code>.`,
  },
  {
    direction: 'Forward',
    focus: 'Add bias',
    activeNodes: ['mul', 'b', 'z'],
    activeEdges: ['mul-z', 'b-z'],
    formula: 'z = wx + b',
    headline: (s) => 'Adding the bias produces the logit.',
    explanation: (s) => `The add node combines the weighted input and the bias to produce <code>z = ${formatNumber(s.values.z)}</code>. Addition is simple on the backward pass: it copies the upstream gradient to both parents.`,
  },
  {
    direction: 'Forward',
    focus: 'Negate',
    activeNodes: ['z', 'neg'],
    activeEdges: ['z-neg'],
    formula: '-z = -1 \\cdot z',
    headline: (s) => 'Sigmoid starts by flipping the sign of the logit.',
    explanation: (s) => `The graph now stores <code>-z = ${formatNumber(s.values.neg)}</code>. This sign flip is small, but it matters later because it contributes a local derivative of <code>-1</code>.`,
  },
  {
    direction: 'Forward',
    focus: 'Exponent',
    activeNodes: ['neg', 'exp'],
    activeEdges: ['neg-exp'],
    formula: 'e^{-z} = \\exp(-z)',
    headline: (s) => 'Exponentiation turns the negated logit into a positive scale.',
    explanation: (s) => `The exponential node stores <code>e^{-z} = ${formatNumber(s.values.exp)}</code>. Because the derivative of <code>e^a</code> is itself, the forward output is exactly what the backward rule will need.`,
  },
  {
    direction: 'Forward',
    focus: 'Denominator',
    activeNodes: ['exp', 'denom'],
    activeEdges: ['exp-denom'],
    formula: 'd = 1 + e^{-z}',
    headline: (s) => 'A denominator appears for the reciprocal step.',
    explanation: (s) => `Adding <code>1</code> produces <code>d = ${formatNumber(s.values.denom)}</code>. The hidden constant is not visually important, so the diagram omits it, but the numeric state still reflects it.`,
  },
  {
    direction: 'Forward',
    focus: 'Sigmoid probability',
    activeNodes: ['denom', 'p'],
    activeEdges: ['denom-p'],
    formula: 'p = \\frac{1}{d}',
    headline: (s) => 'The reciprocal converts the denominator into a probability.',
    explanation: (s) => `The sigmoid output is <code>p = ${formatNumber(s.values.p)}</code>. For this positive example, larger probabilities mean smaller loss.`,
  },
  {
    direction: 'Forward',
    focus: 'Log',
    activeNodes: ['p', 'logp'],
    activeEdges: ['p-logp'],
    formula: '\\log p = \\log(p)',
    headline: (s) => 'The loss path takes a logarithm.',
    explanation: (s) => `The graph stores <code>\\log(p) = ${formatNumber(s.values.logp)}</code>. The log turns a probability into a quantity that grows sharply when the model is wrong.`,
  },
  {
    direction: 'Forward',
    focus: 'Loss',
    activeNodes: ['logp', 'L'],
    activeEdges: ['logp-L'],
    formula: 'L = -\\log p',
    headline: (s) => 'The scalar loss is now ready.',
    explanation: (s) => `Negating the log produces <code>L = ${formatNumber(s.values.L)}</code>. Only now can the backward pass begin, because the engine has a scalar target to differentiate.`,
  },
  {
    direction: 'Backward',
    focus: 'Seed the loss',
    activeNodes: ['L'],
    activeEdges: [],
    formula: '\\frac{\\partial L}{\\partial L} = 1',
    headline: (s) => 'Backprop starts by seeding the loss with gradient 1.',
    explanation: (s) => `Every reverse pass starts with <code>\\partial L / \\partial L = 1</code>. That seed says: if the loss changes by a little, it changes by exactly that much.`,
  },
  {
    direction: 'Backward',
    focus: 'Back through the final negation',
    activeNodes: ['L', 'logp'],
    activeEdges: ['logp-L'],
    formula: '\\frac{\\partial L}{\\partial \\log p} = \\frac{\\partial L}{\\partial L} \\cdot (-1)',
    headline: (s) => 'The final negate sends a clean gradient of -1 to log p.',
    explanation: (s) => `Because <code>L = -\\log(p)</code>, the gradient at <code>\\log p</code> becomes <code>${formatNumber(s.grads.logp)}</code>. The structure is already visible: upstream gradient times local derivative.`,
  },
  {
    direction: 'Backward',
    focus: 'Back through the log',
    activeNodes: ['logp', 'p'],
    activeEdges: ['p-logp'],
    formula: '\\frac{\\partial L}{\\partial p} = \\frac{\\partial L}{\\partial \\log p} \\cdot \\frac{1}{p}',
    headline: (s) => 'The log converts the gradient into a reciprocal scale.',
    explanation: (s) => `The local derivative of <code>\\log(p)</code> is <code>1/p</code>, so the probability receives gradient <code>${formatNumber(s.grads.p)}</code>. Confident wrong probabilities get punished the most.`,
  },
  {
    direction: 'Backward',
    focus: 'Back through the reciprocal',
    activeNodes: ['p', 'denom'],
    activeEdges: ['denom-p'],
    formula: '\\frac{\\partial L}{\\partial d} = \\frac{\\partial L}{\\partial p} \\cdot \\left(-\\frac{1}{d^2}\\right)',
    headline: (s) => 'The reciprocal flips the sign and squares the denominator.',
    explanation: (s) => `The denominator inherits gradient <code>${formatNumber(s.grads.denom)}</code>. This is the step where the reciprocal rule does its full job: negative sign, squared denominator.`,
  },
  {
    direction: 'Backward',
    focus: 'Back through the exponent',
    activeNodes: ['denom', 'exp', 'neg'],
    activeEdges: ['exp-denom', 'neg-exp'],
    formula: '\\frac{\\partial L}{\\partial (-z)} = \\frac{\\partial L}{\\partial e^{-z}} \\cdot e^{-z}',
    headline: (s) => 'Exponentiation scales the upstream message by its own output.',
    explanation: (s) => `Since the derivative of <code>e^{-z}</code> is <code>e^{-z}</code>, the negated logit receives gradient <code>${formatNumber(s.grads.neg)}</code>. This is why caching the forward output is so useful.`,
  },
  {
    direction: 'Backward',
    focus: 'Undo the sign flip',
    activeNodes: ['neg', 'z'],
    activeEdges: ['z-neg'],
    formula: '\\frac{\\partial L}{\\partial z} = \\frac{\\partial L}{\\partial (-z)} \\cdot (-1)',
    headline: (s) => 'The negate node flips the sign once more.',
    explanation: (s) => `The gradient at the logit becomes <code>${formatNumber(s.grads.z)}</code>. For a positive example, this value equals <code>p - 1</code>, which is why the optimizer will push the logit upward when the probability is still below one.`,
  },
  {
    direction: 'Backward',
    focus: 'Split across add',
    activeNodes: ['z', 'mul', 'b'],
    activeEdges: ['mul-z', 'b-z'],
    formula: '\\frac{\\partial L}{\\partial wx} = \\frac{\\partial L}{\\partial z}, \\qquad \\frac{\\partial L}{\\partial b} = \\frac{\\partial L}{\\partial z}',
    headline: (s) => 'The add node copies the same message to both parents.',
    explanation: (s) => `Both <code>wx</code> and <code>b</code> receive <code>${formatNumber(s.grads.z)}</code>. This is the cleanest rule in the table: addition simply forwards the upstream gradient unchanged.`,
  },
  {
    direction: 'Backward',
    focus: 'Back through multiply',
    activeNodes: ['mul', 'w', 'x'],
    activeEdges: ['x-mul', 'w-mul'],
    formula: '\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial wx} \\cdot x, \\qquad \\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial wx} \\cdot w',
    headline: (s) => 'The multiply node turns the logit gradient into parameter gradients.',
    explanation: (s) => `Because the other parent acts as the local derivative, the weight receives <code>${formatNumber(s.grads.w)}</code> and the input would receive <code>${formatNumber(s.grads.x)}</code> if we needed it.`,
  },
  {
    direction: 'Backward',
    focus: 'Optimizer-ready gradients',
    activeNodes: ['w', 'b', 'x'],
    activeEdges: [],
    formula: '\\nabla_{w,b} L = \\left[\\frac{\\partial L}{\\partial w}, \\frac{\\partial L}{\\partial b}\\right]',
    headline: (s) => 'The backward pass ends with gradients on the leaves.',
    explanation: (s) => `For this example, the optimizer sees <code>dL/dw = ${formatNumber(s.grads.w)}</code> and <code>dL/db = ${formatNumber(s.grads.b)}</code>. In a batch, those messages would be accumulated across many examples before the update.`,
  },
];

const MODULE_VIEWS = {
  micro: {
    headline: 'Atomic graph: every local derivative is explicit.',
    body: 'This is ideal for learning the engine itself. Every node is tiny, every dependency is visible, and every backward rule is a one-line local fact.',
    bullets: [
      { title: 'Linear score is decomposed', text: 'The weight-input multiply and the bias add appear as separate steps.' },
      { title: 'Sigmoid is not mysterious', text: 'It is just negate, exponent, add one, and reciprocal.' },
      { title: 'The loss stays honest', text: 'A log and a final negate complete the scalar objective.' },
    ],
    nodes: [
      { id: 'x', label: 'x', x: 90, y: 130, w: 96, h: 56, fill: '#eef7ff' },
      { id: 'w', label: 'w', x: 90, y: 250, w: 96, h: 56, fill: '#eef3ff' },
      { id: 'mul', label: '×', x: 250, y: 190, w: 104, h: 62, fill: '#ffffff' },
      { id: 'b', label: 'b', x: 250, y: 310, w: 96, h: 56, fill: '#eef3ff' },
      { id: 'add', label: '+', x: 410, y: 250, w: 104, h: 62, fill: '#ffffff' },
      { id: 'neg', label: '−', x: 570, y: 250, w: 104, h: 62, fill: '#ffffff' },
      { id: 'exp', label: 'exp', x: 730, y: 250, w: 120, h: 62, fill: '#ffffff' },
      { id: 'inv', label: '1/x', x: 890, y: 250, w: 120, h: 62, fill: '#ffffff' },
      { id: 'log', label: 'log', x: 1030, y: 170, w: 110, h: 62, fill: '#ffffff' },
      { id: 'loss', label: 'L', x: 1030, y: 330, w: 110, h: 62, fill: '#fff2ef' },
    ],
    edges: [
      ['x', 'mul'], ['w', 'mul'], ['mul', 'add'], ['b', 'add'], ['add', 'neg'], ['neg', 'exp'], ['exp', 'inv'], ['inv', 'log'], ['log', 'loss'],
    ],
  },
  modules: {
    headline: 'Framework blocks: the same math, fewer visible nodes.',
    body: 'PyTorch, JAX, and friends do not delete the chain rule. They package many tiny operations behind larger interfaces, each of which still knows how to backpropagate.',
    bullets: [
      { title: 'nn.Linear', text: 'Hides the multiply-and-add details while still producing a differentiable output.' },
      { title: 'Sigmoid', text: 'Encapsulates several scalar ops but can still return the correct gradient.' },
      { title: 'Cross-entropy style loss', text: 'Acts like one clean block even though its backward rule may be optimized internally.' },
    ],
    nodes: [
      { id: 'x', label: 'x', x: 90, y: 180, w: 96, h: 56, fill: '#eef7ff' },
      { id: 'params', label: 'w, b', x: 90, y: 300, w: 120, h: 56, fill: '#eef3ff' },
      { id: 'linear', label: 'nn.Linear', x: 320, y: 240, w: 190, h: 78, fill: '#ecf2ff' },
      { id: 'sigmoid', label: 'Sigmoid', x: 620, y: 240, w: 170, h: 78, fill: '#eefcf7' },
      { id: 'loss', label: 'Cross-Entropy', x: 910, y: 240, w: 190, h: 78, fill: '#fff2ef' },
      { id: 'out', label: 'L', x: 1075, y: 120, w: 90, h: 56, fill: '#fff2ef' },
    ],
    edges: [
      ['x', 'linear'], ['params', 'linear'], ['linear', 'sigmoid'], ['sigmoid', 'loss'], ['loss', 'out'],
    ],
  },
};

const BATCH_DATASETS = {
  easy: {
    label: 'Clean separation',
    description: 'Two negatives sit on the left and one positive sits on the right. The batch mainly asks for a steeper positive slope.',
    defaults: { w: 1.2, b: -0.1 },
    points: [
      { name: 'A', x: -2.2, y: 0 },
      { name: 'B', x: -0.6, y: 0 },
      { name: 'C', x: 2.0, y: 1 },
    ],
  },
  edge: {
    label: 'Near the boundary',
    description: 'All three points live near the decision boundary, so the update has to move both the slope and the offset carefully.',
    defaults: { w: 0.65, b: 0.15 },
    points: [
      { name: 'A', x: -0.9, y: 0 },
      { name: 'B', x: 0.15, y: 1 },
      { name: 'C', x: 0.85, y: 1 },
    ],
  },
  outlier: {
    label: 'One disagreeing point',
    description: 'An outlier pushes against the other two rows, which is exactly the kind of tension that makes gradient averaging feel real.',
    defaults: { w: 1.0, b: -0.2 },
    points: [
      { name: 'A', x: -2.0, y: 0 },
      { name: 'B', x: 0.2, y: 1 },
      { name: 'C', x: 1.9, y: 0 },
    ],
  },
};

const state = {
  selectedRule: RULES[0].id,
  graphStage: 0,
  moduleMode: 'micro',
  batch: {
    datasetKey: 'easy',
    w: BATCH_DATASETS.easy.defaults.w,
    b: BATCH_DATASETS.easy.defaults.b,
    lr: 0.2,
  },
};

let autoPlayHandle = null;
const graphSnapshot = computeSingleExample(SINGLE_INPUT);

const els = {
  progressFill: document.getElementById('progressFill'),
  tocLinks: Array.from(document.querySelectorAll('.toc__link')),
  sections: Array.from(document.querySelectorAll('[data-nav]')),
  ruleChips: document.getElementById('ruleChips'),
  ruleGrid: document.getElementById('ruleGrid'),
  ruleTitle: document.getElementById('ruleTitle'),
  ruleFormula: document.getElementById('ruleFormula'),
  ruleExplanation: document.getElementById('ruleExplanation'),
  graphSvg: document.getElementById('graphSvg'),
  stageIndex: document.getElementById('stageIndex'),
  stageDirection: document.getElementById('stageDirection'),
  stageFocus: document.getElementById('stageFocus'),
  stageHeadline: document.getElementById('stageHeadline'),
  stageExplanation: document.getElementById('stageExplanation'),
  chainRuleMath: document.getElementById('chainRuleMath'),
  nodeTable: document.getElementById('nodeTable'),
  stageScrubber: document.getElementById('stageScrubber'),
  resetGraph: document.getElementById('resetGraph'),
  stepForward: document.getElementById('stepForward'),
  stepBackward: document.getElementById('stepBackward'),
  autoPlay: document.getElementById('autoPlay'),
  exportStage: document.getElementById('exportStage'),
  exportTimeline: document.getElementById('exportTimeline'),
  moduleSvg: document.getElementById('moduleSvg'),
  moduleHeadline: document.getElementById('moduleHeadline'),
  moduleBody: document.getElementById('moduleBody'),
  moduleList: document.getElementById('moduleList'),
  modeChips: Array.from(document.querySelectorAll('[data-mode]')),
  datasetSwitcher: document.getElementById('datasetSwitcher'),
  batchSvg: document.getElementById('batchSvg'),
  meanLoss: document.getElementById('meanLoss'),
  gradW: document.getElementById('gradW'),
  gradB: document.getElementById('gradB'),
  learningRateValue: document.getElementById('learningRateValue'),
  paramW: document.getElementById('paramW'),
  paramWValue: document.getElementById('paramWValue'),
  paramB: document.getElementById('paramB'),
  paramBValue: document.getElementById('paramBValue'),
  learningRate: document.getElementById('learningRate'),
  learningRateInline: document.getElementById('learningRateInline'),
  batchNarrative: document.getElementById('batchNarrative'),
  exampleTableBody: document.getElementById('exampleTableBody'),
  resetBatch: document.getElementById('resetBatch'),
  applyBatchStep: document.getElementById('applyBatchStep'),
  exportBatch: document.getElementById('exportBatch'),
};

function stripMathDelimiters(text) {
  return text.replace(/^\s*\\\[/, '').replace(/\\\]\s*$/, '').trim();
}

function renderTex(element, tex, displayMode = true) {
  if (!element) return;
  if (window.katex && typeof window.katex.render === 'function') {
    window.katex.render(stripMathDelimiters(tex), element, {
      displayMode,
      throwOnError: false,
    });
  } else {
    element.textContent = stripMathDelimiters(tex);
  }
}

function renderStaticMath() {
  document.querySelectorAll('[data-katex-display]').forEach((element) => {
    renderTex(element, element.textContent, true);
  });
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function clampProbability(value) {
  return Math.min(1 - 1e-7, Math.max(1e-7, value));
}

function formatNumber(value, digits = 3) {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  const rounded = Number(value.toFixed(digits));
  const normalized = Object.is(rounded, -0) ? 0 : rounded;
  return normalized.toFixed(digits);
}

function formatSigned(value, digits = 3) {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  const text = formatNumber(Math.abs(value), digits);
  return value >= 0 ? `+${text}` : `-${text}`;
}

function computeSingleExample({ x, w, b }) {
  const values = {
    x,
    w,
    b,
  };
  values.mul = w * x;
  values.z = values.mul + b;
  values.neg = -values.z;
  values.exp = Math.exp(values.neg);
  values.denom = 1 + values.exp;
  values.p = 1 / values.denom;
  values.logp = Math.log(values.p);
  values.L = -values.logp;

  const grads = {
    L: 1,
  };
  grads.logp = grads.L * -1;
  grads.p = grads.logp * (1 / values.p);
  grads.denom = grads.p * (-1 / (values.denom * values.denom));
  grads.exp = grads.denom;
  grads.neg = grads.exp * values.exp;
  grads.z = grads.neg * -1;
  grads.mul = grads.z;
  grads.b = grads.z;
  grads.w = grads.mul * values.x;
  grads.x = grads.mul * values.w;

  return { values, grads };
}

function graphVisibleState(stage) {
  return GRAPH_NODES.reduce((acc, node) => {
    acc[node.id] = {
      value: stage >= node.forwardStep ? graphSnapshot.values[node.id] : null,
      grad: stage >= node.backwardStep ? graphSnapshot.grads[node.id] : null,
      active: GRAPH_STAGES[stage].activeNodes.includes(node.id),
    };
    return acc;
  }, {});
}

function graphNodeColors(node, visible, direction) {
  const activeForward = direction === 'Forward' && visible.active;
  const activeBackward = direction === 'Backward' && visible.active;
  const fills = {
    input: '#eef7ff',
    parameter: '#eef3ff',
    op: '#ffffff',
    loss: '#fff4ef',
  };
  const fill = activeForward ? '#e8f0ff' : activeBackward ? '#fff0eb' : fills[node.kind] || '#ffffff';
  const stroke = activeForward ? '#345ff6' : activeBackward || visible.grad !== null ? '#f0624d' : '#b9c4cf';
  const glow = activeForward
    ? 'drop-shadow(0 18px 22px rgba(52,95,246,0.18))'
    : activeBackward
      ? 'drop-shadow(0 18px 22px rgba(240,98,77,0.18))'
      : 'none';
  return { fill, stroke, glow };
}

function edgePath(edge) {
  const from = GRAPH_NODES.find((node) => node.id === edge.from);
  const to = GRAPH_NODES.find((node) => node.id === edge.to);
  if (!from || !to) return '';
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const startX = from.x + 76;
  const startY = from.y;
  const endX = to.x - 76;
  const endY = to.y;
  if (edge.id === 'p-logp') {
    return `M ${from.x} ${from.y - 44} C ${from.x} ${from.y - 120}, ${to.x} ${to.y + 120}, ${to.x} ${to.y + 44}`;
  }
  if (edge.id === 'logp-L') {
    return `M ${from.x} ${from.y + 44} C ${from.x + 90} ${from.y + 120}, ${to.x + 90} ${to.y - 120}, ${to.x} ${to.y - 44}`;
  }
  return `M ${startX} ${startY} C ${startX + dx * 0.38} ${startY + dy * 0.08}, ${endX - dx * 0.2} ${endY - dy * 0.08}, ${endX} ${endY}`;
}

function nodeMarkup(node, visible, direction) {
  const { fill, stroke, glow } = graphNodeColors(node, visible, direction);
  const badgeY = 55;
  const valueBadge = visible.value === null
    ? ''
    : `<g transform="translate(14 ${badgeY})"><rect width="52" height="18" rx="9" fill="#345ff6" opacity="0.9"></rect><text class="node-badge" x="10" y="12.5">v ${formatNumber(visible.value, 2)}</text></g>`;
  const gradBadge = visible.grad === null
    ? ''
    : `<g transform="translate(72 ${badgeY})"><rect width="54" height="18" rx="9" fill="#f0624d" opacity="0.9"></rect><text class="node-badge" x="8" y="12.5">∂ ${formatNumber(visible.grad, 2)}</text></g>`;

  return `
    <g transform="translate(${node.x - 70} ${node.y - 42})" style="filter:${glow}">
      <rect width="140" height="84" rx="24" fill="${fill}" stroke="${stroke}" stroke-width="${visible.active ? 3 : 2}"></rect>
      <text class="node-label" x="16" y="31">${node.label}</text>
      <text class="node-sub" x="16" y="49">${node.sub}</text>
      ${valueBadge}
      ${gradBadge}
    </g>
  `;
}

function renderGraphFigure() {
  const stage = state.graphStage;
  const stageMeta = GRAPH_STAGES[stage];
  const visibleState = graphVisibleState(stage);
  const direction = stageMeta.direction;

  const edgesMarkup = GRAPH_EDGES.map((edge) => {
    const active = stageMeta.activeEdges.includes(edge.id);
    const color = direction === 'Backward' && active ? '#f0624d' : active ? '#345ff6' : '#bfc7d1';
    return `<path d="${edgePath(edge)}" fill="none" stroke="${color}" stroke-width="${active ? 4 : 2.4}" stroke-linecap="round" opacity="${active ? 0.98 : 0.64}"></path>`;
  }).join('');

  const nodesMarkup = GRAPH_NODES.map((node) => nodeMarkup(node, visibleState[node.id], direction)).join('');

  els.graphSvg.innerHTML = `
    <defs>
      <linearGradient id="graphWash" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#f8fbff"></stop>
        <stop offset="100%" stop-color="#fff7f3"></stop>
      </linearGradient>
    </defs>
    <rect x="18" y="18" width="1384" height="584" rx="30" fill="url(#graphWash)" stroke="#d7dde6"></rect>
    <text x="46" y="62" class="node-sub">Single positive example: x = ${formatNumber(graphSnapshot.values.x)}, w = ${formatNumber(graphSnapshot.values.w)}, b = ${formatNumber(graphSnapshot.values.b)}</text>
    ${edgesMarkup}
    ${nodesMarkup}
  `;

  els.stageIndex.textContent = `${stage} / ${FINAL_STAGE}`;
  els.stageDirection.textContent = stageMeta.direction;
  els.stageFocus.textContent = stageMeta.focus;
  els.stageHeadline.textContent = stageMeta.headline(graphSnapshot);
  els.stageExplanation.innerHTML = stageMeta.explanation(graphSnapshot);
  renderTex(els.chainRuleMath, stageMeta.formula, true);
  els.stageScrubber.value = String(stage);

  renderNodeTable(visibleState);
}

function renderNodeTable(visibleState) {
  const header = `
    <div class="ledger-row" style="font-weight:700;background:rgba(12,127,120,0.08)">
      <div><strong>Node</strong><span>role</span></div>
      <div><strong>Value</strong><span>forward cache</span></div>
      <div><strong>Gradient</strong><span>backward message</span></div>
    </div>
  `;
  const rows = GRAPH_NODES.map((node) => {
    const visible = visibleState[node.id];
    return `
      <div class="ledger-row" style="border:${visible.active ? '1px solid rgba(52,95,246,0.16)' : 'none'}; background:${visible.active ? 'rgba(52,95,246,0.05)' : 'rgba(250,247,241,0.72)'}">
        <div><strong>${node.label}</strong><span>${node.sub}</span></div>
        <div><strong>${formatNumber(visible.value, 3)}</strong><span>${visible.value === null ? 'not created yet' : 'stored from forward'}</span></div>
        <div><strong>${formatNumber(visible.grad, 3)}</strong><span>${visible.grad === null ? 'not reached yet' : 'available for parent update'}</span></div>
      </div>
    `;
  }).join('');
  els.nodeTable.innerHTML = header + rows;
}

function moveGraphStage(nextStage) {
  state.graphStage = Math.max(0, Math.min(FINAL_STAGE, nextStage));
  renderGraphFigure();
}

function stepForwardGraph() {
  if (state.graphStage < FORWARD_STAGE_END) {
    moveGraphStage(state.graphStage + 1);
  }
}

function stepBackwardGraph() {
  if (state.graphStage < FORWARD_STAGE_END) {
    moveGraphStage(FORWARD_STAGE_END + 1);
    return;
  }
  if (state.graphStage < FINAL_STAGE) {
    moveGraphStage(state.graphStage + 1);
  }
}

function toggleAutoplay() {
  if (autoPlayHandle) {
    window.clearInterval(autoPlayHandle);
    autoPlayHandle = null;
    els.autoPlay.textContent = 'Auto play';
    return;
  }
  if (state.graphStage >= FINAL_STAGE) {
    moveGraphStage(0);
  }
  els.autoPlay.textContent = 'Pause';
  autoPlayHandle = window.setInterval(() => {
    if (state.graphStage >= FINAL_STAGE) {
      window.clearInterval(autoPlayHandle);
      autoPlayHandle = null;
      els.autoPlay.textContent = 'Auto play';
      return;
    }
    moveGraphStage(state.graphStage + 1);
  }, 1100);
}

function exportGraphStage() {
  const stage = state.graphStage;
  downloadJson(`autograd-stage-${stage}.json`, createGraphExport(stage));
}

function exportGraphTimeline() {
  const timeline = Array.from({ length: FINAL_STAGE + 1 }, (_, stage) => createGraphExport(stage));
  downloadJson('autograd-graph-timeline.json', timeline);
}

function createGraphExport(stage) {
  const stageMeta = GRAPH_STAGES[stage];
  const visible = graphVisibleState(stage);
  return {
    stage,
    direction: stageMeta.direction,
    focus: stageMeta.focus,
    formula: stageMeta.formula,
    inputs: SINGLE_INPUT,
    nodes: GRAPH_NODES.map((node) => ({
      id: node.id,
      label: node.label,
      sub: node.sub,
      value: visible[node.id].value,
      grad: visible[node.id].grad,
      active: visible[node.id].active,
    })),
  };
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function renderRules() {
  els.ruleChips.innerHTML = RULES.map((rule) => `
    <button type="button" class="rule-chip ${rule.id === state.selectedRule ? 'is-active' : ''}" data-rule-id="${rule.id}">${rule.label}</button>
  `).join('');

  els.ruleGrid.innerHTML = RULES.map((rule) => `
    <article class="rule-card ${rule.id === state.selectedRule ? 'is-active' : ''}" data-rule-id="${rule.id}">
      <h4>${rule.label}</h4>
      <p>${rule.summary}</p>
    </article>
  `).join('');

  const selected = RULES.find((rule) => rule.id === state.selectedRule) || RULES[0];
  els.ruleTitle.textContent = selected.label;
  els.ruleExplanation.textContent = selected.explanation;
  renderTex(els.ruleFormula, selected.tex, true);

  document.querySelectorAll('[data-rule-id]').forEach((element) => {
    element.addEventListener('click', () => {
      state.selectedRule = element.getAttribute('data-rule-id') || RULES[0].id;
      renderRules();
    });
  });
}

function modulePath(fromNode, toNode) {
  const startX = fromNode.x + fromNode.w / 2;
  const startY = fromNode.y;
  const endX = toNode.x - toNode.w / 2;
  const endY = toNode.y;
  const dx = endX - startX;
  return `M ${startX} ${startY} C ${startX + dx * 0.4} ${startY}, ${endX - dx * 0.3} ${endY}, ${endX} ${endY}`;
}

function renderModuleView() {
  const view = MODULE_VIEWS[state.moduleMode];
  els.modeChips.forEach((chip) => {
    chip.classList.toggle('is-active', chip.dataset.mode === state.moduleMode);
  });

  const edges = view.edges.map(([fromId, toId]) => {
    const fromNode = view.nodes.find((node) => node.id === fromId);
    const toNode = view.nodes.find((node) => node.id === toId);
    return `<path d="${modulePath(fromNode, toNode)}" fill="none" stroke="#c5ced8" stroke-width="3" stroke-linecap="round"></path>`;
  }).join('');

  const nodes = view.nodes.map((node) => `
    <g transform="translate(${node.x - node.w / 2} ${node.y - node.h / 2})">
      <rect width="${node.w}" height="${node.h}" rx="22" fill="${node.fill}" stroke="#c8d2de" stroke-width="2"></rect>
      <text class="node-label" x="16" y="${node.h / 2 + 6}" style="font-size:${node.w > 150 ? 18 : 20}px">${node.label}</text>
    </g>
  `).join('');

  els.moduleSvg.innerHTML = `
    <rect x="18" y="18" width="1084" height="324" rx="28" fill="#fbfbfd" stroke="#dae1e9"></rect>
    ${edges}
    ${nodes}
  `;

  els.moduleHeadline.textContent = view.headline;
  els.moduleBody.textContent = view.body;
  els.moduleList.innerHTML = view.bullets.map((bullet) => `
    <div class="module-list__item">
      <strong>${bullet.title}</strong>
      <span>${bullet.text}</span>
    </div>
  `).join('');
}

function datasetButtonsMarkup() {
  return Object.entries(BATCH_DATASETS).map(([key, dataset]) => `
    <button type="button" class="dataset-chip ${state.batch.datasetKey === key ? 'is-active' : ''}" data-dataset="${key}">${dataset.label}</button>
  `).join('');
}

function syncBatchControls() {
  els.paramW.value = String(state.batch.w);
  els.paramB.value = String(state.batch.b);
  els.learningRate.value = String(state.batch.lr);
  els.paramWValue.textContent = formatNumber(state.batch.w, 2);
  els.paramBValue.textContent = formatNumber(state.batch.b, 2);
  els.learningRateInline.textContent = formatNumber(state.batch.lr, 2);
  els.learningRateValue.textContent = formatNumber(state.batch.lr, 2);
}

function computeBatchSnapshot() {
  const dataset = BATCH_DATASETS[state.batch.datasetKey];
  const rows = dataset.points.map((point) => {
    const z = state.batch.w * point.x + state.batch.b;
    const p = clampProbability(sigmoid(z));
    const loss = -(point.y * Math.log(p) + (1 - point.y) * Math.log(1 - p));
    const gradZ = p - point.y;
    return {
      ...point,
      z,
      p,
      loss,
      gradW: gradZ * point.x,
      gradB: gradZ,
    };
  });

  const meanLoss = rows.reduce((sum, row) => sum + row.loss, 0) / rows.length;
  const meanGradW = rows.reduce((sum, row) => sum + row.gradW, 0) / rows.length;
  const meanGradB = rows.reduce((sum, row) => sum + row.gradB, 0) / rows.length;

  return {
    dataset,
    rows,
    meanLoss,
    meanGradW,
    meanGradB,
    nextW: state.batch.w - state.batch.lr * meanGradW,
    nextB: state.batch.b - state.batch.lr * meanGradB,
  };
}

function renderBatchSvg(snapshot) {
  const xScale = (value) => 90 + ((value + 3) / 6) * 610;
  const yScale = (value) => 360 - value * 260;
  const curvePoints = Array.from({ length: 120 }, (_, index) => {
    const x = -3 + (index / 119) * 6;
    const y = sigmoid(state.batch.w * x + state.batch.b);
    return `${index === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(y)}`;
  }).join(' ');

  const guides = [0, 0.5, 1].map((y) => `
    <g>
      <line x1="90" y1="${yScale(y)}" x2="700" y2="${yScale(y)}" stroke="#e3e8ee" stroke-width="1.5" stroke-dasharray="5 8"></line>
      <text x="52" y="${yScale(y) + 5}" class="node-sub">${y.toFixed(1)}</text>
    </g>
  `).join('');

  const xTicks = [-2, -1, 0, 1, 2].map((x) => `
    <g>
      <line x1="${xScale(x)}" y1="360" x2="${xScale(x)}" y2="372" stroke="#aeb8c2" stroke-width="1.4"></line>
      <text x="${xScale(x) - 8}" y="394" class="node-sub">${x}</text>
    </g>
  `).join('');

  const examples = snapshot.rows.map((row, index) => {
    const baseY = yScale(row.y);
    const predY = yScale(row.p);
    const x = xScale(row.x);
    const color = row.y === 1 ? '#0c7f78' : '#f0624d';
    return `
      <g>
        <line x1="${x}" y1="${baseY}" x2="${x}" y2="${predY}" stroke="${color}" stroke-width="2.4" opacity="0.5"></line>
        <circle cx="${x}" cy="${predY}" r="10" fill="#ffffff" stroke="${color}" stroke-width="3"></circle>
        <circle cx="${x}" cy="${baseY}" r="6" fill="${color}" opacity="0.85"></circle>
        <text x="${x + 12}" y="${Math.min(baseY, predY) - 12 - index * 2}" class="node-sub">${row.name}: p=${formatNumber(row.p, 2)}</text>
      </g>
    `;
  }).join('');

  const paramsTag = `w = ${formatNumber(state.batch.w, 2)}, b = ${formatNumber(state.batch.b, 2)}`;

  els.batchSvg.innerHTML = `
    <rect x="18" y="18" width="724" height="394" rx="28" fill="#fbfbfd" stroke="#dae1e9"></rect>
    <text x="42" y="54" class="node-sub">${snapshot.dataset.label} • ${paramsTag}</text>
    <text x="42" y="74" class="node-sub">Circles: model probability. Dots: target labels. Lines: per-example error signal.</text>
    ${guides}
    <line x1="90" y1="360" x2="700" y2="360" stroke="#aeb8c2" stroke-width="2"></line>
    ${xTicks}
    <path d="${curvePoints}" fill="none" stroke="#345ff6" stroke-width="5" stroke-linecap="round"></path>
    ${examples}
    <text x="676" y="394" class="node-sub">x</text>
  `;
}

function batchNarrative(snapshot) {
  const directionW = snapshot.meanGradW > 0 ? 'downward' : 'upward';
  const directionB = snapshot.meanGradB > 0 ? 'downward' : 'upward';
  return `${snapshot.dataset.description} The current batch gradient wants to move <code>w</code> ${directionW} and <code>b</code> ${directionB}. After one step, the parameters would become <code>w = ${formatNumber(snapshot.nextW, 2)}</code> and <code>b = ${formatNumber(snapshot.nextB, 2)}</code>.`;
}

function renderBatchFigure() {
  els.datasetSwitcher.innerHTML = datasetButtonsMarkup();
  syncBatchControls();
  const snapshot = computeBatchSnapshot();

  els.meanLoss.textContent = formatNumber(snapshot.meanLoss, 3);
  els.gradW.textContent = formatSigned(snapshot.meanGradW, 3);
  els.gradB.textContent = formatSigned(snapshot.meanGradB, 3);
  els.batchNarrative.innerHTML = batchNarrative(snapshot);

  els.exampleTableBody.innerHTML = snapshot.rows.map((row) => `
    <tr>
      <td>${row.name}</td>
      <td>${formatNumber(row.x, 2)}</td>
      <td>${row.y}</td>
      <td>${formatNumber(row.p, 3)}</td>
      <td>${formatNumber(row.loss, 3)}</td>
      <td>${formatSigned(row.gradW, 3)}</td>
    </tr>
  `).join('');

  renderBatchSvg(snapshot);

  document.querySelectorAll('[data-dataset]').forEach((button) => {
    button.addEventListener('click', () => {
      const datasetKey = button.getAttribute('data-dataset');
      if (!datasetKey || !BATCH_DATASETS[datasetKey]) return;
      state.batch.datasetKey = datasetKey;
      state.batch.w = BATCH_DATASETS[datasetKey].defaults.w;
      state.batch.b = BATCH_DATASETS[datasetKey].defaults.b;
      renderBatchFigure();
    });
  });
}

function applyBatchUpdate() {
  const snapshot = computeBatchSnapshot();
  state.batch.w = snapshot.nextW;
  state.batch.b = snapshot.nextB;
  renderBatchFigure();
}

function resetBatchParams() {
  const defaults = BATCH_DATASETS[state.batch.datasetKey].defaults;
  state.batch.w = defaults.w;
  state.batch.b = defaults.b;
  renderBatchFigure();
}

function exportBatchSnapshot() {
  const snapshot = computeBatchSnapshot();
  downloadJson('autograd-batch-snapshot.json', {
    dataset: state.batch.datasetKey,
    params: {
      w: state.batch.w,
      b: state.batch.b,
      lr: state.batch.lr,
    },
    summary: {
      meanLoss: snapshot.meanLoss,
      meanGradW: snapshot.meanGradW,
      meanGradB: snapshot.meanGradB,
      nextW: snapshot.nextW,
      nextB: snapshot.nextB,
    },
    rows: snapshot.rows,
  });
}

function updateProgress() {
  const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollHeight <= 0 ? 0 : (window.scrollY / scrollHeight) * 100;
  els.progressFill.style.width = `${Math.min(100, Math.max(0, progress))}%`;
}

function updateToc() {
  let bestId = els.sections[0]?.id;
  let bestDistance = Number.POSITIVE_INFINITY;
  els.sections.forEach((section) => {
    const distance = Math.abs(section.getBoundingClientRect().top - 120);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestId = section.id;
    }
  });
  els.tocLinks.forEach((link) => {
    const active = link.getAttribute('href') === `#${bestId}`;
    link.classList.toggle('is-active', active);
  });
}

function bindEvents() {
  els.stageScrubber.addEventListener('input', (event) => {
    moveGraphStage(Number(event.target.value));
  });
  els.resetGraph.addEventListener('click', () => moveGraphStage(0));
  els.stepForward.addEventListener('click', stepForwardGraph);
  els.stepBackward.addEventListener('click', stepBackwardGraph);
  els.autoPlay.addEventListener('click', toggleAutoplay);
  els.exportStage.addEventListener('click', exportGraphStage);
  els.exportTimeline.addEventListener('click', exportGraphTimeline);

  els.modeChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      state.moduleMode = chip.dataset.mode || 'micro';
      renderModuleView();
    });
  });

  els.paramW.addEventListener('input', (event) => {
    state.batch.w = Number(event.target.value);
    renderBatchFigure();
  });
  els.paramB.addEventListener('input', (event) => {
    state.batch.b = Number(event.target.value);
    renderBatchFigure();
  });
  els.learningRate.addEventListener('input', (event) => {
    state.batch.lr = Number(event.target.value);
    renderBatchFigure();
  });
  els.resetBatch.addEventListener('click', resetBatchParams);
  els.applyBatchStep.addEventListener('click', applyBatchUpdate);
  els.exportBatch.addEventListener('click', exportBatchSnapshot);

  window.addEventListener('scroll', () => {
    updateProgress();
    updateToc();
  }, { passive: true });
  window.addEventListener('resize', updateToc);
}

function exposeApi() {
  window.AutogradExplainer = {
    exportGraphStage: () => createGraphExport(state.graphStage),
    exportGraphTimeline: () => Array.from({ length: FINAL_STAGE + 1 }, (_, stage) => createGraphExport(stage)),
    exportBatchSnapshot: () => computeBatchSnapshot(),
    setGraphStage: moveGraphStage,
  };
}

function init() {
  renderStaticMath();
  renderRules();
  renderGraphFigure();
  renderModuleView();
  renderBatchFigure();
  bindEvents();
  updateProgress();
  updateToc();
  exposeApi();
}

init();
