// ============================================================
// Attention, Calculated — interactive
// Dot products, softmax, and value blending computed live.
// ============================================================

// ---------- Scenarios ----------
// Each scenario has a focus word that wants context, plus the surrounding
// words. The Q/K/V vectors are hand-picked 2-D stand-ins that give the
// "sensible" disambiguation (e.g. river bank → water).
const SCENARIOS = {
  riverBank: {
    sentence: '"The river bank"',
    focus: 'bank',
    caption: 'Focus word "bank" — is it water or money?',
    words: ['The', 'river', 'bank'],
    Q: { bank: { x: 0.85, y: 0.53 } },
    K: {
      The: { x: -0.60, y: 0.80 },
      river: { x: 0.95, y: 0.31 },
      bank: { x: 0.40, y: 0.92 }
    },
    V: {
      The: { x: -0.30, y: -0.20 },
      river: { x: 0.90, y: 0.10 },
      bank: { x: 0.20, y: 0.98 }
    }
  },
  moneyBank: {
    sentence: '"I deposit money at the bank"',
    focus: 'bank',
    caption: 'Same word "bank" — but "money" should now dominate.',
    words: ['deposit', 'money', 'bank'],
    Q: { bank: { x: 0.26, y: 0.97 } },
    K: {
      deposit: { x: 0.20, y: 0.98 },
      money: { x: 0.30, y: 0.95 },
      bank: { x: 0.95, y: 0.31 }
    },
    V: {
      deposit: { x: 0.10, y: 0.99 },
      money: { x: 0.35, y: 0.94 },
      bank: { x: 0.90, y: 0.40 }
    }
  },
  appleFruit: {
    sentence: '"I ate an apple pie"',
    focus: 'apple',
    caption: 'Focus word "apple" — fruit or tech company?',
    words: ['ate', 'apple', 'pie'],
    Q: { apple: { x: 0.92, y: 0.39 } },
    K: {
      ate: { x: 0.85, y: 0.53 },
      apple: { x: 0.30, y: 0.95 },
      pie: { x: 0.96, y: 0.28 }
    },
    V: {
      ate: { x: 0.75, y: 0.66 },
      apple: { x: 0.25, y: 0.97 },
      pie: { x: 0.92, y: 0.39 }
    }
  },
  appleCorp: {
    sentence: '"Apple released a phone"',
    focus: 'Apple',
    caption: 'Same focus word "Apple" — but "phone" should win now.',
    words: ['Apple', 'released', 'phone'],
    Q: { Apple: { x: -0.60, y: 0.80 } },
    K: {
      Apple: { x: 0.95, y: 0.31 },
      released: { x: 0.40, y: 0.92 },
      phone: { x: -0.65, y: 0.76 }
    },
    V: {
      Apple: { x: 0.99, y: 0.14 },
      released: { x: 0.40, y: 0.92 },
      phone: { x: -0.70, y: 0.71 }
    }
  },
  bassFish: {
    sentence: '"I caught a bass at the lake"',
    focus: 'bass',
    caption: 'Focus word "bass" — fish or musical instrument?',
    words: ['caught', 'bass', 'lake'],
    Q: { bass: { x: 0.94, y: 0.34 } },
    K: {
      caught: { x: 0.50, y: 0.87 },
      bass: { x: 0.30, y: 0.95 },
      lake: { x: 0.97, y: 0.24 }
    },
    V: {
      caught: { x: 0.30, y: 0.95 },
      bass: { x: 0.15, y: 0.99 },
      lake: { x: 0.99, y: 0.14 }
    }
  },
  selfBank: {
    sentence: '"The bank itself stood firm"',
    focus: 'bank',
    caption: 'When self-attention favors the focus token itself · "bank" carries its own meaning forward.',
    words: ['The', 'bank', 'itself'],
    Q: { bank: { x: 0.95, y: 0.31 } },
    K: {
      The: { x: -0.80, y: 0.60 },
      bank: { x: 0.96, y: 0.28 },
      itself: { x: 0.85, y: 0.53 }
    },
    V: {
      The: { x: -0.50, y: -0.50 },
      bank: { x: 0.99, y: 0.14 },
      itself: { x: 0.10, y: 0.10 }
    }
  }
};

const OTHER_COLOR = '#2c6fb7';
const Q_COLOR = '#d9622b';
const V_COLOR = '#1e7770';
const WARM_COLOR = '#d9622b';

// ---------- State ----------
let state = {
  scenarioKey: 'riverBank',
  // Current (user-draggable) vectors copied from scenario on load
  Q: {},
  K: {},
  V: {}
};

function loadScenario(key) {
  state.scenarioKey = key;
  state.matrixRow = null;
  state.permuteOrder = null;
  const s = SCENARIOS[key];
  // Deep copy so dragging doesn't mutate the scenario definition
  state.Q = {};
  state.K = {};
  state.V = {};
  Object.keys(s.Q).forEach((w) => { state.Q[w] = { ...s.Q[w] }; });
  Object.keys(s.K).forEach((w) => { state.K[w] = { ...s.K[w] }; });
  Object.keys(s.V).forEach((w) => { state.V[w] = { ...s.V[w] }; });

  document.querySelectorAll('#scenario-buttons [data-scenario]').forEach((b) => {
    b.classList.toggle('is-active', b.dataset.scenario === key);
  });
  const caption = document.getElementById('scenario-caption');
  if (caption) {
    caption.innerHTML =
      `Working with the sentence <strong>${s.sentence}</strong>. ${s.caption}`;
  }

  drawQKCanvas();
  drawValueCanvas();
  updateTables();
  updateLiveMath();
  renderMatrixView();
  renderMaskedHeatmaps();
  renderMultiHead();
  renderPermuteDemo();
}

// ---------- Math helpers ----------
function dot(a, b) {
  return a.x * b.x + a.y * b.y;
}

function normalize(v) {
  const len = Math.hypot(v.x, v.y) || 1;
  return { x: v.x / len, y: v.y / len };
}

function softmax(scores) {
  const maxS = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - maxS));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function computeAttention() {
  const s = SCENARIOS[state.scenarioKey];
  const q = state.Q[s.focus];
  const words = s.words;
  const scores = words.map((w) => dot(q, state.K[w]));
  const weights = softmax(scores);
  // Blend values
  let Vf = { x: 0, y: 0 };
  words.forEach((w, i) => {
    Vf.x += weights[i] * state.V[w].x;
    Vf.y += weights[i] * state.V[w].y;
  });
  return { words, q, scores, weights, Vf };
}

// ---------- Canvas helpers ----------
function makeCanvasTransform(canvas, logicalWidth, logicalHeight, radius) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = logicalWidth * dpr;
  canvas.height = logicalHeight * dpr;
  canvas.style.width = logicalWidth + 'px';
  canvas.style.height = logicalHeight + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cx = logicalWidth / 2;
  const cy = logicalHeight / 2;
  const scale = radius;
  return {
    ctx,
    w: logicalWidth,
    h: logicalHeight,
    cx,
    cy,
    scale,
    toScreen: (v) => ({ x: cx + v.x * scale, y: cy - v.y * scale }),
    fromScreen: (x, y) => ({ x: (x - cx) / scale, y: (cy - y) / scale })
  };
}

function drawFrame(ctx, cx, cy, scale) {
  // Grid
  ctx.strokeStyle = '#f0ebe1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, cy - scale * 1.35);
  ctx.lineTo(cx, cy + scale * 1.35);
  ctx.moveTo(cx - scale * 1.35, cy);
  ctx.lineTo(cx + scale * 1.35, cy);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, scale, 0, Math.PI * 2);
  ctx.strokeStyle = '#d6cdb8';
  ctx.setLineDash([5, 5]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawArrow(ctx, x0, y0, x1, y1, color, width = 3) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();

  const angle = Math.atan2(y1 - y0, x1 - x0);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - 12 * Math.cos(angle - Math.PI / 6), y1 - 12 * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x1 - 12 * Math.cos(angle + Math.PI / 6), y1 - 12 * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function drawGrabDot(ctx, x, y, color) {
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawLabel(ctx, x, y, text, color = '#1a1815') {
  ctx.font = "bold 13px Manrope, sans-serif";
  ctx.fillStyle = color;
  ctx.textAlign = 'left';
  ctx.fillText(text, x + 12, y - 6);
}

// ---------- Q/K canvas ----------
let qkDragging = null;

function drawQKCanvas() {
  const canvas = document.getElementById('qkCanvas');
  if (!canvas) return;
  const T = makeCanvasTransform(canvas, 880, 440, 150);
  const { ctx, w, h, cx, cy, scale, toScreen } = T;

  ctx.fillStyle = '#fdfcf9';
  ctx.fillRect(0, 0, w, h);
  drawFrame(ctx, cx, cy, scale);

  const s = SCENARIOS[state.scenarioKey];

  // Draw keys first
  s.words.forEach((word) => {
    const k = state.K[word];
    const p = toScreen(k);
    drawArrow(ctx, cx, cy, p.x, p.y, OTHER_COLOR, 3);
    drawGrabDot(ctx, p.x, p.y, OTHER_COLOR);
    drawLabel(ctx, p.x, p.y, `K(${word})`, OTHER_COLOR);
  });

  // Draw query on top
  const q = state.Q[s.focus];
  const qp = toScreen(q);
  drawArrow(ctx, cx, cy, qp.x, qp.y, Q_COLOR, 4);
  drawGrabDot(ctx, qp.x, qp.y, Q_COLOR);
  drawLabel(ctx, qp.x, qp.y, `Q(${s.focus})`, Q_COLOR);

  // Title/hint
  ctx.font = "600 12px Manrope, sans-serif";
  ctx.fillStyle = '#9a917f';
  ctx.textAlign = 'left';
  ctx.fillText('Drag any dot', 16, 22);
  ctx.fillText('Q: focus word\'s search', 16, 38);
  ctx.fillText('K: each word\'s label', 16, 54);
}

function installQKDrag() {
  const canvas = document.getElementById('qkCanvas');
  if (!canvas) return;
  const logicalWidth = 880;
  const logicalHeight = 440;
  const radius = 150;
  const cx = logicalWidth / 2;
  const cy = logicalHeight / 2;

  function hit(mx, my) {
    const s = SCENARIOS[state.scenarioKey];
    // Query first
    const q = state.Q[s.focus];
    const qp = { x: cx + q.x * radius, y: cy - q.y * radius };
    if (Math.hypot(qp.x - mx, qp.y - my) < 15) return { kind: 'q', word: s.focus };
    for (const w of s.words) {
      const k = state.K[w];
      const kp = { x: cx + k.x * radius, y: cy - k.y * radius };
      if (Math.hypot(kp.x - mx, kp.y - my) < 15) return { kind: 'k', word: w };
    }
    return null;
  }

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    qkDragging = hit(mx, my);
  });

  canvas.addEventListener('touchstart', (e) => {
    if (!e.touches[0]) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.touches[0].clientX - rect.left;
    const my = e.touches[0].clientY - rect.top;
    qkDragging = hit(mx, my);
    if (qkDragging) e.preventDefault();
  }, { passive: false });

  function handleMove(mx, my) {
    if (!qkDragging) return;
    const vx = (mx - cx) / radius;
    const vy = (cy - my) / radius;
    const n = normalize({ x: vx, y: vy });
    if (qkDragging.kind === 'q') state.Q[qkDragging.word] = n;
    else state.K[qkDragging.word] = n;
    drawQKCanvas();
    updateTables();
    drawValueCanvas();
    updateLiveMath();
  }

  window.addEventListener('mousemove', (e) => {
    if (!qkDragging) return;
    const rect = canvas.getBoundingClientRect();
    const mx = Math.max(0, Math.min(logicalWidth, e.clientX - rect.left));
    const my = Math.max(0, Math.min(logicalHeight, e.clientY - rect.top));
    handleMove(mx, my);
  });

  window.addEventListener('touchmove', (e) => {
    if (!qkDragging || !e.touches[0]) return;
    const rect = canvas.getBoundingClientRect();
    const mx = Math.max(0, Math.min(logicalWidth, e.touches[0].clientX - rect.left));
    const my = Math.max(0, Math.min(logicalHeight, e.touches[0].clientY - rect.top));
    handleMove(mx, my);
    e.preventDefault();
  }, { passive: false });

  window.addEventListener('mouseup', () => { qkDragging = null; });
  window.addEventListener('touchend', () => { qkDragging = null; });
}

// ---------- Value canvas ----------
let valueDragging = null;

function drawValueCanvas() {
  const canvas = document.getElementById('valueCanvas');
  if (!canvas) return;
  const T = makeCanvasTransform(canvas, 880, 440, 150);
  const { ctx, w, h, cx, cy, scale, toScreen } = T;

  ctx.fillStyle = '#fdfcf9';
  ctx.fillRect(0, 0, w, h);
  drawFrame(ctx, cx, cy, scale);

  const s = SCENARIOS[state.scenarioKey];
  const { words, weights, Vf } = computeAttention();

  // Value vectors with alpha = softmax weight
  words.forEach((word, i) => {
    const v = state.V[word];
    const p = toScreen(v);
    const alpha = 0.25 + 0.75 * weights[i];
    const rgba = `rgba(30, 119, 112, ${alpha})`;
    drawArrow(ctx, cx, cy, p.x, p.y, rgba, 2 + 3 * weights[i]);
    drawGrabDot(ctx, p.x, p.y, V_COLOR);
    drawLabel(ctx, p.x, p.y, `V(${word}) · ${(weights[i] * 100).toFixed(0)}%`, V_COLOR);
  });

  // Final blended V (no normalization — it's a real weighted sum)
  const fp = toScreen(Vf);
  ctx.setLineDash([6, 4]);
  drawArrow(ctx, cx, cy, fp.x, fp.y, WARM_COLOR, 4);
  ctx.setLineDash([]);
  drawGrabDot(ctx, fp.x, fp.y, WARM_COLOR);
  ctx.font = "bold 13px Manrope, sans-serif";
  ctx.fillStyle = WARM_COLOR;
  ctx.textAlign = 'left';
  ctx.fillText('V_final', fp.x + 12, fp.y - 6);

  ctx.font = "600 12px Manrope, sans-serif";
  ctx.fillStyle = '#9a917f';
  ctx.textAlign = 'left';
  ctx.fillText('Drag any V vector', 16, 22);
  ctx.fillText('Arrow alpha = attention weight', 16, 38);
}

function installValueDrag() {
  const canvas = document.getElementById('valueCanvas');
  if (!canvas) return;
  const logicalWidth = 880;
  const logicalHeight = 440;
  const radius = 150;
  const cx = logicalWidth / 2;
  const cy = logicalHeight / 2;

  function hit(mx, my) {
    const s = SCENARIOS[state.scenarioKey];
    for (const w of s.words) {
      const v = state.V[w];
      const vp = { x: cx + v.x * radius, y: cy - v.y * radius };
      if (Math.hypot(vp.x - mx, vp.y - my) < 15) return w;
    }
    return null;
  }

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    valueDragging = hit(mx, my);
  });
  canvas.addEventListener('touchstart', (e) => {
    if (!e.touches[0]) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.touches[0].clientX - rect.left;
    const my = e.touches[0].clientY - rect.top;
    valueDragging = hit(mx, my);
    if (valueDragging) e.preventDefault();
  }, { passive: false });

  function handleMove(mx, my) {
    if (!valueDragging) return;
    const vx = (mx - cx) / radius;
    const vy = (cy - my) / radius;
    const n = normalize({ x: vx, y: vy });
    state.V[valueDragging] = n;
    drawValueCanvas();
    updateLiveMath();
  }

  window.addEventListener('mousemove', (e) => {
    if (!valueDragging) return;
    const rect = canvas.getBoundingClientRect();
    const mx = Math.max(0, Math.min(logicalWidth, e.clientX - rect.left));
    const my = Math.max(0, Math.min(logicalHeight, e.clientY - rect.top));
    handleMove(mx, my);
  });
  window.addEventListener('touchmove', (e) => {
    if (!valueDragging || !e.touches[0]) return;
    const rect = canvas.getBoundingClientRect();
    const mx = Math.max(0, Math.min(logicalWidth, e.touches[0].clientX - rect.left));
    const my = Math.max(0, Math.min(logicalHeight, e.touches[0].clientY - rect.top));
    handleMove(mx, my);
    e.preventDefault();
  }, { passive: false });

  window.addEventListener('mouseup', () => { valueDragging = null; });
  window.addEventListener('touchend', () => { valueDragging = null; });
}

// ---------- Tables ----------
function updateTables() {
  const s = SCENARIOS[state.scenarioKey];
  const { words, scores, weights } = computeAttention();
  const q = state.Q[s.focus];

  // Raw scores table
  const rawTbody = document.querySelector('#rawScoresTable tbody');
  if (rawTbody) {
    rawTbody.innerHTML = words.map((w, i) => {
      const k = state.K[w];
      return `
        <tr>
          <td><strong>${w}</strong></td>
          <td>(${q.x.toFixed(2)}, ${q.y.toFixed(2)})</td>
          <td>(${k.x.toFixed(2)}, ${k.y.toFixed(2)})</td>
          <td><code>${scores[i].toFixed(2)}</code></td>
        </tr>
      `;
    }).join('');
  }

  // Attention table: exp, softmax, bar, then ∑ row
  const attnTbody = document.querySelector('#attentionTable tbody');
  if (attnTbody) {
    const maxS = Math.max(...scores);
    const expShifted = scores.map((sc) => Math.exp(sc - maxS));
    const sumExp = expShifted.reduce((a, b) => a + b, 0);
    attnTbody.innerHTML = words.map((w, i) => {
      const weightPct = (weights[i] * 100).toFixed(1);
      const barW = Math.max(2, weights[i] * 160);
      const display = (expShifted[i]).toFixed(3);
      return `
        <tr>
          <td><strong>${w}</strong></td>
          <td><code>${scores[i].toFixed(2)}</code></td>
          <td><code>${display}</code></td>
          <td><code>${weightPct}%</code></td>
          <td class="weight-bar-cell"><div class="weight-bar" style="width:${barW}px"></div></td>
        </tr>
      `;
    }).join('');
    const foot = document.getElementById('attentionTableFoot');
    if (foot) {
      foot.innerHTML = `
        <tr>
          <td>∑</td>
          <td></td>
          <td><code>${sumExp.toFixed(3)}</code></td>
          <td><code>100.0%</code></td>
          <td></td>
        </tr>
      `;
    }
  }

  // Summary table across all scenarios
  const summaryBody = document.getElementById('summaryTableBody');
  if (summaryBody) {
    let html = '';
    Object.keys(SCENARIOS).forEach((key) => {
      const sc = SCENARIOS[key];
      const scoresS = sc.words.map((w) => dot(sc.Q[sc.focus], sc.K[w]));
      const weightsS = softmax(scoresS);
      let bestIdx = 0;
      weightsS.forEach((w, i) => { if (w > weightsS[bestIdx]) bestIdx = i; });
      const bestWord = sc.words[bestIdx];
      const bestPct = (weightsS[bestIdx] * 100).toFixed(0);
      const interp = interpretation(key, bestWord);
      html += `
        <tr${key === state.scenarioKey ? ' style="background:rgba(44,111,183,0.05)"' : ''}>
          <td>${sc.sentence}</td>
          <td><strong>${sc.focus}</strong></td>
          <td><strong>${bestWord}</strong></td>
          <td>${bestPct}%</td>
          <td>${interp}</td>
        </tr>
      `;
    });
    summaryBody.innerHTML = html;
  }
}

function interpretation(key, bestWord) {
  const map = {
    riverBank: { river: 'water-side bank', bank: 'self-reference', The: 'no help' },
    moneyBank: { money: 'financial bank', deposit: 'financial action', bank: 'self-reference' },
    appleFruit: { pie: 'apple the fruit', ate: 'eating context', apple: 'self-reference' },
    appleCorp: { phone: 'Apple the company', released: 'product launch', Apple: 'self-reference' },
    bassFish: { lake: 'bass the fish', caught: 'fishing action', bass: 'self-reference' },
    selfBank: { bank: 'self-reference (carries own meaning)', itself: 'reflexive marker', The: 'no help' }
  };
  return (map[key] && map[key][bestWord]) || 'neighbor wins';
}

// ---------- Live math ----------
function updateLiveMath() {
  if (!window.katex) return;

  const s = SCENARIOS[state.scenarioKey];
  const { words, scores, weights, Vf, q } = computeAttention();

  // Softmax live
  const softmaxEl = document.getElementById('math-softmax-live');
  if (softmaxEl) {
    const pieces = words.map((w, i) =>
      `w_{${w}} = ${weights[i].toFixed(3)}`
    ).join(', \\quad ');
    katex.render(pieces, softmaxEl, { displayMode: true, throwOnError: false });
  }

  // Final V live
  const finalEl = document.getElementById('math-final-live');
  if (finalEl) {
    const terms = words.map((w, i) => {
      const v = state.V[w];
      return `${weights[i].toFixed(2)}\\,\\begin{pmatrix}${v.x.toFixed(2)}\\\\${v.y.toFixed(2)}\\end{pmatrix}`;
    }).join(' + ');
    const vf = `V_{\\text{final}} = ${terms} = \\begin{pmatrix}${Vf.x.toFixed(2)}\\\\${Vf.y.toFixed(2)}\\end{pmatrix}`;
    katex.render(vf, finalEl, { displayMode: true, throwOnError: false });
  }
}

// ---------- Static math ----------
function renderStaticMath() {
  if (!window.katex) return;
  const blocks = {
    'math-dot-product': '\\text{Score}(Q, K) = Q \\cdot K = q_x k_x + q_y k_y',
    'math-softmax':
      'w_i = \\frac{e^{s_i}}{\\sum_j e^{s_j}} \\qquad \\text{so that} \\qquad \\sum_i w_i = 1',
    'math-final-v':
      'V_{\\text{final}} = \\sum_i w_i\\, V_i'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try {
      katex.render(blocks[id], el, { displayMode: true, throwOnError: false });
    } catch (_) { /* no-op */ }
  });
}

// ============================================================
// PART 2 — extensions
//   Step 2: where Q, K, V come from (projection demo)
//   Step 6: matrix view of attention
//   Step 7: three knobs (sqrt(d_k), causal mask, multi-head)
//   Step 8: positional encodings + permute demo
//   Step 9: live training of a tiny attention head
// ============================================================

// ---------- Math helpers (vectors + matrices as nested arrays) ----------
function softmaxArr(arr) {
  let m = -Infinity;
  for (const v of arr) if (Number.isFinite(v) && v > m) m = v;
  if (!Number.isFinite(m)) m = 0;
  const exps = arr.map((v) => Number.isFinite(v) ? Math.exp(v - m) : 0);
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

function matMul(A, B) {
  const rows = A.length;
  const inner = A[0].length;
  const cols = B[0].length;
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols).fill(0);
    for (let k = 0; k < inner; k++) {
      const aik = A[i][k];
      const Bk = B[k];
      for (let j = 0; j < cols; j++) row[j] += aik * Bk[j];
    }
    out[i] = row;
  }
  return out;
}

function transposeMat(M) {
  const rows = M.length;
  const cols = M[0].length;
  const out = new Array(cols);
  for (let j = 0; j < cols; j++) {
    out[j] = new Array(rows);
    for (let i = 0; i < rows; i++) out[j][i] = M[i][j];
  }
  return out;
}

function makeMat(rows, cols, fill = 0) {
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) out[i] = new Array(cols).fill(fill);
  return out;
}

function randn() {
  const u1 = Math.random() || 1e-12;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function randMat(rows, cols, scale = 1) {
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols);
    for (let j = 0; j < cols; j++) row[j] = randn() * scale;
    out[i] = row;
  }
  return out;
}

function shuffleArr(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function rotate2d(v, theta) {
  return {
    x: v.x * Math.cos(theta) - v.y * Math.sin(theta),
    y: v.x * Math.sin(theta) + v.y * Math.cos(theta)
  };
}

// ---------- Render helpers ----------
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

function valueToColor(v, maxAbs, accent = 'mixed') {
  const t = clamp(v / maxAbs, -1, 1);
  if (accent === 'q') return `rgba(217,98,43,${0.10 + 0.7 * Math.abs(t)})`;
  if (accent === 'k') return `rgba(44,111,183,${0.10 + 0.7 * Math.abs(t)})`;
  if (accent === 'v') return `rgba(30,119,112,${0.10 + 0.7 * Math.abs(t)})`;
  return t >= 0
    ? `rgba(44,111,183,${0.10 + 0.7 * t})`
    : `rgba(217,98,43,${0.10 + 0.7 * Math.abs(t)})`;
}

function renderMatrixCellGrid(elementId, matrix, options = {}) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (!matrix.length) { el.innerHTML = ''; return; }
  const rows = matrix.length;
  const cols = matrix[0].length;
  let maxAbs = 0;
  for (const row of matrix) for (const v of row) {
    if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
  }
  maxAbs = Math.max(maxAbs, options.maxAbs || 0.01);
  el.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  el.style.gridTemplateRows = `repeat(${rows}, auto)`;
  let html = '';
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const v = matrix[i][j];
      const color = valueToColor(v, maxAbs, options.accent || 'mixed');
      const display = Number.isFinite(v) ? v.toFixed(2) : '−∞';
      html += `<div class="matrix-cell" style="background:${color}">${display}</div>`;
    }
  }
  el.innerHTML = html;
}

function renderHeatmapGrid(elementId, matrix, labels, options = {}) {
  const el = document.getElementById(elementId);
  if (!el || !matrix.length) return;
  const N = matrix.length;
  let maxVal = 0;
  for (const row of matrix) for (const v of row) {
    if (Number.isFinite(v)) maxVal = Math.max(maxVal, Math.abs(v));
  }
  maxVal = Math.max(maxVal, 0.001);
  const rowsHigh = options.rowsHigh;
  el.style.gridTemplateColumns = `auto repeat(${N}, minmax(0, 1fr))`;
  let html = '<div class="hm-corner"></div>';
  for (let j = 0; j < N; j++) html += `<div class="hm-colhead">${labels[j]}</div>`;
  for (let i = 0; i < N; i++) {
    html += `<div class="hm-rowhead">${labels[i]}</div>`;
    for (let j = 0; j < N; j++) {
      const v = matrix[i][j];
      const color = Number.isFinite(v)
        ? valueToColor(v, maxVal, options.accent || 'mixed')
        : 'rgba(0,0,0,0.55)';
      const dim = (rowsHigh != null && i !== rowsHigh) ? 'opacity:0.35;' : '';
      const display = Number.isFinite(v) ? v.toFixed(2) : '−∞';
      html += `<div class="hm-cell" style="background:${color};${dim}">${display}</div>`;
    }
  }
  el.innerHTML = html;
}

// ============================================================
// Step 2 — Where Q, K, V come from (projection demo)
// ============================================================
const QKV_TOKENS = [
  { id: 'A', label: 'Token A', x: [0.8, 0.1, -0.3, 0.5] },
  { id: 'B', label: 'Token B', x: [0.2, 0.9, 0.4, -0.1] },
  { id: 'C', label: 'Token C', x: [-0.4, 0.3, 0.7, 0.2] }
];
// 4 x 2 projection matrices (hand-picked, not random)
const QKV_WQ = [
  [ 0.30,  0.40],
  [-0.50,  0.20],
  [ 0.60, -0.30],
  [ 0.10,  0.70]
];
const QKV_WK = [
  [ 0.20, -0.40],
  [ 0.50,  0.30],
  [-0.20,  0.60],
  [ 0.40,  0.10]
];
const QKV_WV = [
  [ 0.45,  0.10],
  [-0.10,  0.55],
  [ 0.30, -0.20],
  [ 0.50,  0.40]
];
let qkvFocus = 'A';

function renderQKVSource() {
  const tokensEl = document.getElementById('qkv-tokens');
  if (!tokensEl) return;
  tokensEl.innerHTML = QKV_TOKENS.map((t) => `
    <button class="qkv-token-btn ${t.id === qkvFocus ? 'is-active' : ''}" data-token="${t.id}">${t.label}</button>
  `).join('');
  tokensEl.querySelectorAll('[data-token]').forEach((b) => {
    b.addEventListener('click', () => {
      qkvFocus = b.dataset.token;
      renderQKVSource();
    });
  });

  const focus = QKV_TOKENS.find((t) => t.id === qkvFocus);
  const x = focus.x;

  renderMatrixCellGrid('qkv-x', [x]);
  renderMatrixCellGrid('qkv-wq', QKV_WQ, { accent: 'q' });
  renderMatrixCellGrid('qkv-wk', QKV_WK, { accent: 'k' });
  renderMatrixCellGrid('qkv-wv', QKV_WV, { accent: 'v' });

  const q = matMul([x], QKV_WQ)[0];
  const k = matMul([x], QKV_WK)[0];
  const v = matMul([x], QKV_WV)[0];

  renderMatrixCellGrid('qkv-q-out', [q], { accent: 'q' });
  renderMatrixCellGrid('qkv-k-out', [k], { accent: 'k' });
  renderMatrixCellGrid('qkv-v-out', [v], { accent: 'v' });

  const explain = document.getElementById('qkv-explain');
  if (explain) {
    explain.innerHTML =
      `<strong>${focus.label}</strong> &middot; ` +
      `q=(${q[0].toFixed(2)}, ${q[1].toFixed(2)}) &middot; ` +
      `k=(${k[0].toFixed(2)}, ${k[1].toFixed(2)}) &middot; ` +
      `v=(${v[0].toFixed(2)}, ${v[1].toFixed(2)}). ` +
      `Same input embedding, three rotations, three roles.`;
  }
}

// ============================================================
// Step 6 — Matrix form: Y = softmax(QK^T / sqrt(d)) V
// For pedagogy we let every token act as both query and key
// (self-attention). The focus token's Q is the user-dragged value;
// every other token's Q is just its K (so they self-match). It gives
// a varied N x N heatmap that reflects all the same structure.
// ============================================================
function buildSentenceMatrices() {
  const s = SCENARIOS[state.scenarioKey];
  const N = s.words.length;
  const Qrows = s.words.map((w) =>
    w === s.focus ? [state.Q[w].x, state.Q[w].y]
                  : [state.K[w].x, state.K[w].y]
  );
  const Krows = s.words.map((w) => [state.K[w].x, state.K[w].y]);
  const Vrows = s.words.map((w) => [state.V[w].x, state.V[w].y]);
  const dk = 2;
  const scale = 1 / Math.sqrt(dk);
  const S = makeMat(N, N, 0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      S[i][j] = (Qrows[i][0] * Krows[j][0] + Qrows[i][1] * Krows[j][1]) * scale;
    }
  }
  const A = S.map(softmaxArr);
  const Y = matMul(A, Vrows);
  return { N, S, A, Y, words: s.words };
}

function renderMatrixView() {
  const el = document.getElementById('hm-scores');
  if (!el) return;
  const { N, S, A, Y, words } = buildSentenceMatrices();
  renderHeatmapGrid('hm-scores', S, words, { rowsHigh: state.matrixRow });
  renderHeatmapGrid('hm-weights', A, words, { rowsHigh: state.matrixRow });
  renderMatrixCellGrid('hm-output', Y, { accent: 'v' });

  const buttons = document.getElementById('matrix-row-buttons');
  if (buttons) {
    buttons.innerHTML = words.map((w, i) =>
      `<button class="mini-btn ${state.matrixRow === i ? 'is-active' : ''}" data-row="${i}">${w}</button>`
    ).join('');
    buttons.querySelectorAll('[data-row]').forEach((b) => {
      b.addEventListener('click', () => {
        const r = parseInt(b.dataset.row, 10);
        state.matrixRow = state.matrixRow === r ? null : r;
        renderMatrixView();
      });
    });
  }
  const explain = document.getElementById('matrix-row-explain');
  if (explain) {
    if (state.matrixRow == null) {
      explain.textContent = 'Pick a row to see one token\'s outgoing attention.';
    } else {
      const row = A[state.matrixRow];
      let topIdx = 0;
      for (let j = 0; j < row.length; j++) if (row[j] > row[topIdx]) topIdx = j;
      explain.innerHTML =
        `Row <strong>${words[state.matrixRow]}</strong> sends ` +
        `${(row[topIdx] * 100).toFixed(0)}% of its attention to ` +
        `<strong>${words[topIdx]}</strong>.`;
    }
  }
}

// ============================================================
// Step 7 — Three knobs
// ============================================================
function dkDemoUpdate() {
  const slider = document.getElementById('dk-slider');
  const scaleBox = document.getElementById('dk-scale');
  if (!slider) return;
  const dk = parseInt(slider.value, 10);
  const scaled = scaleBox.checked;
  const dkValEl = document.getElementById('dk-val');
  if (dkValEl) dkValEl.textContent = dk;

  const scaleFactor = scaled ? 1 / Math.sqrt(dk) : 1;
  const N = 8;
  const sample = drawDkSample(dk, N, scaleFactor);
  const barRow = document.getElementById('dk-bars');
  if (barRow) {
    barRow.innerHTML = sample.weights.map((w, i) =>
      `<div class="bar-cell" title="weight ${(w*100).toFixed(1)}%"><div class="bar" style="height:${(w*100).toFixed(1)}%"></div><div class="bar-label">k<sub>${i}</sub></div></div>`
    ).join('');
  }
  const stats = document.getElementById('dk-stats');
  if (stats) {
    const maxW = Math.max(...sample.weights);
    const entropy = -sample.weights.reduce((s, w) => s + (w > 0 ? w * Math.log(w) : 0), 0);
    stats.innerHTML =
      `max weight = <strong>${(maxW * 100).toFixed(1)}%</strong>, ` +
      `entropy = ${entropy.toFixed(2)} (max ${Math.log(N).toFixed(2)} = uniform)`;
  }
  drawDkHistogram(dk, N, scaleFactor);
}

function drawDkSample(dk, N, scaleFactor) {
  const q = Array.from({ length: dk }, randn);
  const keys = Array.from({ length: N }, () =>
    Array.from({ length: dk }, randn)
  );
  const scores = keys.map((k) => {
    let s = 0;
    for (let i = 0; i < dk; i++) s += k[i] * q[i];
    return s * scaleFactor;
  });
  return { weights: softmaxArr(scores), scores };
}

function drawDkHistogram(dk, N, scaleFactor) {
  const canvas = document.getElementById('dk-hist');
  if (!canvas) return;
  const W = 380, H = 180;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fdfcf9';
  ctx.fillRect(0, 0, W, H);

  const trials = 500;
  const bins = 20;
  const counts = new Array(bins).fill(0);
  let avgMax = 0;
  for (let t = 0; t < trials; t++) {
    const sample = drawDkSample(dk, N, scaleFactor);
    const m = Math.max(...sample.weights);
    avgMax += m;
    const bin = Math.min(bins - 1, Math.floor(m * bins));
    counts[bin]++;
  }
  avgMax /= trials;
  const maxCount = Math.max(1, ...counts);
  const m = { l: 30, r: 8, t: 8, b: 24 };
  const px = W - m.l - m.r;
  const py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(m.l, m.t, px, py);
  const bw = px / bins;
  ctx.fillStyle = '#2c6fb7';
  for (let i = 0; i < bins; i++) {
    const h = (counts[i] / maxCount) * py;
    ctx.fillRect(m.l + i * bw + 1, m.t + py - h, bw - 2, h);
  }
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'left';
  ctx.fillText('0', m.l - 4, m.t + py + 14);
  ctx.fillText('1', m.l + px - 4, m.t + py + 14);
  ctx.textAlign = 'center';
  ctx.fillText('peak weight (1 = one-hot)', m.l + px / 2, H - 6);
  const stats = document.getElementById('dk-hist-stats');
  if (stats) {
    stats.innerHTML = `Average peak weight across 500 random draws: <strong>${(avgMax * 100).toFixed(1)}%</strong>`;
  }
}

function renderMaskedHeatmaps() {
  const { N, S, A, words } = buildSentenceMatrices();
  if (!N) return;
  const Smasked = S.map((row, i) => row.map((v, j) => j > i ? -Infinity : v));
  const Amasked = Smasked.map(softmaxArr);
  renderHeatmapGrid('hm-mask-before', A, words);
  renderHeatmapGrid('hm-mask-after', Amasked, words);
}

function renderMultiHead() {
  const grid = document.getElementById('multihead-grid');
  if (!grid) return;
  const s = SCENARIOS[state.scenarioKey];
  const heads = [
    { name: 'Head 1', rotQ: 0,            rotK: 0,             theme: 'identity rotation — same Q/K-space as steps 1–6.' },
    { name: 'Head 2', rotQ: Math.PI / 4,  rotK: -Math.PI / 4,  theme: 'rotates Q and K opposite ways — bonds tokens whose features point against each other.' },
    { name: 'Head 3', rotQ: Math.PI / 2,  rotK: 0,             theme: '90° Q rotation — picks tokens whose K is perpendicular to the original Q (often syntactic adjacency).' },
    { name: 'Head 4', rotQ: -Math.PI / 6, rotK: Math.PI / 6,   theme: 'small mirror twist — sharpens self-similarity, useful for "stay put" heads.' }
  ];
  const N = s.words.length;
  let html = '';
  heads.forEach((h, idx) => {
    html += `<div class="multihead-cell">
      <div class="matrix-label">${h.name}</div>
      <div class="heatmap-grid" id="hm-multi-${idx}"></div>
      <div class="multihead-theme">${h.theme}</div>
    </div>`;
  });
  grid.innerHTML = html;
  heads.forEach((h, idx) => {
    const Qrows = s.words.map((w) => {
      const base = w === s.focus ? state.Q[w] : state.K[w];
      const r = rotate2d(base, h.rotQ);
      return [r.x, r.y];
    });
    const Krows = s.words.map((w) => {
      const r = rotate2d(state.K[w], h.rotK);
      return [r.x, r.y];
    });
    const dk = 2;
    const scale = 1 / Math.sqrt(dk);
    const S = makeMat(N, N, 0);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        S[i][j] = (Qrows[i][0] * Krows[j][0] + Qrows[i][1] * Krows[j][1]) * scale;
      }
    }
    const A = S.map(softmaxArr);
    renderHeatmapGrid(`hm-multi-${idx}`, A, s.words);
  });
}

// ============================================================
// Step 8 — Positional encodings
// ============================================================
function sinusoidalPE(T, d) {
  const out = makeMat(T, d, 0);
  for (let pos = 0; pos < T; pos++) {
    for (let i = 0; i < d; i++) {
      const dim2 = Math.floor(i / 2) * 2;
      const angle = pos / Math.pow(10000, dim2 / d);
      out[pos][i] = (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
    }
  }
  return out;
}

function renderPECanvas() {
  const canvas = document.getElementById('pe-canvas');
  if (!canvas) return;
  const Tslider = document.getElementById('pe-T');
  const Dslider = document.getElementById('pe-d');
  const Pslider = document.getElementById('pe-pos');
  const T = parseInt(Tslider.value, 10);
  const d = parseInt(Dslider.value, 10);
  Pslider.max = String(T - 1);
  if (parseInt(Pslider.value, 10) > T - 1) Pslider.value = String(Math.floor(T / 2));
  const focus = parseInt(Pslider.value, 10);
  document.getElementById('pe-T-val').textContent = T;
  document.getElementById('pe-d-val').textContent = d;
  document.getElementById('pe-pos-val').textContent = focus;

  const PE = sinusoidalPE(T, d);
  const W = 720, H = 280;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const margin = { l: 60, r: 12, t: 12, b: 28 };
  const px = W - margin.l - margin.r;
  const py = H - margin.t - margin.b;
  const cellW = px / d;
  const cellH = py / T;
  for (let pos = 0; pos < T; pos++) {
    for (let i = 0; i < d; i++) {
      const v = PE[pos][i]; // -1..1
      const t = (v + 1) / 2;
      const r = Math.round(217 + (44 - 217) * t);
      const g = Math.round(98 + (111 - 98) * t);
      const b = Math.round(43 + (183 - 43) * t);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(margin.l + i * cellW, margin.t + pos * cellH, cellW + 1, cellH + 1);
    }
  }
  // Highlight focus row
  ctx.strokeStyle = '#1a1815';
  ctx.lineWidth = 2;
  ctx.strokeRect(margin.l, margin.t + focus * cellH, px, cellH);

  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('dimension index →', margin.l + px / 2, H - 8);
  ctx.save();
  ctx.translate(14, margin.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('position ↓', 0, 0);
  ctx.restore();
  ctx.textAlign = 'left';
  ctx.fillText('low →', margin.l + 4, margin.t - 2);
  ctx.textAlign = 'right';
  ctx.fillText('→ high frequency dies off', margin.l + px - 4, margin.t - 2);
}

function renderPermuteDemo() {
  const tokensEl = document.getElementById('permute-tokens');
  const heatmapEl = document.getElementById('permute-heatmap');
  if (!tokensEl || !heatmapEl) return;
  const s = SCENARIOS[state.scenarioKey];
  if (!state.permuteOrder) state.permuteOrder = s.words.map((_, i) => i);
  const order = state.permuteOrder;
  const usePE = document.getElementById('permute-pe').checked;
  const labels = order.map((i) => s.words[i]);

  // X = K vectors for each word, plus tiny PE if enabled
  function pe(t, dim) {
    const angle = t / Math.pow(10, dim);
    return dim === 0 ? Math.sin(angle) : Math.cos(angle * 0.5);
  }
  const xs = order.map((idx, t) => {
    const k = state.K[s.words[idx]];
    if (!usePE) return [k.x, k.y];
    return [k.x + 0.4 * pe(t, 0), k.y + 0.4 * pe(t, 1)];
  });
  const N = xs.length;
  const dk = 2;
  const scaleFactor = 1 / Math.sqrt(dk);
  const S = makeMat(N, N, 0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      S[i][j] = (xs[i][0] * xs[j][0] + xs[i][1] * xs[j][1]) * scaleFactor;
    }
  }
  const A = S.map(softmaxArr);
  tokensEl.innerHTML = labels.map((w, i) =>
    `<span class="permute-chip${s.words[order[i]] === s.focus ? ' is-focus' : ''}">${i + 1}. ${w}</span>`
  ).join('');
  renderHeatmapGrid('permute-heatmap', A, labels);
}

function wirePermuteControls() {
  const shuffleBtn = document.getElementById('permute-shuffle');
  const resetBtn = document.getElementById('permute-reset');
  const peBox = document.getElementById('permute-pe');
  if (shuffleBtn) shuffleBtn.addEventListener('click', () => {
    const s = SCENARIOS[state.scenarioKey];
    state.permuteOrder = shuffleArr(s.words.map((_, i) => i));
    renderPermuteDemo();
  });
  if (resetBtn) resetBtn.addEventListener('click', () => {
    const s = SCENARIOS[state.scenarioKey];
    state.permuteOrder = s.words.map((_, i) => i);
    renderPermuteDemo();
  });
  if (peBox) peBox.addEventListener('change', renderPermuteDemo);
}

function wirePEControls() {
  ['pe-T', 'pe-d', 'pe-pos'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', renderPECanvas);
  });
}

function wireKnobsControls() {
  const dk = document.getElementById('dk-slider');
  const scaleBox = document.getElementById('dk-scale');
  if (dk) dk.addEventListener('input', dkDemoUpdate);
  if (scaleBox) scaleBox.addEventListener('change', dkDemoUpdate);
}

// ============================================================
// Step 9 — Live training of a tiny attention head.
// Toy task "soft lookup": 4 tokens (3 candidates + 1 query),
// candidates carry id (one-hot, 3-d) + payload (random, 3-d).
// Query carries target id (one-hot) + zero payload.
// Target output at the query position = matching candidate's payload.
// Manual SGD over W_Q, W_K, W_V — every gradient computed by hand.
// ============================================================
let trainState = null;
let trainRAF = null;

function makeTrainingExample(D, dv) {
  const ids = shuffleArr([0, 1, 2]); // unique ids per candidate
  const payloads = ids.map(() => Array.from({ length: dv }, () => randn() * 0.6));
  const targetSlot = Math.floor(Math.random() * 3);
  const targetId = ids[targetSlot];
  const target = payloads[targetSlot];
  const x = [];
  for (let i = 0; i < 3; i++) {
    const oh = [0, 0, 0]; oh[ids[i]] = 1;
    x.push([...oh, ...payloads[i]]);
  }
  const oh = [0, 0, 0]; oh[targetId] = 1;
  x.push([...oh, ...new Array(dv).fill(0)]);
  return { x, target, ids, targetSlot, targetId };
}

function makeFixedExample(seed) {
  // Reproducible-ish fixed example for the heatmap viz
  const oldRandom = Math.random;
  let s = seed >>> 0;
  Math.random = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xFFFFFFFF;
  };
  const ex = makeTrainingExample(6, 3);
  Math.random = oldRandom;
  return ex;
}

function initTrainingState() {
  const D = 6, dk = 4, dv = 3, T = 4;
  trainState = {
    D, dk, dv, T,
    Wq: randMat(D, dk, 0.5),
    Wk: randMat(D, dk, 0.5),
    Wv: randMat(D, dv, 0.5),
    losses: [],
    step: 0,
    running: false,
    lr: 0.10,
    fixed: makeFixedExample(7)
  };
}

function forwardAttn(W, x) {
  const T = x.length;
  const dk = W.Wq[0].length;
  const dv = W.Wv[0].length;
  const q = matMul(x, W.Wq);
  const k = matMul(x, W.Wk);
  const v = matMul(x, W.Wv);
  const scaleFactor = 1 / Math.sqrt(dk);
  const s = makeMat(T, T, 0);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      let dot = 0;
      for (let c = 0; c < dk; c++) dot += q[i][c] * k[j][c];
      s[i][j] = dot * scaleFactor;
    }
  }
  const a = s.map(softmaxArr);
  const y = matMul(a, v);
  return { x, q, k, v, s, a, y };
}

function backwardAttn(W, fwd, target, queryPos) {
  const T = fwd.x.length;
  const D = fwd.x[0].length;
  const dk = W.Wq[0].length;
  const dv = W.Wv[0].length;
  const dy = makeMat(T, dv, 0);
  let loss = 0;
  for (let c = 0; c < dv; c++) {
    const diff = fwd.y[queryPos][c] - target[c];
    dy[queryPos][c] = diff;
    loss += 0.5 * diff * diff;
  }
  // dV = a^T dy ; da = dy v^T
  const dV = matMul(transposeMat(fwd.a), dy);
  const da = matMul(dy, transposeMat(fwd.v));
  // softmax row-wise backward
  const ds = makeMat(T, T, 0);
  for (let i = 0; i < T; i++) {
    let dot = 0;
    for (let kk = 0; kk < T; kk++) dot += fwd.a[i][kk] * da[i][kk];
    for (let j = 0; j < T; j++) ds[i][j] = fwd.a[i][j] * (da[i][j] - dot);
  }
  const scaleFactor = 1 / Math.sqrt(dk);
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) ds[i][j] *= scaleFactor;
  // dQ = ds k ; dK = ds^T q
  const dQ = matMul(ds, fwd.k);
  const dK = matMul(transposeMat(ds), fwd.q);
  const dWq = matMul(transposeMat(fwd.x), dQ);
  const dWk = matMul(transposeMat(fwd.x), dK);
  const dWv = matMul(transposeMat(fwd.x), dV);
  return { dWq, dWk, dWv, loss };
}

function trainStepBatch(batchSize = 16) {
  const accumQ = makeMat(trainState.D, trainState.dk, 0);
  const accumK = makeMat(trainState.D, trainState.dk, 0);
  const accumV = makeMat(trainState.D, trainState.dv, 0);
  let totalLoss = 0;
  for (let b = 0; b < batchSize; b++) {
    const ex = makeTrainingExample(trainState.D, trainState.dv);
    const fwd = forwardAttn(trainState, ex.x);
    const bw = backwardAttn(trainState, fwd, ex.target, trainState.T - 1);
    for (let i = 0; i < trainState.D; i++) {
      for (let j = 0; j < trainState.dk; j++) {
        accumQ[i][j] += bw.dWq[i][j];
        accumK[i][j] += bw.dWk[i][j];
      }
      for (let j = 0; j < trainState.dv; j++) {
        accumV[i][j] += bw.dWv[i][j];
      }
    }
    totalLoss += bw.loss;
  }
  const lr = trainState.lr;
  for (let i = 0; i < trainState.D; i++) {
    for (let j = 0; j < trainState.dk; j++) {
      trainState.Wq[i][j] -= lr * accumQ[i][j] / batchSize;
      trainState.Wk[i][j] -= lr * accumK[i][j] / batchSize;
    }
    for (let j = 0; j < trainState.dv; j++) {
      trainState.Wv[i][j] -= lr * accumV[i][j] / batchSize;
    }
  }
  trainState.step++;
  trainState.losses.push(totalLoss / batchSize);
  if (trainState.losses.length > 1500) {
    trainState.losses = trainState.losses.slice(-1500);
  }
}

function drawLossCurve() {
  const canvas = document.getElementById('train-loss-canvas');
  if (!canvas || !trainState) return;
  const W = 500, H = 240;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fdfcf9';
  ctx.fillRect(0, 0, W, H);
  const m = { l: 56, r: 14, t: 14, b: 30 };
  const px = W - m.l - m.r;
  const py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6';
  ctx.lineWidth = 1;
  ctx.strokeRect(m.l, m.t, px, py);

  if (trainState.losses.length === 0) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText('Press Start training to begin.', m.l + px / 2, m.t + py / 2);
    return;
  }
  const logL = trainState.losses.map((v) => Math.log10(Math.max(v, 1e-6)));
  let minLog = Math.floor(Math.min(...logL));
  let maxLog = Math.ceil(Math.max(...logL));
  if (minLog === maxLog) maxLog = minLog + 1;
  const range = maxLog - minLog;

  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = minLog; v <= maxLog; v++) {
    const y = m.t + (1 - (v - minLog) / range) * py;
    ctx.fillText(`10^${v}`, m.l - 6, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath();
    ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.textAlign = 'center';
  const N = trainState.losses.length;
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const x = m.l + (i / ticks) * px;
    const step = Math.round((i / ticks) * (N - 1));
    ctx.fillText(String(step), x, m.t + py + 16);
  }
  ctx.strokeStyle = '#2c6fb7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  logL.forEach((y, i) => {
    const xx = m.l + (i / Math.max(1, N - 1)) * px;
    const yy = m.t + (1 - (y - minLog) / range) * py;
    if (i === 0) ctx.moveTo(xx, yy);
    else ctx.lineTo(xx, yy);
  });
  ctx.stroke();

  ctx.fillStyle = '#6e665b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('step', m.l + px / 2, H - 6);
  ctx.save();
  ctx.translate(14, m.t + py / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('loss (log)', 0, 0);
  ctx.restore();
}

function renderTraining() {
  if (!trainState) return;
  drawLossCurve();
  const fwd = forwardAttn(trainState, trainState.fixed.x);
  const labels = ['c1', 'c2', 'c3', 'query'];
  renderHeatmapGrid('train-heatmap', fwd.a, labels);
  renderMatrixCellGrid('train-wq', trainState.Wq, { accent: 'q' });
  renderMatrixCellGrid('train-wk', trainState.Wk, { accent: 'k' });
  renderMatrixCellGrid('train-wv', trainState.Wv, { accent: 'v' });
  document.getElementById('train-step').textContent = trainState.step;
  const lastLoss = trainState.losses.length > 0 ? trainState.losses[trainState.losses.length - 1] : null;
  document.getElementById('train-loss').textContent = lastLoss == null ? '—' : lastLoss.toFixed(4);
  const ex = trainState.fixed;
  const labelOf = (i) => 'ABC'[i];
  const cap = document.getElementById('train-example-caption');
  if (cap) {
    cap.innerHTML =
      `Candidates carry ids [<strong>${ex.ids.map(labelOf).join(', ')}</strong>]; ` +
      `query asks for "<strong>${labelOf(ex.targetId)}</strong>" → ` +
      `the query row should peak at column <strong>c${ex.targetSlot + 1}</strong>.`;
  }
  const lossCap = document.getElementById('train-loss-caption');
  if (lossCap) {
    if (trainState.step === 0) {
      lossCap.innerHTML = 'Press <em>Start training</em> to begin.';
    } else if (trainState.running) {
      lossCap.innerHTML = `Training&hellip; gradient steps land roughly 6× per frame.`;
    } else {
      lossCap.innerHTML = `Paused at step ${trainState.step}. Press <em>Start training</em> to keep going.`;
    }
  }
}

function trainLoop() {
  if (!trainState || !trainState.running) return;
  for (let i = 0; i < 6; i++) trainStepBatch(16);
  renderTraining();
  trainRAF = requestAnimationFrame(trainLoop);
}

function wireTrainingControls() {
  const startBtn = document.getElementById('train-toggle');
  const resetBtn = document.getElementById('train-reset');
  const lr = document.getElementById('lr-slider');
  if (lr) lr.addEventListener('input', () => {
    if (!trainState) initTrainingState();
    trainState.lr = parseFloat(lr.value);
    document.getElementById('lr-val').textContent = trainState.lr.toFixed(2);
  });
  if (startBtn) startBtn.addEventListener('click', () => {
    if (!trainState) initTrainingState();
    trainState.running = !trainState.running;
    startBtn.textContent = trainState.running ? 'Pause training' : 'Start training';
    if (trainState.running) trainLoop();
    else if (trainRAF) cancelAnimationFrame(trainRAF);
    renderTraining();
  });
  if (resetBtn) resetBtn.addEventListener('click', () => {
    if (trainRAF) cancelAnimationFrame(trainRAF);
    initTrainingState();
    if (startBtn) startBtn.textContent = 'Start training';
    renderTraining();
  });
}

// ============================================================
// Boot
// ============================================================
function init() {
  if (window.katex) {
    renderStaticMath();
    renderExtraMath();
  } else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', () => {
      renderStaticMath();
      renderExtraMath();
      updateLiveMath();
    });
  }

  document.querySelectorAll('#scenario-buttons [data-scenario]').forEach((btn) => {
    btn.addEventListener('click', () => loadScenario(btn.dataset.scenario));
  });

  installQKDrag();
  installValueDrag();

  // Initialise the new sections' interactivity
  renderQKVSource();
  wireKnobsControls();
  dkDemoUpdate();
  wirePEControls();
  renderPECanvas();
  wirePermuteControls();
  initTrainingState();
  wireTrainingControls();
  renderTraining();

  // Multi-head specialisation demo
  initMHState();
  wireMHControls();
  renderMH();

  loadScenario('riverBank');
}

// ============================================================
// Step 9 ½ — Multi-head specialisation, live.
// 4-token sequence; first 3 tokens form (type, key, val) entries
// of two types A/B, plus a fourth query token carrying two key
// queries (one for type A, one for type B). Target = val_A + val_B.
// Two heads with their own W_Q^h / W_K^h / W_V^h, then a final
// linear that combines [head_A_out ; head_B_out] -> dv-dim output.
// All gradients computed by hand.
// ============================================================
let mhState = null;
let mhRAF = null;

function initMHState() {
  // Token format (D = 8): [typeA flag, typeB flag, key0, key1, key2, val0, val1, val2]
  // Query token: [typeA flag, typeB flag, queryA0, queryA1, queryA2, queryB0, queryB1, queryB2]
  // We tile the inputs so the same D matches both candidate and query.
  const D = 8, dk = 4, dv = 3, T = 4;
  mhState = {
    D, dk, dv, T,
    Wq: [randMat(D, dk, 0.4), randMat(D, dk, 0.4)],
    Wk: [randMat(D, dk, 0.4), randMat(D, dk, 0.4)],
    Wv: [randMat(D, dv, 0.4), randMat(D, dv, 0.4)],
    Wo: randMat(2 * dv, dv, 0.3), // combine concat(head_A, head_B)
    losses: [],
    spec: [], // mean of (peakA on tokenA - peakA on tokenB) etc
    step: 0,
    running: false,
    lr: 0.05,
    singleHead: false,
    fixed: makeMHExample(7)
  };
}

function makeMHExample(seed) {
  const oldR = Math.random;
  let s = seed >>> 0;
  Math.random = () => {
    s = (s * 1103515245 + 12345) >>> 0;
    return s / 0xFFFFFFFF;
  };
  const ex = makeMHRandomExample();
  Math.random = oldR;
  return ex;
}

function makeMHRandomExample() {
  // Generate 2 type-A pairs and 2 type-B pairs - but we only have 3
  // candidates; sample 1 A pair and 2 B pair (or 2 A 1 B). Ensure both
  // types present so both heads are needed.
  const D = 8;
  const T = 4;
  const x = []; // T x D
  const placement = Math.random() < 0.5 ? ['A','B','A'] : ['B','A','B'];
  // Random keys/vals
  const keyA = [randn() * 0.6, randn() * 0.6, randn() * 0.6];
  const valA = [randn() * 0.6, randn() * 0.6, randn() * 0.6];
  const keyB = [randn() * 0.6, randn() * 0.6, randn() * 0.6];
  const valB = [randn() * 0.6, randn() * 0.6, randn() * 0.6];
  for (let i = 0; i < 3; i++) {
    const tp = placement[i];
    if (tp === 'A') {
      x.push([1, 0, ...keyA, ...valA]);
    } else {
      x.push([0, 1, ...keyB, ...valB]);
    }
  }
  // Query token: queries both A and B. We supply both keys
  // concatenated; heads pick out their relevant 3-d slice.
  // Layout: [1, 1, keyA[0], keyA[1], keyA[2], keyB[0], keyB[1], keyB[2]]
  x.push([1, 1, ...keyA.slice(0, 3), ...keyB.slice(0, 3)].slice(0, D));
  // Fix length: above has D=8 (1+1+3+3=8). Good.
  const target = [valA[0] + valB[0], valA[1] + valB[1], valA[2] + valB[2]];
  return { x, target, placement };
}

// Forward through a single head h (0 or 1), returning attention probs and head output (T x dv)
function forwardOneHead(h, x, W) {
  const T = x.length;
  const dk = W.Wq[h][0].length;
  const dv = W.Wv[h][0].length;
  const q = matMul(x, W.Wq[h]);
  const k = matMul(x, W.Wk[h]);
  const v = matMul(x, W.Wv[h]);
  const scaleF = 1 / Math.sqrt(dk);
  const s = makeMat(T, T, 0);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      let dot = 0;
      for (let c = 0; c < dk; c++) dot += q[i][c] * k[j][c];
      s[i][j] = dot * scaleF;
    }
  }
  const a = s.map(softmaxArr);
  const out = matMul(a, v);
  return { q, k, v, s, a, out };
}

function mhForward(W, x) {
  const f0 = forwardOneHead(0, x, W);
  const f1 = forwardOneHead(1, x, W);
  // Concatenate along feature dim -> T x (2*dv)
  const T = x.length;
  const dv = f0.v[0].length;
  const concat = new Array(T);
  for (let i = 0; i < T; i++) {
    concat[i] = [];
    for (let c = 0; c < dv; c++) concat[i].push(f0.out[i][c]);
    for (let c = 0; c < dv; c++) concat[i].push(W.singleHead ? 0 : f1.out[i][c]);
  }
  // Final linear -> T x dv
  const y = matMul(concat, W.Wo);
  return { y, concat, f0, f1 };
}

function mhBackward(W, fwd, target, queryPos) {
  const T = fwd.y.length;
  const D = fwd.f0.q.length === 0 ? 0 : null; // not used directly
  const dv = W.Wv[0][0].length;
  const dk = W.Wq[0][0].length;
  const Din = W.Wq[0].length; // D (input dim)
  // Loss: 0.5 * sum_c (y[query, c] - target[c])^2  (only at queryPos)
  let loss = 0;
  const dy = makeMat(T, dv, 0);
  for (let c = 0; c < dv; c++) {
    const diff = fwd.y[queryPos][c] - target[c];
    dy[queryPos][c] = diff;
    loss += 0.5 * diff * diff;
  }
  // d(concat) = dy @ Wo^T, d(Wo) = concat^T @ dy
  const dConcat = matMul(dy, transposeMat(W.Wo));
  const dWo = matMul(transposeMat(fwd.concat), dy);
  // Split dConcat into d(head0_out) and d(head1_out)
  const dHead0 = new Array(T), dHead1 = new Array(T);
  for (let i = 0; i < T; i++) {
    dHead0[i] = dConcat[i].slice(0, dv);
    dHead1[i] = dConcat[i].slice(dv, 2 * dv);
    if (W.singleHead) dHead1[i] = new Array(dv).fill(0);
  }

  function backHead(headIdx, fwdH, dOut) {
    // y_h = a_h @ v_h ; dV_h = a^T dy_h ; da = dy v^T
    const Th = T;
    const dVh = matMul(transposeMat(fwdH.a), dOut);
    const da = matMul(dOut, transposeMat(fwdH.v));
    // softmax row-wise backward
    const ds = makeMat(Th, Th, 0);
    for (let i = 0; i < Th; i++) {
      let dot = 0;
      for (let kk = 0; kk < Th; kk++) dot += fwdH.a[i][kk] * da[i][kk];
      for (let j = 0; j < Th; j++) ds[i][j] = fwdH.a[i][j] * (da[i][j] - dot);
    }
    const scaleF = 1 / Math.sqrt(dk);
    for (let i = 0; i < Th; i++) for (let j = 0; j < Th; j++) ds[i][j] *= scaleF;
    const dQh = matMul(ds, fwdH.k);
    const dKh = matMul(transposeMat(ds), fwdH.q);
    const dWqh = matMul(transposeMat(fwd.f0.x || fwd.input), dQh); // we'll pass x instead
    return { dQh, dKh, dVh, ds };
  }

  return { loss, dHead0, dHead1, dWo };
}

// Forward + backward end-to-end for batch step
function mhTrainStep(batchSize = 16) {
  const W = mhState;
  const D = W.D, dk = W.dk, dv = W.dv, T = W.T;
  const accumWq = [makeMat(D, dk, 0), makeMat(D, dk, 0)];
  const accumWk = [makeMat(D, dk, 0), makeMat(D, dk, 0)];
  const accumWv = [makeMat(D, dv, 0), makeMat(D, dv, 0)];
  const accumWo = makeMat(2 * dv, dv, 0);
  let totalLoss = 0;
  const queryPos = T - 1;
  // Track specialisation
  let specA = 0, specB = 0;
  for (let b = 0; b < batchSize; b++) {
    const ex = makeMHRandomExample();
    const x = ex.x;
    // Forward two heads
    const f0 = forwardOneHead(0, x, W);
    const f1 = forwardOneHead(1, x, W);
    const concat = new Array(T);
    for (let i = 0; i < T; i++) {
      concat[i] = [];
      for (let c = 0; c < dv; c++) concat[i].push(f0.out[i][c]);
      for (let c = 0; c < dv; c++) concat[i].push(W.singleHead ? 0 : f1.out[i][c]);
    }
    const y = matMul(concat, W.Wo);
    const dy = makeMat(T, dv, 0);
    let loss = 0;
    for (let c = 0; c < dv; c++) {
      const diff = y[queryPos][c] - ex.target[c];
      dy[queryPos][c] = diff;
      loss += 0.5 * diff * diff;
    }
    totalLoss += loss;
    // dWo += concat^T dy ; dConcat = dy Wo^T
    for (let i = 0; i < 2 * dv; i++) for (let c = 0; c < dv; c++) {
      accumWo[i][c] += concat[queryPos][i] * dy[queryPos][c];
    }
    const dConcat = matMul(dy, transposeMat(W.Wo));
    const dHead0 = new Array(T), dHead1 = new Array(T);
    for (let i = 0; i < T; i++) {
      dHead0[i] = dConcat[i].slice(0, dv);
      dHead1[i] = W.singleHead ? new Array(dv).fill(0) : dConcat[i].slice(dv, 2 * dv);
    }
    function backH(fwdH, dOut, accQ, accK, accV) {
      const dVh = matMul(transposeMat(fwdH.a), dOut);
      const da = matMul(dOut, transposeMat(fwdH.v));
      const ds = makeMat(T, T, 0);
      for (let i = 0; i < T; i++) {
        let dot = 0;
        for (let kk = 0; kk < T; kk++) dot += fwdH.a[i][kk] * da[i][kk];
        for (let j = 0; j < T; j++) ds[i][j] = fwdH.a[i][j] * (da[i][j] - dot);
      }
      const scaleF = 1 / Math.sqrt(dk);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) ds[i][j] *= scaleF;
      const dQh = matMul(ds, fwdH.k);
      const dKh = matMul(transposeMat(ds), fwdH.q);
      // Accumulate gradients into the W matrices: dWq += x^T dQh, etc.
      const xT = transposeMat(x);
      const tWq = matMul(xT, dQh);
      const tWk = matMul(xT, dKh);
      const tWv = matMul(xT, dVh);
      for (let i = 0; i < D; i++) {
        for (let c = 0; c < dk; c++) {
          accQ[i][c] += tWq[i][c];
          accK[i][c] += tWk[i][c];
        }
        for (let c = 0; c < dv; c++) accV[i][c] += tWv[i][c];
      }
    }
    backH(f0, dHead0, accumWq[0], accumWk[0], accumWv[0]);
    if (!W.singleHead) backH(f1, dHead1, accumWq[1], accumWk[1], accumWv[1]);
    // Specialisation: head 0's attention from query position to type-A token vs type-B token
    const queryRowA = f0.a[queryPos];
    const queryRowB = f1.a[queryPos];
    // Identify which tokens are type A vs type B from the placement
    let aIdx = -1, bIdx = -1;
    for (let i = 0; i < 3; i++) {
      if (ex.placement[i] === 'A' && aIdx < 0) aIdx = i;
      if (ex.placement[i] === 'B' && bIdx < 0) bIdx = i;
    }
    if (aIdx >= 0) specA += queryRowA[aIdx];
    if (bIdx >= 0) specB += queryRowB[bIdx];
  }
  const lr = W.lr;
  for (let h = 0; h < 2; h++) {
    for (let i = 0; i < D; i++) {
      for (let c = 0; c < dk; c++) {
        W.Wq[h][i][c] -= lr * accumWq[h][i][c] / batchSize;
        W.Wk[h][i][c] -= lr * accumWk[h][i][c] / batchSize;
      }
      for (let c = 0; c < dv; c++) W.Wv[h][i][c] -= lr * accumWv[h][i][c] / batchSize;
    }
  }
  for (let i = 0; i < 2 * dv; i++) for (let c = 0; c < dv; c++) W.Wo[i][c] -= lr * accumWo[i][c] / batchSize;
  W.step++;
  W.losses.push(totalLoss / batchSize);
  W.spec.push({ a: specA / batchSize, b: specB / batchSize });
  if (W.losses.length > 1500) {
    W.losses = W.losses.slice(-1500);
    W.spec = W.spec.slice(-1500);
  }
}

function renderMH() {
  if (!mhState) return;
  // Render heatmaps for both heads on the fixed example
  const W = mhState;
  const ex = W.fixed;
  const f0 = forwardOneHead(0, ex.x, W);
  const f1 = forwardOneHead(1, ex.x, W);
  const labels = ex.placement.concat(['query']);
  renderHeatmapGrid('mh-heatA', f0.a, labels);
  renderHeatmapGrid('mh-heatB', f1.a, labels);
  document.getElementById('mh-step').textContent = W.step;
  document.getElementById('mh-loss').textContent = W.losses.length ? W.losses[W.losses.length - 1].toFixed(4) : '—';
  drawMHLossCurve();
  drawMHSpecCurve();
}

function drawMHLossCurve() {
  const canvas = document.getElementById('mh-loss-canvas');
  if (!canvas || !mhState) return;
  const Wd = 500, Hd = 240;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Wd * dpr; canvas.height = Hd * dpr;
  canvas.style.width = Wd + 'px'; canvas.style.height = Hd + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, Wd, Hd);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, Wd, Hd);
  const m = { l: 56, r: 14, t: 14, b: 30 };
  const px = Wd - m.l - m.r, py = Hd - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  if (mhState.losses.length === 0) {
    ctx.fillStyle = '#9a917f';
    ctx.font = '13px Manrope';
    ctx.textAlign = 'center';
    ctx.fillText('Press Start training to begin.', m.l + px / 2, m.t + py / 2);
    return;
  }
  const logL = mhState.losses.map((v) => Math.log10(Math.max(v, 1e-6)));
  let minLog = Math.floor(Math.min(...logL));
  let maxLog = Math.ceil(Math.max(...logL));
  if (minLog === maxLog) maxLog = minLog + 1;
  const range = maxLog - minLog;
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let v = minLog; v <= maxLog; v++) {
    const y = m.t + (1 - (v - minLog) / range) * py;
    ctx.fillText(`10^${v}`, m.l - 6, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  ctx.strokeStyle = '#2c6fb7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const N = mhState.losses.length;
  logL.forEach((y, i) => {
    const xx = m.l + (i / Math.max(1, N - 1)) * px;
    const yy = m.t + (1 - (y - minLog) / range) * py;
    if (i === 0) ctx.moveTo(xx, yy);
    else ctx.lineTo(xx, yy);
  });
  ctx.stroke();
}

function drawMHSpecCurve() {
  const canvas = document.getElementById('mh-spec-canvas');
  if (!canvas || !mhState) return;
  const Wd = 500, Hd = 240;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Wd * dpr; canvas.height = Hd * dpr;
  canvas.style.width = Wd + 'px'; canvas.style.height = Hd + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, Wd, Hd);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, Wd, Hd);
  const m = { l: 56, r: 14, t: 14, b: 30 };
  const px = Wd - m.l - m.r, py = Hd - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  ctx.fillStyle = '#9a917f';
  ctx.font = '11px IBM Plex Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = i / 4;
    const y = m.t + (1 - v) * py;
    ctx.fillText(v.toFixed(2), m.l - 4, y + 3);
    ctx.strokeStyle = '#f0ebe1';
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + px, y); ctx.stroke();
  }
  if (mhState.spec.length === 0) return;
  const N = mhState.spec.length;
  function plot(arr, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const xx = m.l + (i / Math.max(1, N - 1)) * px;
      const yy = m.t + (1 - Math.max(0, Math.min(1, v))) * py;
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
  }
  plot(mhState.spec.map((s) => s.a), '#2c6fb7');
  plot(mhState.spec.map((s) => s.b), '#d9622b');
  ctx.fillStyle = '#3b342b';
  ctx.font = '11px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('Head A on type-A token', m.l + 8, m.t + 14);
  ctx.fillStyle = '#d9622b';
  ctx.fillText('Head B on type-B token', m.l + 220, m.t + 14);
}

function mhLoop() {
  if (!mhState || !mhState.running) return;
  for (let i = 0; i < 4; i++) mhTrainStep(16);
  renderMH();
  mhRAF = requestAnimationFrame(mhLoop);
}

function wireMHControls() {
  const tog = document.getElementById('mh-toggle');
  const reset = document.getElementById('mh-reset');
  const lr = document.getElementById('mh-lr');
  const sh = document.getElementById('mh-singlehead');
  if (lr) lr.addEventListener('input', () => {
    if (!mhState) initMHState();
    mhState.lr = parseFloat(lr.value);
    document.getElementById('mh-lr-val').textContent = mhState.lr.toFixed(3);
  });
  if (sh) sh.addEventListener('change', () => {
    if (!mhState) initMHState();
    mhState.singleHead = sh.checked;
    renderMH();
  });
  if (tog) tog.addEventListener('click', () => {
    if (!mhState) initMHState();
    mhState.running = !mhState.running;
    tog.textContent = mhState.running ? 'Pause' : 'Start training';
    if (mhState.running) mhLoop();
    else if (mhRAF) cancelAnimationFrame(mhRAF);
  });
  if (reset) reset.addEventListener('click', () => {
    if (mhRAF) cancelAnimationFrame(mhRAF);
    initMHState();
    if (tog) tog.textContent = 'Start training';
    renderMH();
  });
}

function renderExtraMath() {
  if (!window.katex) return;
  const blocks = {
    'math-qkv-projection':
      'q_i = x_i\\, W_Q,\\qquad k_i = x_i\\, W_K,\\qquad v_i = x_i\\, W_V',
    'math-attention-full':
      '\\mathrm{Attn}(Q, K, V) = \\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V',
    'math-pe':
      '\\mathrm{PE}_{(t, 2i)} = \\sin\\!\\left(\\frac{t}{10000^{2i/d}}\\right),\\quad \\mathrm{PE}_{(t, 2i+1)} = \\cos\\!\\left(\\frac{t}{10000^{2i/d}}\\right)',
    'math-train-task':
      '\\mathcal{L} = \\tfrac{1}{2}\\,\\bigl\\lVert\\, y_{\\text{query}} - v_{\\text{matching candidate}} \\,\\bigr\\rVert^2'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
