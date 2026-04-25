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

// ---------- Boot ----------
function init() {
  if (window.katex) {
    renderStaticMath();
  } else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', () => {
      renderStaticMath();
      updateLiveMath();
    });
  }

  document.querySelectorAll('#scenario-buttons [data-scenario]').forEach((btn) => {
    btn.addEventListener('click', () => loadScenario(btn.dataset.scenario));
  });

  installQKDrag();
  installValueDrag();
  loadScenario('riverBank');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
