// ============================================================
// Concept Bottleneck Models — synthetic 4-class shape problem.
// ============================================================

const CONCEPTS = [
  { key: 'dark',    label: 'is-dark' },
  { key: 'large',   label: 'is-large' },
  { key: 'striped', label: 'has-stripes' },
  { key: 'warm',    label: 'is-warm' }
];
const CLASS_NAMES = ['Class A', 'Class B', 'Class C', 'Class D'];

// Class rule: deterministic boolean recipe
function classFromConcepts(c) {
  // c: {dark, large, striped, warm} all booleans
  if (c.dark && c.warm) return 0;        // A
  if (c.large && !c.striped) return 1;   // B
  if (c.striped && !c.dark) return 2;    // C
  return 3;                              // D
}

const STATE = {
  shape: null,
  concepts: null,
  override: null
};

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

function newShape() {
  const dark = Math.random() < 0.5;
  const large = Math.random() < 0.5;
  const striped = Math.random() < 0.5;
  const warm = Math.random() < 0.5;
  return { dark, large, striped, warm };
}

// "Bottleneck" predictor — for the demo we observe the actual concept
// values with small noise (a perfectly-trained CBM); user can override.
function predictConcepts(shape) {
  // p in (0, 1) close to the truth
  return {
    dark: shape.dark ? 0.85 + 0.1 * Math.random() : 0.15 + 0.1 * Math.random(),
    large: shape.large ? 0.82 + 0.1 * Math.random() : 0.18 + 0.1 * Math.random(),
    striped: shape.striped ? 0.88 + 0.1 * Math.random() : 0.12 + 0.1 * Math.random(),
    warm: shape.warm ? 0.86 + 0.1 * Math.random() : 0.14 + 0.1 * Math.random()
  };
}

function applyOverride(probs, override) {
  const out = Object.assign({}, probs);
  Object.keys(override).forEach((k) => {
    if (override[k] != null) out[k] = override[k] ? 0.99 : 0.01;
  });
  return out;
}

function predictedClassFromConcepts(probs) {
  // Hard-thresholded class rule, mirroring the truth rule.
  const c = {
    dark: probs.dark > 0.5,
    large: probs.large > 0.5,
    striped: probs.striped > 0.5,
    warm: probs.warm > 0.5
  };
  return classFromConcepts(c);
}

// ---------- Render ----------
function drawImage() {
  const canvas = document.getElementById('cb-img');
  if (!canvas) return;
  const W = 220, H = 220;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const sh = STATE.shape;
  const cx = W / 2, cy = H / 2;
  const size = sh.large ? 80 : 50;
  // base color
  let r, g, b;
  if (sh.warm) { r = 220; g = 100; b = 60; } else { r = 70; g = 130; b = 200; }
  if (sh.dark) { r *= 0.6; g *= 0.6; b *= 0.6; }
  ctx.fillStyle = `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
  ctx.beginPath();
  ctx.arc(cx, cy, size, 0, Math.PI * 2);
  ctx.fill();
  if (sh.striped) {
    ctx.strokeStyle = 'rgba(0,0,0,0.45)';
    ctx.lineWidth = 4;
    for (let i = -size; i <= size; i += 12) {
      ctx.beginPath();
      ctx.moveTo(cx + i, cy - size); ctx.lineTo(cx + i + 12, cy + size);
      ctx.stroke();
    }
  }
  ctx.strokeStyle = '#1a1815';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, size, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = '#e2d8c6';
  ctx.strokeRect(0, 0, W, H);
}

function renderConcepts() {
  const root = document.getElementById('cb-concepts');
  if (!root) return;
  const probs = applyOverride(STATE.concepts, STATE.override);
  let html = '';
  CONCEPTS.forEach((c) => {
    const v = probs[c.key];
    const isOverride = STATE.override[c.key] != null;
    const w = Math.round(v * 100);
    html += `
      <div class="cb-concept" data-key="${c.key}">
        <div class="cb-concept-label">${c.label}${isOverride ? ' <span class="cb-override-tag">overridden</span>' : ''}</div>
        <div class="cb-concept-bar">
          <div class="cb-concept-fill" style="width:${w}%"></div>
        </div>
        <div class="cb-concept-buttons">
          <button data-action="off" data-key="${c.key}" class="${STATE.override[c.key] === false ? 'is-active' : ''}">force 0</button>
          <button data-action="on" data-key="${c.key}" class="${STATE.override[c.key] === true ? 'is-active' : ''}">force 1</button>
          <button data-action="clear" data-key="${c.key}">clear</button>
        </div>
      </div>
    `;
  });
  root.innerHTML = html;
  root.querySelectorAll('button').forEach((b) => {
    b.addEventListener('click', () => {
      const key = b.dataset.key;
      const action = b.dataset.action;
      if (action === 'on') STATE.override[key] = true;
      else if (action === 'off') STATE.override[key] = false;
      else STATE.override[key] = null;
      refreshAll();
    });
  });
}

function refreshAll() {
  drawImage();
  renderConcepts();
  const probs = applyOverride(STATE.concepts, STATE.override);
  const trueCls = classFromConcepts(STATE.shape);
  const predCls = predictedClassFromConcepts(probs);
  document.getElementById('cb-true').textContent = CLASS_NAMES[trueCls];
  document.getElementById('cb-pred').textContent = CLASS_NAMES[predCls];
  document.getElementById('cb-pred').style.color =
    (trueCls === predCls && Object.values(STATE.override).every((v) => v == null))
      ? '#1e7770' : '#c47c1f';
}

function reroll() {
  STATE.shape = newShape();
  STATE.concepts = predictConcepts(STATE.shape);
  STATE.override = { dark: null, large: null, striped: null, warm: null };
  refreshAll();
}

function wire() {
  reroll();
  document.getElementById('cb-reroll').addEventListener('click', reroll);
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-cb':
      'x \\xrightarrow{\\;g\\;} \\hat c \\in \\mathbb{R}^{|\\text{concepts}|} \\xrightarrow{\\;f\\;} \\hat y'
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
