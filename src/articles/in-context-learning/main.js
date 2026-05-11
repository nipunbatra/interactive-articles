// ============================================================
// In-Context Learning — induction-head copy task.
// We don't train a real Transformer; instead we hand-craft the
// induction-head circuit and simulate its behaviour on (key, value)
// pair prompts. The point is to expose the mechanism, not the
// optimisation.
// ============================================================

const STATE = { k: 3, V: 6, prompt: null, attn: null, accuracyHistory: null };

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

const TOKENS = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ'];

function makePrompt(k, V) {
  // k (key, value) pairs + 1 query key.
  // Build sequence: K1 V1 K2 V2 ... Kk Vk QueryKey ?
  // QueryKey is chosen from one of K1..Kk so the answer is well-defined.
  const keys = [], vals = [], seq = [];
  const used = new Set();
  for (let i = 0; i < k; i++) {
    let kIdx, vIdx;
    do { kIdx = Math.floor(Math.random() * V); } while (used.has(kIdx));
    used.add(kIdx);
    vIdx = Math.floor(Math.random() * V);
    keys.push(kIdx); vals.push(vIdx);
    seq.push({ tok: kIdx, role: 'k' });
    seq.push({ tok: vIdx, role: 'v' });
  }
  const qIdx = Math.floor(Math.random() * k);
  seq.push({ tok: keys[qIdx], role: 'q' });
  return { seq, query: keys[qIdx], answer: vals[qIdx], keys, vals, qIdx };
}

// Simulated induction-head: at the last position, attend strongly
// to previous occurrences of the same token (these are the "key"
// positions in the prompt); then predict the token immediately
// following that key (i.e., the value).
function simulateInductionHead(seq) {
  const T = seq.length;
  const queryPos = T - 1;
  const queryTok = seq[queryPos].tok;
  // Layer-1 previous-token head: encodes "what came before me"
  // We use the property that at position t, the representation
  // remembers seq[t-1].
  // Layer-2 induction head: at queryPos, compute logits over earlier
  // positions where seq[t-1] (the position's "previous token" info)
  // equals queryTok. Among those, attend most to the position
  // immediately after such a match — that's the value.
  const logits = new Array(T).fill(-Infinity);
  // The induction head wants to point at the value tokens whose
  // preceding token == queryTok.
  for (let t = 1; t < T - 1; t++) {
    if (seq[t - 1].tok === queryTok) {
      logits[t] = 5;
    } else {
      logits[t] = -1;
    }
  }
  // Softmax
  let m = -Infinity; for (const v of logits) if (Number.isFinite(v) && v > m) m = v;
  const exps = logits.map((v) => Number.isFinite(v) ? Math.exp(v - m) : 0);
  const s = exps.reduce((a, b) => a + b, 0) || 1;
  const attn = exps.map((e) => e / s);
  // Predicted token = weighted vote on seq[i].tok by attn[i]
  const vocab = {};
  for (let i = 0; i < T; i++) {
    if (attn[i] < 1e-6) continue;
    vocab[seq[i].tok] = (vocab[seq[i].tok] || 0) + attn[i];
  }
  let bestTok = 0, bestScore = -Infinity;
  Object.keys(vocab).forEach((k) => {
    const s = vocab[k];
    if (s > bestScore) { bestScore = s; bestTok = parseInt(k); }
  });
  return { attn, predicted: bestTok };
}

function computeAccuracyAcrossK() {
  // Run T trials at each k, plot accuracy
  const Ks = [];
  const accs = [];
  for (let k = 1; k <= 12; k++) {
    let correct = 0;
    const trials = 100;
    for (let t = 0; t < trials; t++) {
      const p = makePrompt(k, STATE.V);
      const { predicted } = simulateInductionHead(p.seq);
      if (predicted === p.answer) correct++;
    }
    Ks.push(k);
    accs.push(correct / trials);
  }
  return { Ks, accs };
}

// ---------- Render ----------
function renderPrompt() {
  const el = document.getElementById('icl-prompt');
  if (!el) return;
  const p = STATE.prompt;
  let html = '<div class="icl-tokens">';
  for (let i = 0; i < p.seq.length; i++) {
    const item = p.seq[i];
    const role = item.role;
    const tok = TOKENS[item.tok];
    html += `<span class="icl-tok icl-tok-${role}"><span class="icl-tok-label">${role}</span><span class="icl-tok-glyph">${tok}</span></span>`;
  }
  html += `<span class="icl-tok icl-tok-pred"><span class="icl-tok-label">?</span><span class="icl-tok-glyph">${TOKENS[STATE.attn.predicted]}</span></span>`;
  html += '</div>';
  const correct = STATE.attn.predicted === p.answer;
  html += `<div class="icl-pred-result ${correct ? 'is-correct' : 'is-wrong'}">predicted <strong>${TOKENS[STATE.attn.predicted]}</strong> · correct answer <strong>${TOKENS[p.answer]}</strong> · ${correct ? '✓' : '✗'}</div>`;
  el.innerHTML = html;
}

function renderAttnHeatmap() {
  const el = document.getElementById('icl-attn');
  if (!el) return;
  const T = STATE.prompt.seq.length;
  const cells = STATE.attn.attn;
  const labels = STATE.prompt.seq.map((x, i) => `${i + 1}\n${TOKENS[x.tok]}`);
  el.style.display = 'grid';
  el.style.gridTemplateColumns = `repeat(${T}, minmax(0, 1fr))`;
  el.style.gap = '2px';
  el.style.padding = '4px';
  el.style.background = '#f7f4ec';
  el.style.border = '1px solid var(--border)';
  el.style.borderRadius = '8px';
  let html = '';
  for (let i = 0; i < T; i++) {
    const w = cells[i] || 0;
    const alpha = 0.1 + 0.7 * w;
    const isLast = i === T - 1;
    html += `<div class="icl-cell ${isLast ? 'icl-cell-q' : ''}" style="background: rgba(217, 98, 43, ${alpha});">
      <div class="icl-cell-tok">${TOKENS[STATE.prompt.seq[i].tok]}</div>
      <div class="icl-cell-w">${(w * 100).toFixed(0)}%</div>
    </div>`;
  }
  el.innerHTML = html;
}

function renderCurve() {
  const canvas = document.getElementById('icl-curve');
  if (!canvas) return;
  const W = 880, H = 280;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 60, r: 14, t: 18, b: 36 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const { Ks, accs } = STATE.accuracyHistory || computeAccuracyAcrossK();
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
  ctx.textAlign = 'center';
  for (let i = 0; i < Ks.length; i++) {
    const x = m.l + (i / (Ks.length - 1)) * px;
    ctx.fillText(Ks[i].toString(), x, m.t + py + 16);
  }
  ctx.strokeStyle = '#2c6fb7'; ctx.lineWidth = 2;
  ctx.beginPath();
  accs.forEach((a, i) => {
    const x = m.l + (i / (Ks.length - 1)) * px;
    const y = m.t + (1 - a) * py;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // current k marker
  const ki = Ks.indexOf(STATE.k);
  if (ki >= 0) {
    const x = m.l + (ki / (Ks.length - 1)) * px;
    ctx.strokeStyle = '#d9622b';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + py); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.fillStyle = '#3b342b';
  ctx.font = '12px Manrope';
  ctx.textAlign = 'center';
  ctx.fillText('k (number of in-context examples)', m.l + px / 2, H - 8);
}

function refresh() {
  STATE.prompt = makePrompt(STATE.k, STATE.V);
  STATE.attn = simulateInductionHead(STATE.prompt.seq);
  STATE.accuracyHistory = computeAccuracyAcrossK();
  renderPrompt();
  renderAttnHeatmap();
  renderCurve();
  const last = STATE.accuracyHistory.accs[STATE.k - 1];
  document.getElementById('icl-acc').textContent = `${(last * 100).toFixed(1)}% (k=${STATE.k})`;
}

function wire() {
  document.getElementById('icl-k').addEventListener('input', (e) => {
    STATE.k = parseInt(e.target.value, 10);
    document.getElementById('icl-k-val').textContent = STATE.k;
    refresh();
  });
  document.getElementById('icl-V').addEventListener('input', (e) => {
    STATE.V = parseInt(e.target.value, 10);
    document.getElementById('icl-V-val').textContent = STATE.V;
    refresh();
  });
  document.getElementById('icl-newprompt').addEventListener('click', refresh);
  refresh();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-induction':
      '\\text{For query token }q\\text{ at the last position, attend to position }t\\text{ where }\\mathrm{seq}[t-1] = q,\\;\\text{and predict }\\mathrm{seq}[t]\\text{ (the token after the match).}',
    'math-icl-gd':
      'p(y \\mid x_*, \\{(x_i, y_i)\\}_i) \\;\\approx\\; \\mathcal{N}\\!\\bigl(x_*^\\top \\hat w_k,\\; \\sigma^2\\bigr),\\quad \\hat w_k = \\hat w_{k-1} - \\eta \\,\\nabla_w \\sum_i (y_i - x_i^\\top \\hat w_{k-1})^2'
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
