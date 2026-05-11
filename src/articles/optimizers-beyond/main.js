// ============================================================
// Beyond Gradient Descent — five optimisers race on a 2D loss.
// All trajectories rendered live with simple analytic gradients/Hessians.
// ============================================================

const X_MIN = -2.5, X_MAX = 2.5;
const SURFACE = {
  quadratic: {
    f: (x, y) => 0.5 * (4 * x * x + 0.4 * y * y),
    g: (x, y) => [4 * x, 0.4 * y],
    h: (x, y) => [[4, 0], [0, 0.4]],
    optimum: [0, 0],
    init: [-2, 1.8]
  },
  rosenbrock: {
    f: (x, y) => Math.pow(1 - x, 2) + 5 * Math.pow(y - x * x, 2),
    g: (x, y) => [-2 * (1 - x) - 20 * (y - x * x) * x, 10 * (y - x * x)],
    h: (x, y) => [[2 - 20 * (y - 3 * x * x), -20 * x], [-20 * x, 10]],
    optimum: [1, 1],
    init: [-1.5, 1.6]
  },
  bowls: {
    f: (x, y) => 0.5 * Math.pow((x + 1.2), 2) + 0.5 * (y * y) + 0.6 * Math.exp(-((x - 1.0) * (x - 1.0) + (y - 0.4) * (y - 0.4))) * (-2),
    g: (x, y) => {
      const dx = (x + 1.2);
      const dy = y;
      const e = Math.exp(-((x - 1.0) * (x - 1.0) + (y - 0.4) * (y - 0.4)));
      return [dx + 1.2 * (x - 1.0) * 2 * e, dy + 1.2 * (y - 0.4) * 2 * e];
    },
    h: (x, y) => [[1, 0], [0, 1]], // approximate
    optimum: [-1.2, 0],
    init: [1.6, -1.2]
  }
};

const OPTIMS = ['newton', 'bfgs', 'natural', 'cmaes', 'spsa'];
const STATE = {
  surface: 'rosenbrock',
  step: 0,
  trajectories: {}
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
function inv2(M) {
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  if (Math.abs(det) < 1e-9) return [[1, 0], [0, 1]];
  return [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]];
}

function reset() {
  const surf = SURFACE[STATE.surface];
  STATE.step = 0;
  STATE.trajectories = {};
  OPTIMS.forEach((k) => {
    STATE.trajectories[k] = {
      pts: [surf.init.slice()],
      // BFGS state
      H: [[1, 0], [0, 1]],
      // CMA-ES state
      sigma: 1.0,
      // gradient-free spsa step
      a: 0.4
    };
  });
}

function stepOnce() {
  const surf = SURFACE[STATE.surface];
  OPTIMS.forEach((k) => {
    const tr = STATE.trajectories[k];
    const cur = tr.pts[tr.pts.length - 1];
    const [x, y] = cur;
    let next;
    if (k === 'newton') {
      const g = surf.g(x, y);
      const H = surf.h(x, y);
      // damp Hessian to keep PD
      const Hd = [[H[0][0] + 0.05, H[0][1]], [H[1][0], H[1][1] + 0.05]];
      const Hi = inv2(Hd);
      const dx = Hi[0][0] * g[0] + Hi[0][1] * g[1];
      const dy = Hi[1][0] * g[0] + Hi[1][1] * g[1];
      next = [x - 0.7 * dx, y - 0.7 * dy];
    } else if (k === 'bfgs') {
      const g = surf.g(x, y);
      const H = tr.H;
      // search direction p = -H g
      const pdx = -(H[0][0] * g[0] + H[0][1] * g[1]);
      const pdy = -(H[1][0] * g[0] + H[1][1] * g[1]);
      const lr = 0.3;
      next = [x + lr * pdx, y + lr * pdy];
      // update H via BFGS rank-2 update (s, y diff)
      const s = [next[0] - x, next[1] - y];
      const gNext = surf.g(next[0], next[1]);
      const yvec = [gNext[0] - g[0], gNext[1] - g[1]];
      const sy = s[0] * yvec[0] + s[1] * yvec[1];
      if (Math.abs(sy) > 1e-9) {
        const rho = 1 / sy;
        // H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
        function outer(a, b) { return [[a[0]*b[0], a[0]*b[1]], [a[1]*b[0], a[1]*b[1]]]; }
        function eye() { return [[1,0],[0,1]]; }
        function sub(A, B) { return [[A[0][0]-B[0][0], A[0][1]-B[0][1]], [A[1][0]-B[1][0], A[1][1]-B[1][1]]]; }
        function mul(A, B) {
          return [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
                  [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]];
        }
        function add(A, B) { return [[A[0][0]+B[0][0], A[0][1]+B[0][1]], [A[1][0]+B[1][0], A[1][1]+B[1][1]]]; }
        function scale(A, k) { return [[A[0][0]*k, A[0][1]*k], [A[1][0]*k, A[1][1]*k]]; }
        const sy_outer = outer(s, yvec);
        const ys_outer = outer(yvec, s);
        const ss_outer = outer(s, s);
        const I = eye();
        const left = sub(I, scale(sy_outer, rho));
        const right = sub(I, scale(ys_outer, rho));
        tr.H = add(mul(mul(left, H), right), scale(ss_outer, rho));
      }
    } else if (k === 'natural') {
      // Approximate Fisher with scaled gradient outer product (toy)
      const g = surf.g(x, y);
      const F = [[g[0] * g[0] + 0.05, g[0] * g[1]], [g[0] * g[1], g[1] * g[1] + 0.05]];
      const Fi = inv2(F);
      const dx = Fi[0][0] * g[0] + Fi[0][1] * g[1];
      const dy = Fi[1][0] * g[0] + Fi[1][1] * g[1];
      const norm = Math.hypot(dx, dy) || 1;
      next = [x - 0.6 * dx / norm * 0.3, y - 0.6 * dy / norm * 0.3];
    } else if (k === 'cmaes') {
      // mini CMA-ES with population size 6
      const popN = 6;
      const samples = [];
      for (let i = 0; i < popN; i++) {
        const sx = x + tr.sigma * randn();
        const sy = y + tr.sigma * randn();
        samples.push({ x: sx, y: sy, f: surf.f(sx, sy) });
      }
      samples.sort((a, b) => a.f - b.f);
      const top = samples.slice(0, 3);
      let mx = 0, my = 0;
      top.forEach((s) => { mx += s.x; my += s.y; });
      mx /= top.length; my /= top.length;
      // shrink sigma
      tr.sigma *= 0.92;
      next = [mx, my];
    } else if (k === 'spsa') {
      const c = 0.05;
      const dx = Math.random() < 0.5 ? -1 : 1;
      const dy = Math.random() < 0.5 ? -1 : 1;
      const fp = surf.f(x + c * dx, y + c * dy);
      const fm = surf.f(x - c * dx, y - c * dy);
      const ghx = (fp - fm) / (2 * c * dx);
      const ghy = (fp - fm) / (2 * c * dy);
      next = [x - tr.a * ghx, y - tr.a * ghy];
      tr.a *= 0.985;
    }
    // Clamp
    next[0] = Math.max(X_MIN, Math.min(X_MAX, next[0]));
    next[1] = Math.max(X_MIN, Math.min(X_MAX, next[1]));
    tr.pts.push(next);
    if (tr.pts.length > 300) tr.pts.shift();
  });
  STATE.step++;
}

// ---------- Render ----------
function renderCanvas() {
  const canvas = document.getElementById('op-canvas');
  if (!canvas) return;
  const W = 880, H = 440;
  const ctx = setupCanvas(canvas, W, H);
  ctx.fillStyle = '#fdfcf9'; ctx.fillRect(0, 0, W, H);
  const m = { l: 50, r: 220, t: 14, b: 30 };
  const px = W - m.l - m.r, py = H - m.t - m.b;
  ctx.strokeStyle = '#e2d8c6'; ctx.strokeRect(m.l, m.t, px, py);
  const surf = SURFACE[STATE.surface];
  // Loss heatmap
  const step = 4;
  let lo = Infinity, hi = -Infinity;
  const cells = [];
  for (let yi = 0; yi < py; yi += step) {
    for (let xi = 0; xi < px; xi += step) {
      const x = X_MIN + (X_MAX - X_MIN) * (xi / px);
      const y = X_MAX - (X_MAX - X_MIN) * (yi / py);
      const v = surf.f(x, y);
      cells.push({ xi, yi, v });
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
  }
  const range = Math.max(1e-6, hi - lo);
  cells.forEach(({ xi, yi, v }) => {
    const t = Math.min(1, (v - lo) / range);
    const r = Math.round(253 - 100 * t);
    const g = Math.round(252 - 90 * t);
    const b = Math.round(249 - 60 * t);
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.fillRect(m.l + xi, m.t + yi, step, step);
  });
  // Trajectories
  const colors = { newton: '#2c6fb7', bfgs: '#1e7770', natural: '#9b59b6', cmaes: '#d9622b', spsa: '#7c4d2a' };
  const labels = { newton: "Newton", bfgs: "BFGS", natural: "Natural grad", cmaes: "CMA-ES", spsa: "SPSA (free)" };
  function project(p) {
    return [m.l + (p[0] - X_MIN) / (X_MAX - X_MIN) * px, m.t + (X_MAX - p[1]) / (X_MAX - X_MIN) * py];
  }
  OPTIMS.forEach((k) => {
    const tr = STATE.trajectories[k];
    ctx.strokeStyle = colors[k];
    ctx.lineWidth = 2;
    ctx.beginPath();
    tr.pts.forEach((p, i) => {
      const [px2, py2] = project(p);
      if (i === 0) ctx.moveTo(px2, py2); else ctx.lineTo(px2, py2);
    });
    ctx.stroke();
    const last = tr.pts[tr.pts.length - 1];
    const [px2, py2] = project(last);
    ctx.beginPath();
    ctx.arc(px2, py2, 4, 0, Math.PI * 2);
    ctx.fillStyle = colors[k];
    ctx.fill();
  });
  // Optimum marker
  const opt = surf.optimum;
  const [px2, py2] = project(opt);
  ctx.strokeStyle = '#1a1815';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px2 - 6, py2); ctx.lineTo(px2 + 6, py2);
  ctx.moveTo(px2, py2 - 6); ctx.lineTo(px2, py2 + 6);
  ctx.stroke();
  // Legend
  ctx.fillStyle = '#3b342b';
  ctx.font = 'bold 13px Manrope';
  ctx.textAlign = 'left';
  ctx.fillText('current loss values', m.l + px + 16, m.t + 14);
  ctx.font = '12px Manrope';
  let ly = m.t + 36;
  OPTIMS.forEach((k) => {
    const tr = STATE.trajectories[k];
    const last = tr.pts[tr.pts.length - 1];
    const v = surf.f(last[0], last[1]);
    ctx.strokeStyle = colors[k]; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(m.l + px + 16, ly - 4); ctx.lineTo(m.l + px + 30, ly - 4); ctx.stroke();
    ctx.fillStyle = '#3b342b';
    ctx.fillText(`${labels[k]}: ${v.toFixed(3)}`, m.l + px + 36, ly);
    ly += 22;
  });
  document.getElementById('op-step-count').textContent = STATE.step;
}

function wire() {
  reset();
  document.getElementById('op-surface').addEventListener('change', (e) => {
    STATE.surface = e.target.value;
    reset();
    renderCanvas();
  });
  document.getElementById('op-step').addEventListener('click', () => { stepOnce(); renderCanvas(); });
  document.getElementById('op-step50').addEventListener('click', () => {
    for (let i = 0; i < 50; i++) stepOnce();
    renderCanvas();
  });
  document.getElementById('op-reset').addEventListener('click', () => { reset(); renderCanvas(); });
  renderCanvas();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-newton-taylor':
      'f(w_t + \\Delta w) \\approx f(w_t) + \\nabla f^\\top \\Delta w + \\tfrac{1}{2} \\Delta w^\\top H \\Delta w',
    'math-bfgs':
      'B_{t+1} = \\left(I - \\frac{s_t y_t^\\top}{y_t^\\top s_t}\\right) B_t \\left(I - \\frac{y_t s_t^\\top}{y_t^\\top s_t}\\right) + \\frac{s_t s_t^\\top}{y_t^\\top s_t}',
    'math-natural':
      'w_{t+1} = w_t - \\eta\\, F(w_t)^{-1} \\nabla L(w_t),\\qquad F = \\mathbb{E}_x\\,\\nabla \\log p_\\theta(x)\\,\\nabla \\log p_\\theta(x)^\\top',
    'math-spsa':
      '\\hat g_t = \\frac{f(w_t + c_t\\Delta_t) - f(w_t - c_t\\Delta_t)}{2c_t}\\, \\Delta_t^{-1}\\qquad (\\Delta_t^{-1} \\text{ elementwise})'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

function boot() {
  wire();
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
