function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-landscape': ['L(x, y) = \\frac{x^2}{20} + y^2', true],
    'math-sgd': ['W_{t+1} = W_t - \\alpha \\nabla L(W_t)', true],
    'math-momentum': ['V_{t+1} = \\beta V_t + \\nabla L(W_t) \\\\ W_{t+1} = W_t - \\alpha V_{t+1}', true],
    'math-adam': ['m_{t+1} = \\beta_1 m_t + (1-\\beta_1) \\nabla L(W_t) \\\\ v_{t+1} = \\beta_2 v_t + (1-\\beta_2) \\nabla L(W_t)^2 \\\\ W_{t+1} = W_t - \\alpha \\frac{m_{t+1}}{\\sqrt{v_{t+1}} + \\epsilon}', true],
  };
  Object.keys(blocks).forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    try {
      katex.render(blocks[id][0], el, { displayMode: blocks[id][1], throwOnError: false });
    } catch (_) {}
  });
}

// Optimization landscape function (Beale-like or simple ravine)
// f(x,y) = x^2 / 20 + y^2
function f(x, y) {
  return (x * x) / 20 + y * y;
}
function grad(x, y) {
  return { dx: x / 10, dy: 2 * y };
}

let animFrame;
let paths = { sgd: [], momentum: [], adam: [] };
let states = {};
let running = false;
let lr = 0.05;

function resetOptimizers(startX, startY) {
  paths = { sgd: [{x: startX, y: startY}], momentum: [{x: startX, y: startY}], adam: [{x: startX, y: startY}] };
  states = {
    sgd: { x: startX, y: startY },
    momentum: { x: startX, y: startY, vx: 0, vy: 0 },
    adam: { x: startX, y: startY, m_x: 0, m_y: 0, v_x: 0, v_y: 0, t: 0 }
  };
  running = true;
  if (animFrame) cancelAnimationFrame(animFrame);
  step();
}

function step() {
  if (!running) return;
  let allDone = true;
  
  // SGD
  let s = states.sgd;
  let g = grad(s.x, s.y);
  s.x -= lr * g.dx;
  s.y -= lr * g.dy;
  paths.sgd.push({x: s.x, y: s.y});
  if (Math.hypot(s.x, s.y) > 0.01) allDone = false;

  // Momentum
  let m = states.momentum;
  let gm = grad(m.x, m.y);
  m.vx = 0.9 * m.vx + lr * gm.dx;
  m.vy = 0.9 * m.vy + lr * gm.dy;
  m.x -= m.vx;
  m.y -= m.vy;
  paths.momentum.push({x: m.x, y: m.y});
  if (Math.hypot(m.x, m.y) > 0.01) allDone = false;

  // Adam
  let a = states.adam;
  a.t += 1;
  let ga = grad(a.x, a.y);
  a.m_x = 0.9 * a.m_x + 0.1 * ga.dx;
  a.m_y = 0.9 * a.m_y + 0.1 * ga.dy;
  a.v_x = 0.999 * a.v_x + 0.001 * (ga.dx * ga.dx);
  a.v_y = 0.999 * a.v_y + 0.001 * (ga.dy * ga.dy);
  let m_hat_x = a.m_x / (1 - Math.pow(0.9, a.t));
  let m_hat_y = a.m_y / (1 - Math.pow(0.9, a.t));
  let v_hat_x = a.v_x / (1 - Math.pow(0.999, a.t));
  let v_hat_y = a.v_y / (1 - Math.pow(0.999, a.t));
  a.x -= lr * 10 * m_hat_x / (Math.sqrt(v_hat_x) + 1e-8); // Boosted lr for adam for visual effect
  a.y -= lr * 10 * m_hat_y / (Math.sqrt(v_hat_y) + 1e-8);
  paths.adam.push({x: a.x, y: a.y});
  if (Math.hypot(a.x, a.y) > 0.01) allDone = false;

  drawCanvas();
  
  if (!allDone && paths.sgd.length < 500) {
    animFrame = requestAnimationFrame(step);
  } else {
    running = false;
  }
}

function initCanvas() {
  const canvas = document.getElementById('optCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  const logicalWidth = 800;
  const logicalHeight = 500;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = logicalWidth * dpr;
  canvas.height = logicalHeight * dpr;
  canvas.style.width = logicalWidth + 'px';
  canvas.style.height = logicalHeight + 'px';
  ctx.scale(dpr, dpr);
  
  const cx = logicalWidth / 2;
  const cy = logicalHeight / 2;
  const scale = 30; // pixels per unit space

  function toScreen(x, y) {
    return { x: cx + x * scale, y: cy - y * scale };
  }

  function drawBackground() {
    // Draw simple contour lines for f(x,y)
    ctx.fillStyle = '#fdfcf9';
    ctx.fillRect(0, 0, logicalWidth, logicalHeight);
    
    ctx.strokeStyle = 'rgba(0,0,0,0.05)';
    ctx.lineWidth = 1;
    for(let r = 1; r <= 15; r++) {
      ctx.beginPath();
      // Ellipse for contour
      const a = Math.sqrt(r * 20) * scale;
      const b = Math.sqrt(r) * scale;
      ctx.ellipse(cx, cy, a, b, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    
    // Draw global minimum star
    ctx.fillStyle = '#1a1815';
    ctx.font = '20px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('★', cx, cy);
  }

  window.drawCanvas = function() {
    drawBackground();
    
    // Draw paths
    const opts = [
      { id: 'sgd', color: '#ca5b2a' },
      { id: 'momentum', color: '#245b8f' },
      { id: 'adam', color: '#1e7770' }
    ];
    
    opts.forEach(opt => {
      const p = paths[opt.id];
      if (!p || p.length === 0) return;
      
      ctx.beginPath();
      const start = toScreen(p[0].x, p[0].y);
      ctx.moveTo(start.x, start.y);
      for(let i=1; i<p.length; i++) {
        const pt = toScreen(p[i].x, p[i].y);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.strokeStyle = opt.color;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw head
      const last = toScreen(p[p.length-1].x, p[p.length-1].y);
      ctx.beginPath();
      ctx.arc(last.x, last.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = opt.color;
      ctx.fill();
    });
  }

  drawBackground();

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    const startX = (mx - cx) / scale;
    const startY = (cy - my) / scale;
    
    resetOptimizers(startX, startY);
  });
  
  const lrSlider = document.getElementById('lr-slider');
  if (lrSlider) {
    lrSlider.addEventListener('input', (e) => {
      lr = parseFloat(e.target.value);
      document.getElementById('lr-val').textContent = lr.toFixed(3);
    });
  }
  
  const resetBtn = document.getElementById('reset-btn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      paths = { sgd: [], momentum: [], adam: [] };
      running = false;
      drawCanvas();
    });
  }
}

function init() {
  if (window.katex) {
    renderMath();
  } else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  initCanvas();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}