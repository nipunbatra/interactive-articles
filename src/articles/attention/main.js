const words = ["The", "river", "bank"];

// Initial vector states
let vectors = {
  "q_bank": { x: 0.8, y: 0.2 },
  "k_The": { x: -0.5, y: 0.5 },
  "k_river": { x: 0.7, y: 0.4 },
  "k_bank": { x: 0.2, y: 0.9 },
  "v_The": { x: -0.2, y: -0.2 },
  "v_river": { x: 0.9, y: 0.1 },
  "v_bank": { x: 0.1, y: 0.8 }
};

let dragging = null;

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-dot-product': ['\\text{Score} = Q \\cdot K = (q_x \\times k_x) + (q_y \\times k_y)', true],
    'math-softmax': ['\\text{Weight}_i = \\frac{e^{\\text{Score}_i}}{\\sum e^{\\text{Score}_j}}', true],
    'math-final-v': ['V_{\\text{final}} = (W_{\\text{The}} \\times V_{\\text{The}}) + (W_{\\text{river}} \\times V_{\\text{river}}) + (W_{\\text{bank}} \\times V_{\\text{bank}})', true],
  };
  Object.keys(blocks).forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    try {
      katex.render(blocks[id][0], el, { displayMode: blocks[id][1], throwOnError: false });
    } catch (_) {}
  });
}

function initCanvas() {
  const canvas = document.getElementById('qkCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  // Setup logical scaling
  const logicalWidth = 800;
  const logicalHeight = 400;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = logicalWidth * dpr;
  canvas.height = logicalHeight * dpr;
  canvas.style.width = logicalWidth + 'px';
  canvas.style.height = logicalHeight + 'px';
  ctx.scale(dpr, dpr);
  
  const cx = logicalWidth / 2;
  const cy = logicalHeight / 2;
  const scale = 150; // pixels per unit

  function toScreen(vec) {
    return { x: cx + vec.x * scale, y: cy - vec.y * scale };
  }
  function fromScreen(x, y) {
    return { x: (x - cx) / scale, y: (cy - y) / scale };
  }

  function draw() {
    ctx.clearRect(0, 0, logicalWidth, logicalHeight);
    
    // Draw Grid
    ctx.strokeStyle = '#f0ebe1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, logicalHeight);
    ctx.moveTo(0, cy); ctx.lineTo(logicalWidth, cy);
    ctx.stroke();

    // Draw unit circle
    ctx.beginPath();
    ctx.arc(cx, cy, scale, 0, Math.PI * 2);
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Keys
    const ks = [
      { id: "k_The", label: "K(The)", color: "#2c6fb7" },
      { id: "k_river", label: "K(river)", color: "#2c6fb7" },
      { id: "k_bank", label: "K(bank)", color: "#2c6fb7" }
    ];
    ks.forEach(k => {
      const v = vectors[k.id];
      const p = toScreen(v);
      drawVector(cx, cy, p.x, p.y, k.color, k.label);
    });

    // Draw Query
    const q = vectors["q_bank"];
    const qp = toScreen(q);
    drawVector(cx, cy, qp.x, qp.y, "#d9622b", "Q(bank)");
    
    updateTables();
  }

  function drawVector(x0, y0, x1, y1, color, label) {
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Draw arrow head
    const angle = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - 10 * Math.cos(angle - Math.PI/6), y1 - 10 * Math.sin(angle - Math.PI/6));
    ctx.lineTo(x1 - 10 * Math.cos(angle + Math.PI/6), y1 - 10 * Math.sin(angle + Math.PI/6));
    ctx.fillStyle = color;
    ctx.fill();

    // Draw label
    ctx.font = "bold 14px Manrope, sans-serif";
    ctx.fillText(label, x1 + 10, y1);
    
    // Draw grab dot
    ctx.beginPath();
    ctx.arc(x1, y1, 8, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    // Check hit
    for (let id of ["q_bank", "k_The", "k_river", "k_bank"]) {
      const p = toScreen(vectors[id]);
      const dist = Math.hypot(p.x - mx, p.y - my);
      if (dist < 15) {
        dragging = id;
        return;
      }
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const rect = canvas.getBoundingClientRect();
    const mx = Math.max(0, Math.min(logicalWidth, e.clientX - rect.left));
    const my = Math.max(0, Math.min(logicalHeight, e.clientY - rect.top));
    const v = fromScreen(mx, my);
    
    // Normalize to unit vector for simplicity
    const len = Math.hypot(v.x, v.y) || 1;
    vectors[dragging] = { x: v.x / len, y: v.y / len };
    draw();
  });

  window.addEventListener('mouseup', () => { dragging = null; });
  
  draw();
}

function updateTables() {
  const q = vectors["q_bank"];
  const scores = {};
  words.forEach(w => {
    const k = vectors["k_" + w];
    scores[w] = (q.x * k.x) + (q.y * k.y);
  });

  // Raw Scores Table
  const rawTbody = document.querySelector('#rawScoresTable tbody');
  if (rawTbody) {
    rawTbody.innerHTML = words.map(w => `
      <tr>
        <td><strong>${w}</strong></td>
        <td>${scores[w].toFixed(2)}</td>
      </tr>
    `).join('');
  }

  // Softmax
  let sumExp = 0;
  words.forEach(w => sumExp += Math.exp(scores[w]));
  
  const weights = {};
  words.forEach(w => weights[w] = Math.exp(scores[w]) / sumExp);

  // Attention Table
  const attnTbody = document.querySelector('#attentionTable tbody');
  if (attnTbody) {
    attnTbody.innerHTML = words.map(w => {
      const weightPct = (weights[w] * 100).toFixed(1);
      return `
        <tr>
          <td><strong>${w}</strong></td>
          <td>${scores[w].toFixed(2)}</td>
          <td>
            <div style="display:flex; align-items:center; gap:10px;">
              <span style="width:40px">${weightPct}%</span>
              <div style="background:var(--accent); height:10px; border-radius:5px; width:${weightPct}px;"></div>
            </div>
          </td>
        </tr>
      `;
    }).join('');
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