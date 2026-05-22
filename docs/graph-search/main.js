// ============================================================
// Graph Search: BFS / DFS / Dijkstra / A* / Greedy best-first on
// an editable grid maze.
// ============================================================

const COLS = 36, ROWS = 22;
const CELL_PX = 18; // logical CSS pixels per cell

// Cell types
const EMPTY = 0, WALL = 1, MUD = 2;

const STATE = {
  grid: null,             // ROWS x COLS of ints
  start: { x: 3, y: 11 },
  goal:  { x: 32, y: 11 },
  tool: 'wall',
  speed: 20,
  running: false,
  // Per-search overlay state for the main canvas:
  search: null            // { algo, parent[], gScore[], frontierSet, exploredSet, generator, finalPath, expanded }
};

function ix(x, y) { return y * COLS + x; }
function inBounds(x, y) { return x >= 0 && x < COLS && y >= 0 && y < ROWS; }

function newGrid() {
  return new Uint8Array(COLS * ROWS);
}

function cellCost(t) {
  if (t === MUD) return 5;
  return 1;
}

// ---------------- Min-heap (binary) ----------------
class MinHeap {
  constructor() { this.a = []; }
  size() { return this.a.length; }
  push(priority, value) {
    this.a.push({ p: priority, v: value });
    this._siftUp(this.a.length - 1);
  }
  pop() {
    if (!this.a.length) return null;
    const top = this.a[0];
    const last = this.a.pop();
    if (this.a.length) { this.a[0] = last; this._siftDown(0); }
    return top;
  }
  _siftUp(i) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.a[i].p < this.a[p].p) { [this.a[i], this.a[p]] = [this.a[p], this.a[i]]; i = p; }
      else break;
    }
  }
  _siftDown(i) {
    const n = this.a.length;
    while (true) {
      const l = 2 * i + 1, r = 2 * i + 2;
      let pick = i;
      if (l < n && this.a[l].p < this.a[pick].p) pick = l;
      if (r < n && this.a[r].p < this.a[pick].p) pick = r;
      if (pick === i) break;
      [this.a[i], this.a[pick]] = [this.a[pick], this.a[i]];
      i = pick;
    }
  }
}

// ---------------- Search generators ----------------
// Each yields once per "step" (one node expanded). The shared state
// in the closure tracks parent, gScore, frontier set, explored set.
// On completion, the path is reconstructed.

function* bfsGen(grid, start, goal) {
  const parent = new Int32Array(COLS * ROWS).fill(-1);
  const visited = new Uint8Array(COLS * ROWS);
  const frontier = new Set();
  const explored = new Set();
  const q = [start];
  visited[ix(start.x, start.y)] = 1;
  frontier.add(ix(start.x, start.y));

  while (q.length) {
    const cur = q.shift();
    const ci = ix(cur.x, cur.y);
    frontier.delete(ci);
    explored.add(ci);
    if (cur.x === goal.x && cur.y === goal.y) {
      yield { parent, frontier, explored, done: true, expanded: explored.size };
      return;
    }
    for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const nx = cur.x + dx, ny = cur.y + dy;
      if (!inBounds(nx, ny)) continue;
      if (grid[ix(nx, ny)] === WALL) continue;
      const ni = ix(nx, ny);
      if (visited[ni]) continue;
      visited[ni] = 1;
      parent[ni] = ci;
      frontier.add(ni);
      q.push({ x: nx, y: ny });
    }
    yield { parent, frontier, explored, done: false };
  }
  yield { parent, frontier, explored, done: true, expanded: explored.size, unreachable: true };
}

function* dfsGen(grid, start, goal) {
  const parent = new Int32Array(COLS * ROWS).fill(-1);
  const visited = new Uint8Array(COLS * ROWS);
  const frontier = new Set();
  const explored = new Set();
  const stack = [start];
  visited[ix(start.x, start.y)] = 1;
  frontier.add(ix(start.x, start.y));

  while (stack.length) {
    const cur = stack.pop();
    const ci = ix(cur.x, cur.y);
    frontier.delete(ci);
    explored.add(ci);
    if (cur.x === goal.x && cur.y === goal.y) {
      yield { parent, frontier, explored, done: true, expanded: explored.size };
      return;
    }
    // Reverse iteration order so the visual is a deterministic spiral
    for (const [dx, dy] of [[0,-1],[1,0],[0,1],[-1,0]]) {
      const nx = cur.x + dx, ny = cur.y + dy;
      if (!inBounds(nx, ny)) continue;
      if (grid[ix(nx, ny)] === WALL) continue;
      const ni = ix(nx, ny);
      if (visited[ni]) continue;
      visited[ni] = 1;
      parent[ni] = ci;
      frontier.add(ni);
      stack.push({ x: nx, y: ny });
    }
    yield { parent, frontier, explored, done: false };
  }
  yield { parent, frontier, explored, done: true, expanded: explored.size, unreachable: true };
}

function* dijkstraGen(grid, start, goal) {
  const N = COLS * ROWS;
  const parent = new Int32Array(N).fill(-1);
  const gScore = new Float64Array(N).fill(Infinity);
  const closed = new Uint8Array(N);
  const frontier = new Set();
  const explored = new Set();
  const pq = new MinHeap();
  gScore[ix(start.x, start.y)] = 0;
  pq.push(0, { x: start.x, y: start.y });
  frontier.add(ix(start.x, start.y));

  while (pq.size()) {
    const { v: cur, p: pri } = pq.pop();
    const ci = ix(cur.x, cur.y);
    if (closed[ci]) continue;
    if (pri > gScore[ci]) continue; // stale entry
    closed[ci] = 1;
    frontier.delete(ci);
    explored.add(ci);
    if (cur.x === goal.x && cur.y === goal.y) {
      yield { parent, frontier, explored, done: true, expanded: explored.size, gScore };
      return;
    }
    for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const nx = cur.x + dx, ny = cur.y + dy;
      if (!inBounds(nx, ny)) continue;
      const t = grid[ix(nx, ny)];
      if (t === WALL) continue;
      const ni = ix(nx, ny);
      const tentative = gScore[ci] + cellCost(t);
      if (tentative < gScore[ni]) {
        gScore[ni] = tentative;
        parent[ni] = ci;
        pq.push(tentative, { x: nx, y: ny });
        frontier.add(ni);
      }
    }
    yield { parent, frontier, explored, done: false };
  }
  yield { parent, frontier, explored, done: true, expanded: explored.size, unreachable: true, gScore };
}

function manhattan(a, b) { return Math.abs(a.x - b.x) + Math.abs(a.y - b.y); }

function* astarGen(grid, start, goal) {
  const N = COLS * ROWS;
  const parent = new Int32Array(N).fill(-1);
  const gScore = new Float64Array(N).fill(Infinity);
  const closed = new Uint8Array(N);
  const frontier = new Set();
  const explored = new Set();
  const pq = new MinHeap();
  const si = ix(start.x, start.y);
  gScore[si] = 0;
  pq.push(manhattan(start, goal), { x: start.x, y: start.y });
  frontier.add(si);

  while (pq.size()) {
    const { v: cur, p: pri } = pq.pop();
    const ci = ix(cur.x, cur.y);
    if (closed[ci]) continue;
    if (gScore[ci] + manhattan(cur, goal) < pri - 1e-9) continue; // stale (heuristic-aware)
    closed[ci] = 1;
    frontier.delete(ci);
    explored.add(ci);
    if (cur.x === goal.x && cur.y === goal.y) {
      yield { parent, frontier, explored, done: true, expanded: explored.size, gScore };
      return;
    }
    for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const nx = cur.x + dx, ny = cur.y + dy;
      if (!inBounds(nx, ny)) continue;
      const t = grid[ix(nx, ny)];
      if (t === WALL) continue;
      const ni = ix(nx, ny);
      const tentative = gScore[ci] + cellCost(t);
      if (tentative < gScore[ni]) {
        gScore[ni] = tentative;
        parent[ni] = ci;
        const f = tentative + manhattan({ x: nx, y: ny }, goal);
        pq.push(f, { x: nx, y: ny });
        frontier.add(ni);
      }
    }
    yield { parent, frontier, explored, done: false };
  }
  yield { parent, frontier, explored, done: true, expanded: explored.size, unreachable: true, gScore };
}

function* greedyGen(grid, start, goal) {
  const N = COLS * ROWS;
  const parent = new Int32Array(N).fill(-1);
  const visited = new Uint8Array(N);
  const frontier = new Set();
  const explored = new Set();
  const pq = new MinHeap();
  pq.push(manhattan(start, goal), { x: start.x, y: start.y });
  visited[ix(start.x, start.y)] = 1;
  frontier.add(ix(start.x, start.y));

  while (pq.size()) {
    const { v: cur } = pq.pop();
    const ci = ix(cur.x, cur.y);
    frontier.delete(ci);
    explored.add(ci);
    if (cur.x === goal.x && cur.y === goal.y) {
      yield { parent, frontier, explored, done: true, expanded: explored.size };
      return;
    }
    for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const nx = cur.x + dx, ny = cur.y + dy;
      if (!inBounds(nx, ny)) continue;
      const t = grid[ix(nx, ny)];
      if (t === WALL) continue;
      const ni = ix(nx, ny);
      if (visited[ni]) continue;
      visited[ni] = 1;
      parent[ni] = ci;
      pq.push(manhattan({ x: nx, y: ny }, goal), { x: nx, y: ny });
      frontier.add(ni);
    }
    yield { parent, frontier, explored, done: false };
  }
  yield { parent, frontier, explored, done: true, expanded: explored.size, unreachable: true };
}

const GENS = { bfs: bfsGen, dfs: dfsGen, dijkstra: dijkstraGen, astar: astarGen, greedy: greedyGen };

function reconstructPath(parent, start, goal) {
  const path = [];
  let i = ix(goal.x, goal.y);
  if (parent[i] === -1 && !(start.x === goal.x && start.y === goal.y)) return null;
  while (i !== -1) {
    path.push({ x: i % COLS, y: Math.floor(i / COLS) });
    i = parent[i];
  }
  path.reverse();
  return path;
}

// ---------------- Rendering ----------------
function setCanvasSize(canvas) {
  const cssW = COLS * CELL_PX;
  const cssH = ROWS * CELL_PX;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function render() {
  const canvas = document.getElementById('gs-canvas');
  setCanvasSize(canvas);
  const ctx = canvas.getContext('2d');
  const search = STATE.search;
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const i = ix(x, y);
      let fill = '#ffffff';
      const t = STATE.grid[i];
      if (t === WALL) fill = '#2f2a22';
      else if (t === MUD) fill = '#b08a52';
      if (search) {
        if (search.explored && search.explored.has(i)) fill = '#c8d8ee';
        if (search.frontier && search.frontier.has(i)) fill = '#f2cf6a';
        if (search.finalPath && search.finalPath.some(p => p.x === x && p.y === y)) fill = '#d9622b';
      }
      if (x === STATE.start.x && y === STATE.start.y) fill = '#1e7770';
      if (x === STATE.goal.x && y === STATE.goal.y) fill = '#d9622b';
      ctx.fillStyle = fill;
      ctx.fillRect(x * CELL_PX, y * CELL_PX, CELL_PX, CELL_PX);
      // grid lines
      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x * CELL_PX + 0.5, y * CELL_PX + 0.5, CELL_PX - 1, CELL_PX - 1);
    }
  }
  // Mark start/goal letters
  ctx.fillStyle = 'white';
  ctx.font = `${Math.floor(CELL_PX * 0.55)}px Manrope, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('S', STATE.start.x * CELL_PX + CELL_PX / 2, STATE.start.y * CELL_PX + CELL_PX / 2);
  ctx.fillText('G', STATE.goal.x * CELL_PX + CELL_PX / 2, STATE.goal.y * CELL_PX + CELL_PX / 2);
}

function setStatus(html) {
  document.getElementById('gs-status').innerHTML = html;
}

// ---------------- Interaction ----------------
function pixelToCell(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / rect.width * COLS);
  const y = Math.floor((e.clientY - rect.top) / rect.height * ROWS);
  return { x: Math.max(0, Math.min(COLS - 1, x)), y: Math.max(0, Math.min(ROWS - 1, y)) };
}

let dragging = false;
function paintCell(x, y) {
  const i = ix(x, y);
  if (STATE.tool === 'wall') STATE.grid[i] = WALL;
  else if (STATE.tool === 'mud') STATE.grid[i] = MUD;
  else if (STATE.tool === 'erase') STATE.grid[i] = EMPTY;
  // start / goal handled separately on mousedown
  if ((x === STATE.start.x && y === STATE.start.y) || (x === STATE.goal.x && y === STATE.goal.y)) {
    if (STATE.tool === 'wall' || STATE.tool === 'mud') {
      STATE.grid[i] = EMPTY;
    }
  }
}

function wireCanvas() {
  const canvas = document.getElementById('gs-canvas');
  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    dragging = true;
    const c = pixelToCell(e, canvas);
    if (STATE.tool === 'start') { STATE.start = c; STATE.grid[ix(c.x, c.y)] = EMPTY; }
    else if (STATE.tool === 'goal') { STATE.goal = c; STATE.grid[ix(c.x, c.y)] = EMPTY; }
    else paintCell(c.x, c.y);
    STATE.search = null;
    render();
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const c = pixelToCell(e, canvas);
    if (STATE.tool === 'start') STATE.start = c;
    else if (STATE.tool === 'goal') STATE.goal = c;
    else paintCell(c.x, c.y);
    STATE.search = null;
    render();
  });
  ['mouseup', 'mouseleave'].forEach((ev) => canvas.addEventListener(ev, () => dragging = false));
  // Touch
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    dragging = true;
    const t = e.touches[0];
    const c = pixelToCell(t, canvas);
    if (STATE.tool === 'start') STATE.start = c;
    else if (STATE.tool === 'goal') STATE.goal = c;
    else paintCell(c.x, c.y);
    STATE.search = null;
    render();
  });
  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!dragging) return;
    const t = e.touches[0];
    const c = pixelToCell(t, canvas);
    if (STATE.tool === 'start') STATE.start = c;
    else if (STATE.tool === 'goal') STATE.goal = c;
    else paintCell(c.x, c.y);
    STATE.search = null;
    render();
  });
  canvas.addEventListener('touchend', () => dragging = false);
}

// ---------------- Search runner ----------------
function startSearch() {
  if (STATE.running) return;
  const algo = document.getElementById('gs-algo').value;
  STATE.search = {
    algo,
    parent: null, frontier: new Set(), explored: new Set(),
    finalPath: null, expanded: 0,
    generator: GENS[algo](STATE.grid, STATE.start, STATE.goal)
  };
  STATE.running = true;
  setStatus(`<strong>${algo}</strong> — running…`);
  const speed = STATE.speed;
  const stepsPerTick = speed <= 4 ? 6 : 1;
  const tick = () => {
    if (!STATE.running) return;
    for (let k = 0; k < stepsPerTick; k++) {
      const r = STATE.search.generator.next();
      if (r.done) { STATE.running = false; break; }
      const v = r.value;
      STATE.search.parent = v.parent;
      STATE.search.frontier = v.frontier;
      STATE.search.explored = v.explored;
      if (v.done) {
        STATE.running = false;
        finishSearch(v);
        break;
      }
    }
    render();
    if (STATE.running) {
      if (speed <= 0) setTimeout(tick, 0);
      else setTimeout(tick, speed);
    }
  };
  tick();
}

function finishSearch(v) {
  const algo = STATE.search.algo;
  if (v.unreachable) {
    STATE.search.finalPath = null;
    const totalCells = COLS * ROWS;
    setStatus(`<strong>${algo}</strong> — goal unreachable. expanded <strong>${v.expanded}</strong> / ${totalCells} cells.`);
    return;
  }
  const path = reconstructPath(v.parent, STATE.start, STATE.goal);
  STATE.search.finalPath = path;
  // Compute path cost
  let cost = 0;
  if (path && path.length > 1) {
    for (let i = 1; i < path.length; i++) {
      const p = path[i];
      cost += cellCost(STATE.grid[ix(p.x, p.y)]);
    }
  }
  setStatus(`<strong>${algo}</strong> done. path length <strong>${path ? path.length - 1 : 0}</strong> steps, cost <strong>${cost}</strong>, explored <strong>${v.expanded}</strong> cells.`);
}

function stepSearch() {
  if (!STATE.search) {
    startSearch();
    STATE.running = false;
    return;
  }
  const r = STATE.search.generator.next();
  if (r.done) return;
  const v = r.value;
  STATE.search.parent = v.parent;
  STATE.search.frontier = v.frontier;
  STATE.search.explored = v.explored;
  if (v.done) finishSearch(v);
  render();
}

function clearSearch() {
  STATE.search = null;
  STATE.running = false;
  setStatus('cleared search. draw walls or press run.');
  render();
}

function clearWalls() {
  STATE.grid = newGrid();
  STATE.search = null;
  setStatus('grid cleared.');
  render();
}

function randomMaze() {
  // Simple random wall pattern with some mud zones
  STATE.grid = newGrid();
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      if ((x === STATE.start.x && y === STATE.start.y) || (x === STATE.goal.x && y === STATE.goal.y)) continue;
      const r = Math.random();
      if (r < 0.22) STATE.grid[ix(x, y)] = WALL;
      else if (r < 0.32) STATE.grid[ix(x, y)] = MUD;
    }
  }
  STATE.search = null;
  setStatus('random maze generated.');
  render();
}

// ---------------- Wiring ----------------
function wire() {
  STATE.grid = newGrid();
  setCanvasSize(document.getElementById('gs-canvas'));
  wireCanvas();

  document.getElementById('gs-tools').addEventListener('click', (e) => {
    const btn = e.target.closest('.tool-btn');
    if (!btn) return;
    document.querySelectorAll('.tool-btn').forEach((b) => b.classList.remove('is-active'));
    btn.classList.add('is-active');
    STATE.tool = btn.dataset.tool;
  });

  document.getElementById('gs-speed').addEventListener('change', (e) => {
    STATE.speed = parseInt(e.target.value, 10);
  });
  document.getElementById('gs-run').addEventListener('click', () => {
    clearSearch();
    startSearch();
  });
  document.getElementById('gs-step').addEventListener('click', stepSearch);
  document.getElementById('gs-clear-search').addEventListener('click', clearSearch);
  document.getElementById('gs-clear-walls').addEventListener('click', clearWalls);
  document.getElementById('gs-maze').addEventListener('click', randomMaze);

  // Default scenario: a simple wall in the middle
  for (let y = 4; y <= 18; y++) STATE.grid[ix(18, y)] = WALL;
  STATE.grid[ix(18, 11)] = EMPTY;
  STATE.grid[ix(18, 10)] = EMPTY;
  render();
  setStatus('default scenario: a wall with a gap. press <strong>run</strong>.');
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-bfs':
      '\\text{BFS guarantee: } \\text{dist}(s, n) = \\text{(depth at which } n \\text{ is first popped)}',
    'math-dijkstra':
      'g(v) \\leftarrow \\min\\!\\left( g(v),\\; g(u) + w(u, v) \\right) \\text{ for every edge } (u, v)',
    'math-astar':
      'f(n) = g(n) + h(n), \\quad \\text{expand } n \\text{ with smallest } f \\text{ first}'
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
