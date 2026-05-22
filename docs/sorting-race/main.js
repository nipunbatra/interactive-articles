// ============================================================
// Sorting Race — five algorithms stepping in lockstep on the same array.
//
// Each algorithm is a generator yielding one "step" per call. The
// step records what just changed: a comparison between indices, a
// write to a position, a swap between two positions. The renderer
// reads each algorithm's array snapshot and the currently-highlighted
// indices, and draws a bar chart.
// ============================================================

const ALGOS = [
  { id: 'insertion', name: 'insertion sort',         color: '#2c6fb7' },
  { id: 'bubble',    name: 'bubble sort',            color: '#7e4ea3' },
  { id: 'merge',     name: 'mergesort',              color: '#1e7770' },
  { id: 'quick',     name: 'quicksort (med-of-3)',   color: '#b86b2a' },
  { id: 'heap',      name: 'heapsort',               color: '#d9622b' }
];

const STATE = {
  base: [],          // immutable starting array (each algo gets a copy)
  runners: [],       // per-algorithm runner objects
  playing: false,
  speed: 16,         // ms per tick
  loopHandle: null,
  finishOrder: []    // ranks
};

// ---------------- Array generation ----------------
function rngSeed(seed) {
  let s = seed >>> 0;
  return function() {
    s = (s + 0x6D2B79F5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeArray(size, dist, seed = 0xC0FFEE) {
  const rng = rngSeed(seed);
  const arr = new Array(size);
  for (let i = 0; i < size; i++) arr[i] = i + 1;
  if (dist === 'random') {
    for (let i = size - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  } else if (dist === 'reversed') {
    arr.reverse();
  } else if (dist === 'sorted') {
    // already 1..n
  } else if (dist === 'nearly') {
    // a few random swaps
    for (let k = 0; k < Math.max(2, Math.floor(size / 10)); k++) {
      const i = Math.floor(rng() * size);
      const j = Math.floor(rng() * size);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  } else if (dist === 'few') {
    // only ~5 distinct values
    for (let i = 0; i < size; i++) {
      arr[i] = 1 + Math.floor(rng() * 5) * Math.floor(size / 5);
    }
  }
  return arr;
}

// ---------------- Algorithm generators ----------------
// Each yields step objects. The runner records highlight indices
// without changing the algorithm's array (the gen mutates it directly).

function* insertionSort(a, stats) {
  for (let i = 1; i < a.length; i++) {
    let j = i;
    while (j > 0) {
      stats.cmps++;
      yield { hi: [j, j - 1] };
      if (a[j] < a[j - 1]) {
        [a[j], a[j - 1]] = [a[j - 1], a[j]];
        stats.writes += 2;
        yield { hi: [j, j - 1], swap: true };
        j--;
      } else break;
    }
  }
}

function* bubbleSort(a, stats) {
  let n = a.length;
  let swapped = true;
  while (swapped) {
    swapped = false;
    for (let i = 1; i < n; i++) {
      stats.cmps++;
      yield { hi: [i - 1, i] };
      if (a[i - 1] > a[i]) {
        [a[i - 1], a[i]] = [a[i], a[i - 1]];
        stats.writes += 2;
        swapped = true;
        yield { hi: [i - 1, i], swap: true };
      }
    }
    n--; // last element is sorted
  }
}

function* mergeSort(a, stats) {
  // Bottom-up, iterative merge — easier to step through.
  const n = a.length;
  const buf = new Array(n);
  for (let width = 1; width < n; width *= 2) {
    for (let lo = 0; lo < n; lo += 2 * width) {
      const mid = Math.min(lo + width, n);
      const hi = Math.min(lo + 2 * width, n);
      // Merge a[lo..mid) and a[mid..hi)
      for (let i = lo; i < hi; i++) buf[i] = a[i];
      let i = lo, j = mid, k = lo;
      while (i < mid && j < hi) {
        stats.cmps++;
        yield { hi: [i, j] };
        if (buf[i] <= buf[j]) {
          a[k] = buf[i]; stats.writes++; yield { hi: [k], write: true };
          i++; k++;
        } else {
          a[k] = buf[j]; stats.writes++; yield { hi: [k], write: true };
          j++; k++;
        }
      }
      while (i < mid) { a[k] = buf[i]; stats.writes++; yield { hi: [k], write: true }; i++; k++; }
      while (j < hi)  { a[k] = buf[j]; stats.writes++; yield { hi: [k], write: true }; j++; k++; }
    }
  }
}

function* quickSort(a, stats) {
  // Median-of-three pivot, Lomuto partition.
  const stk = [[0, a.length - 1]];
  while (stk.length) {
    const [lo, hi] = stk.pop();
    if (lo >= hi) continue;
    // Median of three
    const mid = (lo + hi) >> 1;
    const candidates = [[a[lo], lo], [a[mid], mid], [a[hi], hi]].sort((x, y) => x[0] - y[0]);
    const pivotIdx = candidates[1][1];
    // Move pivot to end
    [a[pivotIdx], a[hi]] = [a[hi], a[pivotIdx]];
    stats.writes += 2;
    const pivot = a[hi];
    let i = lo - 1;
    yield { hi: [hi], pivot: true };
    for (let j = lo; j < hi; j++) {
      stats.cmps++;
      yield { hi: [j], pivot: hi };
      if (a[j] <= pivot) {
        i++;
        if (i !== j) {
          [a[i], a[j]] = [a[j], a[i]];
          stats.writes += 2;
          yield { hi: [i, j], swap: true, pivot: hi };
        }
      }
    }
    [a[i + 1], a[hi]] = [a[hi], a[i + 1]];
    stats.writes += 2;
    yield { hi: [i + 1, hi], swap: true };
    const p = i + 1;
    stk.push([p + 1, hi]);
    stk.push([lo, p - 1]);
  }
}

function* heapSort(a, stats) {
  const n = a.length;
  // Build max-heap (Floyd's heapify)
  for (let i = (n >> 1) - 1; i >= 0; i--) {
    yield* siftDownGen(a, n, i, stats);
  }
  // Repeatedly extract
  for (let end = n - 1; end > 0; end--) {
    [a[0], a[end]] = [a[end], a[0]];
    stats.writes += 2;
    yield { hi: [0, end], swap: true };
    yield* siftDownGen(a, end, 0, stats);
  }
}
function* siftDownGen(a, n, i, stats) {
  while (true) {
    const l = 2 * i + 1, r = 2 * i + 2;
    let best = i;
    if (l < n) { stats.cmps++; yield { hi: [l, best] }; if (a[l] > a[best]) best = l; }
    if (r < n) { stats.cmps++; yield { hi: [r, best] }; if (a[r] > a[best]) best = r; }
    if (best === i) break;
    [a[i], a[best]] = [a[best], a[i]];
    stats.writes += 2;
    yield { hi: [i, best], swap: true };
    i = best;
  }
}

const ALGO_GENS = {
  insertion: insertionSort,
  bubble: bubbleSort,
  merge: mergeSort,
  quick: quickSort,
  heap: heapSort
};

// ---------------- Runners ----------------
function makeRunners() {
  STATE.runners = ALGOS.map((meta) => {
    const arr = STATE.base.slice();
    const stats = { cmps: 0, writes: 0, steps: 0 };
    const gen = ALGO_GENS[meta.id](arr, stats);
    return {
      ...meta,
      arr, stats, gen,
      done: false,
      lastStep: null
    };
  });
  STATE.finishOrder = [];
}

function stepAll() {
  let anyAlive = false;
  for (const r of STATE.runners) {
    if (r.done) continue;
    const result = r.gen.next();
    r.stats.steps++;
    if (result.done) {
      r.done = true;
      r.lastStep = null;
      STATE.finishOrder.push(r.id);
    } else {
      r.lastStep = result.value || null;
      anyAlive = true;
    }
  }
  return anyAlive;
}

// ---------------- Rendering ----------------
function renderCard(r, idx) {
  const card = document.getElementById(`race-${r.id}`);
  card.classList.toggle('done', r.done);
  const rank = STATE.finishOrder.indexOf(r.id);
  if (rank >= 0) {
    if (!card.querySelector('.race-card-rank')) {
      const badge = document.createElement('span');
      badge.className = 'race-card-rank';
      badge.textContent = `#${rank + 1}`;
      card.querySelector('.race-card-head').appendChild(badge);
    }
    if (rank === 0) card.classList.add('is-winner');
  } else {
    card.classList.remove('is-winner');
    const old = card.querySelector('.race-card-rank');
    if (old) old.remove();
  }

  // Stats text
  card.querySelector('.race-stats-cmp').textContent = r.stats.cmps.toLocaleString();
  card.querySelector('.race-stats-wr').textContent  = r.stats.writes.toLocaleString();
  card.querySelector('.race-stats-st').textContent  = r.stats.steps.toLocaleString();

  // Bars
  drawBars(card.querySelector('canvas'), r);
}

function drawBars(canvas, r) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = Math.max(1, w * dpr);
    canvas.height = Math.max(1, h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const a = r.arr;
  const n = a.length;
  const maxV = n; // values are 1..n
  const barW = w / n;
  // Highlight indices
  const hi = new Set((r.lastStep && r.lastStep.hi) || []);
  const isSwap = r.lastStep && r.lastStep.swap;
  const isWrite = r.lastStep && r.lastStep.write;
  const isPivot = r.lastStep && r.lastStep.pivot;
  const pivotIdx = (typeof isPivot === 'number') ? isPivot : (isPivot === true && hi.size === 1 ? Array.from(hi)[0] : -1);

  for (let i = 0; i < n; i++) {
    const v = a[i];
    const bh = Math.max(2, (v / maxV) * (h - 4));
    let color = '#2c6fb7';
    if (r.done) color = '#1e7770';
    if (hi.has(i)) color = isSwap ? '#d9622b' : (isWrite ? '#b86b2a' : '#d9622b');
    if (i === pivotIdx) color = '#7e4ea3';
    ctx.fillStyle = color;
    ctx.fillRect(i * barW + 0.5, h - bh - 1, Math.max(1, barW - 1), bh);
  }
}

function renderAll() {
  STATE.runners.forEach((r, i) => renderCard(r, i));
}

// ---------------- Loop ----------------
function startLoop() {
  if (STATE.playing) return;
  STATE.playing = true;
  document.getElementById('race-play').disabled = true;
  document.getElementById('race-pause').disabled = false;
  const speed = STATE.speed;
  // For very small speed (turbo), do multiple steps per RAF.
  const stepsPerTick = speed <= 4 ? 8 : speed <= 16 ? 2 : 1;
  STATE.loopHandle = setInterval(() => {
    let alive = false;
    for (let k = 0; k < stepsPerTick; k++) {
      alive = stepAll() || alive;
      if (!alive) break;
    }
    renderAll();
    if (!alive) stopLoop(true);
  }, Math.max(8, speed));
}

function stopLoop(allDone = false) {
  if (STATE.loopHandle) clearInterval(STATE.loopHandle);
  STATE.loopHandle = null;
  STATE.playing = false;
  document.getElementById('race-play').disabled = allDone;
  document.getElementById('race-pause').disabled = true;
}

// ---------------- Wiring ----------------
function buildGrid() {
  const grid = document.getElementById('race-grid');
  grid.innerHTML = '';
  for (const meta of ALGOS) {
    const card = document.createElement('div');
    card.className = 'race-card';
    card.id = `race-${meta.id}`;
    card.innerHTML = `
      <div class="race-card-head">
        <span class="race-card-name">${meta.name}</span>
        <span class="race-card-cx" id="cx-${meta.id}"></span>
      </div>
      <canvas></canvas>
      <div class="race-card-stats">
        <span>cmp <strong class="race-stats-cmp">0</strong></span>
        <span>write <strong class="race-stats-wr">0</strong></span>
        <span>step <strong class="race-stats-st">0</strong></span>
      </div>
    `;
    grid.appendChild(card);
  }
}

function resetRace() {
  stopLoop();
  const size = parseInt(document.getElementById('race-size').value, 10);
  const dist = document.getElementById('race-dist').value;
  STATE.base = makeArray(size, dist);
  makeRunners();
  document.getElementById('race-play').disabled = false;
  document.getElementById('race-pause').disabled = true;
  renderAll();
}

function wire() {
  buildGrid();

  document.getElementById('race-dist').addEventListener('change', resetRace);
  document.getElementById('race-size').addEventListener('change', resetRace);
  document.getElementById('race-speed').addEventListener('change', (e) => {
    STATE.speed = parseInt(e.target.value, 10);
    if (STATE.playing) { stopLoop(); startLoop(); }
  });
  document.getElementById('race-play').addEventListener('click', startLoop);
  document.getElementById('race-pause').addEventListener('click', () => stopLoop(false));
  document.getElementById('race-reset').addEventListener('click', resetRace);

  resetRace();
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-insertion':
      '\\text{insertion: best } O(n) \\text{ (sorted in), worst } O(n^2) \\text{ (reverse-sorted)}',
    'math-merge':
      'T(n) = 2 T(n/2) + O(n) \\;\\Rightarrow\\; T(n) = O(n \\log n)',
    'math-quick':
      '\\mathbb{E}[T(n)] = O(n \\log n), \\quad T_{\\max}(n) = O(n^2) \\text{ if pivot is unlucky}',
    'math-heapsort':
      '\\text{heapsort} = \\underbrace{\\text{heapify}}_{O(n)} + n \\cdot \\underbrace{\\text{extract-max}}_{O(\\log n)} = O(n \\log n)',
    'math-bound':
      '\\text{any comparison sort} \\;\\geq\\; \\Omega(n \\log n) \\text{ comparisons (information-theoretic lower bound)}'
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
