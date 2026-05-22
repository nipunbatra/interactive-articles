// ============================================================
// Trees, Branch by Branch
// Two interactive widgets:
//   (a) BST — insert / search / traversal animation
//   (b) Heap — insert (sift-up) / extract root (sift-down) / heapify
// Each tree is rendered as SVG; we re-layout on every change.
// ============================================================

// ---------------- Generic tree layout ----------------
//
// Strategy: compute x-positions by in-order index (BST) or by
// complete-tree slot (heap), then y by depth. Works fine up to
// ~30 nodes which is well above the demo scale.

function layoutTreeInorder(root) {
  // Used for BSTs. Each node gets x = its in-order rank.
  if (!root) return { nodes: [], edges: [], width: 0, depth: 0 };
  const nodes = [];
  const edges = [];
  let counter = 0;
  let maxDepth = 0;

  function visit(node, depth, parent) {
    if (!node) return;
    visit(node.left, depth + 1, node);
    node._idx = counter++;
    node._depth = depth;
    nodes.push(node);
    if (parent) edges.push({ from: parent, to: node });
    if (depth > maxDepth) maxDepth = depth;
    visit(node.right, depth + 1, node);
  }
  visit(root, 0, null);
  return { nodes, edges, width: counter, depth: maxDepth };
}

function layoutHeapArray(arr) {
  // For complete trees stored in an array: layout by slot in the
  // implicit complete tree (left child = 2i+1, right child = 2i+2).
  if (!arr.length) return { nodes: [], edges: [], width: 0, depth: 0 };
  const nodes = arr.map((v, i) => ({ value: v, idx: i, _depth: Math.floor(Math.log2(i + 1)) }));
  const depth = nodes[nodes.length - 1]._depth;
  // Assign x by spreading each level uniformly.
  const xByDepth = {};
  for (const n of nodes) {
    const d = n._depth;
    if (!(d in xByDepth)) xByDepth[d] = 0;
    n._slot = xByDepth[d]++;
  }
  // Compute total slot width
  const slotsPerLevel = {};
  for (const n of nodes) slotsPerLevel[n._depth] = (slotsPerLevel[n._depth] || 0) + 1;
  const maxLevelSlots = Math.max(...Object.values(slotsPerLevel), 1);
  for (const n of nodes) {
    // Distribute slots within the available level width.
    const slotsAtLevel = slotsPerLevel[n._depth];
    const x = (n._slot + 0.5) * (maxLevelSlots / slotsAtLevel);
    n._x = x;
  }
  const edges = [];
  for (const n of nodes) {
    const l = 2 * n.idx + 1, r = 2 * n.idx + 2;
    if (l < arr.length) edges.push({ from: n, to: nodes[l] });
    if (r < arr.length) edges.push({ from: n, to: nodes[r] });
  }
  return { nodes, edges, width: maxLevelSlots, depth };
}

// ---------------- Common SVG render ----------------
function renderTreeSVG(svg, layout, opts = {}) {
  // Map abstract x (slot index) and depth → screen pixels.
  const W = svg.clientWidth || 600;
  const H = svg.clientHeight || 320;
  const padX = 28, padY = 28;
  const innerW = Math.max(40, W - 2 * padX);
  const innerH = Math.max(40, H - 2 * padY);
  const maxX = Math.max(1, layout.width - (opts.heap ? 0 : 1));
  const maxD = Math.max(1, layout.depth);

  function nodeX(n) {
    if (opts.heap) return padX + (n._x / layout.width) * innerW;
    return padX + (n._idx / maxX) * innerW;
  }
  function nodeY(n) {
    return padY + (n._depth / maxD) * innerH;
  }

  // Clear
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const NS = 'http://www.w3.org/2000/svg';

  // Edges first (under nodes)
  for (const e of layout.edges) {
    const line = document.createElementNS(NS, 'line');
    line.setAttribute('class', 'edge' + ((opts.activeEdges && opts.activeEdges.has(edgeKey(e))) ? ' active' : ''));
    line.setAttribute('x1', nodeX(e.from));
    line.setAttribute('y1', nodeY(e.from));
    line.setAttribute('x2', nodeX(e.to));
    line.setAttribute('y2', nodeY(e.to));
    svg.appendChild(line);
  }

  // Nodes
  for (const n of layout.nodes) {
    const g = document.createElementNS(NS, 'g');
    let cls = 'node';
    if (opts.activeIds && opts.activeIds.has(nodeKey(n))) cls += ' active';
    if (opts.foundIds && opts.foundIds.has(nodeKey(n))) cls += ' found';
    if (opts.visitedIds && opts.visitedIds.has(nodeKey(n))) cls += ' visited';
    g.setAttribute('class', cls);
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', nodeX(n));
    c.setAttribute('cy', nodeY(n));
    c.setAttribute('r', 17);
    const t = document.createElementNS(NS, 'text');
    t.setAttribute('x', nodeX(n));
    t.setAttribute('y', nodeY(n));
    t.textContent = String(n.value ?? n.val ?? '');
    g.appendChild(c);
    g.appendChild(t);
    svg.appendChild(g);
  }
}
function nodeKey(n) { return n.idx !== undefined ? `i:${n.idx}` : `v:${n._idx}`; }
function edgeKey(e) { return `${nodeKey(e.from)}->${nodeKey(e.to)}`; }

// =====================================================
//                       PART A — BST
// =====================================================

class BSTNode {
  constructor(value) { this.value = value; this.left = null; this.right = null; }
}

const BST = {
  root: null,
  highlight: { active: new Set(), found: new Set() }
};

function bstInsert(value) {
  if (BST.root === null) { BST.root = new BSTNode(value); return; }
  let cur = BST.root;
  while (true) {
    if (value === cur.value) return; // no duplicates
    if (value < cur.value) {
      if (cur.left) cur = cur.left;
      else { cur.left = new BSTNode(value); return; }
    } else {
      if (cur.right) cur = cur.right;
      else { cur.right = new BSTNode(value); return; }
    }
  }
}

function bstSearch(value) {
  // Returns { path: [nodes...], found: bool, target: node|null }
  const path = [];
  let cur = BST.root;
  while (cur) {
    path.push(cur);
    if (value === cur.value) return { path, found: true, target: cur };
    cur = value < cur.value ? cur.left : cur.right;
  }
  return { path, found: false, target: null };
}

function bstClear() {
  BST.root = null;
  BST.highlight = { active: new Set(), found: new Set() };
}

function renderBST(opts = {}) {
  const svg = document.getElementById('bst-svg');
  const layout = layoutTreeInorder(BST.root);
  const activeIds = new Set();
  const foundIds = new Set();
  if (opts.activePath) {
    for (const n of opts.activePath) activeIds.add(`v:${n._idx}`);
  }
  if (opts.foundNode) foundIds.add(`v:${opts.foundNode._idx}`);
  renderTreeSVG(svg, layout, { activeIds, foundIds });
}

function bstHeight(node) {
  if (!node) return -1;
  return 1 + Math.max(bstHeight(node.left), bstHeight(node.right));
}
function bstSize(node) {
  if (!node) return 0;
  return 1 + bstSize(node.left) + bstSize(node.right);
}

function setBstStatus(html) {
  document.getElementById('bst-status').innerHTML = html;
}

function bstSummary() {
  const n = bstSize(BST.root);
  const h = bstHeight(BST.root);
  const optimal = n > 0 ? Math.ceil(Math.log2(n + 1)) - 1 : 0;
  return `n = <strong>${n}</strong>, height = <strong>${Math.max(0, h)}</strong> (optimal for this size: ${optimal}).`;
}

async function animateSearch(value) {
  const { path, found, target } = bstSearch(value);
  // Step through path
  for (let i = 0; i < path.length; i++) {
    renderBST({ activePath: path.slice(0, i + 1) });
    setBstStatus(`searching for <strong>${value}</strong> — visiting ${path[i].value}.`);
    await sleep(380);
  }
  if (found) {
    renderBST({ activePath: path, foundNode: target });
    setBstStatus(`found <strong>${value}</strong> in ${path.length} step${path.length === 1 ? '' : 's'}. ${bstSummary()}`);
  } else {
    renderBST({ activePath: path });
    setBstStatus(`<strong>${value}</strong> not in tree (walked ${path.length} step${path.length === 1 ? '' : 's'}). ${bstSummary()}`);
  }
}

function bstAdd(value) {
  bstInsert(value);
  renderBST();
  setBstStatus(`inserted <strong>${value}</strong>. ${bstSummary()}`);
}

function bstRandom8() {
  bstClear();
  const used = new Set();
  while (used.size < 8) used.add(1 + Math.floor(Math.random() * 30));
  for (const v of used) bstInsert(v);
  renderBST();
  setBstStatus(`inserted 8 random values. ${bstSummary()}`);
}

function bstSorted8() {
  bstClear();
  for (let v = 1; v <= 8; v++) bstInsert(v);
  renderBST();
  setBstStatus(`inserted 1..8 in order → degenerate right-spine. ${bstSummary()}`);
}

// ---------------- Traversals on BST ----------------
function inOrder(node, out) {
  if (!node) return;
  inOrder(node.left, out); out.push(node); inOrder(node.right, out);
}
function preOrder(node, out) {
  if (!node) return;
  out.push(node); preOrder(node.left, out); preOrder(node.right, out);
}
function postOrder(node, out) {
  if (!node) return;
  postOrder(node.left, out); postOrder(node.right, out); out.push(node);
}
function levelOrder(root) {
  const out = []; if (!root) return out;
  const q = [root];
  while (q.length) { const n = q.shift(); out.push(n); if (n.left) q.push(n.left); if (n.right) q.push(n.right); }
  return out;
}

let TRAV_TIMER = null;
async function runTraversal() {
  cancelTraversal();
  const mode = document.getElementById('trav-mode').value;
  const speed = parseInt(document.getElementById('trav-speed').value, 10) || 320;
  if (!BST.root) {
    setBstStatus('insert some values first.');
    return;
  }
  const out = [];
  if (mode === 'pre') preOrder(BST.root, out);
  else if (mode === 'in') inOrder(BST.root, out);
  else if (mode === 'post') postOrder(BST.root, out);
  else levelOrder(BST.root).forEach((n) => out.push(n));

  const trail = document.getElementById('trav-trail');
  trail.innerHTML = '';

  // Step through, highlight each node briefly
  for (let i = 0; i < out.length; i++) {
    if (TRAV_CANCELLED) return;
    renderBST({ activePath: [out[i]] });
    if (i > 0) {
      const arrow = document.createElement('span');
      arrow.className = 'tok tok-arrow';
      arrow.textContent = '→';
      trail.appendChild(arrow);
    }
    const tok = document.createElement('span');
    tok.className = 'tok';
    tok.textContent = String(out[i].value);
    trail.appendChild(tok);
    await sleep(speed);
  }
  // Final: clear highlight, leave trail
  renderBST();
  setBstStatus(`<strong>${mode}-order</strong> traversal complete: visited ${out.length} node${out.length === 1 ? '' : 's'}.`);
}

let TRAV_CANCELLED = false;
function cancelTraversal() {
  TRAV_CANCELLED = true;
  // Allow microtask flush; next call resets.
  setTimeout(() => { TRAV_CANCELLED = false; }, 0);
}
function resetTraversal() {
  cancelTraversal();
  document.getElementById('trav-trail').innerHTML = '';
  renderBST();
}

// =====================================================
//                      PART B — Heap
// =====================================================

const HEAP = {
  arr: [],
  kind: 'min', // or 'max'
};

function heapCompare(a, b) {
  return HEAP.kind === 'min' ? a < b : a > b;
}

function heapPush(value, opts = {}) {
  HEAP.arr.push(value);
  // Sift up — animated step by step
  return siftUp(HEAP.arr.length - 1, opts);
}
async function siftUp(i, opts) {
  const trail = [];
  while (i > 0) {
    const parent = Math.floor((i - 1) / 2);
    if (heapCompare(HEAP.arr[i], HEAP.arr[parent])) {
      trail.push({ a: i, b: parent });
      [HEAP.arr[i], HEAP.arr[parent]] = [HEAP.arr[parent], HEAP.arr[i]];
      i = parent;
    } else break;
  }
  if (opts && opts.animate) {
    await animateSwapTrail(trail, `sifting <strong>${HEAP.arr[i]}</strong> up`);
  }
}

async function heapExtract(opts) {
  if (HEAP.arr.length === 0) return null;
  const root = HEAP.arr[0];
  const last = HEAP.arr.pop();
  if (HEAP.arr.length > 0) {
    HEAP.arr[0] = last;
    await siftDown(0, opts);
  }
  return root;
}

async function siftDown(i, opts) {
  const trail = [];
  const n = HEAP.arr.length;
  while (true) {
    const l = 2 * i + 1, r = 2 * i + 2;
    let pick = i;
    if (l < n && heapCompare(HEAP.arr[l], HEAP.arr[pick])) pick = l;
    if (r < n && heapCompare(HEAP.arr[r], HEAP.arr[pick])) pick = r;
    if (pick === i) break;
    trail.push({ a: i, b: pick });
    [HEAP.arr[i], HEAP.arr[pick]] = [HEAP.arr[pick], HEAP.arr[i]];
    i = pick;
  }
  if (opts && opts.animate) {
    await animateSwapTrail(trail, `sifting down`);
  }
}

async function animateSwapTrail(trail, label) {
  for (const sw of trail) {
    renderHeap({ highlight: new Set([sw.a, sw.b]) });
    setHeapStatus(`${label}: swap indices ${sw.a} ↔ ${sw.b} (values ${HEAP.arr[sw.b]} ↔ ${HEAP.arr[sw.a]}).`);
    await sleep(360);
  }
  renderHeap();
}

async function heapifyAll() {
  // Floyd's bottom-up heapify, animated.
  for (let i = Math.floor(HEAP.arr.length / 2) - 1; i >= 0; i--) {
    await siftDown(i, { animate: true });
  }
  setHeapStatus(`heapified ${HEAP.arr.length} elements in O(n). ${heapSummary()}`);
}

function heapClear() { HEAP.arr = []; }
function heapSummary() {
  const n = HEAP.arr.length;
  if (n === 0) return 'empty heap.';
  const h = Math.floor(Math.log2(n));
  return `${HEAP.kind}-heap: n = <strong>${n}</strong>, height = <strong>${h}</strong>, root = <strong>${HEAP.arr[0]}</strong>.`;
}

function setHeapStatus(html) { document.getElementById('heap-status').innerHTML = html; }

function renderHeap(opts = {}) {
  // SVG tree
  const svg = document.getElementById('heap-svg');
  const layout = layoutHeapArray(HEAP.arr);
  const activeIds = new Set();
  const foundIds = new Set();
  if (opts.highlight) {
    for (const i of opts.highlight) activeIds.add(`i:${i}`);
  }
  renderTreeSVG(svg, layout, { heap: true, activeIds, foundIds });

  // Array view
  const arrEl = document.getElementById('heap-array');
  arrEl.innerHTML = '';
  HEAP.arr.forEach((v, i) => {
    const cell = document.createElement('div');
    let cls = 'heap-cell';
    if (opts.highlight && opts.highlight.has(i)) cls += ' swap';
    if (i === 0) cls += ' root';
    cell.className = cls;
    cell.innerHTML = `<div>${v}</div><div class="heap-cell-idx">[${i}]</div>`;
    arrEl.appendChild(cell);
  });
}

// =====================================================
//                  Wiring & helpers
// =====================================================

function parseVal(input) {
  const s = String(input).trim();
  if (!s) return null;
  const n = Number(s);
  if (!Number.isFinite(n)) return null;
  return Math.round(n);
}

function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

function wireBST() {
  document.getElementById('bst-add').addEventListener('click', () => {
    const el = document.getElementById('bst-insert');
    const v = parseVal(el.value);
    if (v === null) { setBstStatus('enter an integer.'); return; }
    bstAdd(v);
    el.value = '';
  });
  document.getElementById('bst-insert').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') document.getElementById('bst-add').click();
  });
  document.getElementById('bst-find').addEventListener('click', () => {
    const el = document.getElementById('bst-search');
    const v = parseVal(el.value);
    if (v === null) { setBstStatus('enter an integer to search.'); return; }
    if (!BST.root) { setBstStatus('tree is empty.'); return; }
    animateSearch(v);
  });
  document.getElementById('bst-search').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') document.getElementById('bst-find').click();
  });
  document.getElementById('bst-random').addEventListener('click', () => {
    bstRandom8();
  });
  document.getElementById('bst-sorted').addEventListener('click', () => {
    bstSorted8();
  });
  document.getElementById('bst-clear').addEventListener('click', () => {
    bstClear();
    renderBST();
    setBstStatus('empty tree.');
    document.getElementById('trav-trail').innerHTML = '';
  });

  document.getElementById('trav-play').addEventListener('click', runTraversal);
  document.getElementById('trav-reset').addEventListener('click', resetTraversal);

  // Initial demo: a small balanced tree
  [8, 4, 12, 2, 6, 10, 14].forEach((v) => bstInsert(v));
  renderBST();
  setBstStatus(`demo tree pre-populated. ${bstSummary()}`);
}

function wireHeap() {
  document.getElementById('heap-kind').addEventListener('change', (e) => {
    HEAP.kind = e.target.value;
    // Re-heapify under the new invariant.
    (async () => {
      for (let i = Math.floor(HEAP.arr.length / 2) - 1; i >= 0; i--) await siftDown(i);
      renderHeap();
      setHeapStatus(`switched to <strong>${HEAP.kind}-heap</strong>. ${heapSummary()}`);
    })();
  });
  document.getElementById('heap-add').addEventListener('click', async () => {
    const el = document.getElementById('heap-insert');
    const v = parseVal(el.value);
    if (v === null) { setHeapStatus('enter an integer.'); return; }
    await heapPush(v, { animate: true });
    renderHeap();
    setHeapStatus(`inserted <strong>${v}</strong>. ${heapSummary()}`);
    el.value = '';
  });
  document.getElementById('heap-insert').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') document.getElementById('heap-add').click();
  });
  document.getElementById('heap-extract').addEventListener('click', async () => {
    if (!HEAP.arr.length) { setHeapStatus('heap is empty.'); return; }
    const root = HEAP.arr[0];
    setHeapStatus(`extracting root <strong>${root}</strong> — moving last element to root, then sifting down.`);
    await sleep(280);
    const popped = await heapExtract({ animate: true });
    renderHeap();
    setHeapStatus(`extracted <strong>${popped}</strong>. ${heapSummary()}`);
  });
  document.getElementById('heap-random').addEventListener('click', async () => {
    heapClear();
    for (let k = 0; k < 8; k++) await heapPush(1 + Math.floor(Math.random() * 50));
    renderHeap();
    setHeapStatus(`inserted 8 random values one at a time. ${heapSummary()}`);
  });
  document.getElementById('heap-heapify').addEventListener('click', async () => {
    HEAP.arr = [];
    for (let k = 0; k < 12; k++) HEAP.arr.push(1 + Math.floor(Math.random() * 99));
    renderHeap();
    setHeapStatus(`unheaped random array of 12 values. running Floyd's bottom-up heapify…`);
    await sleep(420);
    await heapifyAll();
  });
  document.getElementById('heap-clear').addEventListener('click', () => {
    heapClear(); renderHeap(); setHeapStatus('empty heap.');
  });

  // Initial demo
  [9, 4, 6, 2, 11, 7, 1].forEach((v) => { HEAP.arr.push(v); siftUp(HEAP.arr.length - 1); });
  renderHeap();
  setHeapStatus(`demo heap pre-populated. ${heapSummary()}`);
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-cost':
      '\\text{BST op cost} = O(\\text{depth of touched node}) = \\begin{cases} O(\\log n) & \\text{balanced} \\\\ O(n) & \\text{worst case (degenerate)} \\end{cases}',
    'math-heap-index':
      '\\text{parent}(i) = \\lfloor (i - 1)/2 \\rfloor, \\quad \\text{left}(i) = 2i + 1, \\quad \\text{right}(i) = 2i + 2',
    'math-heapify-sum':
      '\\sum_{d=0}^{h} \\left\\lceil \\frac{n}{2^{d+1}} \\right\\rceil \\cdot d \\;=\\; O\\!\\left( n \\sum_{d=0}^{\\infty} \\frac{d}{2^{d+1}} \\right) \\;=\\; O(n)'
  };
  Object.keys(blocks).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    try { katex.render(blocks[id], el, { displayMode: true, throwOnError: false }); } catch (_) {}
  });
}

function boot() {
  wireBST();
  wireHeap();
  if (window.katex) renderMath();
  else {
    const s = document.querySelector('script[src*="katex"]');
    if (s) s.addEventListener('load', renderMath);
  }
  // Re-render on resize so SVG width is right
  let resizeT = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => { renderBST(); renderHeap(); }, 120);
  });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
