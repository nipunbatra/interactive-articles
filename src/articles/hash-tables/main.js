// ============================================================
// Hash Tables — chaining + three open-addressing variants on a
// 16-slot toy table. The interactive widget exposes load factor,
// average probe length, and tombstone count live.
// ============================================================

const M = 16;            // table size
const TOMB = '__tomb__'; // tombstone marker for open-addressing

const STATE = {
  strategy: 'linear', // 'chain' | 'linear' | 'quadratic' | 'robin'
  hashName: 'fnv',
  slots: null,             // for OA: array of {key, dist} | null | TOMB
  chains: null,            // for chaining: array of arrays of {key, h}
  totalProbes: 0,
  count: 0,                // number of live keys
  // animation hint (last action highlights)
  lastInsertedIdx: null,
  lastProbedPath: [],
  lastFound: null,
  lastTomb: null
};

// ---------------- Hash functions ----------------
function hashFNV(key) {
  let h = 0x811c9dc5;
  const s = String(key);
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}
function hashBad(key) {
  // First char only — adversarial collision factory.
  const s = String(key);
  if (!s.length) return 0;
  return (s.charCodeAt(0) - 97) >>> 0;
}
function hash(key) {
  const fn = STATE.hashName === 'fnv' ? hashFNV : hashBad;
  return fn(key);
}
function home(key) { return hash(key) % M; }

function quadStep(i) { return (i + i * i) >>> 0; }

// ---------------- Strategies ----------------
function reset() {
  STATE.slots = new Array(M).fill(null);
  STATE.chains = new Array(M).fill(null).map(() => []);
  STATE.totalProbes = 0;
  STATE.count = 0;
  STATE.lastInsertedIdx = null;
  STATE.lastProbedPath = [];
  STATE.lastFound = null;
  STATE.lastTomb = null;
}

function probeIdx(i, h0) {
  if (STATE.strategy === 'linear' || STATE.strategy === 'robin') {
    return (h0 + i) % M;
  } else if (STATE.strategy === 'quadratic') {
    return (h0 + quadStep(i)) % M;
  }
  return (h0 + i) % M;
}

function insert(key) {
  if (STATE.strategy === 'chain') {
    const h0 = home(key);
    const chain = STATE.chains[h0];
    // Skip duplicates
    if (chain.some((e) => e.key === key)) {
      return { ok: false, reason: 'duplicate', idx: h0 };
    }
    chain.push({ key, h: h0 });
    STATE.count++;
    STATE.lastInsertedIdx = h0;
    STATE.lastProbedPath = [h0];
    return { ok: true, idx: h0, probes: 1 };
  }

  // Open addressing
  const h0 = home(key);
  let firstTomb = -1;
  const path = [];
  for (let i = 0; i < M; i++) {
    const idx = probeIdx(i, h0);
    path.push(idx);
    const slot = STATE.slots[idx];
    if (slot === null) {
      const ins = (firstTomb >= 0) ? firstTomb : idx;
      // Robin-hood handling on the way down — only for 'robin' strategy
      if (STATE.strategy === 'robin') {
        let curKey = key, curHome = h0, curDist = i;
        // Walk the probe sequence forward again, performing swaps.
        let walkI = 0;
        let writeIdx = h0;
        while (walkI < M) {
          writeIdx = (curHome + walkI) % M;
          const s = STATE.slots[writeIdx];
          if (s === null || s === TOMB) {
            STATE.slots[writeIdx] = { key: curKey, dist: walkI };
            STATE.count++;
            STATE.lastInsertedIdx = writeIdx;
            STATE.lastProbedPath = path;
            return { ok: true, idx: writeIdx, probes: path.length };
          }
          // If the incumbent has shorter dist, swap.
          if (s.dist < walkI) {
            const evicted = s;
            STATE.slots[writeIdx] = { key: curKey, dist: walkI };
            curKey = evicted.key;
            // The evicted key now needs to be placed; its new "home" for the walk
            // is (writeIdx - evicted.dist), with starting dist = evicted.dist + 1.
            curHome = (writeIdx - evicted.dist + M) % M;
            walkI = evicted.dist + 1;
          } else {
            walkI++;
          }
        }
        return { ok: false, reason: 'full' };
      }
      STATE.slots[ins] = { key, dist: (ins - h0 + M) % M };
      STATE.count++;
      STATE.lastInsertedIdx = ins;
      STATE.lastProbedPath = path;
      return { ok: true, idx: ins, probes: path.length };
    }
    if (slot === TOMB) {
      if (firstTomb < 0) firstTomb = idx;
      continue;
    }
    if (slot.key === key) {
      // duplicate
      return { ok: false, reason: 'duplicate', idx };
    }
  }
  return { ok: false, reason: 'full' };
}

function search(key) {
  if (STATE.strategy === 'chain') {
    const h0 = home(key);
    const chain = STATE.chains[h0];
    const idx = chain.findIndex((e) => e.key === key);
    STATE.lastProbedPath = [h0];
    return { ok: idx >= 0, idx: h0, probes: 1, posInChain: idx };
  }
  const h0 = home(key);
  const path = [];
  for (let i = 0; i < M; i++) {
    const idx = probeIdx(i, h0);
    path.push(idx);
    const slot = STATE.slots[idx];
    if (slot === null) {
      STATE.lastProbedPath = path;
      return { ok: false, probes: path.length };
    }
    if (slot !== TOMB && slot.key === key) {
      STATE.lastProbedPath = path;
      return { ok: true, idx, probes: path.length };
    }
  }
  STATE.lastProbedPath = path;
  return { ok: false, probes: path.length };
}

function remove(key) {
  if (STATE.strategy === 'chain') {
    const h0 = home(key);
    const chain = STATE.chains[h0];
    const idx = chain.findIndex((e) => e.key === key);
    if (idx >= 0) {
      chain.splice(idx, 1);
      STATE.count--;
      STATE.lastProbedPath = [h0];
      return { ok: true, idx: h0 };
    }
    return { ok: false };
  }
  const r = search(key);
  if (!r.ok) return { ok: false };
  STATE.slots[r.idx] = TOMB;
  STATE.count--;
  STATE.lastTomb = r.idx;
  return { ok: true, idx: r.idx };
}

// ---------------- Stats ----------------
function avgProbeLength() {
  if (STATE.strategy === 'chain') {
    let s = 0, n = 0;
    for (const chain of STATE.chains) {
      for (let i = 0; i < chain.length; i++) { s += i + 1; n++; }
    }
    return n ? s / n : 0;
  }
  let s = 0, n = 0;
  for (const slot of STATE.slots) {
    if (slot && slot !== TOMB) { s += slot.dist + 1; n++; }
  }
  return n ? s / n : 0;
}

function loadFactor() {
  return STATE.count / M;
}

// ---------------- Rendering ----------------
function render() {
  const tbl = document.getElementById('ht-table');
  tbl.innerHTML = '';
  if (STATE.strategy === 'chain') {
    for (let i = 0; i < M; i++) {
      const chain = STATE.chains[i];
      const div = document.createElement('div');
      div.className = 'ht-chain' + (chain.length ? ' has-items' : '');
      div.innerHTML = `<span class="ht-chain-idx">[${i}]</span>`;
      for (const node of chain) {
        const n = document.createElement('span');
        n.className = 'ht-chain-node';
        if (STATE.lastProbedPath.includes(i) && STATE.lastFoundKey === node.key) n.classList.add('found');
        else if (STATE.lastInsertedIdx === i && node.key === STATE.lastInsertedKey) n.classList.add('active');
        n.textContent = node.key;
        div.appendChild(n);
      }
      tbl.appendChild(div);
    }
  } else {
    for (let i = 0; i < M; i++) {
      const slot = STATE.slots[i];
      const div = document.createElement('div');
      let cls = 'ht-slot';
      if (slot === TOMB) cls += ' tomb';
      else if (slot) cls += ' filled';
      if (STATE.lastInsertedIdx === i) cls += ' active';
      else if (STATE.lastProbedPath.includes(i) && i !== STATE.lastInsertedIdx) cls += ' probing';
      if (STATE.lastFound === i) cls += ' active';
      div.className = cls;
      let label = '';
      let dist = '';
      if (slot === TOMB) { label = '×'; }
      else if (slot) {
        label = slot.key;
        if (STATE.strategy === 'robin' && slot.dist > 0) dist = `+${slot.dist}`;
      }
      div.innerHTML = `
        <span class="ht-slot-idx">${i}</span>
        <span class="ht-slot-key">${label || '·'}</span>
        ${dist ? `<span class="ht-slot-dist">${dist}</span>` : ''}
      `;
      tbl.appendChild(div);
    }
  }
  // Stats
  const lf = loadFactor();
  const ap = avgProbeLength();
  document.getElementById('ht-load').textContent = lf.toFixed(2);
  document.getElementById('ht-count').textContent = STATE.count;
  document.getElementById('ht-probe').textContent = ap.toFixed(2);
  const loadCard = document.getElementById('ht-load-card');
  loadCard.classList.remove('is-warn', 'is-bad');
  if (lf > 0.9) loadCard.classList.add('is-bad');
  else if (lf > 0.7) loadCard.classList.add('is-warn');
  const probeCard = document.getElementById('ht-probe-card');
  probeCard.classList.remove('is-warn', 'is-bad');
  if (ap > 4) probeCard.classList.add('is-bad');
  else if (ap > 2.5) probeCard.classList.add('is-warn');
}

function setStatus(html) {
  document.getElementById('ht-status').innerHTML = html;
}

// ---------------- Wiring ----------------
function changeStrategy(s) {
  STATE.strategy = s;
  reset();
  setStatus(`switched strategy to <strong>${s}</strong>. table cleared.`);
  render();
}

function wire() {
  reset();
  render();

  document.getElementById('ht-strategy').addEventListener('change', (e) => changeStrategy(e.target.value));
  document.getElementById('ht-hash').addEventListener('change', (e) => {
    STATE.hashName = e.target.value;
    setStatus(`switched hash to <strong>${e.target.value}</strong>. existing keys keep their slots; new inserts use the new hash.`);
  });

  const keyEl = document.getElementById('ht-key');
  function doAdd() {
    const k = keyEl.value.trim();
    if (!k) return;
    STATE.lastFound = null;
    STATE.lastInsertedKey = k;
    const r = insert(k);
    if (!r.ok) {
      if (r.reason === 'duplicate') setStatus(`<strong>${k}</strong> already in table (at slot ${r.idx}).`);
      else setStatus(`table full — cannot insert <strong>${k}</strong>. clear or resize.`);
    } else {
      const h0 = home(k);
      const probeStr = r.probes > 1 ? `, probed ${r.probes} slots` : '';
      setStatus(`inserted <strong>${k}</strong> (hash → slot ${h0}) at slot <strong>${r.idx}</strong>${probeStr}.`);
    }
    keyEl.value = '';
    render();
  }
  document.getElementById('ht-add').addEventListener('click', doAdd);
  keyEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') doAdd(); });

  document.getElementById('ht-find').addEventListener('click', () => {
    const k = keyEl.value.trim();
    if (!k) return;
    STATE.lastInsertedIdx = null;
    STATE.lastFound = null;
    STATE.lastFoundKey = null;
    const r = search(k);
    if (r.ok) {
      STATE.lastFound = r.idx;
      STATE.lastFoundKey = k;
      setStatus(`found <strong>${k}</strong> at slot <strong>${r.idx}</strong> in ${r.probes} probe${r.probes === 1 ? '' : 's'}.`);
    } else {
      setStatus(`<strong>${k}</strong> not in table (walked ${r.probes} probe${r.probes === 1 ? '' : 's'}).`);
    }
    render();
  });

  document.getElementById('ht-del').addEventListener('click', () => {
    const k = keyEl.value.trim();
    if (!k) return;
    const r = remove(k);
    if (r.ok) setStatus(`deleted <strong>${k}</strong>${STATE.strategy === 'chain' ? '' : ' (slot becomes a tombstone)'}.`);
    else setStatus(`<strong>${k}</strong> not in table.`);
    render();
  });

  document.getElementById('ht-burst').addEventListener('click', () => {
    const sample = ['cat','dog','bat','ant','ape','bee','car','arc','bee2','elk','owl','fox','jay','cow','pig','rat','crow','cob','can','ash'];
    let inserted = 0;
    while (inserted < 10) {
      const k = sample[Math.floor(Math.random() * sample.length)] + Math.floor(Math.random() * 999);
      const r = insert(k);
      if (r.ok) inserted++;
      if (!r.ok && r.reason === 'full') break;
    }
    setStatus(`inserted ${inserted} random keys via <strong>${STATE.strategy}</strong>.`);
    render();
  });

  document.getElementById('ht-clear').addEventListener('click', () => {
    reset();
    setStatus('table cleared.');
    render();
  });
}

function renderMath() {
  if (!window.katex) return;
  const blocks = {
    'math-mod':
      'h(\\text{key}) \\bmod m: \\quad \\text{uniformity over slots} \\;\\Leftrightarrow\\; \\text{collision rate} \\sim \\binom{n}{2} / m',
    'math-chain':
      '\\mathbb{E}[\\text{lookup length}] = 1 + \\frac{\\alpha}{2}, \\quad \\text{worst case } O(n)',
    'math-linear':
      '\\mathbb{E}[\\text{probes}] = \\frac{1}{1 - \\alpha} \\text{ (Knuth, 1962)} \\;\\to\\; \\infty \\text{ as } \\alpha \\to 1',
    'math-robin':
      '\\text{Robin Hood: keep slot dist} = (\\text{idx} - \\text{home}) \\bmod m \\text{ minimised} \\;\\Rightarrow\\; \\text{Var}(\\text{probe}) \\to 0'
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
