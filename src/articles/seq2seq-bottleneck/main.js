// Seq2Seq bottleneck · pedagogical interactive
const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const k of Object.keys(attrs)) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

const SCENARIOS = {
  news: {
    short: "The minister announced new tariffs today.",
    medium: "The minister announced new tariffs today, citing rising imports and the need to protect domestic industry during the coming fiscal year.",
    long: "The minister announced new tariffs today, citing rising imports and the need to protect domestic industry during the coming fiscal year, though critics immediately pointed out that similar measures introduced five years ago under a different administration had failed to produce the promised results, and economists warned of retaliation from trading partners that could ultimately harm exporters."
  },
  story: {
    short: "She walked into the forest at dawn.",
    medium: "She walked into the forest at dawn, her boots crunching the frost and her breath clouding in the cold air as she followed the narrow trail upward.",
    long: "She walked into the forest at dawn, her boots crunching the frost and her breath clouding in the cold air as she followed the narrow trail upward, past the lightning-scarred oak where she had played as a child, toward the ridge where, if the old maps could be trusted, a hidden spring had fed the village for centuries before the road was built."
  },
  academic: {
    short: "We evaluate the model on three benchmarks.",
    medium: "We evaluate the model on three benchmarks spanning classification, regression, and structured prediction, using standard train/validation splits and reporting results averaged over five random seeds.",
    long: "We evaluate the model on three benchmarks spanning classification, regression, and structured prediction, using standard train/validation splits and reporting results averaged over five random seeds; hyperparameters are selected via grid search on the validation set, and all comparisons are performed under equivalent compute budgets to ensure that observed improvements reflect genuine algorithmic differences rather than artefacts of tuning."
  }
};
let currentScenario = "news";

function sentenceTokens(len) {
  const s = SCENARIOS[currentScenario];
  const words = (s.long || "").split(/\s+/);
  const snap = Math.min(len, words.length);
  return words.slice(0, snap);
}

// ============== Step 1: source sentence + bits ==============
function renderStep1() {
  const len = parseInt(document.getElementById("len").value);
  document.getElementById("len-val").textContent = len;
  const tokens = sentenceTokens(len);
  const svg = document.getElementById("step1-plot");
  clear(svg);
  svg.appendChild(el("rect", { x: 0, y: 0, width: 880, height: 220, fill: "#fdfcf9" }));

  // Render tokens as box row
  const maxPerRow = 18;
  let x = 20, y = 30;
  let boxH = 30;
  let boxGap = 4;
  const maxWidth = 840;
  const estW = (t) => Math.max(34, t.length * 8 + 10);
  let rowX = x;
  for (let i = 0; i < tokens.length; i++) {
    const w = estW(tokens[i]);
    if (rowX + w > x + maxWidth) {
      y += boxH + 8;
      rowX = x;
    }
    const r = el("rect", { x: rowX, y, width: w, height: boxH, fill: "#fff3ee", stroke: "#d9622b", "stroke-width": 1, rx: 3 });
    svg.appendChild(r);
    svg.appendChild(el("text", { x: rowX + w/2, y: y + 20, "text-anchor": "middle", "font-family": "Source Serif 4, serif", "font-size": 13, fill: "#1a1815" }, tokens[i]));
    rowX += w + boxGap;
  }

  document.getElementById("src-tokens").textContent = tokens.length;
  const bitsPerToken = 10;
  document.getElementById("src-bits").textContent = `≈ ${tokens.length * bitsPerToken} bits`;
}

// ============== Step 2: compression visualization ==============
function renderStep2() {
  const len = parseInt(document.getElementById("len").value);
  const ctx = parseInt(document.getElementById("ctx-dim").value);
  const svg = document.getElementById("step2-plot");
  clear(svg);
  svg.appendChild(el("rect", { x: 0, y: 0, width: 880, height: 260, fill: "#fdfcf9" }));

  // Encoder boxes across top
  const nTok = Math.min(len, 40);
  const startX = 40, endX = 840;
  const dx = (endX - startX) / Math.max(nTok - 1, 1);

  // Vertical axis label
  svg.appendChild(el("text", { x: 15, y: 60, "font-family": "Manrope, sans-serif", "font-size": 11, fill: "#6e665b" }, "encoder steps →"));

  // Draw encoder steps as small boxes, with hidden state intensity fading
  for (let i = 0; i < nTok; i++) {
    const tx = startX + i * dx;
    svg.appendChild(el("rect", { x: tx - 12, y: 50, width: 24, height: 20, fill: "#fff3ee", stroke: "#d9622b", "stroke-width": 0.7, rx: 2 }));
    svg.appendChild(el("text", { x: tx, y: 63, "text-anchor": "middle", "font-family": "Manrope, sans-serif", "font-size": 9, fill: "#1a1815" }, `t${i+1}`));
  }

  // Context vector as a single row of cells
  svg.appendChild(el("text", { x: 15, y: 140, "font-family": "Manrope, sans-serif", "font-size": 11, fill: "#6e665b" }, "context vector"));

  // Draw ctx-dim cells (cap display at 64 cells, intensity by how recent the information is)
  const displayDim = Math.min(ctx, 64);
  const cellW = 760 / displayDim;
  for (let k = 0; k < displayDim; k++) {
    // intensity proportional to where this cell's info came from
    const originToken = Math.floor((k / displayDim) * nTok);
    const recency = 1 - (nTok - originToken - 1) / Math.max(nTok, 1);
    const saturation = Math.max(0, 0.15 + recency * 0.7);
    svg.appendChild(el("rect", {
      x: 60 + k * cellW, y: 130, width: cellW - 1, height: 20,
      fill: `rgba(217, 98, 43, ${saturation.toFixed(2)})`,
      stroke: "#c9c4b5", "stroke-width": 0.4
    }));
  }

  // Arrow from encoders to context vector
  svg.appendChild(el("path", { d: "M 440 75 L 440 125", stroke: "#1a1815", "stroke-width": 1.2, fill: "none", "marker-end": "url(#arr)" }));

  // Info loss curve (stylized)
  const capacity = Math.log2(ctx) * 32;  // stylized
  const entropy = nTok * 10;
  const loss = Math.max(0, (entropy - capacity) / Math.max(entropy, 1)) * 100;

  // Bar chart at bottom
  svg.appendChild(el("text", { x: 440, y: 200, "text-anchor": "middle", "font-family": "Manrope, sans-serif", "font-size": 12, fill: "#6e665b" }, `sentence entropy ≈ ${entropy} bits`));
  svg.appendChild(el("text", { x: 440, y: 220, "text-anchor": "middle", "font-family": "Manrope, sans-serif", "font-size": 12, fill: "#6e665b" }, `context capacity ≈ ${Math.round(capacity)} bits`));

  // Add arrow marker
  const defs = el("defs");
  const marker = el("marker", { id: "arr", markerWidth: 8, markerHeight: 8, refX: 7, refY: 4, orient: "auto" });
  marker.appendChild(el("polygon", { points: "0 0, 8 4, 0 8", fill: "#1a1815" }));
  defs.appendChild(marker);
  svg.appendChild(defs);

  document.getElementById("ctx-cap").textContent = `${Math.round(capacity)} bits`;
  document.getElementById("sent-ent").textContent = `${entropy} bits`;
  document.getElementById("comp-loss").textContent = `${loss.toFixed(1)}%`;
}

// ============== Step 3: BLEU curves ==============
function renderStep3() {
  const svg = document.getElementById("step3-plot");
  clear(svg);
  svg.appendChild(el("rect", { x: 0, y: 0, width: 880, height: 360, fill: "#fdfcf9" }));

  const M = { l: 60, r: 30, t: 30, b: 50 };
  const pW = 880 - M.l - M.r, pH = 360 - M.t - M.b;
  const xMin = 5, xMax = 60, yMin = 0, yMax = 40;
  const xS = x => M.l + (x - xMin)/(xMax-xMin) * pW;
  const yS = y => 360 - M.b - (y - yMin)/(yMax-yMin) * pH;

  // axes
  svg.appendChild(el("line", { x1: M.l, y1: 360 - M.b, x2: 880 - M.r, y2: 360 - M.b, stroke: "#1a1815", "stroke-width": 1 }));
  svg.appendChild(el("line", { x1: M.l, y1: M.t, x2: M.l, y2: 360 - M.b, stroke: "#1a1815", "stroke-width": 1 }));
  // x ticks
  [5, 15, 25, 35, 45, 55].forEach(x => {
    svg.appendChild(el("line", { x1: xS(x), y1: 360-M.b, x2: xS(x), y2: 360-M.b+4, stroke: "#6e665b" }));
    svg.appendChild(el("text", { x: xS(x), y: 360-M.b+18, "text-anchor": "middle", "font-family": "IBM Plex Mono, monospace", "font-size": 11, fill: "#6e665b" }, x));
  });
  // y ticks
  [0, 10, 20, 30, 40].forEach(y => {
    svg.appendChild(el("line", { x1: M.l-4, y1: yS(y), x2: M.l, y2: yS(y), stroke: "#6e665b" }));
    svg.appendChild(el("text", { x: M.l-8, y: yS(y)+3, "text-anchor": "end", "font-family": "IBM Plex Mono, monospace", "font-size": 11, fill: "#6e665b" }, y));
  });
  // labels
  svg.appendChild(el("text", { x: 440, y: 350, "text-anchor": "middle", "font-family": "Manrope, sans-serif", "font-size": 12, fill: "#6e665b" }, "source sentence length (tokens)"));
  svg.appendChild(el("text", { x: 20, y: 180, "text-anchor": "middle", "font-family": "Manrope, sans-serif", "font-size": 12, fill: "#6e665b", transform: "rotate(-90, 20, 180)" }, "BLEU"));

  // Seq2Seq plain curve: drops from 28 at len=5 to 15 at len=60
  const seq2seq = x => 30 - 0.35 * (x - 5);
  // Reversed: ~4 points higher
  const reversed = x => 34 - 0.28 * (x - 5);
  // Bahdanau attention: stays high
  const attn = x => 34 - 0.05 * (x - 5);

  const drawCurve = (fn, color, width = 2) => {
    let d = "", first = true;
    for (let x = xMin; x <= xMax; x += 1) {
      const xp = xS(x), yp = yS(Math.max(0, fn(x)));
      d += (first ? `M ${xp} ${yp}` : ` L ${xp} ${yp}`);
      first = false;
    }
    svg.appendChild(el("path", { d, stroke: color, "stroke-width": width, fill: "none" }));
  };
  drawCurve(seq2seq, "#d9622b", 2.5);
  drawCurve(reversed, "#7b9e89", 2.2);
  drawCurve(attn, "#4a6670", 2.5);

  // highlight cliff region
  svg.appendChild(el("rect", { x: xS(30), y: M.t, width: xS(60)-xS(30), height: pH, fill: "#a8324b", "fill-opacity": 0.05 }));
  svg.appendChild(el("text", { x: xS(45), y: 45, "text-anchor": "middle", "font-family": "Manrope, sans-serif", "font-size": 11, fill: "#a8324b" }, "bottleneck region"));
}

// ============== Step 4: recall by position ==============
function renderStep4() {
  const svg = document.getElementById("step4-plot");
  clear(svg);
  svg.appendChild(el("rect", { x: 0, y: 0, width: 880, height: 240, fill: "#fdfcf9" }));

  const nTok = 21;
  const recallIdx = parseInt(document.getElementById("recall").value);
  document.getElementById("recall-val").textContent = recallIdx;
  document.getElementById("pos-val").textContent = recallIdx;

  const startX = 40, endX = 840;
  const dx = (endX - startX) / (nTok - 1);
  const sample = sentenceTokens(nTok);

  // Recall accuracy shaped like a U (high at ends, low in middle)
  const recallAcc = pos => {
    const norm = (pos - (nTok-1)/2) / ((nTok-1)/2);
    return 0.35 + 0.55 * Math.abs(norm);
  };

  for (let i = 0; i < nTok; i++) {
    const tx = startX + i * dx;
    const acc = recallAcc(i);
    const color = acc > 0.7 ? "#7b9e89" : (acc > 0.5 ? "#c9a961" : "#a8324b");
    svg.appendChild(el("rect", { x: tx - 18, y: 80, width: 36, height: 30, fill: color, "fill-opacity": 0.3, stroke: color, "stroke-width": 1.2, rx: 2 }));
    svg.appendChild(el("text", { x: tx, y: 100, "text-anchor": "middle", "font-family": "Source Serif 4, serif", "font-size": 10, fill: "#1a1815" }, (sample[i]||`t${i+1}`).substring(0,6)));

    // Accuracy bar
    const barH = 70 * acc;
    svg.appendChild(el("rect", { x: tx - 12, y: 200 - barH, width: 24, height: barH, fill: color, "fill-opacity": 0.7 }));
  }

  // highlight the queried position
  const tx = startX + recallIdx * dx;
  svg.appendChild(el("rect", { x: tx - 20, y: 78, width: 40, height: 34, fill: "none", stroke: "#1a1815", "stroke-width": 2, rx: 3 }));

  svg.appendChild(el("text", { x: 15, y: 95, "font-family": "Manrope, sans-serif", "font-size": 11, fill: "#6e665b" }, "tokens"));
  svg.appendChild(el("text", { x: 15, y: 180, "font-family": "Manrope, sans-serif", "font-size": 11, fill: "#6e665b" }, "recall"));
  svg.appendChild(el("text", { x: 440, y: 228, "text-anchor": "middle", "font-family": "Manrope, sans-serif", "font-size": 12, fill: "#6e665b" }, "U-shaped recall · middle tokens lost first"));

  document.getElementById("recall-acc").textContent = `${Math.round(recallAcc(recallIdx) * 100)}%`;
}

// ============== Step 5: attention weights ==============
function renderStep5() {
  const svg = document.getElementById("step5-plot");
  clear(svg);
  svg.appendChild(el("rect", { x: 0, y: 0, width: 880, height: 260, fill: "#fdfcf9" }));

  const src = ["The", "cat", "sat", "on", "the", "mat"];
  const tgt = ["Le", "chat", "était", "sur", "le", "tapis"];

  // target on left, source on top, attention weights in matrix
  const cellW = 90, cellH = 28;
  const xOff = 160, yOff = 60;

  // Source headers
  for (let i = 0; i < src.length; i++) {
    const tx = xOff + i * cellW + cellW/2;
    svg.appendChild(el("text", { x: tx, y: 45, "text-anchor": "middle", "font-family": "Source Serif 4, serif", "font-size": 13, fill: "#1a1815" }, src[i]));
  }

  // Target rows + attention cells
  for (let t = 0; t < tgt.length; t++) {
    svg.appendChild(el("text", { x: xOff - 12, y: yOff + t * cellH + 18, "text-anchor": "end", "font-family": "Source Serif 4, serif", "font-size": 13, fill: "#1a1815" }, tgt[t]));
    for (let i = 0; i < src.length; i++) {
      // attention weight: peaked at aligned position (slightly rough)
      const diff = Math.abs(t - i);
      const w = Math.exp(-diff * diff * 0.5);
      const norm = Math.exp(-0 * 0 * 0.5) + Math.exp(-1 * 1 * 0.5) + Math.exp(-2 * 2 * 0.5) + Math.exp(-3 * 3 * 0.5) + Math.exp(-4 * 4 * 0.5) + Math.exp(-5 * 5 * 0.5);
      const a = w / norm;
      svg.appendChild(el("rect", { x: xOff + i * cellW, y: yOff + t * cellH, width: cellW - 3, height: cellH - 3, fill: `rgba(217, 98, 43, ${Math.min(0.95, a * 3).toFixed(2)})`, stroke: "#c9c4b5", "stroke-width": 0.4 }));
      svg.appendChild(el("text", { x: xOff + i * cellW + cellW/2, y: yOff + t * cellH + 17, "text-anchor": "middle", "font-family": "IBM Plex Mono, monospace", "font-size": 9, fill: "#1a1815", "fill-opacity": 0.8 }, a.toFixed(2)));
    }
  }

  // Labels
  svg.appendChild(el("text", { x: xOff + (src.length * cellW)/2, y: 20, "text-anchor": "middle", "font-family": "Manrope, sans-serif", "font-size": 12, fill: "#6e665b" }, "source ·"));
  svg.appendChild(el("text", { x: 80, y: yOff - 8, "font-family": "Manrope, sans-serif", "font-size": 12, fill: "#6e665b" }, "target ↓"));
  svg.appendChild(el("text", { x: 440, y: 240, "text-anchor": "middle", "font-family": "Manrope, sans-serif", "font-size": 12, fill: "#6e665b" }, "each target row = soft attention distribution over sources"));
}

document.querySelectorAll("#scenario-buttons .mode-button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#scenario-buttons .mode-button").forEach(b => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    currentScenario = btn.dataset.case;
    renderStep1();
    renderStep2();
  });
});
document.getElementById("len").addEventListener("input", () => { renderStep1(); renderStep2(); });
document.getElementById("ctx-dim").addEventListener("change", renderStep2);
document.getElementById("recall").addEventListener("input", renderStep4);

renderStep1();
renderStep2();
renderStep3();
renderStep4();
renderStep5();
