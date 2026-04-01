export function renderMath() {
  if (!window.katex) return;
  var blocks = {
    'math-choose':    ['\\binom{10}{5} = \\frac{10!}{5!\\;5!} = 252', true],
    'math-pval':      ['p = \\frac{\\text{\\# splits with gap} \\ge \\text{observed gap}}{\\text{total splits (252)}}', true],
    'math-bigchoose': ['\\binom{40}{20} = 137{,}846{,}528{,}640', true],
    'math-se-formula':['SE = \\sqrt{\\frac{s_A^2}{n_A} + \\frac{s_B^2}{n_B}}', true]
  };
  Object.keys(blocks).forEach(function (id) {
    var el = document.getElementById(id);
    if (!el) return;
    try {
      katex.render(blocks[id][0], el, { displayMode: blocks[id][1], throwOnError: false });
    } catch (_) {}
  });
}

function erf(x) {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

export function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

export function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

export function meanGap(a, b) {
  return Math.abs(mean(a) - mean(b));
}

export function variance(arr) {
  const m = mean(arr);
  return arr.reduce((acc, val) => acc + Math.pow(val - m, 2), 0) / (arr.length - 1);
}

export function standardError(groupA, groupB) {
  return Math.sqrt(variance(groupA) / groupA.length + variance(groupB) / groupB.length);
}

export function combinations(array, choose) {
  const results = [];
  function helper(start, combo) {
    if (combo.length === choose) {
      results.push([...combo]);
      return;
    }
    for (let i = start; i < array.length; i++) {
      combo.push(array[i]);
      helper(i + 1, combo);
      combo.pop();
    }
  }
  helper(0, []);
  return results;
}

export function generateAllResplits(values) {
  const indexed = values.map((value, index) => ({ value, index }));
  const groupAs = combinations(indexed, values.length / 2);

  return groupAs.map(groupA => {
    const groupAIds = new Set(groupA.map(item => item.index));
    const groupB = indexed.filter(item => !groupAIds.has(item.index));
    const sampleA = groupA.map(item => item.value);
    const sampleB = groupB.map(item => item.value);
    const mA = mean(sampleA);
    const mB = mean(sampleB);
    return {
      sampleA,
      sampleB,
      gap: Math.abs(mA - mB),
      diff: mA - mB
    };
  });
}

export function shuffle(array) {
  let currentIndex = array.length, randomIndex;
  while (currentIndex !== 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
  }
  return array;
}