const TAU = Math.PI * 2;
const SQRT_TWO_PI = Math.sqrt(TAU);

const oneDState = {
  mu: 0,
  sigma: 1,
  a: -1,
  b: 1,
};

const twoDState = {
  muX: 0,
  muY: 0,
  sigmaX: 1.2,
  sigmaY: 0.9,
  rho: 0.55,
  xSliceOffset: 0,
  ySliceOffset: 0,
};

const elements = {
  progressFill: document.getElementById('progressFill'),
  navLinks: Array.from(document.querySelectorAll('.toc__link')),
  sections: Array.from(document.querySelectorAll('[data-nav]')),

  oneDCanvas: document.getElementById('oneDCanvas'),
  mu1: document.getElementById('mu1'),
  mu1Value: document.getElementById('mu1Value'),
  sigma1: document.getElementById('sigma1'),
  sigma1Value: document.getElementById('sigma1Value'),
  intervalA: document.getElementById('intervalA'),
  intervalAValue: document.getElementById('intervalAValue'),
  intervalB: document.getElementById('intervalB'),
  intervalBValue: document.getElementById('intervalBValue'),
  oneDProbability: document.getElementById('oneDProbability'),
  oneDProbabilityCopy: document.getElementById('oneDProbabilityCopy'),
  oneDZRange: document.getElementById('oneDZRange'),
  oneDPeak: document.getElementById('oneDPeak'),

  muX: document.getElementById('muX'),
  muXValue: document.getElementById('muXValue'),
  muY: document.getElementById('muY'),
  muYValue: document.getElementById('muYValue'),
  sigmaX: document.getElementById('sigmaX'),
  sigmaXValue: document.getElementById('sigmaXValue'),
  sigmaY: document.getElementById('sigmaY'),
  sigmaYValue: document.getElementById('sigmaYValue'),
  rho: document.getElementById('rho'),
  rhoValue: document.getElementById('rhoValue'),
  covarianceMatrix: document.getElementById('covarianceMatrix'),
  ellipseAngle: document.getElementById('ellipseAngle'),
  majorSpread: document.getElementById('majorSpread'),
  twoDCanvas: document.getElementById('twoDCanvas'),

  xSliceOffset: document.getElementById('xSliceOffset'),
  xSliceOffsetValue: document.getElementById('xSliceOffsetValue'),
  ySliceOffset: document.getElementById('ySliceOffset'),
  ySliceOffsetValue: document.getElementById('ySliceOffsetValue'),
  jointDistributionMath: document.getElementById('jointDistributionMath'),
  marginalTopCanvas: document.getElementById('marginalTopCanvas'),
  marginalMainCanvas: document.getElementById('marginalMainCanvas'),
  marginalRightCanvas: document.getElementById('marginalRightCanvas'),
  marginalTopSummary: document.getElementById('marginalTopSummary'),
  marginalXMath: document.getElementById('marginalXMath'),
  marginalYMath: document.getElementById('marginalYMath'),
  marginalCorrelationNote: document.getElementById('marginalCorrelationNote'),
  jointNarrative: document.getElementById('jointNarrative'),
  conditionalMainCanvas: document.getElementById('conditionalMainCanvas'),
  conditionalTopCanvas: document.getElementById('conditionalTopCanvas'),
  conditionalRightCanvas: document.getElementById('conditionalRightCanvas'),
  conditionalTopSummary: document.getElementById('conditionalTopSummary'),
  sliceReadout: document.getElementById('sliceReadout'),
  xConditionalFormula: document.getElementById('xConditionalFormula'),
  yConditionalFormula: document.getElementById('yConditionalFormula'),
  shrinkageFormula: document.getElementById('shrinkageFormula'),
  jointSummary: document.getElementById('jointSummary'),
};

const baseSamples = Array.from({ length: 280 }, () => [randn(), randn()]);
const conditionalInteraction = {
  plane: null,
  dragMode: null,
};

function normalPdf(x, mu = 0, sigma = 1) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * SQRT_TWO_PI);
}

function erf(x) {
  const sign = Math.sign(x) || 1;
  const absX = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * absX);
  const poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t);
  return sign * (1 - poly * Math.exp(-absX * absX));
}

function normalCdf(x, mu = 0, sigma = 1) {
  return 0.5 * (1 + erf((x - mu) / (sigma * Math.SQRT2)));
}

function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function formatSigned(value, digits = 2) {
  const rounded = round(value, digits).toFixed(digits);
  return value > 0 ? `+${rounded}` : rounded;
}

function resizeCanvas(canvas) {
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const cssWidth = canvas.clientWidth || canvas.width;
  const cssHeight = canvas.clientHeight || canvas.height;
  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: cssWidth, height: cssHeight };
}

function drawAxes(ctx, width, height, xMin, xMax, yMin, yMax, options = {}) {
  const { paddingLeft = 52, paddingRight = 22, paddingTop = 18, paddingBottom = 40 } = options;
  const plotWidth = width - paddingLeft - paddingRight;
  const plotHeight = height - paddingTop - paddingBottom;
  const xToPx = (x) => paddingLeft + ((x - xMin) / (xMax - xMin)) * plotWidth;
  const yToPx = (y) => paddingTop + plotHeight - ((y - yMin) / (yMax - yMin)) * plotHeight;

  ctx.strokeStyle = 'rgba(30, 36, 48, 0.12)';
  ctx.lineWidth = 1;

  const xStep = niceStep((xMax - xMin) / 6);
  const yStep = niceStep((yMax - yMin) / 5);

  ctx.font = '12px "Avenir Next", "Segoe UI", sans-serif';
  ctx.fillStyle = 'rgba(94, 102, 118, 0.85)';

  for (let x = Math.ceil(xMin / xStep) * xStep; x <= xMax + 1e-6; x += xStep) {
    const px = xToPx(x);
    ctx.beginPath();
    ctx.moveTo(px, paddingTop);
    ctx.lineTo(px, height - paddingBottom);
    ctx.stroke();
    ctx.fillText(round(x, 1).toFixed(1), px - 10, height - paddingBottom + 18);
  }

  for (let y = Math.ceil(yMin / yStep) * yStep; y <= yMax + 1e-6; y += yStep) {
    const py = yToPx(y);
    ctx.beginPath();
    ctx.moveTo(paddingLeft, py);
    ctx.lineTo(width - paddingRight, py);
    ctx.stroke();
    if (Math.abs(y) > 1e-6) {
      ctx.fillText(round(y, 2).toFixed(2), 8, py + 4);
    }
  }

  ctx.strokeStyle = 'rgba(30, 36, 48, 0.35)';
  ctx.beginPath();
  ctx.moveTo(paddingLeft, yToPx(0));
  ctx.lineTo(width - paddingRight, yToPx(0));
  ctx.stroke();

  return { xToPx, yToPx, paddingLeft, paddingRight, paddingTop, paddingBottom, plotWidth, plotHeight };
}

function niceStep(rawStep) {
  const exponent = Math.floor(Math.log10(rawStep));
  const fraction = rawStep / 10 ** exponent;
  let niceFraction;
  if (fraction <= 1) niceFraction = 1;
  else if (fraction <= 2) niceFraction = 2;
  else if (fraction <= 5) niceFraction = 5;
  else niceFraction = 10;
  return niceFraction * 10 ** exponent;
}

function drawOneD() {
  const { ctx, width, height } = resizeCanvas(elements.oneDCanvas);
  ctx.clearRect(0, 0, width, height);

  const a = Math.min(oneDState.a, oneDState.b);
  const b = Math.max(oneDState.a, oneDState.b);
  const extent = Math.max(4.2 * oneDState.sigma, Math.abs(a - oneDState.mu) + oneDState.sigma, Math.abs(b - oneDState.mu) + oneDState.sigma);
  const xMin = oneDState.mu - extent;
  const xMax = oneDState.mu + extent;
  const yMax = normalPdf(oneDState.mu, oneDState.mu, oneDState.sigma) * 1.18;

  const axes = drawAxes(ctx, width, height, xMin, xMax, 0, yMax, {
    paddingLeft: 56,
    paddingRight: 24,
    paddingTop: 18,
    paddingBottom: 42,
  });

  const { xToPx, yToPx } = axes;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(xToPx(xMin), yToPx(0));
  for (let x = xMin; x <= xMax; x += (xMax - xMin) / 320) {
    ctx.lineTo(xToPx(x), yToPx(normalPdf(x, oneDState.mu, oneDState.sigma)));
  }
  ctx.lineTo(xToPx(xMax), yToPx(0));
  ctx.closePath();
  ctx.clip();

  ctx.beginPath();
  ctx.moveTo(xToPx(a), yToPx(0));
  for (let x = a; x <= b; x += (xMax - xMin) / 260) {
    ctx.lineTo(xToPx(x), yToPx(normalPdf(x, oneDState.mu, oneDState.sigma)));
  }
  ctx.lineTo(xToPx(b), yToPx(0));
  ctx.closePath();
  ctx.fillStyle = 'rgba(197, 164, 58, 0.28)';
  ctx.fill();
  ctx.restore();

  ctx.beginPath();
  for (let x = xMin; x <= xMax; x += (xMax - xMin) / 420) {
    const px = xToPx(x);
    const py = yToPx(normalPdf(x, oneDState.mu, oneDState.sigma));
    if (x === xMin) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#d8704f';
  ctx.stroke();

  drawVerticalMarker(ctx, xToPx(oneDState.mu), yToPx(0), yToPx(normalPdf(oneDState.mu, oneDState.mu, oneDState.sigma)), '#2d8b88', '\u03bc');
  drawVerticalMarker(ctx, xToPx(a), yToPx(0), yToPx(normalPdf(a, oneDState.mu, oneDState.sigma)), '#7b6210', 'a');
  drawVerticalMarker(ctx, xToPx(b), yToPx(0), yToPx(normalPdf(b, oneDState.mu, oneDState.sigma)), '#7b6210', 'b');

  const probability = normalCdf(b, oneDState.mu, oneDState.sigma) - normalCdf(a, oneDState.mu, oneDState.sigma);
  const zA = (a - oneDState.mu) / oneDState.sigma;
  const zB = (b - oneDState.mu) / oneDState.sigma;
  const peak = normalPdf(oneDState.mu, oneDState.mu, oneDState.sigma);

  elements.oneDProbability.textContent = `${(probability * 100).toFixed(2)}%`;
  elements.oneDProbabilityCopy.textContent = `Probability that X falls between ${a.toFixed(2)} and ${b.toFixed(2)}.`;
  elements.oneDZRange.textContent = `z = [${zA.toFixed(2)}, ${zB.toFixed(2)}]`;
  elements.oneDPeak.textContent = peak.toFixed(3);
}

function drawVerticalMarker(ctx, x, yBottom, yTop, color, label) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, yBottom);
  ctx.lineTo(x, yTop - 6);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, yTop - 6, 3, 0, TAU);
  ctx.fill();
  ctx.font = '12px "JetBrains Mono", monospace';
  ctx.fillText(label, x - 5, yTop - 16);
}

function eigenDecomposition(sigmaX, sigmaY, rho) {
  const a = sigmaX ** 2;
  const d = sigmaY ** 2;
  const b = rho * sigmaX * sigmaY;
  const trace = a + d;
  const detPart = Math.sqrt((a - d) ** 2 + 4 * b * b);
  const lambda1 = (trace + detPart) / 2;
  const lambda2 = (trace - detPart) / 2;
  const angle = 0.5 * Math.atan2(2 * b, a - d);
  return { lambda1, lambda2, angle };
}

function samplePoints(state) {
  const sqrtTerm = Math.sqrt(Math.max(1 - state.rho ** 2, 0.0001));
  return baseSamples.map(([z1, z2]) => ({
    x: state.muX + state.sigmaX * z1,
    y: state.muY + state.sigmaY * (state.rho * z1 + sqrtTerm * z2),
  }));
}

function getViewBounds(state) {
  return {
    xMin: state.muX - Math.max(3.8 * state.sigmaX, 1.6),
    xMax: state.muX + Math.max(3.8 * state.sigmaX, 1.6),
    yMin: state.muY - Math.max(3.8 * state.sigmaY, 1.6),
    yMax: state.muY + Math.max(3.8 * state.sigmaY, 1.6),
  };
}

function drawTwoDOverview() {
  const { ctx, width, height } = resizeCanvas(elements.twoDCanvas);
  ctx.clearRect(0, 0, width, height);

  const bounds = getViewBounds(twoDState);
  const axes = drawPlane(ctx, width, height, bounds, { padding: 28, equalUnits: true });
  drawGaussianCloud(ctx, axes, twoDState, { showConditional: false });

  const covXY = twoDState.rho * twoDState.sigmaX * twoDState.sigmaY;
  const { lambda1, angle } = eigenDecomposition(twoDState.sigmaX, twoDState.sigmaY, twoDState.rho);
  elements.covarianceMatrix.textContent = `[[${(twoDState.sigmaX ** 2).toFixed(2)}, ${covXY.toFixed(2)}],\n [${covXY.toFixed(2)}, ${(twoDState.sigmaY ** 2).toFixed(2)}]]`;
  elements.ellipseAngle.innerHTML = `${(angle * 180 / Math.PI).toFixed(1)}&deg;`;
  elements.majorSpread.textContent = Math.sqrt(lambda1).toFixed(2);
}

function drawPlane(ctx, width, height, bounds, options = {}) {
  const padding = options.padding ?? 26;
  const paddingLeft = options.paddingLeft ?? padding + 14;
  const paddingRight = options.paddingRight ?? padding;
  const paddingTop = options.paddingTop ?? padding;
  const paddingBottom = options.paddingBottom ?? padding + 14;
  const availableWidth = width - paddingLeft - paddingRight;
  const availableHeight = height - paddingTop - paddingBottom;
  const xRange = bounds.xMax - bounds.xMin;
  const yRange = bounds.yMax - bounds.yMin;
  let plotLeft = paddingLeft;
  let plotTop = paddingTop;
  let plotWidth = availableWidth;
  let plotHeight = availableHeight;

  if (options.equalUnits) {
    const unitsToPx = Math.min(availableWidth / xRange, availableHeight / yRange);
    plotWidth = unitsToPx * xRange;
    plotHeight = unitsToPx * yRange;
    plotLeft = paddingLeft + (availableWidth - plotWidth) * (options.xAlign ?? 0.5);
    plotTop = paddingTop + (availableHeight - plotHeight) * (options.yAlign ?? 0.5);
  }

  const plotRight = plotLeft + plotWidth;
  const plotBottom = plotTop + plotHeight;
  const xToPx = (x) => plotLeft + ((x - bounds.xMin) / xRange) * plotWidth;
  const yToPx = (y) => plotTop + plotHeight - ((y - bounds.yMin) / yRange) * plotHeight;
  const pxToX = (px) => bounds.xMin + ((px - plotLeft) / plotWidth) * xRange;
  const pxToY = (py) => bounds.yMin + ((plotHeight - (py - plotTop)) / plotHeight) * yRange;

  ctx.fillStyle = 'rgba(255, 255, 255, 0.68)';
  ctx.fillRect(plotLeft, plotTop, plotWidth, plotHeight);

  const xStep = niceStep(xRange / 6);
  const yStep = niceStep(yRange / 6);
  const yLabelX = Math.max(4, plotLeft - 36);
  ctx.strokeStyle = 'rgba(30, 36, 48, 0.08)';
  ctx.lineWidth = 1;
  ctx.font = '12px "Avenir Next", sans-serif';
  ctx.fillStyle = 'rgba(94, 102, 118, 0.88)';
  for (let x = Math.ceil(bounds.xMin / xStep) * xStep; x <= bounds.xMax + 1e-6; x += xStep) {
    const px = xToPx(x);
    ctx.beginPath();
    ctx.moveTo(px, plotTop);
    ctx.lineTo(px, plotBottom);
    ctx.stroke();
    ctx.fillText(round(x, 1).toFixed(1), px - 11, plotBottom + 18);
  }
  for (let y = Math.ceil(bounds.yMin / yStep) * yStep; y <= bounds.yMax + 1e-6; y += yStep) {
    const py = yToPx(y);
    ctx.beginPath();
    ctx.moveTo(plotLeft, py);
    ctx.lineTo(plotRight, py);
    ctx.stroke();
    ctx.fillText(round(y, 1).toFixed(1), yLabelX, py + 4);
  }

  ctx.strokeStyle = 'rgba(30, 36, 48, 0.28)';
  ctx.strokeRect(plotLeft, plotTop, plotWidth, plotHeight);

  ctx.fillText('x', plotRight - 10, plotBottom + 34);
  ctx.fillText('y', plotLeft - 12, plotTop + 10);

  return {
    ctx,
    width,
    height,
    bounds,
    paddingLeft,
    paddingRight,
    paddingTop,
    paddingBottom,
    plotLeft,
    plotRight,
    plotTop,
    plotBottom,
    plotWidth,
    plotHeight,
    xToPx,
    yToPx,
    pxToX,
    pxToY,
  };
}

function drawGaussianCloud(ctx, plane, state, options = {}) {
  const { xToPx, yToPx } = plane;
  const { lambda1, lambda2, angle } = eigenDecomposition(state.sigmaX, state.sigmaY, state.rho);
  const levels = [3.1, 2.45, 1.8, 1.15];

  ctx.save();
  ctx.translate(xToPx(state.muX), yToPx(state.muY));
  ctx.rotate(-angle);
  levels.forEach((level, index) => {
    const rx = level * Math.sqrt(lambda1) * plane.plotWidth / (plane.bounds.xMax - plane.bounds.xMin);
    const ry = level * Math.sqrt(lambda2) * plane.plotHeight / (plane.bounds.yMax - plane.bounds.yMin);
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, TAU);
    ctx.fillStyle = `rgba(45, 139, 136, ${0.06 + index * 0.04})`;
    ctx.fill();
    ctx.strokeStyle = `rgba(45, 139, 136, ${0.18 + index * 0.1})`;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  });
  ctx.restore();

  const points = samplePoints(state);
  ctx.fillStyle = 'rgba(216, 112, 79, 0.34)';
  points.forEach((point) => {
    const px = xToPx(point.x);
    const py = yToPx(point.y);
    ctx.beginPath();
    ctx.arc(px, py, 2.1, 0, TAU);
    ctx.fill();
  });

  ctx.fillStyle = '#1e2430';
  ctx.beginPath();
  ctx.arc(xToPx(state.muX), yToPx(state.muY), 4.2, 0, TAU);
  ctx.fill();

  if (options.showConditional) {
    const info = getJointConditionals(state);
    const yFromX = (x) => state.muY + state.rho * (state.sigmaY / state.sigmaX) * (x - state.muX);
    const xFromY = (y) => state.muX + state.rho * (state.sigmaX / state.sigmaY) * (y - state.muY);

    ctx.save();
    ctx.setLineDash([7, 6]);
    ctx.lineWidth = 1.5;

    ctx.strokeStyle = 'rgba(216, 112, 79, 0.45)';
    ctx.beginPath();
    ctx.moveTo(xToPx(plane.bounds.xMin), yToPx(clamp(yFromX(plane.bounds.xMin), plane.bounds.yMin, plane.bounds.yMax)));
    ctx.lineTo(xToPx(plane.bounds.xMax), yToPx(clamp(yFromX(plane.bounds.xMax), plane.bounds.yMin, plane.bounds.yMax)));
    ctx.stroke();

    ctx.strokeStyle = 'rgba(197, 164, 58, 0.55)';
    ctx.beginPath();
    ctx.moveTo(xToPx(clamp(xFromY(plane.bounds.yMin), plane.bounds.xMin, plane.bounds.xMax)), yToPx(plane.bounds.yMin));
    ctx.lineTo(xToPx(clamp(xFromY(plane.bounds.yMax), plane.bounds.xMin, plane.bounds.xMax)), yToPx(plane.bounds.yMax));
    ctx.stroke();
    ctx.restore();

    ctx.strokeStyle = 'rgba(216, 112, 79, 0.95)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xToPx(info.observedX), plane.plotTop);
    ctx.lineTo(xToPx(info.observedX), plane.plotBottom);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(197, 164, 58, 0.95)';
    ctx.beginPath();
    ctx.moveTo(plane.plotLeft, yToPx(info.observedY));
    ctx.lineTo(plane.plotRight, yToPx(info.observedY));
    ctx.stroke();

    ctx.fillStyle = '#d8704f';
    ctx.beginPath();
    ctx.arc(xToPx(info.observedX), yToPx(info.yConditionalMu), 5.5, 0, TAU);
    ctx.fill();

    ctx.fillStyle = '#c5a43a';
    ctx.beginPath();
    ctx.arc(xToPx(info.xConditionalMu), yToPx(info.observedY), 5.5, 0, TAU);
    ctx.fill();

    ctx.fillStyle = 'white';
    ctx.strokeStyle = '#1e2430';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(xToPx(info.observedX), yToPx(info.observedY), 4.6, 0, TAU);
    ctx.fill();
    ctx.stroke();
  }
}

function getJointConditionals(state) {
  const observedX = state.muX + state.xSliceOffset * state.sigmaX;
  const observedY = state.muY + state.ySliceOffset * state.sigmaY;
  const shrinkFactor = Math.sqrt(Math.max(1 - state.rho ** 2, 0));
  const yConditionalMu = state.muY + state.rho * (state.sigmaY / state.sigmaX) * (observedX - state.muX);
  const yConditionalSigma = shrinkFactor * state.sigmaY;
  const xConditionalMu = state.muX + state.rho * (state.sigmaX / state.sigmaY) * (observedY - state.muY);
  const xConditionalSigma = shrinkFactor * state.sigmaX;
  return {
    observedX,
    observedY,
    xConditionalMu,
    xConditionalSigma,
    yConditionalMu,
    yConditionalSigma,
    shrinkFactor,
  };
}

function getJointLayout() {
  return {
    mainPaddingLeft: 44,
    mainPaddingRight: 24,
    mainPaddingTop: 24,
    mainPaddingBottom: 42,
    topPaddingLeft: 44,
    topPaddingRight: 24,
    topPaddingTop: 18,
    topPaddingBottom: 32,
    rightPaddingLeft: 18,
    rightPaddingRight: 18,
    rightPaddingTop: 24,
    rightPaddingBottom: 42,
  };
}

function drawJointMath() {
  const covXY = twoDState.rho * twoDState.sigmaX * twoDState.sigmaY;
  elements.jointDistributionMath.innerHTML = `[X, Y]<sup>T</sup> ~ N([${twoDState.muX.toFixed(2)}, ${twoDState.muY.toFixed(2)}]<sup>T</sup>, [[${(twoDState.sigmaX ** 2).toFixed(2)}, ${covXY.toFixed(2)}], [${covXY.toFixed(2)}, ${(twoDState.sigmaY ** 2).toFixed(2)}]])`;
}

function drawMarginalGuides(ctx, plane, state) {
  ctx.save();
  ctx.setLineDash([8, 8]);
  ctx.lineWidth = 1.6;
  ctx.strokeStyle = 'rgba(45, 139, 136, 0.72)';
  ctx.beginPath();
  ctx.moveTo(plane.xToPx(state.muX), plane.plotTop);
  ctx.lineTo(plane.xToPx(state.muX), plane.plotBottom);
  ctx.moveTo(plane.plotLeft, plane.yToPx(state.muY));
  ctx.lineTo(plane.plotRight, plane.yToPx(state.muY));
  ctx.stroke();

  ctx.fillStyle = '#2d8b88';
  ctx.beginPath();
  ctx.arc(plane.xToPx(state.muX), plane.yToPx(state.muY), 5, 0, TAU);
  ctx.fill();
  ctx.restore();
}

function syncRightAxisCanvasHeight(rightCanvas, mainCanvas) {
  const mainHeight = mainCanvas.clientHeight;
  if (mainHeight > 0) {
    rightCanvas.style.height = `${mainHeight}px`;
  }
}

function drawMarginalStage() {
  const layout = getJointLayout();
  const bounds = getViewBounds(twoDState);

  const main = resizeCanvas(elements.marginalMainCanvas);
  main.ctx.clearRect(0, 0, main.width, main.height);
  const plane = drawPlane(main.ctx, main.width, main.height, bounds, {
    paddingLeft: layout.mainPaddingLeft,
    paddingRight: layout.mainPaddingRight,
    paddingTop: layout.mainPaddingTop,
    paddingBottom: layout.mainPaddingBottom,
    equalUnits: true,
  });
  drawGaussianCloud(main.ctx, plane, twoDState, { showConditional: false });
  drawMarginalGuides(main.ctx, plane, twoDState);
  syncRightAxisCanvasHeight(elements.marginalRightCanvas, elements.marginalMainCanvas);

  drawHorizontalDensityOverlay(elements.marginalTopCanvas, {
    boundsMin: bounds.xMin,
    boundsMax: bounds.xMax,
    curves: [
      { mu: twoDState.muX, sigma: twoDState.sigmaX, color: '#2d8b88', fill: 'rgba(45, 139, 136, 0.16)' },
    ],
    paddingLeft: layout.topPaddingLeft,
    paddingRight: layout.topPaddingRight,
    paddingTop: layout.topPaddingTop,
    paddingBottom: layout.topPaddingBottom,
    plotLeft: plane.plotLeft,
    plotRight: plane.plotRight,
    showTickLabels: true,
  });

  drawVerticalDensityOverlay(elements.marginalRightCanvas, {
    boundsMin: bounds.yMin,
    boundsMax: bounds.yMax,
    curves: [
      { mu: twoDState.muY, sigma: twoDState.sigmaY, color: '#2d8b88', fill: 'rgba(45, 139, 136, 0.16)' },
    ],
    paddingLeft: layout.rightPaddingLeft,
    paddingRight: layout.rightPaddingRight,
    paddingTop: layout.rightPaddingTop,
    paddingBottom: layout.rightPaddingBottom,
    plotTop: plane.plotTop,
    plotBottom: plane.plotBottom,
    showTickLabels: true,
  });

  elements.marginalTopSummary.textContent = `X ~ N(${twoDState.muX.toFixed(2)}, ${twoDState.sigmaX.toFixed(2)}²)`;
  elements.marginalXMath.innerHTML = `X ~ N(${twoDState.muX.toFixed(2)}, ${twoDState.sigmaX.toFixed(2)}²)`;
  elements.marginalYMath.innerHTML = `Y ~ N(${twoDState.muY.toFixed(2)}, ${twoDState.sigmaY.toFixed(2)}²)`;
  elements.marginalCorrelationNote.textContent = Math.abs(twoDState.rho) < 0.05
    ? 'With rho near zero, the joint cloud is axis-aligned, so the marginals feel almost obvious.'
    : `Even with rho = ${twoDState.rho.toFixed(2)}, the teal marginals ignore the tilt and only read the diagonal entries of the covariance matrix.`;
}

function drawConditionalStage() {
  const layout = getJointLayout();
  const bounds = getViewBounds(twoDState);
  const info = getJointConditionals(twoDState);

  const main = resizeCanvas(elements.conditionalMainCanvas);
  main.ctx.clearRect(0, 0, main.width, main.height);
  const plane = drawPlane(main.ctx, main.width, main.height, bounds, {
    paddingLeft: layout.mainPaddingLeft,
    paddingRight: layout.mainPaddingRight,
    paddingTop: layout.mainPaddingTop,
    paddingBottom: layout.mainPaddingBottom,
    equalUnits: true,
  });
  drawGaussianCloud(main.ctx, plane, twoDState, { showConditional: true });
  conditionalInteraction.plane = plane;
  syncRightAxisCanvasHeight(elements.conditionalRightCanvas, elements.conditionalMainCanvas);

  drawHorizontalDensityOverlay(elements.conditionalTopCanvas, {
    boundsMin: bounds.xMin,
    boundsMax: bounds.xMax,
    curves: [
      { mu: twoDState.muX, sigma: twoDState.sigmaX, color: '#2d8b88', fill: 'rgba(45, 139, 136, 0.16)' },
      { mu: info.xConditionalMu, sigma: info.xConditionalSigma, color: '#c5a43a', fill: 'rgba(197, 164, 58, 0.18)' },
    ],
    paddingLeft: layout.topPaddingLeft,
    paddingRight: layout.topPaddingRight,
    paddingTop: layout.topPaddingTop,
    paddingBottom: layout.topPaddingBottom,
    plotLeft: plane.plotLeft,
    plotRight: plane.plotRight,
    showTickLabels: true,
  });

  drawVerticalDensityOverlay(elements.conditionalRightCanvas, {
    boundsMin: bounds.yMin,
    boundsMax: bounds.yMax,
    curves: [
      { mu: twoDState.muY, sigma: twoDState.sigmaY, color: '#2d8b88', fill: 'rgba(45, 139, 136, 0.16)' },
      { mu: info.yConditionalMu, sigma: info.yConditionalSigma, color: '#d8704f', fill: 'rgba(216, 112, 79, 0.18)' },
    ],
    paddingLeft: layout.rightPaddingLeft,
    paddingRight: layout.rightPaddingRight,
    paddingTop: layout.rightPaddingTop,
    paddingBottom: layout.rightPaddingBottom,
    plotTop: plane.plotTop,
    plotBottom: plane.plotBottom,
    showTickLabels: true,
  });

  elements.conditionalTopSummary.textContent = `Teal: X ~ N(${twoDState.muX.toFixed(2)}, ${twoDState.sigmaX.toFixed(2)}²). Gold: X | Y = ${info.observedY.toFixed(2)} ~ N(${info.xConditionalMu.toFixed(2)}, ${info.xConditionalSigma.toFixed(2)}²).`;
  elements.sliceReadout.innerHTML = `x<sub>0</sub> = ${info.observedX.toFixed(2)}, y<sub>0</sub> = ${info.observedY.toFixed(2)}`;
  elements.xConditionalFormula.innerHTML = `X ~ N(${twoDState.muX.toFixed(2)}, ${twoDState.sigmaX.toFixed(2)}²)<br />X | Y = ${info.observedY.toFixed(2)} ~ N(${info.xConditionalMu.toFixed(2)}, ${info.xConditionalSigma.toFixed(2)}²)`;
  elements.yConditionalFormula.innerHTML = `Y ~ N(${twoDState.muY.toFixed(2)}, ${twoDState.sigmaY.toFixed(2)}²)<br />Y | X = ${info.observedX.toFixed(2)} ~ N(${info.yConditionalMu.toFixed(2)}, ${info.yConditionalSigma.toFixed(2)}²)`;
  elements.shrinkageFormula.textContent = `Std shrink factor = ${info.shrinkFactor.toFixed(2)}`;
  elements.jointSummary.textContent = Math.abs(twoDState.rho) < 0.05
    ? 'With almost zero correlation, the conditionals nearly sit on top of the marginals.'
    : `At rho = ${twoDState.rho.toFixed(2)}, both conditionals are narrower than their marginals by the same factor, and their means slide linearly with the slice lines.`;
  elements.jointNarrative.innerHTML = `Drag the coral vertical line to set <code>x<sub>0</sub> = ${info.observedX.toFixed(2)}</code> and watch the coral conditional on the right shift to <code>${info.yConditionalMu.toFixed(2)}</code>. Drag the gold horizontal line to set <code>y<sub>0</sub> = ${info.observedY.toFixed(2)}</code> and watch the gold conditional on the top shift to <code>${info.xConditionalMu.toFixed(2)}</code>.`;
}

function drawHorizontalDensityOverlay(canvas, options) {
  const { ctx, width, height } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const yMax = Math.max(...options.curves.map((curve) => normalPdf(curve.mu, curve.mu, curve.sigma))) * 1.2;
  const plotLeft = options.plotLeft ?? options.paddingLeft;
  const plotRight = options.plotRight ?? width - options.paddingRight;
  const plotWidth = plotRight - plotLeft;
  const plotTop = options.paddingTop;
  const plotBottom = height - options.paddingBottom;
  const plotHeight = plotBottom - plotTop;
  const xToPx = (x) => plotLeft + ((x - options.boundsMin) / (options.boundsMax - options.boundsMin)) * plotWidth;
  const yToPx = (y) => plotTop + plotHeight - (y / yMax) * plotHeight;

  const xStep = niceStep((options.boundsMax - options.boundsMin) / 6);
  ctx.strokeStyle = 'rgba(30, 36, 48, 0.08)';
  ctx.lineWidth = 1;
  for (let x = Math.ceil(options.boundsMin / xStep) * xStep; x <= options.boundsMax + 1e-6; x += xStep) {
    const px = xToPx(x);
    ctx.beginPath();
    ctx.moveTo(px, plotTop);
    ctx.lineTo(px, plotBottom);
    ctx.stroke();
    if (options.showTickLabels) {
      ctx.font = '12px "Avenir Next", sans-serif';
      ctx.fillStyle = 'rgba(94, 102, 118, 0.9)';
      ctx.fillText(round(x, 1).toFixed(1), px - 11, plotBottom + 18);
    }
  }

  ctx.strokeStyle = 'rgba(30, 36, 48, 0.28)';
  ctx.beginPath();
  ctx.moveTo(plotLeft, yToPx(0));
  ctx.lineTo(plotRight, yToPx(0));
  ctx.stroke();

  options.curves.forEach((curve) => {
    ctx.beginPath();
    ctx.moveTo(xToPx(options.boundsMin), yToPx(0));
    for (let x = options.boundsMin; x <= options.boundsMax; x += (options.boundsMax - options.boundsMin) / 320) {
      ctx.lineTo(xToPx(x), yToPx(normalPdf(x, curve.mu, curve.sigma)));
    }
    ctx.lineTo(xToPx(options.boundsMax), yToPx(0));
    ctx.closePath();
    ctx.fillStyle = curve.fill;
    ctx.fill();

    ctx.beginPath();
    for (let x = options.boundsMin; x <= options.boundsMax; x += (options.boundsMax - options.boundsMin) / 320) {
      const px = xToPx(x);
      const py = yToPx(normalPdf(x, curve.mu, curve.sigma));
      if (x === options.boundsMin) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = curve.color;
    ctx.stroke();

    ctx.strokeStyle = curve.color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(xToPx(curve.mu), yToPx(0));
    ctx.lineTo(xToPx(curve.mu), yToPx(normalPdf(curve.mu, curve.mu, curve.sigma)));
    ctx.stroke();
  });
}

function drawVerticalDensityOverlay(canvas, options) {
  const { ctx, width, height } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const maxDensity = Math.max(...options.curves.map((curve) => normalPdf(curve.mu, curve.mu, curve.sigma))) * 1.2;
  const densityLeft = options.paddingLeft;
  const densityRight = width - options.paddingRight;
  const densityWidth = densityRight - densityLeft;
  const plotTop = options.plotTop ?? options.paddingTop;
  const plotBottom = options.plotBottom ?? height - options.paddingBottom;
  const plotHeight = plotBottom - plotTop;
  const densityToPx = (density) => densityLeft + (density / maxDensity) * densityWidth;
  const yToPx = (y) => plotTop + plotHeight - ((y - options.boundsMin) / (options.boundsMax - options.boundsMin)) * plotHeight;

  const yStep = niceStep((options.boundsMax - options.boundsMin) / 6);
  ctx.strokeStyle = 'rgba(30, 36, 48, 0.08)';
  ctx.lineWidth = 1;
  for (let y = Math.ceil(options.boundsMin / yStep) * yStep; y <= options.boundsMax + 1e-6; y += yStep) {
    const py = yToPx(y);
    ctx.beginPath();
    ctx.moveTo(densityLeft, py);
    ctx.lineTo(densityRight, py);
    ctx.stroke();
    if (options.showTickLabels) {
      ctx.font = '12px "Avenir Next", sans-serif';
      ctx.fillStyle = 'rgba(94, 102, 118, 0.9)';
      ctx.fillText(round(y, 1).toFixed(1), 2, py + 4);
    }
  }

  ctx.strokeStyle = 'rgba(30, 36, 48, 0.28)';
  ctx.beginPath();
  ctx.moveTo(densityToPx(0), plotTop);
  ctx.lineTo(densityToPx(0), plotBottom);
  ctx.stroke();

  options.curves.forEach((curve) => {
    ctx.beginPath();
    ctx.moveTo(densityToPx(0), yToPx(options.boundsMin));
    for (let y = options.boundsMin; y <= options.boundsMax; y += (options.boundsMax - options.boundsMin) / 320) {
      ctx.lineTo(densityToPx(normalPdf(y, curve.mu, curve.sigma)), yToPx(y));
    }
    ctx.lineTo(densityToPx(0), yToPx(options.boundsMax));
    ctx.closePath();
    ctx.fillStyle = curve.fill;
    ctx.fill();

    ctx.beginPath();
    for (let y = options.boundsMin; y <= options.boundsMax; y += (options.boundsMax - options.boundsMin) / 320) {
      const px = densityToPx(normalPdf(y, curve.mu, curve.sigma));
      const py = yToPx(y);
      if (y === options.boundsMin) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = curve.color;
    ctx.stroke();

    ctx.strokeStyle = curve.color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(densityToPx(0), yToPx(curve.mu));
    ctx.lineTo(densityToPx(normalPdf(curve.mu, curve.mu, curve.sigma)), yToPx(curve.mu));
    ctx.stroke();
  });
}

function syncOneDInputs() {
  elements.mu1Value.textContent = oneDState.mu.toFixed(1);
  elements.sigma1Value.textContent = oneDState.sigma.toFixed(2);
  elements.intervalAValue.textContent = Math.min(oneDState.a, oneDState.b).toFixed(1);
  elements.intervalBValue.textContent = Math.max(oneDState.a, oneDState.b).toFixed(1);
  elements.mu1.value = oneDState.mu;
  elements.sigma1.value = oneDState.sigma;
  elements.intervalA.value = oneDState.a;
  elements.intervalB.value = oneDState.b;
}

function syncTwoDInputs() {
  elements.muXValue.textContent = twoDState.muX.toFixed(1);
  elements.muYValue.textContent = twoDState.muY.toFixed(1);
  elements.sigmaXValue.textContent = twoDState.sigmaX.toFixed(2);
  elements.sigmaYValue.textContent = twoDState.sigmaY.toFixed(2);
  elements.rhoValue.textContent = twoDState.rho.toFixed(2);
  elements.xSliceOffsetValue.textContent = formatSigned(twoDState.xSliceOffset, 2);
  elements.ySliceOffsetValue.textContent = formatSigned(twoDState.ySliceOffset, 2);

  elements.muX.value = twoDState.muX;
  elements.muY.value = twoDState.muY;
  elements.sigmaX.value = twoDState.sigmaX;
  elements.sigmaY.value = twoDState.sigmaY;
  elements.rho.value = twoDState.rho;
  elements.xSliceOffset.value = twoDState.xSliceOffset;
  elements.ySliceOffset.value = twoDState.ySliceOffset;
}

function renderAll() {
  syncOneDInputs();
  syncTwoDInputs();
  drawOneD();
  drawTwoDOverview();
  drawJointMath();
  drawMarginalStage();
  drawConditionalStage();
}

function wireInputs() {
  elements.mu1.addEventListener('input', () => {
    oneDState.mu = Number(elements.mu1.value);
    renderAll();
  });

  elements.sigma1.addEventListener('input', () => {
    oneDState.sigma = Number(elements.sigma1.value);
    renderAll();
  });

  elements.intervalA.addEventListener('input', () => {
    oneDState.a = Number(elements.intervalA.value);
    renderAll();
  });

  elements.intervalB.addEventListener('input', () => {
    oneDState.b = Number(elements.intervalB.value);
    renderAll();
  });

  ['muX', 'muY', 'sigmaX', 'sigmaY', 'rho'].forEach((key) => {
    elements[key].addEventListener('input', () => {
      twoDState[key] = Number(elements[key].value);
      renderAll();
    });
  });

  ['xSliceOffset', 'ySliceOffset'].forEach((key) => {
    elements[key].addEventListener('input', () => {
      twoDState[key] = Number(elements[key].value);
      renderAll();
    });
  });

  document.querySelectorAll('[data-one-d-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      const preset = button.dataset.oneDPreset;
      if (preset === 'standard') Object.assign(oneDState, { mu: 0, sigma: 1, a: -1, b: 1 });
      if (preset === 'wide') Object.assign(oneDState, { mu: 0, sigma: 1.8, a: -1.5, b: 1.5 });
      if (preset === 'shifted') Object.assign(oneDState, { mu: 1.2, sigma: 0.75, a: 0.6, b: 1.8 });
      renderAll();
    });
  });

  document.querySelectorAll('[data-two-d-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      const preset = button.dataset.twoDPreset;
      if (preset === 'independent') Object.assign(twoDState, { muX: 0, muY: 0, sigmaX: 1.2, sigmaY: 1.0, rho: 0, xSliceOffset: 0, ySliceOffset: 0 });
      if (preset === 'positive') Object.assign(twoDState, { muX: 0.1, muY: 0.2, sigmaX: 1.3, sigmaY: 0.85, rho: 0.72, xSliceOffset: 1.1, ySliceOffset: 1.0 });
      if (preset === 'negative') Object.assign(twoDState, { muX: -0.2, muY: 0.1, sigmaX: 1.05, sigmaY: 1.25, rho: -0.76, xSliceOffset: 0.95, ySliceOffset: 0.9 });
      if (preset === 'needle') Object.assign(twoDState, { muX: 0, muY: 0, sigmaX: 1.7, sigmaY: 0.55, rho: 0.88, xSliceOffset: 1.6, ySliceOffset: 1.35 });
      if (preset === 'symmetric') Object.assign(twoDState, { muX: 0, muY: 0, sigmaX: 1.1, sigmaY: 1.1, rho: 0.82, xSliceOffset: 1.0, ySliceOffset: -1.0 });
      if (preset === 'marginal-demo') Object.assign(twoDState, { muX: 0, muY: 0, sigmaX: 1.35, sigmaY: 0.85, rho: -0.9, xSliceOffset: 0.25, ySliceOffset: -0.2 });
      if (preset === 'conditional-demo') Object.assign(twoDState, { muX: 0.25, muY: -0.15, sigmaX: 1.45, sigmaY: 1.1, rho: 0.92, xSliceOffset: 1.8, ySliceOffset: -1.25 });
      if (preset === 'study-habits') Object.assign(twoDState, { muX: 0.2, muY: 0.35, sigmaX: 1.0, sigmaY: 0.9, rho: 0.63, xSliceOffset: 0.9, ySliceOffset: 1.1 });
      if (preset === 'budget') Object.assign(twoDState, { muX: 0.1, muY: -0.15, sigmaX: 1.25, sigmaY: 0.95, rho: -0.68, xSliceOffset: 1.1, ySliceOffset: -0.8 });
      if (preset === 'road-gps') Object.assign(twoDState, { muX: 0.0, muY: 0.0, sigmaX: 1.9, sigmaY: 0.45, rho: 0.86, xSliceOffset: 1.5, ySliceOffset: 0.7 });
      renderAll();
    });
  });

  window.addEventListener('resize', renderAll);
}

function wireConditionalDrag() {
  const canvas = elements.conditionalMainCanvas;
  const pickMode = (event) => {
    const plane = conditionalInteraction.plane;
    if (!plane) return null;
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const info = getJointConditionals(twoDState);
    const dx = Math.abs(px - plane.xToPx(info.observedX));
    const dy = Math.abs(py - plane.yToPx(info.observedY));
    const intersectionDistance = Math.hypot(dx, dy);
    if (intersectionDistance < 16) return 'both';
    if (dx < 10 && dy < 90) return 'x';
    if (dy < 10 && dx < 90) return 'y';
    return null;
  };

  const updateFromPointer = (event) => {
    const plane = conditionalInteraction.plane;
    if (!plane) return;
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    if (conditionalInteraction.dragMode === 'x' || conditionalInteraction.dragMode === 'both') {
      const x = clamp(plane.pxToX(px), plane.bounds.xMin, plane.bounds.xMax);
      twoDState.xSliceOffset = clamp((x - twoDState.muX) / twoDState.sigmaX, -3, 3);
    }
    if (conditionalInteraction.dragMode === 'y' || conditionalInteraction.dragMode === 'both') {
      const y = clamp(plane.pxToY(py), plane.bounds.yMin, plane.bounds.yMax);
      twoDState.ySliceOffset = clamp((y - twoDState.muY) / twoDState.sigmaY, -3, 3);
    }
    renderAll();
  };

  canvas.addEventListener('pointerdown', (event) => {
    const mode = pickMode(event);
    if (!mode) return;
    conditionalInteraction.dragMode = mode;
    canvas.setPointerCapture(event.pointerId);
    updateFromPointer(event);
    event.preventDefault();
  });

  canvas.addEventListener('pointermove', (event) => {
    if (conditionalInteraction.dragMode) {
      updateFromPointer(event);
      return;
    }
    const mode = pickMode(event);
    canvas.style.cursor = mode === 'both' ? 'move' : mode === 'x' ? 'ew-resize' : mode === 'y' ? 'ns-resize' : 'default';
  });

  ['pointerup', 'pointercancel'].forEach((type) => {
    canvas.addEventListener(type, () => {
      conditionalInteraction.dragMode = null;
      canvas.style.cursor = 'default';
    });
  });

  canvas.addEventListener('pointerleave', () => {
    if (!conditionalInteraction.dragMode) {
      canvas.style.cursor = 'default';
    }
  });
}

function wireScrolling() {
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add('is-visible');
    });
  }, { threshold: 0.18 });

  document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

  const navObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const id = `#${entry.target.id}`;
      elements.navLinks.forEach((link) => link.classList.toggle('is-active', link.getAttribute('href') === id));
    });
  }, { threshold: 0.45, rootMargin: '-15% 0px -40% 0px' });

  elements.sections.forEach((section) => navObserver.observe(section));

  const onScroll = () => {
    const total = document.documentElement.scrollHeight - window.innerHeight;
    const progress = total <= 0 ? 0 : window.scrollY / total;
    elements.progressFill.style.width = `${Math.min(100, Math.max(0, progress * 100))}%`;
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

wireInputs();
wireConditionalDrag();
wireScrolling();
renderAll();
