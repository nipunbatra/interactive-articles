#!/usr/bin/env node

import puppeteer from 'puppeteer';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const defaultSource = resolve(here, '../src/articles/xor-feature-lift/index.html');
const source = resolve(process.argv[2] || defaultSource);
const screenshotDirectory = resolve(process.argv[3] || '/private/tmp/xor-feature-lift-screenshots');
mkdirSync(screenshotDirectory, { recursive: true });

if (!existsSync(source)) throw new Error(`XOR feature-lift lab not found: ${source}`);

const browserCandidates = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
  '/Applications/Chromium.app/Contents/MacOS/Chromium'
].filter(Boolean);
const executablePath = browserCandidates.find((candidate) => existsSync(candidate));

const browser = await puppeteer.launch({
  ...(executablePath ? { executablePath } : {}),
  headless: 'new',
  args: ['--no-sandbox', '--force-color-profile=srgb']
});

const failures = [];
const checks = [];

function check(condition, message, detail = '') {
  checks.push(message);
  if (!condition) failures.push(detail ? `${message}: ${detail}` : message);
}

const page = await browser.newPage();
const pageErrors = [];
page.on('pageerror', (error) => pageErrors.push(error.message));
page.on('console', (message) => {
  if (message.type() === 'error') pageErrors.push(message.text());
});

async function load(viewport = { width: 1440, height: 1000, deviceScaleFactor: 1 }) {
  await page.setViewport(viewport);
  await page.goto(pathToFileURL(source).href, { waitUntil: 'load', timeout: 30000 });
  await page.waitForFunction(() => window.XorFeatureLab?.snapshot, { timeout: 10000 });
}

const lab = (script) => page.evaluate(script);
const snapshot = () => lab(() => window.XorFeatureLab.snapshot());

try {
  await load();

  const contract = await lab(() => {
    const requiredIds = [
      'runBtn', 'stepBtn', 'resetBtn', 'stepValue', 'statusText', 'featureHelp',
      'w1Slider', 'w2Slider', 'w3Slider', 'bSlider', 'w1Value', 'w2Value', 'w3Value', 'bValue', 'w3Label',
      'lossValue', 'accuracyValue', 'boundaryValue', 'dimensionValue',
      'inputCanvas', 'featureCanvas', 'historyCanvas', 'boundaryEquation', 'planeEquation', 'workspace'
    ];
    const ids = [...document.querySelectorAll('[id]')].map((element) => element.id);
    const duplicateIds = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
    const unnamedControls = [...document.querySelectorAll('button')]
      .filter((button) => !(button.textContent.trim() || button.getAttribute('aria-label')))
      .map((button) => button.id || button.outerHTML);
    const unlabelledInputs = [...document.querySelectorAll('input')]
      .filter((input) => !(input.labels?.length || input.getAttribute('aria-label')))
      .map((input) => input.id);
    const canvases = [...document.querySelectorAll('canvas')].map((canvas) => ({
      id: canvas.id,
      role: canvas.getAttribute('role'),
      label: canvas.getAttribute('aria-label'),
      tabIndex: canvas.tabIndex
    }));
    return {
      missing: requiredIds.filter((id) => !document.getElementById(id)),
      duplicateIds,
      unnamedControls,
      unlabelledInputs,
      canvases,
      featureButtons: [...document.querySelectorAll('[data-feature]')].map((button) => button.dataset.feature),
      datasetButtons: [...document.querySelectorAll('[data-dataset]')].map((button) => button.dataset.dataset),
      viewButtons: [...document.querySelectorAll('[data-view]')].map((button) => button.dataset.view),
      statusLive: document.getElementById('statusText')?.getAttribute('aria-live'),
      skipTarget: document.querySelector('.skip-link')?.hash,
      rawDollars: (document.body.textContent.match(/\$/g) || []).length
    };
  });
  check(contract.missing.length === 0, 'Required lab DOM IDs should exist', contract.missing.join(', '));
  check(contract.duplicateIds.length === 0, 'DOM IDs should be unique', contract.duplicateIds.join(', '));
  check(contract.unnamedControls.length === 0, 'Every button should have an accessible name', contract.unnamedControls.join(' | '));
  check(contract.unlabelledInputs.length === 0, 'Every slider should be labelled', contract.unlabelledInputs.join(', '));
  check(contract.canvases.length === 3
      && contract.canvases.every((canvas) => canvas.role === 'img' && canvas.label && canvas.tabIndex >= 0),
    'All three canvases should expose an accessible image fallback and keyboard focus', JSON.stringify(contract.canvases));
  check(contract.featureButtons.join('|') === 'none|product|radius|relu',
    'Feature controls should exist in teaching order', contract.featureButtons.join('|'));
  check(contract.datasetButtons.join('|') === 'clusters|corners' && contract.viewButtons.join('|') === 'orbit|edge|top',
    'Dataset and camera controls should exist in teaching order', `${contract.datasetButtons} / ${contract.viewButtons}`);
  check(contract.statusLive === 'polite' && contract.skipTarget === '#workspace',
    'The lab should provide a polite live status and working skip link', JSON.stringify(contract));
  check(contract.rawDollars === 0, 'Page text should not contain stray dollar signs that KaTeX auto-render could mangle', String(contract.rawDollars));

  // --- Line only: the best line is chance.
  const initial = await snapshot();
  check(initial.feature === 'none' && initial.dataset === 'clusters' && initial.steps === 0 && initial.lift === 1,
    'Line-only clusters should be the zero-step default', JSON.stringify(initial));
  check(initial.datasetSize === 100 && initial.classCounts[0] === 50 && initial.classCounts[1] === 50,
    'The cluster dataset should contain 100 balanced points', JSON.stringify(initial.classCounts));
  check(initial.featureDimension === 2 && initial.boundaryKind === 'straight line' && initial.w3SliderDisabled,
    'Line mode should report a 2D straight-line boundary and disable the w3 slider', JSON.stringify(initial));
  check(initial.boundarySegmentCount > 10 && initial.boundaryLineDeviation < 1e-6,
    'The traced boundary should lie exactly on the line w·x + b = 0', `${initial.boundarySegmentCount} segments, deviation ${initial.boundaryLineDeviation}`);
  check(Math.abs(initial.weights.w1 - 0.7) < 1e-12 && Math.abs(initial.weights.w2 + 0.45) < 1e-12 && initial.weights.w3 === 0,
    'Initial weights should be the documented deterministic start', JSON.stringify(initial.weights));

  const lineTrained = await lab(() => window.XorFeatureLab.step(800));
  check(lineTrained.steps === 800 && lineTrained.historyLength === 801,
    'Stepping should advance the counter and record one loss per step', JSON.stringify({ steps: lineTrained.steps, history: lineTrained.historyLength }));
  check(lineTrained.loss < initial.loss && lineTrained.loss > 0.6 && lineTrained.loss < 0.7,
    'A line on XOR clusters should settle at the chance floor near ln 2', `${initial.loss} → ${lineTrained.loss}`);
  check(lineTrained.accuracy <= 0.66, 'A trained line should stay near chance accuracy on XOR', String(lineTrained.accuracy));
  check(lineTrained.weights.w3 === 0, 'The absent third feature should keep an exactly zero weight', String(lineTrained.weights.w3));
  check(Math.hypot(lineTrained.weights.w1, lineTrained.weights.w2) < Math.hypot(0.7, 0.45),
    'Gradient descent should shrink the line weights toward zero', JSON.stringify(lineTrained.weights));

  const reset = await lab(() => window.XorFeatureLab.reset());
  check(reset.steps === 0 && reset.historyLength === 1 && Math.abs(reset.weights.w1 - 0.7) < 1e-12,
    'Reset should restore the deterministic start', JSON.stringify(reset));

  const corners = await lab(() => {
    window.XorFeatureLab.setDataset('corners');
    return window.XorFeatureLab.step(1500);
  });
  check(corners.datasetSize === 4 && corners.classCounts.join('|') === '2|2', 'The Boolean dataset should have four balanced corners', JSON.stringify(corners.classCounts));
  check(Math.abs(corners.loss - corners.chanceLoss) < 0.005 && Math.abs(corners.weights.w1) < 0.05 && Math.abs(corners.weights.w2) < 0.05 && Math.abs(corners.weights.b) < 0.05,
    'On the symmetric corners the convex optimum should be w = 0 with loss ln 2', JSON.stringify({ loss: corners.loss, weights: corners.weights }));
  check(corners.accuracy === 0.5 && corners.ties === 4, 'The optimal line on the four corners should score exactly chance, with all four corners tied at p = 0.5', JSON.stringify({ accuracy: corners.accuracy, ties: corners.ties }));

  // --- Product feature: the lift.
  const lifted = await lab(() => {
    window.XorFeatureLab.setDataset('clusters');
    window.XorFeatureLab.reset();
    return window.XorFeatureLab.setFeature('product', { animate: false });
  });
  check(lifted.feature === 'product' && lifted.lift === 1 && lifted.featureDimension === 3 && lifted.weights.w3 === 0 && lifted.steps === 0,
    'Switching features should reset the new weight, the counter, and finish the lift immediately when asked', JSON.stringify(lifted));
  check(lifted.boundaryKind === 'straight line' && !lifted.w3SliderDisabled,
    'With w3 = 0 the product model should still draw a line and enable the w3 slider', JSON.stringify(lifted));

  const productTrained = await lab(() => window.XorFeatureLab.step(600));
  check(productTrained.accuracy === 1 && productTrained.loss < 0.15,
    'Logistic regression with x1·x2 should classify every XOR point', JSON.stringify({ accuracy: productTrained.accuracy, loss: productTrained.loss }));
  check(productTrained.weights.w3 < -1 && productTrained.boundaryKind === 'hyperbola',
    'The product weight should turn strongly negative and bend the boundary into a hyperbola', JSON.stringify(productTrained.weights));

  const equations = await lab(() => ({
    boundary: document.getElementById('boundaryEquation').textContent,
    plane: document.getElementById('planeEquation').textContent
  }));
  check(equations.boundary.includes('x₁x₂') && equations.boundary.includes('⇒') && equations.plane.includes('u₃'),
    'Live equations should show the product term and the solved hyperbola', JSON.stringify(equations));

  // --- Camera views.
  const edge = await lab(() => window.XorFeatureLab.setView('edge', { animate: false }));
  check(edge.view === 'edge' && edge.planeEdgeAlignment < 1e-6,
    'Edge-on view should place the camera inside the decision plane', String(edge.planeEdgeAlignment));
  const edgeAfterTraining = await lab(() => window.XorFeatureLab.step(60));
  check(edgeAfterTraining.view === 'edge' && edgeAfterTraining.planeEdgeAlignment < 1e-6,
    'Edge-on view should track the plane as training tilts it', String(edgeAfterTraining.planeEdgeAlignment));
  const top = await lab(() => window.XorFeatureLab.setView('top', { animate: false }));
  check(Math.abs(top.camera.elevation - Math.PI / 2) < 1e-9 && Math.abs(top.camera.azimuth + Math.PI / 2) < 1e-9,
    'Top view should look straight down with x1 to the right', JSON.stringify(top.camera));
  const custom = await lab(() => window.XorFeatureLab.setCamera({ azimuth: 0.3, elevation: 0.7, zoom: 1.4 }));
  check(custom.view === 'custom' && Math.abs(custom.camera.zoom - 1.4) < 1e-9, 'Manual camera changes should mark the view as custom', JSON.stringify(custom));
  const lineEdge = await lab(() => {
    window.XorFeatureLab.setFeature('none', { animate: false });
    return window.XorFeatureLab.setView('edge', { animate: false });
  });
  check(lineEdge.planeEdgeAlignment < 1e-6, 'Edge-on view should also work for the vertical plane of the line-only model', String(lineEdge.planeEdgeAlignment));

  // --- Decoy and ReLU features.
  const radius = await lab(() => {
    window.XorFeatureLab.setFeature('radius', { animate: false });
    return window.XorFeatureLab.step(800);
  });
  check(radius.accuracy <= 0.6 && radius.loss > 0.6 && radius.boundaryKind === 'circle',
    'The squared-radius decoy should not separate XOR', JSON.stringify({ accuracy: radius.accuracy, loss: radius.loss, kind: radius.boundaryKind }));

  const relu = await lab(() => {
    window.XorFeatureLab.setFeature('relu', { animate: false });
    return window.XorFeatureLab.step(1500);
  });
  check(relu.accuracy === 1 && relu.boundaryKind === 'bent line',
    'One hand-made ReLU feature plus a line should solve XOR', JSON.stringify({ accuracy: relu.accuracy, loss: relu.loss }));

  const handSolution = await lab(() => {
    window.XorFeatureLab.setDataset('corners');
    return window.XorFeatureLab.setWeights({ w1: 1, w2: 1, w3: -4, b: 1 });
  });
  check(handSolution.accuracy === 1 && handSolution.lastAction === 'hand',
    'The explanation card weights (1, 1, −4, 1) should classify the four corners', JSON.stringify(handSolution));

  // --- Visible controls, including the asynchronous Run/Pause path.
  await load();
  await page.click('[data-feature="product"]');
  await page.waitForFunction(() => window.XorFeatureLab.snapshot().lift === 1, { timeout: 5000 });
  await page.click('#stepBtn');
  let visible = await lab(() => ({
    snapshot: window.XorFeatureLab.snapshot(),
    stepText: document.getElementById('stepValue').textContent.trim(),
    pressed: document.querySelector('[data-feature="product"]').getAttribute('aria-pressed')
  }));
  check(visible.snapshot.feature === 'product' && visible.snapshot.steps === 1 && visible.stepText === '1' && visible.pressed === 'true',
    'Visible feature and One step controls should update public and displayed state', JSON.stringify(visible));

  await lab(() => {
    const slider = document.getElementById('w3Slider');
    slider.value = '-3';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
  });
  visible = await lab(() => ({ snapshot: window.XorFeatureLab.snapshot(), text: document.getElementById('w3Value').textContent }));
  check(visible.snapshot.weights.w3 === -3 && visible.snapshot.lastAction === 'hand' && visible.text === '−3.00',
    'Moving the w3 slider should set the weight by hand and echo it', JSON.stringify(visible));

  await page.click('#runBtn');
  await page.waitForFunction(() => window.XorFeatureLab.snapshot().steps >= 20, { timeout: 5000 });
  await page.click('#runBtn');
  visible = await lab(() => ({
    snapshot: window.XorFeatureLab.snapshot(),
    pressed: document.getElementById('runBtn').getAttribute('aria-pressed'),
    text: document.getElementById('runBtn').textContent.trim()
  }));
  check(!visible.snapshot.running && visible.pressed === 'false' && visible.text === 'Run',
    'Run should pause through the visible button and expose a consistent pressed state', JSON.stringify(visible));

  const box = await lab(() => {
    const rect = document.getElementById('featureCanvas').getBoundingClientRect();
    return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
  });
  const beforeDrag = await snapshot();
  await page.mouse.move(box.x, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x + 80, box.y + 20, { steps: 4 });
  await page.mouse.up();
  const afterDrag = await snapshot();
  check(afterDrag.view === 'custom' && Math.abs(afterDrag.camera.azimuth - beforeDrag.camera.azimuth) > 0.1,
    'Dragging the feature-space canvas should orbit the camera', JSON.stringify({ before: beforeDrag.camera, after: afterDrag.camera }));

  await page.click('[data-view="top"]');
  await page.waitForFunction(() => Math.abs(window.XorFeatureLab.snapshot().camera.elevation - Math.PI / 2) < 1e-6, { timeout: 5000 });
  const topClicked = await snapshot();
  check(topClicked.view === 'top', 'The visible Top button should animate the camera to the top view', JSON.stringify(topClicked.camera));

  await page.screenshot({ path: resolve(screenshotDirectory, 'xor-feature-lift-desktop-1440x1000.png'), fullPage: true });

  await load({ width: 390, height: 844, deviceScaleFactor: 1 });
  const mobile = await lab(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    inputWidth: document.getElementById('inputCanvas').getBoundingClientRect().width,
    featureWidth: document.getElementById('featureCanvas').getBoundingClientRect().width
  }));
  check(mobile.scrollWidth <= mobile.clientWidth + 1, 'Mobile layout should not introduce horizontal page overflow', JSON.stringify(mobile));
  check(mobile.inputWidth > 280 && mobile.featureWidth > 280, 'Both mobile canvases should retain usable width', JSON.stringify(mobile));
  await page.screenshot({ path: resolve(screenshotDirectory, 'xor-feature-lift-mobile-390x844.png'), fullPage: true });

  check(pageErrors.length === 0, 'The page should load and run without browser errors', pageErrors.join(' | '));
} finally {
  await browser.close();
}

for (const message of checks) console.log(`PASS  ${message}`);
if (failures.length) {
  console.error(`\n${failures.length} failure(s):`);
  for (const failure of failures) console.error(`FAIL  ${failure}`);
  process.exitCode = 1;
} else {
  console.log(`\n${checks.length} checks passed.`);
}
