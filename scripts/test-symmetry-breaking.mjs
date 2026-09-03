#!/usr/bin/env node

import puppeteer from 'puppeteer';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const defaultSource = resolve(here, '../src/articles/symmetry-breaking/index.html');
const source = resolve(process.argv[2] || defaultSource);
const screenshotDirectory = resolve(process.argv[3] || '/private/tmp/symmetry-breaking-screenshots');
mkdirSync(screenshotDirectory, { recursive: true });

if (!existsSync(source)) throw new Error(`Symmetry-breaking lab not found: ${source}`);

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

function allEqual(values, tolerance = 0) {
  return values.every((value) => Math.abs(value - values[0]) <= tolerance);
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
  await page.waitForFunction(() => window.SymmetryBreakingLab?.snapshot, { timeout: 10000 });
}

async function snapshot() {
  return page.evaluate(() => window.SymmetryBreakingLab.snapshot());
}

try {
  await load();

  const contract = await page.evaluate(() => {
    const requiredIds = [
      'runBtn', 'stepBtn', 'resetBtn', 'stepValue', 'statusText',
      'lossValue', 'accuracyValue', 'spreadValue', 'rankValue',
      'surfaceCanvas', 'hingeCanvas', 'neuronLegend', 'workspace'
    ];
    const ids = [...document.querySelectorAll('[id]')].map((element) => element.id);
    const duplicateIds = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
    const unnamedControls = [...document.querySelectorAll('button')]
      .filter((button) => !(button.textContent.trim() || button.getAttribute('aria-label')))
      .map((button) => button.id || button.outerHTML);
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
      canvases,
      modeButtons: [...document.querySelectorAll('[data-mode]')].map((button) => button.dataset.mode),
      statusLive: document.getElementById('statusText')?.getAttribute('aria-live'),
      skipTarget: document.querySelector('.skip-link')?.hash
    };
  });
  check(contract.missing.length === 0, 'Required lab DOM IDs should exist', contract.missing.join(', '));
  check(contract.duplicateIds.length === 0, 'DOM IDs should be unique', contract.duplicateIds.join(', '));
  check(contract.unnamedControls.length === 0, 'Every button should have an accessible name', contract.unnamedControls.join(' | '));
  check(contract.canvases.length === 2
      && contract.canvases.every((canvas) => canvas.role === 'img' && canvas.label && canvas.tabIndex >= 0),
    'Both canvases should expose an accessible image fallback and keyboard focus', JSON.stringify(contract.canvases));
  check(contract.modeButtons.join('|') === 'identical|perturbed|independent',
    'All three initialization controls should exist in teaching order', contract.modeButtons.join('|'));
  check(contract.statusLive === 'polite' && contract.skipTarget === '#workspace',
    'The lab should provide a polite live status and working skip link', JSON.stringify(contract));

  const initial = await snapshot();
  check(initial.mode === 'identical' && initial.steps === 0, 'Identical mode should be the zero-step default', JSON.stringify(initial));
  check(initial.datasetSize === 100 && initial.classCounts[0] === 50 && initial.classCounts[1] === 50,
    'The fixed XOR dataset should contain 100 balanced observations', JSON.stringify(initial.classCounts));
  check(initial.separateRowStorage,
    'Identical incoming rows should use separate arrays rather than shared references');
  check(initial.separation === 0 && initial.featureRank === 1 && initial.distinctDirections === 1,
    'Exact copies should begin with zero separation and one effective hidden feature', JSON.stringify(initial));
  check(allEqual(initial.incomingRows.flatMap((row) => row.filter((_, index) => index === 0)))
      && allEqual(initial.incomingRows.flatMap((row) => row.filter((_, index) => index === 1)))
      && allEqual(initial.hiddenBiases) && allEqual(initial.outgoingCoefficients),
    'Every identical-mode parameter group should contain exact copies');

  const identicalTrained = await page.evaluate(() => {
    window.SymmetryBreakingLab.step(1000);
    return window.SymmetryBreakingLab.snapshot();
  });
  check(identicalTrained.steps === 1000 && identicalTrained.parameterSignature !== initial.parameterSignature,
    'One thousand identical-mode steps should update the model deterministically');
  check(identicalTrained.separation === 0 && identicalTrained.featureRank === 1,
    'Full-batch gradient descent should preserve exact hidden-neuron equality', JSON.stringify(identicalTrained));
  check(allEqual(identicalTrained.hiddenBiases) && allEqual(identicalTrained.outgoingCoefficients)
      && allEqual(identicalTrained.incomingRows.map((row) => row[0]))
      && allEqual(identicalTrained.incomingRows.map((row) => row[1])),
    'All copied parameters should remain exactly equal after training');
  check(identicalTrained.accuracy >= 0.73 && identicalTrained.accuracy <= 0.76,
    'The effective width-one model should plateau near three-of-four XOR accuracy', identicalTrained.accuracy);

  await page.evaluate(() => window.SymmetryBreakingLab.reset());
  const identicalReset = await snapshot();
  check(identicalReset.parameterSignature === initial.parameterSignature && identicalReset.steps === 0,
    'Reset should exactly reproduce the identical initialization');

  const perturbed = await page.evaluate(() => {
    window.SymmetryBreakingLab.setMode('perturbed');
    const before = window.SymmetryBreakingLab.snapshot();
    window.SymmetryBreakingLab.step(200);
    return { before, after: window.SymmetryBreakingLab.snapshot() };
  });
  check(perturbed.before.separation > 0.15 && perturbed.before.featureRank > 1,
    'Small deterministic offsets should visibly release more than one hidden feature', JSON.stringify(perturbed.before));
  check(perturbed.after.separation > 0 && perturbed.after.parameterSignature !== perturbed.before.parameterSignature,
    'Perturbed neurons should remain separated while gradient descent updates them', JSON.stringify(perturbed.after));

  const independent = await page.evaluate(() => {
    window.SymmetryBreakingLab.setMode('independent');
    const before = window.SymmetryBreakingLab.snapshot();
    window.SymmetryBreakingLab.step(400);
    return { before, after: window.SymmetryBreakingLab.snapshot() };
  });
  check(independent.before.featureRank >= 3 && independent.before.distinctDirections >= 3,
    'Independent seed 12 should begin with several distinct hidden features', JSON.stringify(independent.before));
  check(independent.after.loss < independent.before.loss,
    'Independent training should reduce the measured binary cross-entropy', `${independent.before.loss} to ${independent.after.loss}`);
  check(independent.after.accuracy === 1 && independent.after.featureRank >= 3,
    'Independent initialization should learn all 100 XOR observations', JSON.stringify(independent.after));

  // Exercise the visible controls, including the asynchronous Run/Pause path.
  await page.click('[data-mode="identical"]');
  await page.click('#stepBtn');
  let visibleControlState = await page.evaluate(() => ({
    snapshot: window.SymmetryBreakingLab.snapshot(),
    stepText: document.getElementById('stepValue').textContent.trim(),
    identicalPressed: document.querySelector('[data-mode="identical"]').getAttribute('aria-pressed')
  }));
  check(visibleControlState.snapshot.steps === 1 && visibleControlState.stepText === '1'
      && visibleControlState.identicalPressed === 'true',
    'The visible One step and mode controls should update public and displayed state', JSON.stringify(visibleControlState));
  await page.click('#runBtn');
  await page.waitForFunction(() => window.SymmetryBreakingLab.snapshot().steps >= 9, { timeout: 5000 });
  await page.click('#runBtn');
  visibleControlState = await page.evaluate(() => ({
    snapshot: window.SymmetryBreakingLab.snapshot(),
    pressed: document.getElementById('runBtn').getAttribute('aria-pressed'),
    text: document.getElementById('runBtn').textContent.trim()
  }));
  check(!visibleControlState.snapshot.running && visibleControlState.pressed === 'false' && visibleControlState.text === 'Run',
    'Run should pause through the visible button and expose a consistent pressed state', JSON.stringify(visibleControlState));

  await page.screenshot({
    path: resolve(screenshotDirectory, 'symmetry-breaking-desktop-1440x1000.png'),
    fullPage: true
  });
  await load({ width: 390, height: 844, deviceScaleFactor: 1 });
  const mobile = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    surfaceWidth: document.getElementById('surfaceCanvas').getBoundingClientRect().width,
    hingeWidth: document.getElementById('hingeCanvas').getBoundingClientRect().width
  }));
  check(mobile.scrollWidth <= mobile.clientWidth + 1,
    'Mobile layout should not introduce horizontal page overflow', JSON.stringify(mobile));
  check(mobile.surfaceWidth > 280 && mobile.hingeWidth > 280,
    'Both mobile canvases should retain usable width', JSON.stringify(mobile));
  await page.screenshot({
    path: resolve(screenshotDirectory, 'symmetry-breaking-mobile-390x844.png'),
    fullPage: true
  });

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
