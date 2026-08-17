#!/usr/bin/env node

import puppeteer from 'puppeteer';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const defaultSource = resolve(here, '../src/articles/mlp-decision-boundary/index.html');
const source = resolve(process.argv[2] || defaultSource);
const screenshotDir = resolve(process.argv[3] || '/private/tmp/relu-lab-screenshots');
mkdirSync(screenshotDir, { recursive: true });

if (!existsSync(source)) {
  throw new Error(`ReLU classification playground not found: ${source}`);
}

const browserCandidates = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
  '/Applications/Chromium.app/Contents/MacOS/Chromium'
].filter(Boolean);
const executablePath = browserCandidates.find(candidate => existsSync(candidate));

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

function closeTo(actual, expected, tolerance = 1e-10) {
  return Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance;
}

const page = await browser.newPage();
const pageErrors = [];
page.on('pageerror', error => pageErrors.push(error.message));
page.on('console', message => {
  if (message.type() === 'error') pageErrors.push(message.text());
});

async function load(viewport = { width: 1440, height: 900, deviceScaleFactor: 1 }) {
  await page.setViewport(viewport);
  await page.goto(pathToFileURL(source).href, { waitUntil: 'load', timeout: 30000 });
  await page.waitForFunction(() => window.ReLUClassificationLab?.snapshot, { timeout: 10000 });
}

async function snapshot() {
  return page.evaluate(() => window.ReLUClassificationLab.snapshot());
}

try {
  await load();

  const apiContract = await page.evaluate(() => ({
    hasSetDataSeed: typeof window.ReLUClassificationLab.setDataSeed === 'function',
    hasSetWeightSeed: typeof window.ReLUClassificationLab.setWeightSeed === 'function',
    hasRecommendedSetup: typeof window.ReLUClassificationLab.loadRecommendedSetup === 'function',
    hasWeightSignature: typeof window.ReLUClassificationLab.snapshot().weightSignature === 'string',
    hasTrainableState: typeof window.ReLUClassificationLab.snapshot().trainable === 'boolean'
  }));
  check(apiContract.hasSetDataSeed && apiContract.hasSetWeightSeed && apiContract.hasRecommendedSetup
      && apiContract.hasWeightSignature && apiContract.hasTrainableState,
    'Public API should expose seeds, recommended setup, trainability, and a weight signature',
    JSON.stringify(apiContract));

  // Flow 1: changing the weight seed must preserve data; changing the data
  // seed must preserve weights. Architecture and dataset remain fixed.
  const seedIsolation = await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    api.setDataset('moons');
    api.setArchitecture({ depth: 1, width: 5 });
    api.setDataSeed(41);
    api.setWeightSeed(71);
    const baseline = api.snapshot();
    api.setWeightSeed(72);
    const weightChanged = api.snapshot();
    api.setDataSeed(42);
    const dataChanged = api.snapshot();
    api.setDataSeed(42);
    const dataRepeated = api.snapshot();
    return { baseline, weightChanged, dataChanged, dataRepeated };
  });
  check(seedIsolation.baseline.dataSeed === 41 && seedIsolation.baseline.weightSeed === 71,
    'Baseline should retain independently selected seed values', JSON.stringify(seedIsolation.baseline));
  check(seedIsolation.weightChanged.dataSignature === seedIsolation.baseline.dataSignature,
    'Changing only the weight seed should preserve every data point');
  check(seedIsolation.weightChanged.weightSignature !== seedIsolation.baseline.weightSignature,
    'Changing only the weight seed should change model weights');
  check(seedIsolation.weightChanged.dataSeed === 41 && seedIsolation.weightChanged.weightSeed === 72,
    'Weight-seed change should not alter the data seed', JSON.stringify(seedIsolation.weightChanged));
  check(seedIsolation.dataChanged.dataSignature !== seedIsolation.weightChanged.dataSignature,
    'Changing only the data seed should resample a stochastic dataset');
  check(seedIsolation.dataChanged.weightSignature === seedIsolation.weightChanged.weightSignature,
    'Changing only the data seed should preserve model weights');
  check(seedIsolation.dataChanged.dataSeed === 42 && seedIsolation.dataChanged.weightSeed === 72,
    'Data-seed change should not alter the weight seed', JSON.stringify(seedIsolation.dataChanged));
  check(seedIsolation.dataRepeated.dataSignature === seedIsolation.dataChanged.dataSignature
      && seedIsolation.dataRepeated.weightSignature === seedIsolation.dataChanged.weightSignature,
    'Repeating a data seed should reproduce data without disturbing weights');

  // Flow 2: both hand-constructed XOR rules retain their intended contracts,
  // and loading a proof never silently replaces the chosen evidence.
  const proofs = await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    api.setDataset('xor4');
    const cornerDataBefore = api.snapshot().dataSignature;
    api.loadCornerRule();
    const corner = api.snapshot();
    const cornerDataAfter = corner.dataSignature;
    const cornerBeforeBlockedTraining = api.snapshot();
    api.step(8);
    api.start();
    const cornerAfterBlockedTraining = api.snapshot();
    const cornerControls = {
      runDisabled: document.getElementById('runBtn').disabled,
      stepDisabled: document.getElementById('stepBtn').disabled,
      status: document.getElementById('statusText').textContent.trim()
    };
    api.stop();
    api.setDataset('xorField');
    const fieldDataBefore = api.snapshot().dataSignature;
    const fieldDistribution = api.snapshot();
    api.loadFieldRule();
    const field = api.snapshot();
    api.setDataset('circles');
    const carriedBeforeBlockedTraining = api.snapshot();
    api.step(8);
    api.start();
    const carriedAfterBlockedTraining = api.snapshot();
    const carriedControls = {
      runDisabled: document.getElementById('runBtn').disabled,
      stepDisabled: document.getElementById('stepBtn').disabled,
      status: document.getElementById('statusText').textContent.trim()
    };
    api.stop();
    return {
      corner, field, cornerDataBefore, cornerDataAfter, fieldDataBefore,
      fieldDataAfter: field.dataSignature, fieldDistribution,
      cornerBeforeBlockedTraining, cornerAfterBlockedTraining, cornerControls,
      carriedBeforeBlockedTraining, carriedAfterBlockedTraining, carriedControls
    };
  });
  check(proofs.corner.dataSize === 4 && proofs.corner.depth === 1 && proofs.corner.width === 2,
    'Corner construction should use four XOR observations and two ReLUs', JSON.stringify(proofs.corner));
  check(proofs.corner.provenance === 'constructed' && proofs.corner.initializer === 'corner',
    'Corner construction should report explicit constructed provenance', JSON.stringify(proofs.corner));
  check(proofs.corner.accuracy === 1,
    'Two-ReLU corner construction should classify all four Boolean points', proofs.corner.accuracy);
  check(proofs.corner.fieldAgreement > 0.74 && proofs.corner.fieldAgreement < 0.78,
    'Corner construction should expose roughly 75% filled-field agreement', proofs.corner.fieldAgreement);
  check(proofs.cornerDataBefore === proofs.cornerDataAfter,
    'Loading the corner construction should leave evidence unchanged');
  check(!proofs.corner.trainable && proofs.cornerControls.runDisabled && proofs.cornerControls.stepDisabled,
    'Constructed proofs should visibly disable Run and One step',
    JSON.stringify({ snapshot: proofs.corner, controls: proofs.cornerControls }));
  check(proofs.cornerAfterBlockedTraining.steps === proofs.cornerBeforeBlockedTraining.steps
      && proofs.cornerAfterBlockedTraining.weightSignature === proofs.cornerBeforeBlockedTraining.weightSignature
      && !proofs.cornerAfterBlockedTraining.running,
    'Constructed proofs should remain unchanged when training is requested through the API',
    JSON.stringify({ before: proofs.cornerBeforeBlockedTraining, after: proofs.cornerAfterBlockedTraining }));
  check(/random|recommended/i.test(proofs.cornerControls.status),
    'Constructed-proof status should explain how to enable training', proofs.cornerControls.status);

  check(proofs.field.dataSize === 256 && proofs.field.depth === 1 && proofs.field.width === 4,
    'Filled-XOR construction should use 256 samples and four ReLUs', JSON.stringify(proofs.field));
  check(proofs.field.provenance === 'constructed' && proofs.field.initializer === 'field',
    'Filled-field construction should report explicit constructed provenance', JSON.stringify(proofs.field));
  check(proofs.field.accuracy === 1 && proofs.field.fieldAgreement > 0.999,
    'Four-ReLU construction should match sampled and dense XOR fields',
    JSON.stringify({ accuracy: proofs.field.accuracy, fieldAgreement: proofs.field.fieldAgreement }));
  check(proofs.fieldDataBefore === proofs.fieldDataAfter,
    'Loading the filled-field construction should leave evidence unchanged');
  check(Array.isArray(proofs.fieldDistribution.classCounts)
      && proofs.fieldDistribution.classCounts.length === 2
      && proofs.fieldDistribution.classCounts.every(count => count === 128),
    'Filled XOR should contain exactly 128 observations from each class',
    JSON.stringify(proofs.fieldDistribution.classCounts));
  check(Array.isArray(proofs.fieldDistribution.quadrantCounts)
      && proofs.fieldDistribution.quadrantCounts.length === 4
      && proofs.fieldDistribution.quadrantCounts.every(count => count === 64),
    'Filled XOR should contain exactly 64 observations in each quadrant',
    JSON.stringify(proofs.fieldDistribution.quadrantCounts));
  check(proofs.carriedBeforeBlockedTraining.provenance === 'carried'
      && !proofs.carriedBeforeBlockedTraining.trainable
      && proofs.carriedControls.runDisabled && proofs.carriedControls.stepDisabled,
    'A model carried to different evidence should be labeled and blocked from accidental training',
    JSON.stringify({ snapshot: proofs.carriedBeforeBlockedTraining, controls: proofs.carriedControls }));
  check(proofs.carriedAfterBlockedTraining.steps === proofs.carriedBeforeBlockedTraining.steps
      && proofs.carriedAfterBlockedTraining.weightSignature === proofs.carriedBeforeBlockedTraining.weightSignature
      && !proofs.carriedAfterBlockedTraining.running,
    'A carried model should remain unchanged until the learner chooses a fresh start',
    JSON.stringify({ before: proofs.carriedBeforeBlockedTraining, after: proofs.carriedAfterBlockedTraining }));

  // Recommended setup is the explicit bridge from inspection to optimization:
  // it chooses a dataset-appropriate random model and enables both train controls.
  await page.evaluate(() => window.ReLUClassificationLab.setDataset('circles'));
  const recommendationBefore = await snapshot();
  await page.click('#recommendedSetupBtn');
  const recommendation = await page.evaluate(() => ({
    snapshot: window.ReLUClassificationLab.snapshot(),
    text: document.getElementById('recommendedSetupText').textContent.trim(),
    weightSeedValue: Number(document.getElementById('weightSeedInput').value),
    runDisabled: document.getElementById('runBtn').disabled,
    stepDisabled: document.getElementById('stepBtn').disabled,
    status: document.getElementById('statusText').textContent.trim()
  }));
  check(recommendation.snapshot.depth === 2 && recommendation.snapshot.width === 8
      && closeTo(recommendation.snapshot.learningRate, 0.1)
      && recommendation.snapshot.weightSeed === 23 && recommendation.weightSeedValue === 23,
    'Circles recommendation should choose two layers, width 8, learning rate 0.1, and seed 23',
    JSON.stringify(recommendation));
  check(recommendation.snapshot.provenance === 'random' && recommendation.snapshot.trainable
      && !recommendation.snapshot.running && recommendation.snapshot.steps === 0
      && !recommendation.runDisabled && !recommendation.stepDisabled,
    'Recommended setup should load a fresh, stopped, trainable random model',
    JSON.stringify(recommendation));
  check(recommendation.snapshot.weightSignature !== recommendationBefore.weightSignature
      && /2(?: hidden)? layers?.*8.*0\.1.*23/i.test(recommendation.text),
    'Recommended control should update both model weights and its visible configuration summary',
    JSON.stringify({ before: recommendationBefore.weightSignature, after: recommendation }));

  // Flow 3: signed output terms and bias must reconstruct the raw logit for
  // constructed and trained/deep models at several probes.
  const decompositions = await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    const points = [[0, 0], [0.25, 0.75], [0.5, 0.5], [0.91, 0.13], [1, 1]];
    api.setDataset('xor4');
    api.loadCornerRule();
    const corner = points.map(([x, y]) => api.getContributionDecomposition(x, y));
    api.setDataset('xorField');
    api.loadFieldRule();
    const field = points.map(([x, y]) => api.getContributionDecomposition(x, y));
    api.setArchitecture({ depth: 2, width: 3 });
    api.setWeightSeed(19);
    api.step(3);
    const deep = points.map(([x, y]) => api.getContributionDecomposition(x, y));
    return { corner, field, deep, deepSnapshot: api.snapshot() };
  });
  for (const [name, values, expectedCount] of [
    ['corner', decompositions.corner, 2],
    ['field', decompositions.field, 4],
    ['deep', decompositions.deep, 3]
  ]) {
    for (const value of values) {
      check(value.contributions.length === expectedCount,
        `${name} decomposition should expose one signed term per final hidden unit`, JSON.stringify(value));
      check(closeTo(value.sum, value.reconstructed),
        `${name} signed contributions plus bias should reconstruct z`, JSON.stringify(value));
      check(closeTo(value.bias + value.contributions.reduce((sum, term) => sum + term, 0), value.sum),
        `${name} decomposition should be numerically additive`, JSON.stringify(value));
    }
  }
  check(decompositions.deepSnapshot.provenance === 'trained' && decompositions.deepSnapshot.steps === 3,
    'Deep decomposition smoke test should use a trained model', JSON.stringify(decompositions.deepSnapshot));

  // Flow 4: the three input-field buttons drive state and expose one pressed
  // view at a time.
  const fieldViews = [];
  for (const view of ['probability', 'logit', 'class']) {
    await page.click(`[data-class-view="${view}"]`);
    fieldViews.push(await page.evaluate(expected => ({
      snapshot: window.ReLUClassificationLab.snapshot(),
      pressed: [...document.querySelectorAll('[data-class-view]')].map(button => ({
        view: button.dataset.classView,
        pressed: button.getAttribute('aria-pressed'),
        active: button.classList.contains('is-active')
      })),
      label: document.getElementById('mainCanvas').getAttribute('aria-label'),
      expected
    }), view));
  }
  for (const result of fieldViews) {
    check(result.snapshot.view === result.expected,
      `${result.expected} field control should update public state`, result.snapshot.view);
    check(result.pressed.filter(item => item.pressed === 'true' && item.active).length === 1
        && result.pressed.find(item => item.view === result.expected)?.pressed === 'true',
      `${result.expected} field control should be the only pressed view`, JSON.stringify(result.pressed));
    check(result.label.toLowerCase().includes(result.expected),
      `${result.expected} field should update the main canvas accessible description`, result.label);
  }

  // Flow 5: exercise the real pointer and wheel listeners on #surfaceCanvas,
  // then restore the canonical isometric camera with the visible Reset button.
  await page.evaluate(() => window.ReLUClassificationLab.setSurfaceView('iso'));
  const beforeOrbit = await snapshot();
  const surface = await page.$('#surfaceCanvas');
  await surface.evaluate(element => {
    document.documentElement.style.scrollBehavior = 'auto';
    element.scrollIntoView({ block: 'center', behavior: 'instant' });
  });
  await page.waitForFunction(() => {
    const bounds = document.getElementById('surfaceCanvas').getBoundingClientRect();
    return bounds.top >= 0 && bounds.bottom <= innerHeight;
  }, { timeout: 3000 });
  const surfaceBox = await surface.boundingBox();
  check(Boolean(surfaceBox), 'Logit surface should expose a measurable orbit target');
  if (surfaceBox) {
    const centerX = surfaceBox.x + surfaceBox.width / 2;
    const centerY = surfaceBox.y + surfaceBox.height / 2;
    await page.mouse.move(centerX, centerY);
    await page.mouse.down();
    await page.mouse.move(centerX + 78, centerY - 34, { steps: 8 });
    await page.mouse.up();
    const afterDrag = await snapshot();
    check(Math.abs(afterDrag.camera.azimuth - beforeOrbit.camera.azimuth) > 0.2
        && Math.abs(afterDrag.camera.elevation - beforeOrbit.camera.elevation) > 0.08,
      'Pointer drag should change surface azimuth and elevation',
      JSON.stringify({ before: beforeOrbit.camera, after: afterDrag.camera }));
    const pressedAfterDrag = await page.$$eval('[data-surface-view]', buttons => buttons.map(button => button.getAttribute('aria-pressed')));
    check(pressedAfterDrag.every(value => value === 'false'),
      'A custom orbit should clear camera-preset pressed states', JSON.stringify(pressedAfterDrag));

    await page.mouse.move(centerX, centerY);
    await page.mouse.wheel({ deltaY: -260 });
    const afterWheel = await snapshot();
    check(afterWheel.camera.zoom > afterDrag.camera.zoom,
      'Wheel-up over the surface should zoom in', JSON.stringify({ afterDrag: afterDrag.camera, afterWheel: afterWheel.camera }));

    await page.click('#surfaceResetBtn');
    const afterReset = await snapshot();
    check(closeTo(afterReset.camera.azimuth, -Math.PI / 4)
        && closeTo(afterReset.camera.elevation, 0.58)
        && closeTo(afterReset.camera.zoom, 1),
      'Surface Reset should restore the canonical isometric camera', JSON.stringify(afterReset.camera));
    const isoPressed = await page.$eval('[data-surface-view="iso"]', button => ({
      pressed: button.getAttribute('aria-pressed'),
      active: button.classList.contains('is-active')
    }));
    check(isoPressed.pressed === 'true' && isoPressed.active,
      'Surface Reset should restore the Iso pressed state', JSON.stringify(isoPressed));
  }

  // Flow 6: Clear custom data through the visible control, then prove that
  // both seed changes and dataset round-trips preserve the intentionally empty
  // custom set.
  await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    api.setDataset('custom');
    api.clearCustomPoints();
    api.addCustomPoint(0.2, 0.2, 0);
    api.addCustomPoint(0.4, 0.3, 0);
    api.addCustomPoint(0.75, 0.8, 1);
  });
  const customBeforeClear = await snapshot();
  check(customBeforeClear.dataset === 'custom' && customBeforeClear.dataSize === 3,
    'Custom setup should contain the three requested points', JSON.stringify(customBeforeClear));
  await page.click('#clearPointsBtn');
  const customCleared = await snapshot();
  check(customCleared.dataSize === 0 && customCleared.dataSignature === '',
    'Clear all should leave a genuinely empty custom dataset', JSON.stringify(customCleared));
  const customPersistence = await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    api.setDataSeed(91);
    const afterDataSeed = api.snapshot();
    api.setWeightSeed(92);
    const afterWeightSeed = api.snapshot();
    api.setDataset('blobs');
    api.setDataset('custom');
    const afterRoundTrip = api.snapshot();
    return { afterDataSeed, afterWeightSeed, afterRoundTrip };
  });
  for (const [name, result] of Object.entries(customPersistence)) {
    check(result.dataset === 'custom' && result.dataSize === 0 && result.dataSignature === '',
      `Cleared custom data should remain empty ${name}`, JSON.stringify(result));
  }

  // Flow 7: exact stepping, Reset, and Run/Pause state stay synchronized with
  // provenance, history, button text, and aria-pressed.
  const trainingInitial = await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    api.setDataset('blobs');
    api.setArchitecture({ depth: 1, width: 4 });
    api.setDataSeed(17);
    api.setWeightSeed(23);
    return api.snapshot();
  });
  await page.click('#stepBtn');
  const afterStep = await snapshot();
  check(afterStep.steps === 1 && afterStep.historyLength === trainingInitial.historyLength + 1,
    'One step should add exactly one gradient step and one history sample',
    JSON.stringify({ before: trainingInitial, after: afterStep }));
  check(afterStep.provenance === 'trained' && afterStep.weightSignature !== trainingInitial.weightSignature,
    'One step should mark and change a trained model');
  check(closeTo(afterStep.historyLastLoss, afterStep.loss, 1e-11),
    'The latest history point should report the current post-update BCE',
    JSON.stringify({ historyLastLoss: afterStep.historyLastLoss, loss: afterStep.loss }));

  await page.click('#resetBtn');
  const afterTrainingReset = await snapshot();
  check(afterTrainingReset.steps === 0 && !afterTrainingReset.running && afterTrainingReset.provenance === 'random',
    'Reset should restore a stopped zero-step random initialization', JSON.stringify(afterTrainingReset));
  check(afterTrainingReset.dataSignature === trainingInitial.dataSignature
      && afterTrainingReset.weightSignature === trainingInitial.weightSignature,
    'Reset weights should preserve data and reproduce the selected initialization');

  await page.click('#runBtn');
  await page.waitForFunction(() => window.ReLUClassificationLab.snapshot().running
    && window.ReLUClassificationLab.snapshot().steps > 0, { timeout: 5000 });
  const whileRunning = await page.evaluate(() => ({
    snapshot: window.ReLUClassificationLab.snapshot(),
    text: document.getElementById('runBtn').textContent.trim(),
    pressed: document.getElementById('runBtn').getAttribute('aria-pressed')
  }));
  check(whileRunning.snapshot.running && whileRunning.text === 'Pause' && whileRunning.pressed === 'true',
    'Running state should expose Pause and aria-pressed=true', JSON.stringify(whileRunning));
  await page.click('#runBtn');
  const paused = await page.evaluate(() => ({
    snapshot: window.ReLUClassificationLab.snapshot(),
    text: document.getElementById('runBtn').textContent.trim(),
    pressed: document.getElementById('runBtn').getAttribute('aria-pressed'),
    status: document.getElementById('statusText').textContent.trim()
  }));
  check(!paused.snapshot.running && paused.text === 'Run' && paused.pressed === 'false',
    'Paused state should expose Run and aria-pressed=false', JSON.stringify(paused));
  check(paused.status.toLowerCase().includes('paused'),
    'Pause should persist an explicit paused status', paused.status);
  const pausedSteps = paused.snapshot.steps;
  await new Promise(resolveWait => setTimeout(resolveWait, 120));
  check((await snapshot()).steps === pausedSteps,
    'Gradient steps should stop advancing after Pause');

  await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    api.setDataset('custom');
    api.clearCustomPoints();
  });
  await page.click('#runBtn');
  const emptyTraining = await page.evaluate(() => ({
    snapshot: window.ReLUClassificationLab.snapshot(),
    status: document.getElementById('statusText').textContent.trim()
  }));
  check(!emptyTraining.snapshot.running && emptyTraining.snapshot.steps === 0,
    'Empty custom data should not enter a running training state', JSON.stringify(emptyTraining));
  check(emptyTraining.status.toLowerCase().includes('add at least one'),
    'Empty custom training should explain the missing evidence', emptyTraining.status);

  // The second accuracy metric must describe what it actually measures. The
  // explanatory copy also makes the optimized BCE/thresholded-accuracy split
  // explicit instead of making a temporary accuracy dip look like divergence.
  const metricSemantics = await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    const label = () => document.getElementById('metric2Label').textContent.trim();
    api.setDataset('circles');
    const circles = label();
    api.setDataset('spirals');
    const spirals = label();
    api.setDataset('custom');
    const custom = label();
    return {
      circles,
      spirals,
      custom,
      objective: document.querySelector('.objective-note')?.textContent.replace(/\s+/g, ' ').trim()
    };
  });
  check(metricSemantics.circles === 'Balanced field accuracy',
    'Analytic datasets should label the dense metric as balanced field accuracy', metricSemantics.circles);
  check(metricSemantics.spirals === 'Holdout accuracy',
    'Spirals should label their independent metric as holdout accuracy', metricSemantics.spirals);
  check(/unavailable/i.test(metricSemantics.custom),
    'Custom evidence should say that a field-accuracy target is unavailable', metricSemantics.custom);
  check(/BCE/i.test(metricSemantics.objective || '') && /0\.5.*threshold/i.test(metricSemantics.objective || ''),
    'Optimize section should explain BCE versus thresholded accuracy', metricSemantics.objective);

  // Regression for the reported "starts okay, progressively worse" failure.
  // Every recommendation should reduce BCE monotonically at fixed checkpoints;
  // accuracy is checked only at the end because it is a thresholded metric.
  const learningSweeps = await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    const plans = [
      { dataset: 'xorField', seed: 11, steps: [25, 100, 400, 1200] },
      { dataset: 'blobs', seed: 11, steps: [25, 100, 400, 800] },
      { dataset: 'circles', seed: 11, steps: [25, 100, 500, 1600] },
      { dataset: 'moons', seed: 11, steps: [25, 100, 600, 2000] },
      { dataset: 'spirals', seed: 11, steps: [25, 100, 600, 1800] }
    ];
    return plans.map(plan => {
      api.setDataset(plan.dataset);
      api.setDataSeed(plan.seed);
      api.loadRecommendedSetup();
      const checkpoints = [api.snapshot()];
      let completed = 0;
      for (const target of plan.steps) {
        api.step(target - completed);
        checkpoints.push(api.snapshot());
        completed = target;
      }
      return { ...plan, checkpoints };
    });
  });
  const minimumFinalAccuracy = { xorField: 0.98, blobs: 0.99, circles: 0.98, moons: 0.96, spirals: 0.90 };
  for (const sweep of learningSweeps) {
    const losses = sweep.checkpoints.map(point => point.loss);
    const final = sweep.checkpoints.at(-1);
    check(losses.every((loss, index) => index === 0 || loss <= losses[index - 1] + 1e-11),
      `${sweep.dataset} recommendation should not increase BCE across training checkpoints`,
      JSON.stringify(losses));
    check(final.loss < losses[0] - 0.01,
      `${sweep.dataset} recommendation should make substantive BCE progress`,
      JSON.stringify({ initial: losses[0], final: final.loss }));
    check(final.accuracy >= minimumFinalAccuracy[sweep.dataset],
      `${sweep.dataset} recommendation should learn a useful classifier`,
      JSON.stringify({ accuracy: final.accuracy, minimum: minimumFinalAccuracy[sweep.dataset], snapshot: final }));
    check(closeTo(final.historyLastLoss, final.loss, 1e-11),
      `${sweep.dataset} history should end at the current BCE`,
      JSON.stringify({ historyLastLoss: final.historyLastLoss, loss: final.loss }));
    if (sweep.dataset === 'spirals') {
      check(final.fieldAgreement >= 0.90,
        'Spirals recommendation should generalize to at least 90% holdout accuracy',
        JSON.stringify({ holdoutAccuracy: final.fieldAgreement, snapshot: final }));
    }
  }

  // Domain-aware initialization should make the recommended filled-XOR model
  // learn across independent weight seeds, not only one favorable example.
  const filledXorSeeds = await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    return [1, 3, 7, 11, 23].map(seed => {
      api.setDataset('xorField');
      api.setDataSeed(11);
      api.loadRecommendedSetup();
      // Recommended deliberately standardizes on seed 23. Reset only the
      // weights afterward to audit the same frozen architecture across seeds.
      api.setWeightSeed(seed);
      const before = api.snapshot();
      api.step(1200);
      return { seed, before, after: api.snapshot() };
    });
  });
  for (const run of filledXorSeeds) {
    check(run.after.loss < run.before.loss * 0.20 && run.after.accuracy >= 0.98,
      `Filled-XOR recommendation should learn from weight seed ${run.seed}`,
      JSON.stringify(run));
    if (typeof run.after.deadUnits === 'number') {
      check(run.after.deadUnits < run.after.width * run.after.depth,
        `Filled-XOR seed ${run.seed} should retain at least one active ReLU`, JSON.stringify(run.after));
    }
  }

  // Accessibility contract after dynamic feature generation.
  await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    api.setDataset('xorField');
    api.setArchitecture({ depth: 2, width: 4 });
  });
  const accessibility = await page.evaluate(() => {
    const ids = [...document.querySelectorAll('[id]')].map(element => element.id);
    const duplicateIds = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
    const controlsWithoutNames = [...document.querySelectorAll('button, input, select')].filter(element => {
      if (element.offsetParent === null) return false;
      const labels = element.labels ? [...element.labels].map(label => label.textContent.trim()).join(' ') : '';
      return !(element.getAttribute('aria-label') || element.getAttribute('aria-labelledby') || labels || element.textContent.trim());
    }).map(element => element.id || element.outerHTML.slice(0, 80));
    const canvasesWithoutNames = [...document.querySelectorAll('canvas')].filter(canvas => (
      canvas.getAttribute('role') !== 'img' || !canvas.getAttribute('aria-label')
    )).map(canvas => canvas.id || canvas.className);
    return {
      duplicateIds,
      controlsWithoutNames,
      canvasesWithoutNames,
      surfaceKeyboardFocusable: document.getElementById('surfaceCanvas').tabIndex >= 0,
      skipTargetExists: document.querySelector('.skip-link')?.hash === '#experiment' && Boolean(document.getElementById('experiment')),
      liveStatus: document.getElementById('statusText')?.getAttribute('role') === 'status'
        && document.getElementById('statusText')?.getAttribute('aria-live') === 'polite'
    };
  });
  await page.evaluate(() => document.activeElement?.blur());
  await page.keyboard.press('Tab');
  const keyboardFocus = await page.evaluate(() => ({
    tag: document.activeElement?.tagName,
    text: document.activeElement?.textContent?.trim(),
    outline: getComputedStyle(document.activeElement).outlineStyle
  }));
  check(accessibility.duplicateIds.length === 0, 'DOM IDs should remain unique', accessibility.duplicateIds.join(', '));
  check(accessibility.controlsWithoutNames.length === 0,
    'Every visible classification control should have an accessible name', accessibility.controlsWithoutNames.join(' | '));
  check(accessibility.canvasesWithoutNames.length === 0,
    'Every classification canvas should expose an image role and current accessible label', accessibility.canvasesWithoutNames.join(', '));
  check(accessibility.surfaceKeyboardFocusable && accessibility.skipTargetExists && accessibility.liveStatus,
    'Surface keyboard access, skip navigation, and polite status should remain wired', JSON.stringify(accessibility));
  check(keyboardFocus.outline !== 'none',
    'Keyboard focus should remain visibly outlined', JSON.stringify(keyboardFocus));

  // Mobile acceptance exercises custom controls, both seed lanes, deep-model
  // inspection, camera controls, and both probe sliders at 390 x 844.
  await load({ width: 390, height: 844, deviceScaleFactor: 1 });
  await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    api.setDataset('custom');
    api.setArchitecture({ depth: 2, width: 4 });
  });
  const mobile = await page.evaluate(() => {
    const controls = [...document.querySelectorAll('button, select, input:not([type="hidden"]), summary')]
      .filter(element => element.offsetParent !== null
        && !element.closest('details:not([open])')
        && getComputedStyle(element).visibility !== 'hidden')
      .map(element => {
        const bounds = element.getBoundingClientRect();
        return {
          tag: element.tagName,
          id: element.id,
          text: element.textContent.trim().replace(/\s+/g, ' ').slice(0, 50),
          width: Math.round(bounds.width * 10) / 10,
          height: Math.round(bounds.height * 10) / 10
        };
      });
    const overflow = [...document.querySelectorAll('body *')].filter(element => {
      const bounds = element.getBoundingClientRect();
      return bounds.right > document.documentElement.clientWidth + 1 || bounds.left < -1;
    }).slice(0, 12).map(element => ({
      tag: element.tagName,
      id: element.id,
      className: String(element.className),
      left: Math.round(element.getBoundingClientRect().left),
      right: Math.round(element.getBoundingClientRect().right)
    }));
    return {
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      mainCanvasWidth: document.getElementById('mainCanvas').getBoundingClientRect().width,
      surfaceCanvasWidth: document.getElementById('surfaceCanvas').getBoundingClientRect().width,
      status: document.getElementById('statusText').textContent.trim(),
      trainDisabled: document.getElementById('runBtn').disabled && document.getElementById('stepBtn').disabled,
      undersized: controls.filter(control => control.width < 44 || control.height < 44),
      overflow
    };
  });
  check(mobile.scrollWidth <= mobile.clientWidth + 1 && mobile.overflow.length === 0,
    '390px classification layout should not overflow horizontally', JSON.stringify(mobile));
  check(mobile.mainCanvasWidth <= mobile.clientWidth && mobile.surfaceCanvasWidth <= mobile.clientWidth,
    'Mobile input and logit canvases should fit the viewport', JSON.stringify(mobile));
  check(mobile.undersized.length === 0,
    'Every visible mobile classification control should provide a 44 x 44 CSS-pixel target',
    JSON.stringify(mobile.undersized));
  check(mobile.trainDisabled && /add at least one/i.test(mobile.status),
    'An empty custom dataset should keep training disabled and explain why after architecture changes',
    JSON.stringify({ trainDisabled: mobile.trainDisabled, status: mobile.status }));

  const mobileScreenshot = resolve(screenshotDir, 'classification-mobile-390x844.png');
  await page.screenshot({ path: mobileScreenshot, fullPage: true });

  await load({ width: 1440, height: 900, deviceScaleFactor: 1 });
  await page.evaluate(() => {
    const api = window.ReLUClassificationLab;
    api.setDataset('xorField');
    api.loadFieldRule();
    api.setView('logit');
    api.setSurfaceView('iso');
  });
  const desktopScreenshot = resolve(screenshotDir, 'classification-desktop-1440x900.png');
  await page.screenshot({ path: desktopScreenshot, fullPage: true });

  check(pageErrors.length === 0,
    'Classification lab should emit no JavaScript or console errors', pageErrors.join(' | '));

  console.log(JSON.stringify({
    pass: failures.length === 0,
    source,
    checks: checks.length,
    failures,
    pageErrors,
    screenshots: { desktop: desktopScreenshot, mobile: mobileScreenshot },
    proofMetrics: {
      corner: { accuracy: proofs.corner.accuracy, fieldAgreement: proofs.corner.fieldAgreement },
      field: { accuracy: proofs.field.accuracy, fieldAgreement: proofs.field.fieldAgreement }
    },
    learningMetrics: Object.fromEntries(learningSweeps.map(sweep => {
      const first = sweep.checkpoints[0], last = sweep.checkpoints.at(-1);
      return [sweep.dataset, {
        steps: last.steps,
        initialLoss: first.loss,
        finalLoss: last.loss,
        finalAccuracy: last.accuracy,
        finalFieldAgreement: last.fieldAgreement
      }];
    })),
    filledXorSeeds: filledXorSeeds.map(run => ({
      seed: run.seed,
      initialLoss: run.before.loss,
      finalLoss: run.after.loss,
      finalAccuracy: run.after.accuracy,
      deadUnits: run.after.deadUnits
    }))
  }, null, 2));
  process.exitCode = failures.length ? 1 : 0;
} finally {
  await browser.close();
}
