#!/usr/bin/env node

import puppeteer from 'puppeteer';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const defaultSource = resolve(here, '../src/articles/relu-function-approximation/index.html');
const source = resolve(process.argv[2] || defaultSource);
const screenshotDirectory = resolve(process.argv[3] || '/private/tmp/relu-lab-screenshots');
const screenshots = {
  desktop: resolve(screenshotDirectory, 'approximation-desktop-1440x900.png'),
  mobile: resolve(screenshotDirectory, 'approximation-mobile-390x844.png')
};

mkdirSync(screenshotDirectory, { recursive: true });

if (!existsSync(source)) {
  throw new Error(`ReLU function-approximation lab not found: ${source}`);
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

function closeTo(actual, expected, tolerance = 1e-12) {
  return Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance;
}

function parseModel(snapshot) {
  const values = snapshot.weightSignature.split('|').map(Number);
  const width = snapshot.width;
  return {
    c: values[0],
    W: values.slice(1, 1 + width),
    b: values.slice(1 + width, 1 + 2 * width),
    v: values.slice(1 + 2 * width, 1 + 3 * width)
  };
}

function contributionSum(model, x) {
  return model.c + model.v.reduce(
    (sum, outputWeight, index) => sum + outputWeight * Math.max(0, model.W[index] * x + model.b[index]),
    0
  );
}

function sine(x) {
  return Math.sin(Math.PI * x);
}

function linearInterpolant(knots, fn, x) {
  const rightIndex = Math.min(
    knots.length - 1,
    Math.max(1, knots.findIndex(knot => knot >= x))
  );
  const left = knots[rightIndex - 1];
  const right = knots[rightIndex];
  const t = (x - left) / (right - left);
  return fn(left) + t * (fn(right) - fn(left));
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
  await page.waitForFunction(() => window.ReLUFunctionLab?.snapshot, { timeout: 10000 });
}

async function snapshot() {
  return page.evaluate(() => window.ReLUFunctionLab.snapshot());
}

try {
  await load();

  // Flow 1: canonical fixed-knot witnesses. These are deterministic 401-grid
  // values for sin(pi x) with uniform interior knots.
  const expectedCanonical = new Map([
    [2, { mse: 0.06591720640024977, gap: 0.437254368063465, maxX: -0.635 }],
    [5, { mse: 0.00476409938985911, gap: 0.1339745962155614, maxX: -0.5 }],
    [15, { mse: 0.00009817895407574258, gap: 0.01882507512327125, maxX: -0.44 }]
  ]);
  const canonical = await page.evaluate(() => [2, 5, 15].map(interiorHinges => {
    window.ReLUFunctionLab.setLane('construct');
    window.ReLUFunctionLab.setTarget('sine');
    window.ReLUFunctionLab.setComplexity(interiorHinges);
    return window.ReLUFunctionLab.snapshot();
  }));

  for (const result of canonical) {
    const expected = expectedCanonical.get(result.interiorHinges);
    check(result.width === result.interiorHinges + 1,
      `${result.interiorHinges}-hinge construction should expose one ReLU per segment`,
      JSON.stringify(result));
    check(closeTo(result.denseGridMse, expected.mse),
      `${result.interiorHinges}-hinge sine construction should retain its canonical dense-grid MSE`,
      `${result.denseGridMse} versus ${expected.mse}`);
    check(closeTo(result.maxGridGap, expected.gap),
      `${result.interiorHinges}-hinge sine construction should retain its canonical maximum gap`,
      `${result.maxGridGap} versus ${expected.gap}`);
    check(closeTo(result.maxGapX, expected.maxX),
      `${result.interiorHinges}-hinge sine construction should retain its canonical maximum-gap location`,
      `${result.maxGapX} versus ${expected.maxX}`);
  }
  check(canonical[0].maxGridGap > canonical[1].maxGridGap
      && canonical[1].maxGridGap > canonical[2].maxGridGap,
    'Canonical sine maximum gaps should shrink from 2 to 5 to 15 hinges');

  // Flow 2: the signed cards must form the displayed interpolant. The public
  // weight signature lets QA sum c + v_j ReLU(w_j x + b_j) independently.
  const constructed = canonical[1];
  const constructedModel = parseModel(constructed);
  const constructedKnots = [-1, ...constructed.interiorKnotPositions, 1];
  for (const x of [...constructedKnots, -0.91, -0.7, -0.25, 0.13, 0.55, 0.93]) {
    const sum = contributionSum(constructedModel, x);
    const expected = linearInterpolant(constructedKnots, sine, x);
    check(closeTo(sum, expected, 2e-5),
      'Bias plus signed ReLU contributions should equal the fixed-knot interpolant',
      `x=${x.toFixed(3)}, sum=${sum}, interpolant=${expected}`);
  }

  // Flow 3: lane controls use .lane-tab[data-lane], and only training exposes
  // active optimizer buttons and .train-only controls.
  await page.evaluate(() => window.ReLUFunctionLab.setLane('construct'));
  let laneState = await page.evaluate(() => {
    const visible = element => element.offsetParent !== null && getComputedStyle(element).visibility !== 'hidden';
    return {
      lane: window.ReLUFunctionLab.snapshot().lane,
      constructPressed: document.querySelector('[data-lane="construct"]').getAttribute('aria-pressed'),
      trainPressed: document.querySelector('[data-lane="train"]').getAttribute('aria-pressed'),
      optimizerDisabled: ['runBtn', 'stepBtn', 'batchBtn'].every(id => document.getElementById(id).disabled),
      resetDisabled: document.getElementById('resetBtn').disabled,
      trainOnlyVisible: [...document.querySelectorAll('.train-only')].some(visible),
      constructOnlyVisible: [...document.querySelectorAll('.construction-only')].every(visible)
    };
  });
  check(laneState.lane === 'construct' && laneState.constructPressed === 'true' && laneState.trainPressed === 'false',
    'Construction lane should expose a consistent pressed state', JSON.stringify(laneState));
  check(laneState.optimizerDisabled && !laneState.resetDisabled,
    'Construction should disable optimizer actions but retain Reset', JSON.stringify(laneState));
  check(!laneState.trainOnlyVisible && laneState.constructOnlyVisible,
    'Construction should show only construction-specific controls', JSON.stringify(laneState));

  await page.click('[data-lane="train"]');
  laneState = await page.evaluate(() => {
    const visible = element => element.offsetParent !== null && getComputedStyle(element).visibility !== 'hidden';
    return {
      lane: window.ReLUFunctionLab.snapshot().lane,
      constructPressed: document.querySelector('[data-lane="construct"]').getAttribute('aria-pressed'),
      trainPressed: document.querySelector('[data-lane="train"]').getAttribute('aria-pressed'),
      optimizerEnabled: ['runBtn', 'stepBtn', 'batchBtn'].every(id => !document.getElementById(id).disabled),
      trainOnlyVisible: [...document.querySelectorAll('.train-only')].every(visible),
      constructOnlyVisible: [...document.querySelectorAll('.construction-only')].some(visible)
    };
  });
  check(laneState.lane === 'train' && laneState.constructPressed === 'false' && laneState.trainPressed === 'true',
    'Training lane should expose a consistent pressed state', JSON.stringify(laneState));
  check(laneState.optimizerEnabled && laneState.trainOnlyVisible && !laneState.constructOnlyVisible,
    'Training should enable optimizer actions and hide construction-only controls', JSON.stringify(laneState));

  // Flow 4: a seed, width, target, and step count fully determine training.
  const deterministic = await page.evaluate(() => {
    const api = window.ReLUFunctionLab;
    api.setLane('train');
    api.setTarget('sine');
    api.setComplexity(6);
    api.setSeed(37);
    const initial = api.snapshot();
    api.step(100);
    return { initial, trained: api.snapshot() };
  });
  check(deterministic.trained.steps === 100,
    'Training API should execute exactly 100 requested steps', deterministic.trained.steps);
  check(deterministic.trained.weightSignature !== deterministic.initial.weightSignature,
    'Training should change the initial weight signature');
  check(deterministic.trained.trainingSampleMse < deterministic.initial.trainingSampleMse,
    'One hundred stable sine steps should reduce training-sample MSE',
    `${deterministic.initial.trainingSampleMse} to ${deterministic.trained.trainingSampleMse}`);

  await page.click('#resetBtn');
  const reset = await snapshot();
  check(reset.steps === 0 && reset.provenance === 'random',
    'Reset should return training to a zero-step random initialization', JSON.stringify(reset));
  check(reset.weightSignature === deterministic.initial.weightSignature,
    'Reset should exactly reproduce the selected deterministic initialization');
  const repeated = await page.evaluate(() => {
    window.ReLUFunctionLab.step(100);
    return window.ReLUFunctionLab.snapshot();
  });
  check(repeated.weightSignature === deterministic.trained.weightSignature,
    'Reset followed by the same 100 steps should reproduce trained weights exactly');
  check(closeTo(repeated.trainingSampleMse, deterministic.trained.trainingSampleMse),
    'Reset followed by the same steps should reproduce training MSE exactly');

  // Flow 5: exercise the actual pointer path for an interior-knot drag, then
  // verify the visible Reset control restores uniform positions.
  await page.evaluate(() => {
    const api = window.ReLUFunctionLab;
    api.setLane('construct');
    api.setTarget('sine');
    api.setComplexity(5);
  });
  const beforeDrag = await snapshot();
  const canvas = await page.$('#mainCanvas');
  await canvas.evaluate(element => element.scrollIntoView({ block: 'center' }));
  const canvasBox = await canvas.boundingBox();
  check(Boolean(canvasBox), 'Main plot should have a measurable pointer target');
  if (canvasBox) {
    const knotIndex = 2;
    const knotX = beforeDrag.interiorKnotPositions[knotIndex];
    const knotY = sine(knotX);
    const startX = canvasBox.x + 58 + (knotX + 1) * (canvasBox.width - 78) / 2;
    const startY = canvasBox.y + 19 + (1.35 - knotY) * (canvasBox.height - 62) / 2.7;
    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(startX + 52, startY, { steps: 8 });
    await page.mouse.up();
    const afterDrag = await snapshot();
    check(Math.abs(afterDrag.interiorKnotPositions[knotIndex] - knotX) > 0.04,
      'Dragging an orange interior knot should update its x position',
      `${knotX} to ${afterDrag.interiorKnotPositions[knotIndex]}`);
    check(afterDrag.interiorKnotPositions.every((value, index, values) => index === 0 || value > values[index - 1]),
      'Dragged interior knots should remain strictly ordered', JSON.stringify(afterDrag.interiorKnotPositions));
    await page.click('#resetBtn');
    const afterKnotReset = await snapshot();
    check(afterKnotReset.interiorKnotPositions.every((value, index) => closeTo(value, -1 + 2 * (index + 1) / 6)),
      'Construction Reset should restore canonical uniform knot positions',
      JSON.stringify(afterKnotReset.interiorKnotPositions));
  }

  // Numerical smoke test every target in both lanes, including the zero-valued
  // initial custom target. No metric or model parameter may become non-finite.
  const targetResults = await page.evaluate(() => {
    const api = window.ReLUFunctionLab;
    const targets = ['sine', 'tent', 'twoBumps', 'smoothStep', 'custom'];
    return targets.map(target => {
      api.setLane('construct');
      api.setTarget(target);
      api.setComplexity(5);
      const construct = api.snapshot();
      api.setLane('train');
      api.setComplexity(6);
      api.setSeed(11);
      api.step(1);
      const train = api.snapshot();
      return { target, construct, train };
    });
  });
  for (const result of targetResults) {
    for (const [lane, value] of [['construct', result.construct], ['train', result.train]]) {
      const signatureFinite = value.weightSignature.split('|').every(part => Number.isFinite(Number(part)));
      check(Number.isFinite(value.denseGridMse) && Number.isFinite(value.maxGridGap) && signatureFinite,
        `${result.target} should remain finite in the ${lane} lane`, JSON.stringify(value));
    }
  }

  // Accessibility contract: names, unique IDs, live status, keyboard focus,
  // and labelled canvas alternatives must survive dynamic contribution cards.
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
    const mainCanvas = document.getElementById('mainCanvas');
    return {
      duplicateIds,
      controlsWithoutNames,
      canvasesWithoutNames,
      mainCanvasKeyboardFocusable: mainCanvas.tabIndex >= 0,
      skipTargetExists: document.querySelector('.skip-link')?.hash === '#workspace' && Boolean(document.getElementById('workspace')),
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
  check(accessibility.controlsWithoutNames.length === 0, 'Every visible form control should have an accessible name', accessibility.controlsWithoutNames.join(' | '));
  check(accessibility.canvasesWithoutNames.length === 0, 'Every canvas should expose an image role and current accessible label', accessibility.canvasesWithoutNames.join(', '));
  check(accessibility.mainCanvasKeyboardFocusable, 'Main plot should remain keyboard focusable');
  check(accessibility.skipTargetExists && accessibility.liveStatus, 'Skip link and polite live training status should be wired');
  check(keyboardFocus.outline !== 'none', 'Keyboard focus should remain visibly outlined', JSON.stringify(keyboardFocus));

  // Desktop acceptance: the complete primary experiment loop—not only its
  // heading—must fit in the initial 1440 x 900 viewport.
  await load({ width: 1440, height: 900, deviceScaleFactor: 1 });
  const desktop = await page.evaluate(() => {
    const rect = selector => {
      const bounds = document.querySelector(selector).getBoundingClientRect();
      return { top: bounds.top, bottom: bounds.bottom, left: bounds.left, right: bounds.right, width: bounds.width, height: bounds.height };
    };
    return {
      viewport: { width: innerWidth, height: innerHeight },
      documentWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      laneTabs: rect('.lane-tabs'),
      controls: rect('.controls'),
      actionGroup: rect('.action-group'),
      metrics: rect('.metric-strip'),
      mainCanvas: rect('#mainCanvas')
    };
  });
  check(desktop.documentWidth <= desktop.clientWidth + 1,
    '1440px layout should not overflow horizontally', JSON.stringify(desktop));
  check(desktop.actionGroup.bottom <= desktop.viewport.height + 1,
    'Desktop action controls should fit in the first 900px viewport', JSON.stringify(desktop.actionGroup));
  check(desktop.metrics.bottom <= desktop.viewport.height + 1 && desktop.mainCanvas.bottom <= desktop.viewport.height + 1,
    'Desktop metrics and complete main canvas should fit in the first 900px viewport',
    JSON.stringify({ metrics: desktop.metrics, mainCanvas: desktop.mainCanvas }));
  await page.screenshot({ path: screenshots.desktop, fullPage: true });

  // Mobile acceptance: no horizontal overflow and every visible native control
  // has at least a 44 x 44 CSS-pixel target.
  await load({ width: 390, height: 844, deviceScaleFactor: 1 });
  const mobile = await page.evaluate(() => {
    document.querySelector('.knot-editor')?.setAttribute('open', '');
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
      undersized: controls.filter(control => control.width < 44 || control.height < 44),
      overflow
    };
  });
  check(mobile.scrollWidth <= mobile.clientWidth + 1 && mobile.overflow.length === 0,
    '390px layout should not overflow horizontally', JSON.stringify(mobile));
  check(mobile.undersized.length === 0,
    'Every visible mobile control should provide a 44 x 44 CSS-pixel target',
    JSON.stringify(mobile.undersized));
  await page.screenshot({ path: screenshots.mobile, fullPage: true });

  check(pageErrors.length === 0, 'Lab should emit no JavaScript or console errors', pageErrors.join(' | '));

  console.log(JSON.stringify({
    pass: failures.length === 0,
    source,
    checks: checks.length,
    failures,
    pageErrors,
    screenshots,
    canonicalGaps: Object.fromEntries(canonical.map(result => [result.interiorHinges, result.maxGridGap]))
  }, null, 2));
  process.exitCode = failures.length ? 1 : 0;
} finally {
  await browser.close();
}
