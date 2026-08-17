#!/usr/bin/env node
import puppeteer from 'puppeteer';
import { existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const BRAVE = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
const repoRoot = resolve(new URL('..', import.meta.url).pathname);
const defaultSource = resolve(repoRoot, 'src/articles/mlp-decision-boundary/index.html');
const source = resolve(process.argv[2] || defaultSource);
const screenshotDir = process.argv[3] ? resolve(process.argv[3]) : null;
if (!existsSync(source)) throw new Error(`Interactive source not found: ${source}`);
if (screenshotDir) mkdirSync(screenshotDir, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: existsSync(BRAVE) ? BRAVE : undefined,
  headless: 'new',
  args: ['--no-sandbox', '--force-color-profile=srgb']
});
const failures = [];
const check = (condition, message, detail = '') => { if (!condition) failures.push(detail ? `${message}: ${detail}` : message); };

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') pageErrors.push(message.text()); });
  await page.goto(pathToFileURL(source).href, { waitUntil: 'networkidle2', timeout: 30000 });
  await page.waitForFunction(() => window.ReLULab && window.ReLULab.snapshot().featureCount === 2);

  let snapshot = await page.evaluate(() => window.ReLULab.snapshot());
  check(snapshot.mode === 'classification', 'Default mode should be classification', JSON.stringify(snapshot));
  check(snapshot.dataset === 'xor4' && snapshot.dataSize === 4, 'Default dataset should be the four Boolean XOR points', JSON.stringify(snapshot));
  check(snapshot.depth === 1 && snapshot.width === 2 && snapshot.parameters === 9, 'Corner rule should use exactly two hidden ReLUs and nine parameters', JSON.stringify(snapshot));
  check(snapshot.accuracy === 1, 'Two-ReLU corner rule should classify all four Boolean points', snapshot.accuracy);
  check(snapshot.fieldAgreement > .73 && snapshot.fieldAgreement < .77, 'Corner rule should agree with about 75% of the filled XOR field', snapshot.fieldAgreement);

  snapshot = await page.evaluate(() => {
    window.ReLULab.setDataset('xorField');
    window.ReLULab.loadFieldRule();
    return window.ReLULab.snapshot();
  });
  check(snapshot.dataSize === 256, 'Filled XOR should contain 256 sampled points', snapshot.dataSize);
  check(snapshot.depth === 1 && snapshot.width === 4, 'Regional XOR rule should use four ReLUs', JSON.stringify(snapshot));
  check(snapshot.accuracy === 1 && snapshot.fieldAgreement > .995, 'Four-ReLU rule should match the filled XOR field away from the axes', JSON.stringify(snapshot));

  const deterministic = await page.evaluate(() => {
    window.ReLULab.setDataset('moons'); window.ReLULab.setSeed(31);
    const a = window.ReLULab.snapshot().dataSignature;
    window.ReLULab.setSeed(31); const b = window.ReLULab.snapshot().dataSignature;
    window.ReLULab.setSeed(32); const c = window.ReLULab.snapshot().dataSignature;
    return { a, b, c };
  });
  check(deterministic.a === deterministic.b, 'A seed should reproduce a sampled dataset');
  check(deterministic.a !== deterministic.c, 'Changing the seed should change a sampled dataset');

  snapshot = await page.evaluate(() => {
    window.ReLULab.setArchitecture({ depth: 0, width: 3 });
    const before = window.ReLULab.snapshot(); window.ReLULab.step(1);
    return { before, after: window.ReLULab.snapshot() };
  });
  check(snapshot.before.depth === 0 && snapshot.before.parameters === 3 && snapshot.before.featureCount === 0, 'Linear baseline should have three parameters and no hidden features', JSON.stringify(snapshot.before));
  check(snapshot.after.steps === snapshot.before.steps + 1, 'One step should perform exactly one gradient update', JSON.stringify(snapshot));

  const approx = await page.evaluate(() => {
    window.ReLULab.setMode('approximation');
    window.ReLULab.setApproximation({ method: 'construct', width: 2, target: 'sine' }); const w2 = window.ReLULab.snapshot();
    window.ReLULab.setApproximation({ method: 'construct', width: 5, target: 'sine' }); const w5 = window.ReLULab.snapshot();
    window.ReLULab.setApproximation({ method: 'construct', width: 15, target: 'sine' }); const w15 = window.ReLULab.snapshot();
    return { w2, w5, w15 };
  });
  check(approx.w2.mode === 'approximation' && approx.w2.approxMethod === 'construct', 'Approximation mode should expose a fixed-knot construction', JSON.stringify(approx.w2));
  check(approx.w2.maxGap > approx.w5.maxGap && approx.w5.maxGap > approx.w15.maxGap, 'Adding fixed knots should reduce the largest sine approximation gap', JSON.stringify(approx));
  check(approx.w15.maxGap < .03, 'Fifteen segments should approximate the displayed sine target closely', approx.w15.maxGap);

  snapshot = await page.evaluate(() => {
    window.ReLULab.setApproximation({ method: 'train', width: 5, target: 'tent' });
    const before = window.ReLULab.snapshot(); window.ReLULab.step(1);
    return { before, after: window.ReLULab.snapshot() };
  });
  check(snapshot.before.approxMethod === 'train', 'Training mode should use random ReLU weights', JSON.stringify(snapshot.before));
  check(snapshot.after.steps === 1 && Number.isFinite(snapshot.after.mse), 'One approximation step should update a finite loss', JSON.stringify(snapshot.after));

  snapshot = await page.evaluate(() => {
    window.ReLULab.setMode('classification'); window.ReLULab.clearCustomPoints();
    window.ReLULab.addCustomPoint(.2, .2, 0); window.ReLULab.addCustomPoint(.8, .8, 1);
    return window.ReLULab.snapshot();
  });
  check(snapshot.dataset === 'custom' && snapshot.dataSize === 2, 'Custom point API should add labeled data to the plot', JSON.stringify(snapshot));

  const accessibility = await page.evaluate(() => {
    const ids = [...document.querySelectorAll('[id]')].map(element => element.id);
    const duplicateIds = ids.filter((id, index) => ids.indexOf(id) !== index);
    const unnamedControls = [...document.querySelectorAll('button, input, select')].filter(element => {
      const label = element.labels && element.labels.length ? element.labels[0].textContent.trim() : '';
      return !(element.getAttribute('aria-label') || label || element.textContent.trim());
    }).map(element => ({ tag: element.tagName, id: element.id, type: element.type }));
    const canvasesWithoutNames = [...document.querySelectorAll('canvas')].filter(canvas => canvas.getAttribute('role') !== 'img' || !canvas.getAttribute('aria-label')).length;
    const firstButton = document.querySelector('button'); firstButton.focus();
    return { duplicateIds, unnamedControls, canvasesWithoutNames, focusOutline: getComputedStyle(firstButton).outlineStyle };
  });
  check(accessibility.duplicateIds.length === 0, 'IDs should be unique', accessibility.duplicateIds.join(', '));
  check(accessibility.unnamedControls.length === 0, 'Every control should have an accessible name', JSON.stringify(accessibility.unnamedControls));
  check(accessibility.canvasesWithoutNames === 0, 'Every canvas should have an accessible image name', accessibility.canvasesWithoutNames);
  check(accessibility.focusOutline !== 'none', 'Keyboard focus should be visible', accessibility.focusOutline);

  if (screenshotDir) {
    await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
    await page.evaluate(() => { window.ReLULab.setMode('classification'); window.ReLULab.setDataset('xorField'); window.ReLULab.loadFieldRule(); });
    await page.screenshot({ path: resolve(screenshotDir, 'relu-lab-classification.png'), fullPage: true });
    await page.evaluate(() => { window.ReLULab.setMode('approximation'); window.ReLULab.setApproximation({ method: 'construct', width: 5, target: 'sine' }); });
    await page.screenshot({ path: resolve(screenshotDir, 'relu-lab-approximation.png'), fullPage: true });
  }

  await page.setViewport({ width: 375, height: 812, deviceScaleFactor: 1 });
  await new Promise(resolveWait => setTimeout(resolveWait, 150));
  const mobile = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    smallestButtons: [...document.querySelectorAll('button')].filter(button => button.offsetParent !== null).map(button => ({ id: button.id, className: button.className, text: button.textContent.trim(), height: button.getBoundingClientRect().height })).sort((a,b) => a.height-b.height).slice(0,5),
    offenders: [...document.querySelectorAll('body *')].filter(element => {
      const rect = element.getBoundingClientRect();
      return rect.right > document.documentElement.clientWidth + 1 || rect.left < -1;
    }).slice(0, 10).map(element => ({ tag: element.tagName, id: element.id, right: Math.round(element.getBoundingClientRect().right) }))
  }));
  check(mobile.scrollWidth <= mobile.clientWidth + 1, 'Mobile layout should not overflow horizontally', JSON.stringify(mobile));
  check(mobile.smallestButtons[0].height >= 32, 'Visible buttons should have a usable touch height', JSON.stringify(mobile.smallestButtons));
  check(pageErrors.length === 0, 'Page should emit no JavaScript or console errors', pageErrors.join(' | '));
  if (screenshotDir) await page.screenshot({ path: resolve(screenshotDir, 'relu-lab-mobile.png'), fullPage: true });

  const result = { pass: failures.length === 0, failures, source, pageErrors };
  console.log(JSON.stringify(result, null, 2));
  process.exitCode = failures.length ? 1 : 0;
} finally {
  await browser.close();
}
