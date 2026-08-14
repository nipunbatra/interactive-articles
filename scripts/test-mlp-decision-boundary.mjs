#!/usr/bin/env node
import puppeteer from 'puppeteer';
import { existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const BRAVE = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
const defaultSource = resolve(new URL('..', import.meta.url).pathname, 'src/articles/mlp-decision-boundary/index.html');
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
function check(condition, message, detail = '') {
  if (!condition) failures.push(detail ? `${message}: ${detail}` : message);
}

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') pageErrors.push(message.text()); });
  await page.goto(pathToFileURL(source).href, { waitUntil: 'networkidle2', timeout: 30000 });
  await page.waitForFunction(() => window.MLPPlayground && window.MLPPlayground.snapshot().mapCount === 2);

  let snapshot = await page.evaluate(() => window.MLPPlayground.snapshot());
  check(snapshot.dataset === 'xor4', 'Initial dataset should be four-point XOR', snapshot.dataset);
  check(snapshot.dataSize === 4, 'Four-point XOR should contain exactly four observations', snapshot.dataSize);
  check(snapshot.depth === 1 && snapshot.width === 2, 'Initial proof should use one layer with two ReLUs', JSON.stringify(snapshot));
  check(snapshot.accuracy === 1, 'Two-ReLU proof should classify all four XOR coordinates', snapshot.accuracy);
  check(snapshot.fieldAgreement > 0.73 && snapshot.fieldAgreement < 0.77, 'Corner proof should agree with about 75% of the filled XOR field', snapshot.fieldAgreement);
  check(snapshot.mapCount === 2, 'One activation map should appear per hidden unit', snapshot.mapCount);

  snapshot = await page.evaluate(() => {
    window.MLPPlayground.setDataset('xorField');
    window.MLPPlayground.loadProof();
    return window.MLPPlayground.snapshot();
  });
  check(snapshot.dataSize === 256, 'Filled XOR should contain 256 region samples', snapshot.dataSize);
  check(snapshot.width === 4 && snapshot.depth === 1, 'Filled-field proof should use four one-layer ReLUs', JSON.stringify(snapshot));
  check(snapshot.accuracy === 1, 'Four-ReLU field proof should classify every filled-XOR sample', snapshot.accuracy);
  check(snapshot.fieldAgreement > 0.995, 'Four-ReLU proof should match the XOR field away from the axes', snapshot.fieldAgreement);
  check(snapshot.mapCount === 4, 'Field proof should expose four activation maps', snapshot.mapCount);

  const signatures = await page.evaluate(() => {
    window.MLPPlayground.setSeed(31);
    const first = window.MLPPlayground.snapshot().dataSignature;
    window.MLPPlayground.setSeed(31);
    const repeat = window.MLPPlayground.snapshot().dataSignature;
    window.MLPPlayground.setSeed(32);
    const changed = window.MLPPlayground.snapshot().dataSignature;
    return { first, repeat, changed };
  });
  check(signatures.first === signatures.repeat, 'The same seed should reproduce the same data');
  check(signatures.first !== signatures.changed, 'A different seed should change the sampled data');

  snapshot = await page.evaluate(() => {
    window.MLPPlayground.setArchitecture({ depth: 0, width: 3 });
    const before = window.MLPPlayground.snapshot();
    window.MLPPlayground.step(1);
    return { before, after: window.MLPPlayground.snapshot() };
  });
  check(snapshot.before.depth === 0 && snapshot.before.parameters === 3, 'Linear baseline should have zero hidden layers and three parameters', JSON.stringify(snapshot.before));
  check(snapshot.before.mapCount === 0, 'Linear baseline should show no hidden maps', snapshot.before.mapCount);
  check(snapshot.after.steps === snapshot.before.steps + 1, 'Single-step control should perform exactly one update', JSON.stringify(snapshot));

  const accessibility = await page.evaluate(() => {
    const ids = [...document.querySelectorAll('[id]')].map(element => element.id);
    const duplicateIds = ids.filter((id, index) => ids.indexOf(id) !== index);
    const controlsWithoutNames = [...document.querySelectorAll('button, input, select')].filter(element => {
      const label = element.labels && element.labels.length ? element.labels[0].textContent.trim() : '';
      return !(element.getAttribute('aria-label') || label || element.textContent.trim());
    }).length;
    const canvasesWithoutNames = [...document.querySelectorAll('canvas')].filter(canvas => canvas.getAttribute('role') !== 'img' || !canvas.getAttribute('aria-label')).length;
    const firstButton = document.querySelector('button'); firstButton.focus();
    const focusStyle = getComputedStyle(firstButton);
    return { duplicateIds, controlsWithoutNames, canvasesWithoutNames, focusOutline: focusStyle.outlineStyle };
  });
  check(accessibility.duplicateIds.length === 0, 'IDs should be unique', accessibility.duplicateIds.join(', '));
  check(accessibility.controlsWithoutNames === 0, 'Every control should have an accessible name', accessibility.controlsWithoutNames);
  check(accessibility.canvasesWithoutNames === 0, 'Every canvas should have an image role and accessible name', accessibility.canvasesWithoutNames);
  check(accessibility.focusOutline !== 'none', 'Keyboard focus should remain visible', accessibility.focusOutline);

  await page.setViewport({ width: 375, height: 812, deviceScaleFactor: 1 });
  await new Promise(resolveWait => setTimeout(resolveWait, 180));
  const mobile = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    smallestButton: Math.min(...[...document.querySelectorAll('button:not([hidden])')].map(button => button.getBoundingClientRect().height)),
    offenders: [...document.querySelectorAll('body *')].filter(element => {
      const rect = element.getBoundingClientRect();
      return rect.right > document.documentElement.clientWidth + 1 || rect.left < -1;
    }).slice(0, 12).map(element => ({ tag: element.tagName, id: element.id, className: String(element.className), right: Math.round(element.getBoundingClientRect().right), scrollWidth: element.scrollWidth }))
  }));
  check(mobile.scrollWidth <= mobile.clientWidth + 1, 'Mobile layout should not overflow horizontally', JSON.stringify(mobile));
  check(mobile.smallestButton >= 32, 'Visible buttons should keep a usable touch height', mobile.smallestButton);
  check(pageErrors.length === 0, 'Page should emit no JavaScript or console errors', pageErrors.join(' | '));

  if (screenshotDir) {
    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
    await page.evaluate(() => window.MLPPlayground.setDataset('xor4'));
    const lab = await page.$('#lab');
    await lab.screenshot({ path: resolve(screenshotDir, 'mlp-playground-lab.png') });
  }

  const result = { pass: failures.length === 0, failures, source, initialChecks: 6, pageErrors };
  console.log(JSON.stringify(result, null, 2));
  process.exitCode = failures.length ? 1 : 0;
} finally {
  await browser.close();
}
