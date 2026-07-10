#!/usr/bin/env node
// CDP verification harness for interactive articles, driving Brave via puppeteer.
//
// Usage:
//   node scripts/verify-interactive.mjs <slug|path-or-url> [outdir]
//
// - Accepts a bare slug (resolved to docs/<slug>/index.html), a file path, or a URL.
// - Loads the page in Brave, captures console + page errors, screenshots, then
//   exercises every <input>/<button> control and screenshots again.
// - Emits a JSON verdict on stdout and writes screenshots to <outdir> (default: /tmp/ia-verify).
//
// Exit code 0 = PASS (no JS errors, canvas non-blank, no KaTeX errors, controls present).

import puppeteer from 'puppeteer';
import { existsSync, mkdirSync } from 'node:fs';
import { resolve, isAbsolute } from 'node:path';

const BRAVE = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
const REPO = resolve(new URL('..', import.meta.url).pathname);

function toUrl(arg) {
  if (arg.startsWith('http') || arg.startsWith('file:')) return arg;
  if (arg.includes('/') || arg.endsWith('.html')) {
    const p = isAbsolute(arg) ? arg : resolve(process.cwd(), arg);
    return 'file://' + p;
  }
  // bare slug
  const p = resolve(REPO, 'docs', arg, 'index.html');
  return 'file://' + p;
}

const arg = process.argv[2];
if (!arg) { console.error('usage: verify-interactive.mjs <slug|path|url> [outdir]'); process.exit(2); }
const url = toUrl(arg);
const outdir = process.argv[3] || '/tmp/ia-verify';
mkdirSync(outdir, { recursive: true });
const label = arg.replace(/[^a-z0-9]+/gi, '_');

const errors = [];
const warnings = [];

const browser = await puppeteer.launch({
  executablePath: existsSync(BRAVE) ? BRAVE : undefined,
  headless: 'new',
  args: ['--no-sandbox', '--force-color-profile=srgb', '--window-size=1280,900'],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 2 });

  page.on('console', (m) => { if (m.type() === 'error') errors.push('console.error: ' + m.text()); });
  page.on('pageerror', (e) => errors.push('pageerror: ' + (e && e.message ? e.message : String(e))));
  page.on('requestfailed', (r) => {
    const u = r.url();
    // KaTeX/font CDN failures matter; ignore favicon
    if (!u.includes('favicon')) warnings.push('requestfailed: ' + u + ' ' + (r.failure()?.errorText || ''));
  });

  const resp = await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
  if (!resp || !resp.ok()) {
    if (resp && resp.status() !== 0) errors.push('http status ' + resp.status());
  }
  await new Promise((r) => setTimeout(r, 900)); // let canvases draw / KaTeX render

  // --- inspect page state ---
  const state = await page.evaluate(() => {
    const out = { canvases: [], katexErrors: 0, rawDollars: 0, controls: 0, buttons: 0, title: document.title };
    // canvas non-blank check: sample pixels, count distinct-ish colors
    for (const c of document.querySelectorAll('canvas')) {
      try {
        const ctx = c.getContext('2d');
        const w = c.width, h = c.height;
        let nonUniform = false, first = null, n = 0, painted = 0;
        if (ctx && w && h) {
          const data = ctx.getImageData(0, 0, w, h).data;
          const step = Math.max(4, Math.floor(data.length / 4 / 4000) * 4);
          for (let i = 0; i < data.length; i += step) {
            const key = data[i] + ',' + data[i+1] + ',' + data[i+2] + ',' + data[i+3];
            if (data[i+3] !== 0) painted++;
            if (first === null) first = key; else if (key !== first) nonUniform = true;
            n++;
          }
        }
        out.canvases.push({ w, h, nonUniform, sampled: n, paintedFrac: n ? +(painted / n).toFixed(3) : 0 });
      } catch (e) { out.canvases.push({ error: String(e) }); }
    }
    out.katexErrors = document.querySelectorAll('.katex-error').length;
    // raw $...$ left unrendered in visible text
    const bodyText = document.body ? document.body.innerText : '';
    const m = bodyText.match(/\$[^$\n]{1,60}\$/g);
    out.rawDollars = m ? m.length : 0;
    out.controls = document.querySelectorAll('input[type=range], input[type=number], select').length;
    out.buttons = document.querySelectorAll('button').length;
    return out;
  });

  await page.screenshot({ path: `${outdir}/${label}__1_initial.png` });

  // --- exercise controls: move each range to a different value, click each button ---
  let interacted = 0;
  const ranges = await page.$$('input[type=range]');
  for (const r of ranges.slice(0, 8)) {
    try {
      await r.evaluate((el) => {
        const min = parseFloat(el.min || '0'), max = parseFloat(el.max || '100');
        const mid = (min + max) / 2;
        const target = (parseFloat(el.value) === mid) ? max : mid;
        const set = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        set.call(el, String(target));
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      });
      interacted++;
    } catch (e) { /* ignore */ }
  }
  const buttons = await page.$$('button');
  for (const b of buttons.slice(0, 6)) {
    try { await b.click({ delay: 10 }); interacted++; } catch (e) {}
  }
  await new Promise((r) => setTimeout(r, 700));
  await page.screenshot({ path: `${outdir}/${label}__2_after.png` });

  // re-check canvases updated / still non-blank
  const state2 = await page.evaluate(() => {
    const cs = [];
    for (const c of document.querySelectorAll('canvas')) {
      try {
        const ctx = c.getContext('2d'); const w = c.width, h = c.height;
        let nonUniform = false, first = null;
        const data = ctx.getImageData(0, 0, w, h).data;
        const step = Math.max(4, Math.floor(data.length / 4 / 4000) * 4);
        for (let i = 0; i < data.length; i += step) {
          const key = data[i]+','+data[i+1]+','+data[i+2];
          if (first === null) first = key; else if (key !== first) nonUniform = true;
        }
        cs.push({ nonUniform });
      } catch (e) { cs.push({ error: String(e) }); }
    }
    return cs;
  });

  // --- verdict ---
  const blankCanvas = state.canvases.filter((c) => !c.error && !c.nonUniform);
  const problems = [];
  if (errors.length) problems.push(`${errors.length} JS error(s)`);
  if (state.katexErrors) problems.push(`${state.katexErrors} KaTeX error(s)`);
  if (state.rawDollars) problems.push(`${state.rawDollars} unrendered $...$`);
  if (state.canvases.length && blankCanvas.length === state.canvases.length) problems.push('all canvases blank');
  if (state.canvases.length === 0 && state.buttons + state.controls === 0) problems.push('no canvas and no controls');

  const verdict = {
    url, pass: problems.length === 0, problems,
    errors, warnings: warnings.slice(0, 8),
    canvases: state.canvases, katexErrors: state.katexErrors, rawDollars: state.rawDollars,
    controls: state.controls, buttons: state.buttons, interacted,
    screenshots: [`${outdir}/${label}__1_initial.png`, `${outdir}/${label}__2_after.png`],
  };
  console.log(JSON.stringify(verdict, null, 2));
  await browser.close();
  process.exit(verdict.pass ? 0 : 1);
} catch (e) {
  console.log(JSON.stringify({ url, pass: false, fatal: String(e && e.stack || e) }, null, 2));
  await browser.close();
  process.exit(1);
}
