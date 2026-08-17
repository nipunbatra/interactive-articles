#!/usr/bin/env node

import puppeteer from 'puppeteer';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = resolve(process.argv[2] || resolve(here, '../src/articles/autograd/index.html'));
const outDir = resolve(process.argv[3] || '/private/tmp/autograd-audit-screenshots');
mkdirSync(outDir, { recursive: true });

if (!existsSync(source)) throw new Error(`Autograd article not found: ${source}`);

const browserCandidates = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
].filter(Boolean);
const executablePath = browserCandidates.find(existsSync);

const browser = await puppeteer.launch({
  ...(executablePath ? { executablePath } : {}),
  headless: 'new',
  args: ['--no-sandbox', '--force-color-profile=srgb'],
});

const failures = [];
const errors = [];
const checks = [];
const page = await browser.newPage();
page.on('pageerror', (error) => errors.push(error.message));
page.on('console', (message) => {
  if (message.type() === 'error') errors.push(message.text());
});

function assert(condition, message) {
  if (condition) checks.push(message);
  else failures.push(message);
}

async function load(width, height) {
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  await page.goto(pathToFileURL(source).href, { waitUntil: 'networkidle0', timeout: 30000 });
  await page.waitForFunction(() => window.AutogradExplainer?.setGraphStage, { timeout: 10000 });
}

async function settle() {
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function auditViewport(name, width, height) {
  await load(width, height);
  const result = await page.evaluate(() => {
    const root = document.documentElement;
    const visible = (element) => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
    };
    const controls = [...document.querySelectorAll('button, a[href], input, summary, [tabindex="0"]')]
      .filter(visible)
      .map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          label: element.textContent.trim() || element.getAttribute('aria-label') || element.id,
          width: rect.width,
          height: rect.height,
        };
      });

    const transformedBox = (svg, node) => {
      let box;
      try { box = node.getBBox(); } catch { return null; }
      const svgMatrix = svg.getScreenCTM();
      const nodeMatrix = node.getScreenCTM();
      if (!svgMatrix || !nodeMatrix) return null;
      const matrix = svgMatrix.inverse().multiply(nodeMatrix);
      const points = [
        new DOMPoint(box.x, box.y),
        new DOMPoint(box.x + box.width, box.y),
        new DOMPoint(box.x, box.y + box.height),
        new DOMPoint(box.x + box.width, box.y + box.height),
      ].map((point) => point.matrixTransform(matrix));
      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);
      return {
        x: Math.min(...xs),
        y: Math.min(...ys),
        width: Math.max(...xs) - Math.min(...xs),
        height: Math.max(...ys) - Math.min(...ys),
      };
    };

    const svgIssues = [...document.querySelectorAll('svg')].flatMap((svg) => {
      const viewBox = svg.viewBox.baseVal;
      if (!viewBox?.width || !viewBox?.height) return [];
      return [...svg.querySelectorAll('text')].flatMap((node) => {
        const box = transformedBox(svg, node);
        if (!box) return [];
        const pad = 1;
        const outside = box.x < viewBox.x - pad || box.y < viewBox.y - pad
          || box.x + box.width > viewBox.x + viewBox.width + pad
          || box.y + box.height > viewBox.y + viewBox.height + pad;
        return outside ? [{
          svg: svg.id || svg.className.baseVal || 'svg',
          text: node.textContent.trim(),
          box: [box.x, box.y, box.width, box.height],
          viewBox: [viewBox.x, viewBox.y, viewBox.width, viewBox.height],
        }] : [];
      });
    });

    const textCollisions = [...document.querySelectorAll('svg')].flatMap((svg) => {
      const textNodes = [...svg.querySelectorAll('text')]
        .filter((node) => node.textContent.trim())
        .map((node) => ({ node, rect: node.getBoundingClientRect() }))
        .filter(({ rect }) => rect.width > 1 && rect.height > 1);
      const collisions = [];
      for (let first = 0; first < textNodes.length; first += 1) {
        for (let second = first + 1; second < textNodes.length; second += 1) {
          const a = textNodes[first];
          const b = textNodes[second];
          const overlapWidth = Math.min(a.rect.right, b.rect.right) - Math.max(a.rect.left, b.rect.left);
          const overlapHeight = Math.min(a.rect.bottom, b.rect.bottom) - Math.max(a.rect.top, b.rect.top);
          if (overlapWidth > 1.5 && overlapHeight > 1.5) {
            collisions.push({
              svg: svg.id || svg.className.baseVal || 'svg',
              first: a.node.textContent.trim(),
              second: b.node.textContent.trim(),
            });
          }
        }
      }
      return collisions;
    });

    const siblingOverlaps = [...document.querySelectorAll('.figure-toolbar, .figure-controls, .dataset-switcher, .summary-strip')]
      .flatMap((container) => {
        const children = [...container.children].filter(visible);
        const collisions = [];
        for (let first = 0; first < children.length; first += 1) {
          for (let second = first + 1; second < children.length; second += 1) {
            const a = children[first].getBoundingClientRect();
            const b = children[second].getBoundingClientRect();
            const overlapWidth = Math.min(a.right, b.right) - Math.max(a.left, b.left);
            const overlapHeight = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
            if (overlapWidth > 1 && overlapHeight > 1) {
              collisions.push(`${children[first].textContent.trim()} / ${children[second].textContent.trim()}`);
            }
          }
        }
        return collisions;
      });

    const clippedLabels = [...document.querySelectorAll('.toolbar-button, .mode-chip, .rule-chip, .dataset-chip, .stage-pill strong, .summary-card strong')]
      .filter(visible)
      .filter((element) => element.scrollWidth > element.clientWidth + 1 || element.scrollHeight > element.clientHeight + 1)
      .map((element) => element.textContent.trim());

    const graphSurface = document.querySelector('.graph-surface');
    return {
      horizontalOverflow: root.scrollWidth - root.clientWidth,
      graphOverflow: graphSurface.scrollWidth - graphSurface.clientWidth,
      smallControls: controls.filter((control) => control.width < 44 || control.height < 44),
      svgIssues,
      textCollisions,
      siblingOverlaps,
      clippedLabels,
      duplicateIds: [...document.querySelectorAll('[id]')]
        .map((element) => element.id)
        .filter((id, index, ids) => ids.indexOf(id) !== index),
    };
  });

  assert(result.horizontalOverflow <= 1, `${name}: no page-level horizontal overflow`);
  if (width <= 760) {
    assert(result.graphOverflow > 20, `${name}: graph uses contained horizontal exploration at the narrow breakpoint`);
  } else {
    assert(result.graphOverflow <= 2, `${name}: the complete graph fits without initial clipping`);
  }
  if (width <= 760) assert(result.smallControls.length === 0, `${name}: every visible control is at least 44×44px`);
  assert(result.svgIssues.length === 0, `${name}: all SVG text stays inside its viewBox${result.svgIssues.length ? ` ${JSON.stringify(result.svgIssues)}` : ''}`);
  assert(result.textCollisions.length === 0, `${name}: SVG text labels do not collide${result.textCollisions.length ? ` ${JSON.stringify(result.textCollisions)}` : ''}`);
  assert(result.siblingOverlaps.length === 0, `${name}: toolbar, switcher, and summary siblings do not overlap${result.siblingOverlaps.length ? ` ${JSON.stringify(result.siblingOverlaps)}` : ''}`);
  assert(result.clippedLabels.length === 0, `${name}: critical control and status labels are not clipped${result.clippedLabels.length ? ` ${JSON.stringify(result.clippedLabels)}` : ''}`);
  assert(result.duplicateIds.length === 0, `${name}: DOM IDs are unique`);

  await page.screenshot({ path: `${outDir}/${name}-full.png`, fullPage: true });
  for (const [selector, slug] of [
    ['.hero', 'hero'],
    ['.diagram-board', 'sketch'],
    ['#single-example .interactive-figure', 'graph'],
    ['#modules .interactive-figure', 'modules'],
    ['#batching .interactive-figure', 'batch'],
  ]) {
    const element = await page.$(selector);
    if (element) await element.screenshot({ path: `${outDir}/${name}-${slug}.png` });
  }
}

async function auditAllGraphStages(name, width, height) {
  await load(width, height);
  for (let stage = 0; stage <= 17; stage += 1) {
    await page.evaluate((value) => window.AutogradExplainer.setGraphStage(value), stage);
    await settle();
    const result = await page.evaluate((expectedStage) => {
      const exported = window.AutogradExplainer.exportGraphStage();
      const previewRows = [...document.querySelectorAll('[data-ledger-scope="preview"]')];
      const allRows = [...document.querySelectorAll('[data-ledger-scope="all"]')];
      const previewIds = previewRows.map((row) => row.dataset.nodeId);
      const activeNodes = exported.nodes.filter((node) => node.active);
      const activeIds = activeNodes.map((node) => node.id);
      const surfaceRect = document.querySelector('.graph-surface').getBoundingClientRect();
      const activeVisibility = activeIds.map((id) => {
        const element = document.querySelector(`[data-graph-node="${id}"]`);
        if (!element) return { id, ratio: 0 };
        const rect = element.getBoundingClientRect();
        const visibleWidth = Math.max(0, Math.min(rect.right, surfaceRect.right) - Math.max(rect.left, surfaceRect.left));
        const visibleHeight = Math.max(0, Math.min(rect.bottom, surfaceRect.bottom) - Math.max(rect.top, surfaceRect.top));
        return { id, ratio: (visibleWidth * visibleHeight) / (rect.width * rect.height) };
      });
      const graphSvg = document.getElementById('graphSvg');
      const viewBox = graphSvg.viewBox.baseVal;
      const svgTextOutside = [...graphSvg.querySelectorAll('text')].some((node) => {
        let box;
        try { box = node.getBBox(); } catch { return false; }
        const svgMatrix = graphSvg.getScreenCTM();
        const nodeMatrix = node.getScreenCTM();
        if (!svgMatrix || !nodeMatrix) return false;
        const matrix = svgMatrix.inverse().multiply(nodeMatrix);
        const points = [
          new DOMPoint(box.x, box.y),
          new DOMPoint(box.x + box.width, box.y),
          new DOMPoint(box.x, box.y + box.height),
          new DOMPoint(box.x + box.width, box.y + box.height),
        ].map((point) => point.matrixTransform(matrix));
        const xs = points.map((point) => point.x);
        const ys = points.map((point) => point.y);
        return Math.min(...xs) < viewBox.x - 1 || Math.min(...ys) < viewBox.y - 1
          || Math.max(...xs) > viewBox.x + viewBox.width + 1
          || Math.max(...ys) > viewBox.y + viewBox.height + 1;
      });
      return {
        exportedStage: exported.stage,
        nodeCount: exported.nodes.length,
        stageLabel: document.getElementById('stageIndex').textContent.trim(),
        scrubberValue: Number(document.getElementById('stageScrubber').value),
        previewCount: previewRows.length,
        allCount: allRows.length,
        activeInPreview: activeIds.every((id) => previewIds.includes(id)),
        activeValuesReady: activeNodes.every((node) => node.value !== null),
        activeGradientsReady: exported.direction !== 'Backward' || activeNodes.every((node) => node.grad !== null),
        activeVisibility,
        svgTextOutside,
        expectedStage,
      };
    }, stage);

    const visibleEnough = result.activeVisibility.every(({ ratio }) => ratio >= 0.82);
    assert(
      result.exportedStage === stage
        && result.nodeCount === 11
        && result.stageLabel === `${stage} / 17`
        && result.scrubberValue === stage
        && result.previewCount >= 1
        && result.previewCount <= 5
        && result.allCount === 11
        && result.activeInPreview
        && result.activeValuesReady
        && result.activeGradientsReady
        && visibleEnough
        && !result.svgTextOutside,
      `${name}: graph stage ${stage} keeps API, scrubber, compact ledger, SVG bounds, and active-node visibility in sync${visibleEnough ? '' : ` ${JSON.stringify(result.activeVisibility)}`}`,
    );
  }
}

async function auditInteractions() {
  await load(1024, 768);
  const result = await page.evaluate(() => {
    const stage = () => window.AutogradExplainer.exportGraphStage().stage;
    document.getElementById('resetGraph').click();
    document.getElementById('stepForward').click();
    const forwardStage = stage();
    document.getElementById('resetGraph').click();
    document.getElementById('stepBackward').click();
    const backwardStage = stage();

    const scrubber = document.getElementById('stageScrubber');
    scrubber.value = '8';
    scrubber.dispatchEvent(new Event('input', { bubbles: true }));
    const scrubberStage = stage();

    window.AutogradExplainer.setGraphStage(17);
    document.getElementById('autoPlay').click();
    const autoplayResetStage = stage();
    const autoplayLabel = document.getElementById('autoPlay').textContent.trim();
    document.getElementById('autoPlay').click();

    document.querySelector('button[data-rule-id="log"]').click();
    const logRuleSelected = document.getElementById('ruleTitle').textContent.trim() === 'Log'
      && document.querySelector('button[data-rule-id="log"]').classList.contains('is-active');

    document.querySelector('[data-mode="modules"]').click();
    const moduleModeSelected = document.querySelector('[data-mode="modules"]').classList.contains('is-active')
      && document.getElementById('moduleHeadline').textContent.includes('Framework blocks');

    document.querySelector('[data-dataset="edge"]').click();
    const edgeSelected = document.querySelector('[data-dataset="edge"]').classList.contains('is-active');
    const beforeLoss = window.AutogradExplainer.exportBatchSnapshot().meanLoss;
    document.getElementById('applyBatchStep').click();
    const afterLoss = window.AutogradExplainer.exportBatchSnapshot().meanLoss;

    const paramW = document.getElementById('paramW');
    paramW.value = '2';
    paramW.dispatchEvent(new Event('input', { bubbles: true }));
    const sliderSynced = document.getElementById('paramWValue').textContent.trim() === '2.00';

    document.getElementById('resetBatch').click();
    const resetW = Number(document.getElementById('paramW').value);

    return {
      forwardStage,
      backwardStage,
      scrubberStage,
      autoplayResetStage,
      autoplayLabel,
      logRuleSelected,
      moduleModeSelected,
      edgeSelected,
      beforeLoss,
      afterLoss,
      sliderSynced,
      resetW,
    };
  });

  assert(result.forwardStage === 1, 'interaction: Forward advances setup to stage 1');
  assert(result.backwardStage === 9, 'interaction: Backward from setup jumps to the loss seed');
  assert(result.scrubberStage === 8, 'interaction: scrubber selects the completed forward pass');
  assert(result.autoplayResetStage === 0 && result.autoplayLabel === 'Pause', 'interaction: autoplay restarts a completed timeline and can pause');
  assert(result.logRuleSelected, 'interaction: local-rule selector updates the spotlight');
  assert(result.moduleModeSelected, 'interaction: module selector switches to framework blocks');
  assert(result.edgeSelected, 'interaction: batch dataset selector switches datasets');
  assert(result.afterLoss < result.beforeLoss, 'interaction: one batch gradient step lowers the selected batch loss');
  assert(result.sliderSynced, 'interaction: parameter slider updates its visible value');
  assert(Math.abs(result.resetW - 0.65) < 1e-9, 'interaction: reset restores the selected dataset defaults');
}

async function auditExpandedLedger() {
  await load(390, 844);
  const result = await page.evaluate(() => {
    window.AutogradExplainer.setGraphStage(17);
    const disclosure = document.querySelector('.ledger-disclosure');
    disclosure.open = true;
    window.AutogradExplainer.setGraphStage(16);
    const all = document.querySelector('.ledger-all');
    return {
      remainsOpen: document.querySelector('.ledger-disclosure').open,
      allRows: document.querySelectorAll('[data-ledger-scope="all"]').length,
      previewRows: document.querySelectorAll('[data-ledger-scope="preview"]').length,
      containedScroll: all.scrollHeight <= all.clientHeight + 1 || getComputedStyle(all).overflowY === 'auto',
      pageOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    };
  });
  assert(
    result.remainsOpen
      && result.allRows === 11
      && result.previewRows <= 5
      && result.containedScroll
      && result.pageOverflow <= 1,
    'expanded ledger: all 11 nodes remain accessible across stage changes without widening the page',
  );
}

async function captureStageSeries(name, width, height, stages) {
  await load(width, height);
  for (const stage of stages) {
    await page.evaluate((value) => window.AutogradExplainer.setGraphStage(value), stage);
    await settle();
    const element = await page.$('#single-example .interactive-figure');
    await element.screenshot({ path: `${outDir}/${name}-graph-stage-${String(stage).padStart(2, '0')}.png` });
  }
}

const viewports = [
  ['desktop-1440x900', 1440, 900],
  ['laptop-1024x768', 1024, 768],
  ['narrow-760x900', 760, 900],
  ['mobile-390x844', 390, 844],
];

for (const [name, width, height] of viewports) {
  await auditViewport(name, width, height);
  await auditAllGraphStages(name, width, height);
}

await auditInteractions();
await auditExpandedLedger();
await captureStageSeries('desktop', 1440, 900, [0, 1, 8, 9, 13, 15, 16, 17]);
await captureStageSeries('mobile', 390, 844, [0, 4, 8, 9, 13, 17]);

await load(390, 844);
await page.evaluate(() => {
  window.AutogradExplainer.setGraphStage(17);
  document.querySelector('.ledger-disclosure').open = true;
});
await settle();
await (await page.$('#single-example .interactive-figure')).screenshot({ path: `${outDir}/mobile-graph-stage-17-expanded-ledger.png` });

await page.evaluate(() => document.querySelector('[data-mode="modules"]').click());
await settle();
await (await page.$('#modules .interactive-figure')).screenshot({ path: `${outDir}/mobile-modules-framework.png` });

await page.evaluate(() => {
  document.querySelector('[data-dataset="edge"]').click();
  document.getElementById('applyBatchStep').click();
});
await settle();
await (await page.$('#batching .interactive-figure')).screenshot({ path: `${outDir}/mobile-batch-after-update.png` });

await browser.close();
console.log(JSON.stringify({ failures, errors, checksPassed: checks.length }, null, 2));
if (failures.length || errors.length) process.exitCode = 1;
