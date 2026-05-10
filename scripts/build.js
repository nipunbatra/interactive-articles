const fs = require('fs');
const path = require('path');

const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'src');
const articlesDir = path.join(srcDir, 'articles');
const sharedDir = path.join(srcDir, 'shared');
const docsDir = path.join(rootDir, 'docs');
const siteConfigPath = path.join(rootDir, 'site.config.json');
const katexDistDir = path.join(rootDir, 'node_modules', 'katex', 'dist');

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function cleanDir(dirPath) {
  fs.rmSync(dirPath, { recursive: true, force: true });
  ensureDir(dirPath);
}

function copyDir(sourceDir, targetDir) {
  ensureDir(targetDir);
  for (const entry of fs.readdirSync(sourceDir, { withFileTypes: true })) {
    const sourcePath = path.join(sourceDir, entry.name);
    const targetPath = path.join(targetDir, entry.name);
    if (entry.isDirectory()) {
      copyDir(sourcePath, targetPath);
    } else {
      fs.copyFileSync(sourcePath, targetPath);
    }
  }
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function loadArticles() {
  const slugs = fs.readdirSync(articlesDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);

  const articles = slugs.map((slug) => {
    const metaPath = path.join(articlesDir, slug, 'meta.json');
    const indexPath = path.join(articlesDir, slug, 'index.html');
    if (!fs.existsSync(metaPath)) {
      throw new Error(`Missing meta.json for article "${slug}"`);
    }
    if (!fs.existsSync(indexPath)) {
      throw new Error(`Missing index.html for article "${slug}"`);
    }
    const meta = readJson(metaPath);
    return {
      ...meta,
      slug,
      url: `${slug}/`,
    };
  });

  return articles.sort((a, b) => (a.order ?? 999) - (b.order ?? 999));
}

function renderHighlights(highlights, compact = false, limit = highlights?.length ?? 0) {
  const items = (highlights || []).slice(0, limit);
  return `
    <ul class="feature-list${compact ? ' feature-list--compact' : ''}">
      ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}
    </ul>
  `;
}

function renderTagPills(tags = []) {
  if (!tags.length) {
    return '';
  }

  return `
    <div class="pill-row">
      ${tags.slice(0, 3).map((tag) => `<span class="tiny-pill">${escapeHtml(tag)}</span>`).join('')}
    </div>
  `;
}

function renderArticleList(articles) {
  // Group by collection; within each collection, sort by date desc.
  const groups = new Map();
  articles.forEach((a) => {
    const col = a.collection || 'Other';
    if (!groups.has(col)) groups.set(col, []);
    groups.get(col).push(a);
  });
  // Order collections: pinned order first, then alphabetical
  const pinned = ['Deep learning', 'Computer Vision', 'Multimodal', 'Probability & inference', 'Time series'];
  const colNames = Array.from(groups.keys()).sort((a, b) => {
    const ia = pinned.indexOf(a), ib = pinned.indexOf(b);
    if (ia >= 0 && ib >= 0) return ia - ib;
    if (ia >= 0) return -1;
    if (ib >= 0) return 1;
    return a.localeCompare(b);
  });

  function rowHtml(article) {
    const dateText = article.date
      ? new Date(article.date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
      : '';
    return `
      <li class="ix-row">
        <a class="ix-row-link" href="${escapeHtml(article.url)}">
          <span class="ix-row-title">${escapeHtml(article.title)}</span>
          <span class="ix-row-tag">${escapeHtml(article.tagline || article.summary || '').slice(0, 110)}</span>
          <span class="ix-row-meta">
            ${article.readingTime ? `<span>${escapeHtml(article.readingTime)}</span>` : ''}
            ${dateText ? `<span class="ix-row-date">${dateText}</span>` : ''}
          </span>
        </a>
      </li>
    `;
  }

  return colNames.map((col) => {
    const items = groups.get(col).slice().sort((a, b) => {
      if (a.date && b.date) return new Date(b.date) - new Date(a.date);
      return (a.order || 99) - (b.order || 99);
    });
    return `
      <section class="ix-collection">
        <h2 class="ix-collection-title">${escapeHtml(col)} <span class="ix-collection-count">${items.length}</span></h2>
        <ul class="ix-list">
          ${items.map(rowHtml).join('')}
        </ul>
      </section>
    `;
  }).join('\n');
}

function renderHomePage(siteConfig, articles) {
  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${escapeHtml(siteConfig.title)}</title>
    <meta name="description" content="${escapeHtml(siteConfig.description)}" />
    <link rel="stylesheet" href="assets/site.css" />
    <style>
      .simple-shell {
        max-width: 920px;
        margin: 0 auto;
        padding: 3rem 1.5rem 5rem;
      }
      .simple-header {
        margin-bottom: 3rem;
        text-align: center;
      }
      .simple-header h1 {
        font-family: var(--font-serif);
        font-size: clamp(2.2rem, 4.5vw, 3rem);
        margin: 0 0 0.6rem;
        color: var(--ink);
        letter-spacing: -0.02em;
        line-height: 1.1;
      }
      .simple-header p {
        font-size: 1.05rem;
        color: var(--muted);
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.55;
      }
      .ix-collection {
        margin: 2.4rem 0;
      }
      .ix-collection-title {
        font-family: var(--font-serif);
        font-size: 1.05rem;
        margin: 0 0 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: var(--ink);
        display: flex;
        align-items: center;
        gap: 0.6rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.45rem;
      }
      .ix-collection-count {
        font-family: var(--font-mono, 'IBM Plex Mono', monospace);
        font-size: 0.78rem;
        color: var(--muted);
        background: var(--panel-soft);
        border: 1px solid var(--border);
        padding: 0.05rem 0.5rem;
        border-radius: 999px;
        font-weight: 500;
      }
      .ix-list {
        list-style: none;
        margin: 0;
        padding: 0;
        display: grid;
        gap: 0;
      }
      .ix-row {
        margin: 0;
      }
      .ix-row-link {
        display: grid;
        grid-template-columns: 1.4fr 2.2fr auto;
        gap: 1rem;
        align-items: baseline;
        padding: 0.7rem 0.6rem;
        border-radius: 6px;
        text-decoration: none;
        color: var(--ink);
        transition: background 120ms ease;
        border-bottom: 1px solid rgba(0,0,0,0.04);
      }
      .ix-row-link:hover {
        background: rgba(44, 111, 183, 0.05);
      }
      .ix-row-title {
        font-family: var(--font-serif);
        font-weight: 600;
        font-size: 1rem;
        color: var(--ink);
      }
      .ix-row-tag {
        font-size: 0.92rem;
        color: var(--muted);
        line-height: 1.4;
        font-family: var(--font-serif);
      }
      .ix-row-meta {
        display: flex;
        gap: 0.7rem;
        font-family: var(--font-mono, 'IBM Plex Mono', monospace);
        font-size: 0.74rem;
        color: var(--muted);
        white-space: nowrap;
        text-align: right;
        justify-content: flex-end;
      }
      .ix-row-date { opacity: 0.7; }
      @media (max-width: 720px) {
        .ix-row-link {
          grid-template-columns: 1fr;
          gap: 0.25rem;
        }
        .ix-row-meta { justify-content: flex-start; }
      }
      .simple-footer {
        margin-top: 4rem;
        text-align: center;
        color: var(--muted);
        font-size: 0.9rem;
        border-top: 1px solid var(--border);
        padding-top: 1.5rem;
      }
    </style>
  </head>
  <body>
    <main class="simple-shell">
      <header class="simple-header">
        <h1>${escapeHtml(siteConfig.title)}</h1>
        <p>${escapeHtml(siteConfig.description)}</p>
      </header>

      ${renderArticleList(articles)}

      <footer class="simple-footer">
        <p>Interactive explainers designed to be manipulated, not skimmed as static notes.</p>
      </footer>
    </main>
  </body>
</html>`;
}

function build() {
  const siteConfig = readJson(siteConfigPath);
  const articles = loadArticles();

  if (!fs.existsSync(katexDistDir)) {
    throw new Error('KaTeX assets not found. Run "npm install" before building the site.');
  }

  cleanDir(docsDir);
  ensureDir(path.join(docsDir, 'assets'));
  ensureDir(path.join(docsDir, 'vendor'));

  copyDir(sharedDir, path.join(docsDir, 'assets'));
  copyDir(katexDistDir, path.join(docsDir, 'vendor', 'katex'));

  for (const article of articles) {
    copyDir(
      path.join(articlesDir, article.slug),
      path.join(docsDir, article.slug)
    );
    injectKaTeXAutoRender(path.join(docsDir, article.slug, 'index.html'));
  }

  fs.writeFileSync(path.join(docsDir, '.nojekyll'), '');
  fs.writeFileSync(path.join(docsDir, 'index.html'), renderHomePage(siteConfig, articles));
}

// Add KaTeX auto-render so $...$ / $$...$$ inside paragraphs renders.
// Idempotent: skips if marker comment is already present.
function injectKaTeXAutoRender(htmlPath) {
  if (!fs.existsSync(htmlPath)) return;
  let html = fs.readFileSync(htmlPath, 'utf8');
  if (html.includes('<!-- katex-auto-render -->')) return;
  const insertion = `
    <!-- katex-auto-render -->
    <script defer src="../vendor/katex/contrib/auto-render.min.js"></script>
    <script>
      document.addEventListener('DOMContentLoaded', function () {
        var go = function () {
          if (!window.renderMathInElement) { setTimeout(go, 50); return; }
          renderMathInElement(document.body, {
            delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '\\\\[', right: '\\\\]', display: true },
              { left: '$', right: '$', display: false },
              { left: '\\\\(', right: '\\\\)', display: false }
            ],
            throwOnError: false,
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option']
          });
        };
        go();
      });
    </script>
  </body>`;
  if (html.includes('</body>')) {
    // Function form bypasses `$$` etc. special handling in replacement strings.
    html = html.replace('</body>', () => insertion);
    fs.writeFileSync(htmlPath, html);
  }
}

build();
