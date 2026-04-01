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
  return articles.map((article) => `
      <article class="simple-article-row">
        <div class="simple-article-header">
          <span class="simple-article-meta">${escapeHtml(article.collection || 'Explainer')} &middot; ${escapeHtml(article.readingTime || '')}</span>
          ${article.difficulty ? `<span class="simple-article-difficulty">${escapeHtml(article.difficulty)}</span>` : ''}
        </div>
        <h2 class="simple-article-title"><a href="${escapeHtml(article.url)}">${escapeHtml(article.title)}</a></h2>
        <p class="simple-article-summary">${escapeHtml(article.summary || '')}</p>
        <div class="simple-article-footer">
          <a class="simple-article-link" href="${escapeHtml(article.url)}">Read interactive explainer &rarr;</a>
        </div>
      </article>
    `).join('\n');
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
        max-width: 800px;
        margin: 0 auto;
        padding: 4rem 1.5rem;
      }
      .simple-header {
        margin-bottom: 4rem;
        text-align: center;
      }
      .simple-header h1 {
        font-family: var(--font-serif);
        font-size: clamp(2.5rem, 5vw, 3.5rem);
        margin: 0 0 1rem;
        color: var(--ink);
        letter-spacing: -0.02em;
        line-height: 1.1;
      }
      .simple-header p {
        font-size: 1.15rem;
        color: var(--muted);
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
      }
      .simple-article-list {
        display: grid;
        gap: 2rem;
      }
      .simple-article-row {
        background: var(--panel);
        padding: 2.5rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border);
        box-shadow: 0 8px 30px rgba(31, 38, 48, 0.04);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
      }
      .simple-article-row:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(31, 38, 48, 0.08);
      }
      .simple-article-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
      }
      .simple-article-meta {
        font-size: 0.85rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }
      .simple-article-difficulty {
        font-size: 0.8rem;
        background: var(--panel-soft);
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        color: var(--muted);
        border: 1px solid var(--border);
      }
      .simple-article-title {
        margin: 0 0 0.8rem;
        font-size: 1.8rem;
        font-family: var(--font-serif);
        line-height: 1.2;
      }
      .simple-article-title a {
        text-decoration: none;
        color: var(--ink);
      }
      .simple-article-title a:hover {
        color: var(--accent);
      }
      .simple-article-summary {
        color: var(--muted);
        font-size: 1.05rem;
        margin: 0 0 1.5rem;
        line-height: 1.6;
      }
      .simple-article-footer {
        margin-top: 1.5rem;
      }
      .simple-article-link {
        display: inline-block;
        text-decoration: none;
        color: var(--accent);
        font-weight: 600;
        font-size: 1rem;
        transition: color 0.2s;
      }
      .simple-article-link:hover {
        color: var(--accent-2);
      }
      .simple-footer {
        margin-top: 5rem;
        text-align: center;
        color: var(--muted);
        font-size: 0.95rem;
        border-top: 1px solid var(--border);
        padding-top: 2rem;
      }
    </style>
  </head>
  <body>
    <main class="simple-shell">
      <header class="simple-header">
        <h1>${escapeHtml(siteConfig.title)}</h1>
        <p>${escapeHtml(siteConfig.description)}</p>
      </header>

      <div class="simple-article-list">
        ${renderArticleList(articles)}
      </div>
      
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
  }

  fs.writeFileSync(path.join(docsDir, '.nojekyll'), '');
  fs.writeFileSync(path.join(docsDir, 'index.html'), renderHomePage(siteConfig, articles));
}

build();
