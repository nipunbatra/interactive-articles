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

function renderMetrics(articles) {
  const collections = new Set(articles.map((article) => article.collection).filter(Boolean));
  const readingTimes = articles.map((article) => article.readingTime).filter(Boolean);

  return `
    <div class="hero__stats">
      <article class="stat-card">
        <p class="guide__label">Now live</p>
        <p class="stat-card__value">${articles.length}</p>
        <p class="guide__copy">Interactive articles published and linked from this front page.</p>
      </article>
      <article class="stat-card">
        <p class="guide__label">Coverage</p>
        <p class="stat-card__value">${collections.size}</p>
        <p class="guide__copy">Collections spanning ${escapeHtml(Array.from(collections).join(', '))}.</p>
      </article>
      <article class="stat-card">
        <p class="guide__label">Reading shape</p>
        <p class="stat-card__value">${escapeHtml(readingTimes[0] || 'Short')}</p>
        <p class="guide__copy">Essay-length explainers designed to be manipulated, not skimmed as static notes.</p>
      </article>
    </div>
  `;
}

function renderFeaturedArticle(article) {
  return `
    <aside class="spotlight-card">
      <div class="spotlight-card__body">
        <div class="spotlight-card__header">
          <p class="card__meta">Featured release</p>
          <span class="preview-pill">${escapeHtml(article.collection || 'Explainer')}</span>
        </div>
        <h2 class="spotlight-card__title">${escapeHtml(article.title)}</h2>
        <p class="card__copy">${escapeHtml(article.summary)}</p>
        <p class="spotlight-card__facts">${escapeHtml(article.readingTime || '')} · ${escapeHtml(article.difficulty || 'All levels')}</p>
        ${renderTagPills(article.tags)}
        ${renderHighlights(article.highlights, false, 4)}
        <div class="spotlight-card__footer">
          <a class="article-link" href="${escapeHtml(article.url)}">Read the featured article</a>
          <span class="ghost-note">Scroll essay, live figures, exportable state</span>
        </div>
      </div>
    </aside>
  `;
}

function renderArticleCards(articles) {
  return articles.map((article) => `
      <article class="article-card">
        <div class="article-card__body">
          <div class="article-card__header">
            <p class="card__meta">${escapeHtml(article.collection || 'Explainer')} · ${escapeHtml(article.readingTime || '')} · ${escapeHtml(article.difficulty || 'General')}</p>
            <span class="tiny-pill">${escapeHtml(article.status || 'Published')}</span>
          </div>
          <h2 class="card__title">${escapeHtml(article.title)}</h2>
          <p class="card__copy">${escapeHtml(article.summary || '')}</p>
          ${renderTagPills(article.tags)}
          ${renderHighlights(article.highlights, true, 2)}
          <div class="article-card__footer">
            <a class="article-link" href="${escapeHtml(article.url)}">Open article</a>
          </div>
        </div>
      </article>
    `).join('\n');
}

function renderHomePage(siteConfig, articles) {
  const featuredArticle = articles[0];

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${escapeHtml(siteConfig.title)}</title>
    <meta name="description" content="${escapeHtml(siteConfig.description)}" />
    <link rel="stylesheet" href="assets/site.css" />
  </head>
  <body>
    <header class="site-header">
      <div class="site-header__brand">
        <p class="eyebrow">Interactive explainers</p>
        <a class="brand-mark" href="./">${escapeHtml(siteConfig.title)}</a>
      </div>
      <div class="site-header__meta">
        <span class="pill">${articles.length} live article${articles.length === 1 ? '' : 's'}</span>
        <span class="pill">${escapeHtml(siteConfig.tagline)}</span>
      </div>
    </header>

    <main class="site-shell">
      <section class="hero">
        <div class="hero__copy-wrap">
          <div class="hero__copy">
            <p class="eyebrow">Curated library</p>
            <h1 class="hero__title">Mechanisms, not slogans.</h1>
            <p class="hero__copy">${escapeHtml(siteConfig.description)}</p>
            <p class="hero__copy">The front page is intentionally small. Each article is meant to feel like a product-quality lab note: editorial pacing, deliberate visuals, and interactivity that exposes the machinery instead of decorating it.</p>
            <div class="hero__actions">
              <a class="article-link" href="${escapeHtml(featuredArticle.url)}">Start with ${escapeHtml(featuredArticle.title)}</a>
              <a class="ghost-link" href="#library">Browse the library</a>
            </div>
          </div>
          ${renderMetrics(articles)}
        </div>
        ${renderFeaturedArticle(featuredArticle)}
      </section>

      <section class="section" id="library">
        <div class="section__head">
          <div>
            <p class="eyebrow">Published</p>
            <h2 class="section__title">The library</h2>
          </div>
          <p class="section__copy">${escapeHtml(siteConfig.tagline)}</p>
        </div>
        <div class="card-grid">
          ${renderArticleCards(articles)}
        </div>
      </section>

      <footer class="site-footer">
        <p>${escapeHtml(siteConfig.title)} is designed as a small, high-signal collection. Additions should feel like flagship explainers, not filler posts.</p>
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
