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
      url: `articles/${slug}/`,
    };
  });

  return articles.sort((a, b) => (a.order ?? 999) - (b.order ?? 999));
}

function renderHighlights(highlights, compact = false) {
  return `
    <ul class="feature-list${compact ? ' feature-list--compact' : ''}">
      ${(highlights || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('')}
    </ul>
  `;
}

function renderFeaturedArticle(article) {
  return `
    <aside class="spotlight-card">
      <div class="spotlight-card__body">
        <p class="card__meta">Start here</p>
        <h2 class="spotlight-card__title">${escapeHtml(article.title)}</h2>
        <p class="card__copy">${escapeHtml(article.summary)}</p>
        <p class="spotlight-card__facts">${escapeHtml(article.collection || 'Explainer')} · ${escapeHtml(article.readingTime || '')}</p>
        ${renderHighlights(article.highlights)}
        <div class="spotlight-card__footer">
          <a class="article-link" href="${escapeHtml(article.url)}">Read the explainer</a>
        </div>
      </div>
    </aside>
  `;
}

function renderArticleCards(articles) {
  return articles.map((article) => `
      <article class="article-card">
        <div class="article-card__body">
          <p class="card__meta">${escapeHtml(article.collection || 'Explainer')} · ${escapeHtml(article.readingTime || '')}</p>
          <h2 class="card__title">${escapeHtml(article.title)}</h2>
          <p class="card__copy">${escapeHtml(article.summary || '')}</p>
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
        <p class="eyebrow">Explainer library</p>
        <a class="brand-mark" href="./">${escapeHtml(siteConfig.title)}</a>
      </div>
    </header>

    <main class="site-shell">
      <section class="hero">
        <div class="hero__copy">
          <p class="eyebrow">Interactive explainers</p>
          <h1 class="hero__title">Ideas first. Interaction second. Math when it helps.</h1>
          <p class="hero__copy">${escapeHtml(siteConfig.description)}</p>
          <p class="hero__copy">Each page is meant to read like a short visual essay: an example, a picture, a live figure, and only then the formal statement.</p>
          <div class="hero__actions">
            <a class="article-link" href="${escapeHtml(featuredArticle.url)}">Start with the featured explainer</a>
            <a class="ghost-link" href="#library">Browse the library</a>
          </div>
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
        <p>${escapeHtml(siteConfig.title)}. Quiet pages, simple builds, and room to add more explainers over time.</p>
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
  ensureDir(path.join(docsDir, 'articles'));
  ensureDir(path.join(docsDir, 'assets'));
  ensureDir(path.join(docsDir, 'vendor'));

  copyDir(sharedDir, path.join(docsDir, 'assets'));
  copyDir(katexDistDir, path.join(docsDir, 'vendor', 'katex'));

  for (const article of articles) {
    copyDir(
      path.join(articlesDir, article.slug),
      path.join(docsDir, 'articles', article.slug)
    );
  }

  fs.writeFileSync(path.join(docsDir, '.nojekyll'), '');
  fs.writeFileSync(path.join(docsDir, 'index.html'), renderHomePage(siteConfig, articles));
}

build();
