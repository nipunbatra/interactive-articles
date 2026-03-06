const fs = require('fs');
const path = require('path');

const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'src');
const articlesDir = path.join(srcDir, 'articles');
const sharedDir = path.join(srcDir, 'shared');
const docsDir = path.join(rootDir, 'docs');
const siteConfigPath = path.join(rootDir, 'site.config.json');

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

function renderArticleCards(articles) {
  return articles.map((article) => {
    const tags = (article.tags || [])
      .map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`)
      .join('');

    return `
      <article class="article-card">
        <div class="article-card__body">
          <div class="article-card__header">
            <div>
              <p class="card__meta">${escapeHtml(article.tagline || '')}</p>
              <h2 class="card__title">${escapeHtml(article.title)}</h2>
            </div>
            <span class="pill">${escapeHtml(article.readingTime || '')}</span>
          </div>
          <p class="card__copy">${escapeHtml(article.summary || '')}</p>
          <div class="pill-row">${tags}</div>
          <div class="article-card__footer">
            <span class="pill">${escapeHtml(article.status || 'published')}</span>
            <a class="article-link" href="${escapeHtml(article.url)}">Open article</a>
          </div>
        </div>
      </article>
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
  </head>
  <body>
    <main class="site-shell">
      <section class="hero">
        <p class="eyebrow">Interactive library</p>
        <h1 class="hero__title">${escapeHtml(siteConfig.title)}</h1>
        <p class="hero__copy">${escapeHtml(siteConfig.description)}</p>

        <div class="hero__stats">
          <article class="stat-card">
            <p class="eyebrow">Articles</p>
            <p class="stat-card__value">${articles.length}</p>
            <p class="hero__copy">Small, self-contained explainers with live controls.</p>
          </article>
          <article class="stat-card">
            <p class="eyebrow">Style</p>
            <p class="stat-card__value">Hands-on</p>
            <p class="hero__copy">Every article should let the reader drag, tweak, and see the math react.</p>
          </article>
          <article class="stat-card">
            <p class="eyebrow">Deploy</p>
            <p class="stat-card__value">GitHub Pages</p>
            <p class="hero__copy">The built site lives in <code>docs/</code> so Pages can serve it directly.</p>
          </article>
        </div>
      </section>

      <section class="section">
        <div>
          <p class="eyebrow">Published</p>
          <h2 class="section__title">Current articles</h2>
          <p class="section__copy">${escapeHtml(siteConfig.tagline)}</p>
        </div>
        <div class="card-grid">
          ${renderArticleCards(articles)}
        </div>
      </section>

      <section class="section">
        <div>
          <p class="eyebrow">Workflow</p>
          <h2 class="section__title">How this project is organized</h2>
        </div>
        <div class="guide-grid">
          <article class="guide-card">
            <p class="guide__label">Source</p>
            <p class="guide__copy">Put each article in <code>src/articles/&lt;slug&gt;/</code> with its own <code>index.html</code>, JS, CSS, and <code>meta.json</code>.</p>
          </article>
          <article class="guide-card">
            <p class="guide__label">Build</p>
            <p class="guide__copy">Run <code>node scripts/build.js</code>. It copies articles into <code>docs/articles/</code> and regenerates the site homepage.</p>
          </article>
          <article class="guide-card">
            <p class="guide__label">Deploy</p>
            <p class="guide__copy">Push the repo to GitHub and point Pages at the <code>docs/</code> folder on your main branch.</p>
          </article>
        </div>
      </section>

      <footer class="site-footer">
        <p>${escapeHtml(siteConfig.title)}. Built as a static site for easy hosting and low maintenance.</p>
      </footer>
    </main>
  </body>
</html>`;
}

function build() {
  const siteConfig = readJson(siteConfigPath);
  const articles = loadArticles();

  cleanDir(docsDir);
  ensureDir(path.join(docsDir, 'articles'));
  ensureDir(path.join(docsDir, 'assets'));

  copyDir(sharedDir, path.join(docsDir, 'assets'));

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
