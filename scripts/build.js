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

function uniqueTags(articles) {
  return [...new Set(articles.flatMap((article) => article.tags || []))];
}

function renderTagPills(tags, extraClass = '') {
  return tags
    .map((tag) => `<span class="pill ${extraClass}">${escapeHtml(tag)}</span>`)
    .join('');
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
      <div class="spotlight-card__preview" aria-hidden="true">
        <span class="preview-pill preview-pill--coral">1D bell</span>
        <span class="preview-pill preview-pill--teal">2D cloud</span>
        <span class="preview-pill preview-pill--gold">Slices</span>
      </div>
      <div class="spotlight-card__body">
        <p class="card__meta">Featured explainer</p>
        <h2 class="spotlight-card__title">${escapeHtml(article.title)}</h2>
        <p class="card__copy">${escapeHtml(article.summary)}</p>
        <div class="pill-row pill-row--soft">
          <span class="pill">${escapeHtml(article.collection || 'Explainer')}</span>
          <span class="pill">${escapeHtml(article.difficulty || 'Open level')}</span>
          <span class="pill">${escapeHtml(article.readingTime || '')}</span>
        </div>
        ${renderHighlights(article.highlights)}
        <div class="spotlight-card__footer">
          <a class="article-link" href="${escapeHtml(article.url)}">Read the explainer</a>
        </div>
      </div>
    </aside>
  `;
}

function renderArticleCards(articles) {
  return articles.map((article, index) => `
      <article class="article-card">
        <div class="article-card__preview" aria-hidden="true">
          <span>${String(index + 1).padStart(2, '0')}</span>
          <p>${escapeHtml(article.tagline || article.collection || '')}</p>
        </div>
        <div class="article-card__body">
          <div class="article-card__header">
            <div>
              <p class="card__meta">${escapeHtml(article.collection || 'Explainer')}</p>
              <h2 class="card__title">${escapeHtml(article.title)}</h2>
            </div>
            <span class="pill">${escapeHtml(article.readingTime || '')}</span>
          </div>
          <p class="card__copy">${escapeHtml(article.summary || '')}</p>
          ${renderHighlights(article.highlights, true)}
          <div class="pill-row">${renderTagPills(article.tags || [])}</div>
          <div class="article-card__footer">
            <span class="pill">${escapeHtml(article.status || 'published')}</span>
            <a class="article-link" href="${escapeHtml(article.url)}">Open article</a>
          </div>
        </div>
      </article>
    `).join('\n');
}

function renderHomePage(siteConfig, articles) {
  const featuredArticle = articles[0];
  const tags = uniqueTags(articles);

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
      <div class="site-header__meta">
        <span class="tiny-pill">${articles.length} published</span>
        <span class="tiny-pill">Narrative + math + interaction</span>
      </div>
    </header>

    <main class="site-shell">
      <section class="hero">
        <div class="hero__copy">
          <p class="eyebrow">Interactive explainers</p>
          <h1 class="hero__title">Articles that let you drag the idea until it clicks.</h1>
          <p class="hero__copy">${escapeHtml(siteConfig.description)}</p>
          <p class="hero__copy">Each piece is designed as a guided experience: start with intuition, adjust live controls, then land on the math once the shape already makes sense.</p>
          <div class="hero__actions">
            <a class="article-link" href="${escapeHtml(featuredArticle.url)}">Start with the featured explainer</a>
            <a class="ghost-link" href="#library">Browse the library</a>
          </div>
          <div class="pill-row pill-row--hero">${renderTagPills(tags)}</div>
          <div class="hero__stats">
            <article class="stat-card">
              <p class="eyebrow">Format</p>
              <p class="stat-card__value">Hands-on</p>
              <p class="hero__copy">Readers should move sliders, drag points, and watch the story change.</p>
            </article>
            <article class="stat-card">
              <p class="eyebrow">Arc</p>
              <p class="stat-card__value">Feel then formalize</p>
              <p class="hero__copy">The goal is not a glossary. It is a sequence that makes the math feel inevitable.</p>
            </article>
            <article class="stat-card">
              <p class="eyebrow">Deploy</p>
              <p class="stat-card__value">Static + fast</p>
              <p class="hero__copy">Everything ships as a lightweight static site and deploys cleanly with GitHub Pages.</p>
            </article>
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

      <section class="section section--tracks">
        <div class="section__head">
          <div>
            <p class="eyebrow">What makes these different</p>
            <h2 class="section__title">Built like explainers, not notes</h2>
          </div>
          <p class="section__copy">The reference bar is an explainer that has a point of view, a rhythm, and a reason to exist interactively.</p>
        </div>
        <div class="guide-grid">
          <article class="guide-card">
            <p class="guide__label">Narrative spine</p>
            <p class="guide__copy">Every explainer should have a visible path: what you know at the start, what changes in the middle, and what you can now answer at the end.</p>
          </article>
          <article class="guide-card">
            <p class="guide__label">Visual payoffs</p>
            <p class="guide__copy">Interactions are not decoration. They should expose the one thing that is hard to understand from static text alone.</p>
          </article>
          <article class="guide-card">
            <p class="guide__label">Library quality</p>
            <p class="guide__copy">The landing page should feel curated: clear spotlight, clear catalog, and a strong sense of what kind of learning experience lives here.</p>
          </article>
        </div>
      </section>

      <footer class="site-footer">
        <p>${escapeHtml(siteConfig.title)}. Static, fast, and organized for adding more interactive explainers without reworking the whole site.</p>
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
