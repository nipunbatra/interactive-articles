# Interactive Articles

A small static site for intuition-first interactive articles. The current article is:

- `Seeing the Multivariate Normal`

## Project structure

```text
src/
  articles/
    multivariate-normal/
      index.html
      app.js
      styles.css
      meta.json
  shared/
    site.css
scripts/
  build.js
docs/
  ...generated site for GitHub Pages
site.config.json
```

## Build

Install dependencies and generate the site into `docs/`:

```bash
npm ci
npm run build
```

## Preview locally

After building:

```bash
python3 -m http.server 4173 -d docs
```

Then open `http://localhost:4173`.

## Add a new article

1. Create `src/articles/<slug>/`
2. Add `index.html`, article JS/CSS files, and `meta.json`
3. Run `npm run build`

The homepage in `docs/index.html` is generated automatically from the article metadata.

## Deploy to GitHub Pages

This project now includes a GitHub Actions workflow:

- [pages.yml](/Users/nipun/git/interactive/.github/workflows/pages.yml)

Recommended setup:

1. Push the repo to GitHub.
2. In `Settings -> Pages`, set `Source` to `GitHub Actions`.
3. Push to `main`.

The workflow rebuilds the site and deploys the generated `docs/` output automatically.
