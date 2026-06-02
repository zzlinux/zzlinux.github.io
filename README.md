# GitHub Personal Blog

Minimal Astro personal blog for Markdown notes and standalone HTML documents.

## Local Development

Install dependencies:

```sh
npm install
```

Start the local development server:

```sh
npm run dev
```

## Build

Build the static site:

```sh
npm run build
```

Astro generates the static site in `dist/`.

## Add a Markdown Article

Create a Markdown file in `src/content/posts/`, for example
`src/content/posts/my-note.md`.

Use frontmatter with `title`, `date`, `tags`, and `summary`:

```md
---
title: "My Note"
date: "2026-06-02"
tags: ["writing", "blog"]
summary: "A short description of the article."
---

Write the article body here.
```

## Add a Standalone HTML Document

1. Put the HTML file in `public/docs/`, such as `public/docs/report.html`.
2. Add metadata to `src/data/externalPosts.ts`.
3. Link back to `/articles/` from the HTML when practical.

Example metadata:

```ts
{
  title: "Report",
  date: "2026-06-02",
  summary: "A standalone HTML report linked from the article index.",
  tags: ["report"],
  href: "/docs/report.html",
  type: "html",
}
```

## Deploy

Push to `main`. GitHub Actions builds the site and deploys `dist/` to GitHub
Pages.

In the GitHub repository settings, set Pages source to GitHub Actions.

The workflow uses tightened permissions and `actions/configure-pages`, so no
extra local deploy command is needed.
