# GitHub 个人博客

这是一个极简 Astro 个人博客，用来记录 Markdown 笔记和独立 HTML 文档。

## 本地开发

安装依赖：

```sh
npm install
```

启动本地开发服务器：

```sh
npm run dev
```

## 构建

构建静态网站：

```sh
npm run build
```

Astro 会把静态站点生成到 `dist/` 目录。

## 添加 Markdown 文章

在 `src/content/posts/` 中创建一个 Markdown 文件，例如
`src/content/posts/my-note.md`。

文件开头使用 frontmatter 填写 `title`、`date`、`tags` 和 `summary`：

```md
---
title: "我的笔记"
date: "2026-06-02"
tags: ["写作", "博客"]
summary: "文章的一句话摘要。"
---

在这里写文章正文。
```

保存后，文章会自动出现在 `/articles/` 列表页，并且首页会自动展示最新的 5 篇内容。

## 添加独立 HTML 或 PDF 文档

1. 把 HTML 或 PDF 文件放到 `public/docs/`，例如 `public/docs/report.html` 或 `public/docs/report.pdf`。
2. 在 `src/data/externalPosts.ts` 中添加这篇文档的元数据。
3. 如果是 HTML 页面，并且方便修改，可以在页面里加一个返回 `/articles/` 的链接。

元数据示例：

```ts
{
  title: "报告标题",
  date: "2026-06-02",
  summary: "这是一篇会出现在文章列表里的独立 HTML 报告。",
  tags: ["报告"],
  href: "/docs/report.html",
  type: "html",
}
```

保存后，这篇 HTML 文档会出现在 `/articles/` 列表页，点击后跳转到 `href` 指向的 HTML 页面。

PDF 文档的写法类似，只需要把 `href` 指向 PDF 文件，并把 `type` 写成 `"pdf"`：

```ts
{
  title: "PDF 报告标题",
  date: "2026-06-03",
  summary: "这是一篇会出现在文章列表里的 PDF 报告。",
  tags: ["报告"],
  href: "/docs/report.pdf",
  type: "pdf",
}
```

保存后，这篇 PDF 文档也会出现在 `/articles/` 列表页，点击后由浏览器直接打开 PDF。

## 发布新文章

发布 Markdown 文章、HTML 文档或 PDF 文档的流程是一样的：

1. 按上面的方式添加 Markdown 文件，或者添加 HTML/PDF 文件和对应元数据。
2. 本地启动预览：

```sh
npm run dev
```

3. 打开终端显示的本地地址，检查首页、`/articles/` 和文章详情页。
4. 构建检查：

```sh
npm run build
```

5. 提交并推送到 `main` 分支：

```sh
git add .
git commit -m "docs: add new article"
git push origin main
```

推送后 GitHub Actions 会自动发布，不需要手动上传 `dist/`。

## 修改个人主页内容

首页内容主要在 `src/pages/index.astro` 中修改：

- 页面标题：修改 `<BaseLayout title="Your Name | Personal Blog">`。
- 首页大标题：修改 `<h1 class="page-title">Your Name</h1>`。
- 个人简介：修改 `<p class="lede">...</p>`。
- 关注方向：修改 `Focus` 区域里的标签列表。
- 最新文章：由 `getAllArticles()` 自动生成，通常不需要手动改。

顶部导航里的站点名称在 `src/layouts/BaseLayout.astro` 中修改：

```astro
<a class="site-brand" href="/">Your Name</a>
```

关于页内容在 `src/pages/about.astro` 中修改，链接页内容在 `src/pages/links.astro` 中修改。

修改个人信息后，建议同样执行：

```sh
npm run build
```

确认构建通过后再提交推送。

## 部署

推送到 `main` 分支后，GitHub Actions 会自动构建站点，并把 `dist/` 部署到 GitHub Pages。

在 GitHub 仓库设置里，把 Pages 的发布来源设置为 GitHub Actions。

当前 workflow 已经配置了收窄后的权限和 `actions/configure-pages`，本地不需要额外执行部署命令。
