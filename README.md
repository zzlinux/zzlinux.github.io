# zzlinux.github.io

这是一个基于 Astro 的 GitHub 个人博客，用来发布普通文章、论文解读、独立 HTML 文档和 PDF 文档。

核心思路是：日常只维护内容文件和 frontmatter，首页、文章表格、论文表格、分类目录、详情页 TOC 会自动生成。

## 本地开发

安装依赖：

```sh
npm install
```

启动本地开发服务器：

```sh
npm run dev
```

构建静态网站：

```sh
npm run build
```

构建产物会生成到 `dist/`。

## 内容结构

Markdown 内容放在 `src/content/posts/` 下，可以直接平铺，也可以按自己的习惯创建多级文件夹，例如：

```text
src/content/posts/
  rl/
    ppo.md
    imgs/
      ppo-clip.png
  papers/
    vision/
      diffusion-policy.md
```

系统会递归扫描 `src/content/posts/`，真正决定文章展示位置的是 frontmatter，不是文件夹名。

## 发布普通文章

普通文章的 frontmatter 示例：

```md
---
title: "通俗易懂的 PPO 算法理解"
date: "2026-06-03"
summary: "沿着一条自然的逻辑链，从最原始的强化学习目标一步步推到 PPO。"
category: "强化学习"
subcategory: "PPO"
tags: ["强化学习", "PPO", "RLHF"]
type: "article"
pinned: true
pinnedRank: 1
---

# 正文标题

这里写正文。
```

字段说明：

- `title`：文章标题，表格里的标题列会自动带链接。
- `date`：站内发布时间，用于首页 Latest 和表格默认排序。
- `summary`：一句话摘要。
- `category`：一级分类，会显示在 Articles 左侧目录。
- `subcategory`：二级分类，表格默认按它分组。
- `tags`：标签数组。
- `type: "article"`：普通文章。
- `pinned`：是否置顶，可省略，默认 `false`。
- `pinnedRank`：置顶排序，可省略；有 rank 时按 rank 排，没有 rank 时按发布时间倒序排。

首页规则：

- `Latest` 默认展示最新 5 篇内容。
- 点击“再看 5 篇”会继续展开 5 篇。
- 置顶内容如果已经出现在最新 5 篇里，不会重复出现在 Pinned 区域；等后续新文章把它挤出最新 5 篇后，才会显示到 Pinned。

## 发布论文解读

论文解读也是 Markdown，只是 `type` 写成 `"paper"`，并增加 `paper` 字段：

```md
---
title: "Paper Title"
date: "2026-06-05"
summary: "这里就是论文的一句话总结。"
category: "多模态大模型"
subcategory: "Video Understanding"
tags: ["paper", "multimodal"]
type: "paper"
paper:
  authors: ["Author A", "Author B"]
  institutions: ["Example Lab"]
  published: "2026"
  venue: "CVPR"
---

# Paper Title

这里写论文解读正文。
```

论文会自动出现在 `/papers/`，表格中会展示标题、meta info、发表时间、会议/期刊、标签和 summary。论文不需要额外写 `paper_name`，`title` 就是论文名；也不需要额外写一句话总结，`summary` 就是该信息。

## 添加 HTML 或 PDF 文档

HTML/PDF 文件放到 `public/docs/`，再在 `src/data/externalPosts.ts` 中登记元数据：

```ts
{
  title: "PDF 报告标题",
  date: "2026-06-03",
  summary: "这是一篇会出现在 Articles 表格里的 PDF 报告。",
  tags: ["报告"],
  href: "/docs/report.pdf",
  type: "pdf",
  category: "模型评测",
  subcategory: "报告",
}
```

`type` 可以是 `"html"` 或 `"pdf"`。这类文档会出现在 `/articles/`，点击标题直接打开对应文件。

## 图片引用

推荐两种方式。

文章局部素材：把图片和 Markdown 放在同一个内容目录附近，例如 `ppo.md` 旁边放 `imgs/ppo.png`，正文中用相对路径引用：

```md
![PPO Clip](./imgs/ppo.png)
```

全局素材：把图片放在 `public/images/`，正文中用站点绝对路径引用：

```md
![头像](/images/avatar.png)
```

## 公式与目录

Markdown 支持 LaTeX 公式。

行内公式：

```md
策略在状态 $s$ 下选择动作 $a$。
```

公式块：

```md
$$
\pi_\theta(a|s)
$$
```

详情页左侧会自动生成 h1、h2、h3 三级 TOC，并且可以打开或隐藏。文章详情页和论文详情页都有返回按钮，分别返回 `/articles/` 和 `/papers/` 表格页。

## 修改首页和个人信息

首页内容在 `src/pages/index.astro`：

- 修改大标题和简介。
- 修改 Focus 标签。
- Latest、Pinned 内容由内容模型自动生成。

个人介绍和链接已经合并到 `src/pages/about.astro`，顶部导航在 `src/layouts/BaseLayout.astro`。

## 切换博客样式主题

全局主题配置在 `src/site.config.ts`：

```ts
export const siteConfig = {
  theme: "github",
} as const;
```

当前支持：

- `"github"`：接近 GitHub 页面风格，白底、蓝色链接、灰色边框，整体更清爽简洁。
- `"classic"`：保留之前的暖色背景和绿色强调色风格。

修改 `theme` 后重新运行 `npm run dev` 或 `npm run build` 即可生效。

## 后续扩展 Book/Wiki

后续如果要做 book/wiki，建议新增一个独立 collection，例如 `src/content/books/`：

```text
src/content/books/
  rl-notes/
    index.md
    01-policy-gradient.md
    02-ppo.md
```

每个章节可以使用 `bookId`、`chapter`、`order`、`title`、`summary` 字段。页面结构可以复用当前详情页 TOC，但左侧目录改成 book 的章节树，点击章节进入对应 Markdown 页面。

## 部署到 GitHub Pages

推送到 `main` 分支后，GitHub Actions 会自动构建并发布到 GitHub Pages。

GitHub 仓库设置里，Pages 的发布来源选择 GitHub Actions。
