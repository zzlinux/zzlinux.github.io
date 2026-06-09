import { defineConfig } from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import rehypeMermaid from "./src/lib/rehypeMermaid.mjs";

export default defineConfig({
  site: "https://zzlinux.github.io",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeMermaid,
      [rehypeKatex, { strict: false, throwOnError: false }],
    ],
  },
});
