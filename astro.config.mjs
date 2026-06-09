import { defineConfig } from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import rehypeMarkmap from "./src/lib/rehypeMarkmap.mjs";

export default defineConfig({
  site: "https://zzlinux.github.io",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeMarkmap,
      [rehypeKatex, { strict: false, throwOnError: false }],
    ],
  },
});
