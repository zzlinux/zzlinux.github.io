import { defineCollection, z } from "astro:content";

const posts = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    tags: z.array(z.string()).default([]),
    summary: z.string(),
    category: z.string().default("未分类"),
    subcategory: z.string().default("默认"),
    type: z.enum(["article", "paper"]).default("article"),
    pinned: z.boolean().default(false),
    pinnedRank: z.coerce.number().optional(),
    paper: z
      .object({
        authors: z.array(z.string()).default([]),
        institutions: z.array(z.string()).default([]),
        published: z.string().optional(),
        venue: z.string().optional(),
      })
      .optional(),
  }),
});

export const collections = { posts };
