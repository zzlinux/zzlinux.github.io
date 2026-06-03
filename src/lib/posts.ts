import { getCollection, type CollectionEntry } from "astro:content";

import {
  type ExternalPostType,
  externalPosts,
  parseExternalPostDate,
} from "../data/externalPosts";

export type ArticleType = "markdown" | ExternalPostType;

export type ArticleListItem = {
  title: string;
  date: Date;
  summary: string;
  tags: string[];
  href: string;
  type: ArticleType;
};

export function markdownPostToListItem(
  post: CollectionEntry<"posts">,
): ArticleListItem {
  return {
    title: post.data.title,
    date: post.data.date,
    summary: post.data.summary,
    tags: post.data.tags,
    href: `/articles/${post.slug}/`,
    type: "markdown",
  };
}

export async function getAllArticles(): Promise<ArticleListItem[]> {
  const markdownPosts = await getCollection("posts");
  const markdownArticles = markdownPosts.map(markdownPostToListItem);
  const externalArticles = externalPosts.map((post) => ({
    ...post,
    date: parseExternalPostDate(post.date),
  }));

  return [...markdownArticles, ...externalArticles].sort(
    (a, b) => b.date.getTime() - a.date.getTime(),
  );
}
