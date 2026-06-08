import { getCollection, type CollectionEntry } from "astro:content";

import {
  type ExternalPost,
  type ExternalPostType,
  externalPosts,
  parseExternalPostDate,
} from "../data/externalPosts";

export type SourceType = "markdown" | ExternalPostType;
export type ContentType = "article" | "paper";

export type PaperMeta = {
  authors: string[];
  institutions: string[];
  published?: string;
  venue?: string;
};

export type ContentItem = {
  id: string;
  title: string;
  date: Date;
  summary: string;
  category: string;
  subcategory: string;
  tags: string[];
  href: string;
  contentType: ContentType;
  sourceType: SourceType;
  pinned: boolean;
  pinnedRank?: number;
  paper?: PaperMeta;
};

export type ContentSubcategoryGroup = {
  subcategory: string;
  items: ContentItem[];
};

export type ContentCategoryGroup = {
  category: string;
  subcategories: ContentSubcategoryGroup[];
};

export type ArticleType = SourceType;
export type ArticleListItem = ContentItem;

export function compareByDateDesc(a: ContentItem, b: ContentItem): number {
  return b.date.getTime() - a.date.getTime();
}

export function comparePinnedWithinGroup(a: ContentItem, b: ContentItem): number {
  if (a.pinned !== b.pinned) {
    return a.pinned ? -1 : 1;
  }

  const aHasRank = typeof a.pinnedRank === "number";
  const bHasRank = typeof b.pinnedRank === "number";

  if (aHasRank && bHasRank) {
    return (a.pinnedRank ?? 0) - (b.pinnedRank ?? 0) || compareByDateDesc(a, b);
  }

  if (aHasRank) {
    return -1;
  }

  if (bHasRank) {
    return 1;
  }

  return compareByDateDesc(a, b);
}

export function getDefaultLatestItems(
  items: ContentItem[],
  count = 5,
): ContentItem[] {
  return [...items].sort(compareByDateDesc).slice(0, count);
}

export function getPinnedFallbackItems(
  items: ContentItem[],
  latestCount = 5,
): ContentItem[] {
  const latestIds = new Set(getDefaultLatestItems(items, latestCount).map((item) => item.id));

  return items
    .filter((item) => item.pinned && !latestIds.has(item.id))
    .sort(comparePinnedWithinGroup);
}

export function groupContentByCategory(items: ContentItem[]): ContentCategoryGroup[] {
  const categories = new Map<string, Map<string, ContentItem[]>>();

  for (const item of items) {
    const category = item.category || "未分类";
    const subcategory = item.subcategory || "默认";

    if (!categories.has(category)) {
      categories.set(category, new Map());
    }

    const subcategories = categories.get(category)!;
    if (!subcategories.has(subcategory)) {
      subcategories.set(subcategory, []);
    }

    subcategories.get(subcategory)!.push(item);
  }

  return [...categories.entries()]
    .sort(([a], [b]) => a.localeCompare(b, "zh-Hans-CN"))
    .map(([category, subcategories]) => ({
      category,
      subcategories: [...subcategories.entries()]
        .sort(([a], [b]) => a.localeCompare(b, "zh-Hans-CN"))
        .map(([subcategory, subcategoryItems]) => ({
          subcategory,
          items: [...subcategoryItems].sort(comparePinnedWithinGroup),
        })),
    }));
}

export function markdownPostToContentItem(post: CollectionEntry<"posts">): ContentItem {
  const contentType = post.data.type;

  return {
    id: `post:${post.slug}`,
    title: post.data.title,
    date: post.data.date,
    summary: post.data.summary,
    category: post.data.category,
    subcategory: post.data.subcategory,
    tags: post.data.tags,
    href: contentType === "paper" ? `/papers/${post.slug}/` : `/articles/${post.slug}/`,
    contentType,
    sourceType: "markdown",
    pinned: post.data.pinned,
    pinnedRank: post.data.pinnedRank,
    paper: post.data.paper,
  };
}

export function externalPostToContentItem(post: ExternalPost): ContentItem {
  return {
    id: `external:${post.href}`,
    title: post.title,
    date: parseExternalPostDate(post.date),
    summary: post.summary,
    category: post.category ?? "外部文档",
    subcategory: post.subcategory ?? post.type.toUpperCase(),
    tags: post.tags,
    href: post.href,
    contentType: "article",
    sourceType: post.type,
    pinned: post.pinned ?? false,
    pinnedRank: post.pinnedRank,
  };
}

export async function getAllContent(): Promise<ContentItem[]> {
  const markdownPosts = await getCollection("posts");
  const markdownContent = markdownPosts.map(markdownPostToContentItem);
  const externalArticles = externalPosts.map(externalPostToContentItem);

  return [...markdownContent, ...externalArticles].sort(compareByDateDesc);
}

export async function getAllArticles(): Promise<ArticleListItem[]> {
  const content = await getAllContent();

  return content.filter((item) => item.contentType === "article");
}

export async function getAllPapers(): Promise<ContentItem[]> {
  const content = await getAllContent();

  return content.filter((item) => item.contentType === "paper");
}
