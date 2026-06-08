import type { ContentItem } from "./posts";
import {
  comparePinnedWithinGroup,
  getDefaultLatestItems,
  getPinnedFallbackItems,
  groupContentByCategory,
} from "./posts";

const oldPinned: ContentItem = {
  id: "old-pinned",
  title: "Old pinned",
  date: new Date("2026-01-01T00:00:00"),
  summary: "Older pinned item.",
  category: "AI",
  subcategory: "PPO",
  tags: ["rl"],
  href: "/articles/old-pinned/",
  contentType: "article",
  sourceType: "markdown",
  pinned: true,
};

const rankedPinned: ContentItem = {
  ...oldPinned,
  id: "ranked-pinned",
  title: "Ranked pinned",
  date: new Date("2025-01-01T00:00:00"),
  pinnedRank: 1,
};

const paper: ContentItem = {
  id: "paper",
  title: "Paper title",
  date: new Date("2026-06-01T00:00:00"),
  summary: "Paper summary.",
  category: "LLM",
  subcategory: "Reasoning",
  tags: ["paper"],
  href: "/papers/paper/",
  contentType: "paper",
  sourceType: "markdown",
  pinned: false,
  paper: {
    authors: ["A"],
    institutions: ["Lab"],
    published: "2025",
    venue: "NeurIPS",
  },
};

const latest = Array.from({ length: 5 }, (_, index): ContentItem => ({
  ...oldPinned,
  id: `latest-${index}`,
  title: `Latest ${index}`,
  date: new Date(`2026-06-0${index + 2}T00:00:00`),
  pinned: index === 0,
}));

const allItems = [...latest, oldPinned, rankedPinned, paper];

if (getDefaultLatestItems(allItems).length !== 5) {
  throw new Error("Default latest should contain five items.");
}

if (getPinnedFallbackItems(allItems).some((item) => item.id === "latest-0")) {
  throw new Error("Pinned fallback should exclude items already in the default latest window.");
}

if (comparePinnedWithinGroup(oldPinned, rankedPinned) <= 0) {
  throw new Error("Pinned items with pinnedRank should sort before unranked pinned items.");
}

const grouped = groupContentByCategory(allItems);
if (!grouped.some((group) => group.category === "LLM" && group.subcategories[0]?.subcategory === "Reasoning")) {
  throw new Error("Content should group by category and subcategory.");
}
