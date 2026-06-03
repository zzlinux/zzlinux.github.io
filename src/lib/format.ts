import type { ArticleType } from "./posts";

export const formatArticleDate = (date: Date) =>
  new Intl.DateTimeFormat("en", { dateStyle: "medium" }).format(date);

export const formatArticleType = (type: ArticleType) => {
  if (type === "html") {
    return "HTML";
  }

  if (type === "pdf") {
    return "PDF";
  }

  return "Markdown";
};
