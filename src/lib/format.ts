export const formatArticleDate = (date: Date) =>
  new Intl.DateTimeFormat("en", { dateStyle: "medium" }).format(date);

export const formatArticleType = (type: "markdown" | "html") =>
  type === "html" ? "HTML" : "Markdown";
