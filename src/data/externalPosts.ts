export type IsoDateString = `${number}-${number}-${number}`;
export type ExternalPostType = "html" | "pdf";

export type ExternalPost = {
  title: string;
  date: IsoDateString;
  summary: string;
  tags: string[];
  href: string;
  type: ExternalPostType;
  category?: string;
  subcategory?: string;
  pinned?: boolean;
  pinnedRank?: number;
};

export function parseExternalPostDate(date: IsoDateString): Date {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    throw new Error("External post dates must use YYYY-MM-DD format.");
  }

  return new Date(`${date}T00:00:00`);
}

export const externalPosts: ExternalPost[] = [
  {
    title: "Model Evaluation Report",
    date: "2026-06-02",
    summary: "A standalone HTML report linked from the article index.",
    tags: ["ai", "evaluation"],
    href: "/docs/model-eval-report.html",
    type: "html",
    category: "模型评测",
    subcategory: "报告",
  },
];
