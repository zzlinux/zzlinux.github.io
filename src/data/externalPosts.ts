export type ExternalPost = {
  title: string;
  date: string;
  summary: string;
  tags: string[];
  href: string;
  type: "html";
};

export const externalPosts: ExternalPost[] = [
  {
    title: "Model Evaluation Report",
    date: "2026-06-02",
    summary: "A standalone HTML report linked from the article index.",
    tags: ["ai", "evaluation"],
    href: "/docs/model-eval-report.html",
    type: "html",
  },
];
