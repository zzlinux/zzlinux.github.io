import { formatArticleType } from "./format";

const articleTypes: Array<[Parameters<typeof formatArticleType>[0], string]> = [
  ["markdown", "Markdown"],
  ["html", "HTML"],
  ["pdf", "PDF"],
];

for (const [type, label] of articleTypes) {
  if (formatArticleType(type) !== label) {
    throw new Error(`Expected ${type} to format as ${label}`);
  }
}
