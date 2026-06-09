import GithubSlugger from "github-slugger";

export type TocHeading = {
  depth: 1 | 2 | 3;
  text: string;
  slug: string;
};

function stripMarkdown(text: string): string {
  return text
    .replace(/\s+#+\s*$/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[`*_~$]/g, "")
    .trim();
}

function stripFencedCodeBlocks(markdown: string): string {
  return markdown.replace(/^```[\s\S]*?^```\s*/gm, "");
}

export function getMarkdownToc(markdown: string): TocHeading[] {
  const body = stripFencedCodeBlocks(markdown.replace(/^---[\s\S]*?---\s*/, ""));
  const slugger = new GithubSlugger();
  const headings: TocHeading[] = [];
  const headingPattern = /^(#{1,3})\s+(.+)$/gm;
  let match: RegExpExecArray | null;

  while ((match = headingPattern.exec(body)) !== null) {
    const text = stripMarkdown(match[2]);

    if (text.length === 0) {
      continue;
    }

    headings.push({
      depth: match[1].length as 1 | 2 | 3,
      text,
      slug: slugger.slug(text),
    });
  }

  return headings;
}
