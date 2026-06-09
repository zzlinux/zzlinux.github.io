function hasClass(node, className) {
  const classes = node?.properties?.className;
  return Array.isArray(classes) && classes.includes(className);
}

function textContent(node) {
  if (!node) return "";
  if (node.type === "text") return node.value || "";
  if (!Array.isArray(node.children)) return "";
  return node.children.map(textContent).join("");
}

function visit(node) {
  if (!node || !Array.isArray(node.children)) return;

  for (let index = 0; index < node.children.length; index += 1) {
    const child = node.children[index];
    const code = child?.children?.[0];

    if (
      child?.type === "element" &&
      child.tagName === "pre" &&
      code?.type === "element" &&
      code.tagName === "code" &&
      hasClass(code, "language-markmap")
    ) {
      node.children[index] = {
        type: "element",
        tagName: "div",
        properties: {
          className: ["markmap-shell"],
          dataMarkmapSource: textContent(code).trim(),
        },
        children: [
          {
            type: "element",
            tagName: "svg",
            properties: {
              className: ["markmap-svg"],
              role: "img",
              ariaLabel: "论文脑图",
            },
            children: [],
          },
        ],
      };
      continue;
    }

    visit(child);
  }
}

export default function rehypeMarkmap() {
  return (tree) => visit(tree);
}
