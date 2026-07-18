export function projectMarkdown(source, sourcePath) {
  if (/^docs\/rfc\/RFC-[0-9]+\.md$/.test(sourcePath)) {
    const changelog = source.search(/\r?\n## Changelog\r?\n/);
    if (changelog !== -1) {
      source = `${source.slice(0, changelog).trimEnd()}\n`;
    }
  }

  const match = /^#\s+([^\r\n]+)$/m.exec(source);
  if (!match) {
    throw new Error(`${sourcePath}: expected a level-one heading`);
  }

  let bodyStart = match.index + match[0].length;
  for (let lineEnding = 0; lineEnding < 2; lineEnding += 1) {
    if (source.startsWith('\r\n', bodyStart)) {
      bodyStart += 2;
    } else if (source[bodyStart] === '\n') {
      bodyStart += 1;
    } else {
      break;
    }
  }

  return {
    title: match[1].trim(),
    body: source.slice(0, match.index) + source.slice(bodyStart),
  };
}
