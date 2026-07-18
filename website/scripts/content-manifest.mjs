import { readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { siteBase } from '../site.config.mjs';

export const websiteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const repositoryRoot = path.resolve(websiteRoot, '..');
export { siteBase };

const fixedEntries = [
  {
    source: 'README.md',
    target: 'src/content/docs/docs/getting-started/installation.md',
    description: 'Install InferLab and run the first workspace-oriented commands.',
  },
  {
    source: 'docs/workspace-authoring.md',
    target: 'src/content/docs/docs/guides/workspace-authoring.md',
    description: 'Author versioned workspaces and machine-local bindings.',
  },
  {
    source: 'docs/tui.md',
    target: 'src/content/docs/docs/guides/tui.md',
    description: 'Use the view-only workspace console.',
  },
  {
    source: 'docs/backend-support.md',
    target: 'src/content/docs/docs/reference/backend-support.md',
    description: 'Qualified backend capabilities exposed by InferLab.',
  },
];

async function renderedEntries(kind) {
  const sourceDirectory = path.join(repositoryRoot, 'docs', kind);
  const names = (await readdir(sourceDirectory))
    .filter((name) => name.endsWith('.md'))
    .sort();

  return names.map((name) => ({
    source: `docs/${kind}/${name}`,
    target: `src/content/docs/docs/architecture/${kind}/${name}`,
    description:
      kind === 'rfc'
        ? 'Current InferLab specification clause set.'
        : 'Current InferLab architecture decision record.',
  }));
}

export async function contentManifest() {
  return [
    ...fixedEntries,
    ...(await renderedEntries('rfc')),
    ...(await renderedEntries('adr')),
  ];
}

export function routeForTarget(target) {
  let route = target.replace(/^src\/content\/docs\//, '').replace(/\.md$/, '');
  route = route.replace(/\/index$/, '');
  return `${siteBase}/${route.toLowerCase()}/`.replace(/\/+/g, '/');
}
