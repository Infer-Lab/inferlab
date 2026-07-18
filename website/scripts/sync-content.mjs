import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  contentManifest,
  repositoryRoot,
  routeForTarget,
  siteBase,
  websiteRoot,
} from './content-manifest.mjs';
import { projectMarkdown } from './content-projection.mjs';
import { repositoryUrl } from '../site.config.mjs';

function frontmatter(title, description) {
  return `---\ntitle: ${JSON.stringify(title)}\ndescription: ${JSON.stringify(description)}\n---\n\n`;
}

const manifest = await contentManifest();
const bySource = new Map(manifest.map((entry) => [entry.source, entry]));
const directoryRoutes = new Map([
  ['docs/rfc', `${siteBase}/docs/architecture/rfc/`],
  ['docs/adr', `${siteBase}/docs/architecture/adr/`],
]);

function splitSuffix(url) {
  const index = url.search(/[?#]/);
  return index === -1 ? [url, ''] : [url.slice(0, index), url.slice(index)];
}

function rewriteUrl(url, sourceEntry) {
  if (
    url === '' ||
    url.startsWith('#') ||
    url.startsWith('/') ||
    /^[a-z][a-z+.-]*:/i.test(url)
  ) {
    return url;
  }

  const [pathname, suffix] = splitSuffix(url);
  const resolved = path
    .normalize(path.join(path.dirname(sourceEntry.source), pathname))
    .replaceAll(path.sep, '/');
  const target = bySource.get(resolved);
  if (target) return `${routeForTarget(target.target)}${suffix}`;

  const directoryRoute = directoryRoutes.get(resolved.replace(/\/$/, ''));
  if (directoryRoute) return `${directoryRoute}${suffix}`;

  const absolute = path.resolve(repositoryRoot, resolved);
  const relative = path.relative(repositoryRoot, absolute).replaceAll(path.sep, '/');
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`${sourceEntry.source}: projected link escapes repository root: ${url}`);
  }
  return `${repositoryUrl}/blob/main/${relative}${suffix}`;
}

function projectLinks(source, entry) {
  return source.replace(/(\]\()([^\s)]+)([^)]*\))/g, (_match, open, url, close) => {
    return `${open}${rewriteUrl(url, entry)}${close}`;
  });
}

function withoutLinkTargets(source) {
  return source.replace(/(\]\()([^\s)]+)([^)]*\))/g, '$1<route>$3');
}
const generatedDirectories = [
  'src/content/docs/docs/architecture/rfc',
  'src/content/docs/docs/architecture/adr',
];

for (const directory of generatedDirectories) {
  const absoluteDirectory = path.join(websiteRoot, directory);
  const names = await readdir(absoluteDirectory).catch((error) => {
    if (error && error.code === 'ENOENT') return [];
    throw error;
  });
  for (const name of names) {
    if (name !== 'index.md') {
      await rm(path.join(absoluteDirectory, name), { recursive: true, force: true });
    }
  }
}

for (const entry of manifest) {
  const sourcePath = path.join(repositoryRoot, entry.source);
  const targetPath = path.join(websiteRoot, entry.target);
  const source = await readFile(sourcePath, 'utf8');
  const projection = projectMarkdown(source, entry.source);
  const projectedBody = projectLinks(projection.body, entry);
  const metadata = frontmatter(projection.title, entry.description);

  await mkdir(path.dirname(targetPath), { recursive: true });
  await writeFile(targetPath, metadata + projectedBody, 'utf8');

  const projected = await readFile(targetPath, 'utf8');
  if (
    withoutLinkTargets(projected.slice(metadata.length)) !==
    withoutLinkTargets(projection.body)
  ) {
    throw new Error(`${entry.target}: projection changed authoritative prose`);
  }
}

console.log(`Projected ${manifest.length} authoritative documents.`);
