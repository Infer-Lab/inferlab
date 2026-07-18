import { access, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { contentManifest, routeForTarget } from './content-manifest.mjs';

const websiteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(websiteRoot, 'dist');
const base = '/inferlab/';
const siteOrigin = 'https://infer-lab.github.io';

async function exists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

for (const required of [
  'index.html',
  'docs/index.html',
  'docs/getting-started/installation/index.html',
  'docs/reference/backend-support/index.html',
  'docs/architecture/rfc/rfc-0011/index.html',
  'docs/architecture/adr/adr-0019/index.html',
  'pagefind/pagefind.js',
]) {
  if (!(await exists(path.join(dist, required)))) {
    throw new Error(`missing built site output: ${required}`);
  }
}

if (await exists(path.join(dist, 'CNAME'))) {
  throw new Error('default GitHub Pages deployment must not contain CNAME');
}

async function htmlFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map((entry) => {
      const child = path.join(directory, entry.name);
      if (entry.isDirectory()) return htmlFiles(child);
      return entry.name.endsWith('.html') ? [child] : [];
    }),
  );
  return nested.flat();
}

function outputPathForUrl(url) {
  const pathname = decodeURIComponent(url.pathname);
  const relative = pathname.slice(base.length);
  if (relative === '') return path.join(dist, 'index.html');
  if (relative === '404/') return path.join(dist, '404.html');
  if (path.extname(relative)) return path.join(dist, relative);
  return path.join(dist, relative, 'index.html');
}

for (const entry of await contentManifest()) {
  const pageUrl = new URL(routeForTarget(entry.target), siteOrigin);
  const outputPath = outputPathForUrl(pageUrl);
  const html = await readFile(outputPath, 'utf8');
  const headingCount = [...html.matchAll(/<h1(?:\s|>)/g)].length;
  if (headingCount !== 1) {
    throw new Error(`${entry.source}: expected one rendered H1, found ${headingCount}`);
  }
}

for (const htmlPath of await htmlFiles(dist)) {
  const html = await readFile(htmlPath, 'utf8');
  const relativePage = path.relative(dist, htmlPath).replaceAll(path.sep, '/');
  const pageUrl = new URL(relativePage.replace(/index\.html$/, ''), `${siteOrigin}${base}`);
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const reference = match[1];
    if (reference.startsWith('#') || /^(?:mailto:|data:)/.test(reference)) continue;
    const url = new URL(reference, pageUrl);
    if (url.origin !== siteOrigin) continue;
    if (!decodeURIComponent(url.pathname).startsWith(base)) {
      throw new Error(`${relativePage}: local reference escapes ${base}: ${reference}`);
    }
    const target = outputPathForUrl(url);
    if (!path.extname(url.pathname) && !url.pathname.endsWith('/')) {
      throw new Error(`${relativePage}: page URL path lacks trailing slash: ${reference}`);
    }
    if (!(await exists(target))) {
      throw new Error(`${relativePage}: unresolved local reference ${reference}`);
    }
  }
}

console.log('Verified GitHub Pages base path, search index, and local site references.');
