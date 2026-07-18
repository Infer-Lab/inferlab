import assert from 'node:assert/strict';
import test from 'node:test';
import { projectMarkdown } from './content-projection.mjs';

test('promotes the leading H1 to page metadata without duplicating it in the body', () => {
  const projection = projectMarkdown('# InferLab\n\nOperator guidance.\n', 'README.md');

  assert.equal(projection.title, 'InferLab');
  assert.equal(projection.body, 'Operator guidance.\n');
});

test('preserves generated comments and headings below the page title', () => {
  const projection = projectMarkdown(
    '<!-- GENERATED -->\n<!-- SIGNATURE -->\n\n# RFC-0011: Website\n\n## Summary\n\nText.\n',
    'docs/rfc/RFC-0011.md',
  );

  assert.equal(projection.title, 'RFC-0011: Website');
  assert.equal(
    projection.body,
    '<!-- GENERATED -->\n<!-- SIGNATURE -->\n\n## Summary\n\nText.\n',
  );
});

test('rejects a projected document without a page heading', () => {
  assert.throws(
    () => projectMarkdown('Operator guidance.\n', 'docs/guide.md'),
    /docs\/guide\.md: expected a level-one heading/,
  );
});

test('omits private RFC changelog history from the public projection', () => {
  const projection = projectMarkdown(
    '# RFC-0011: Website\n\nCurrent contract.\n\n## Changelog\n\nHistorical entry.\n',
    'docs/rfc/RFC-0011.md',
  );

  assert.equal(projection.body, 'Current contract.\n');
});
