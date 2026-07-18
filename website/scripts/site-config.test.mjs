import assert from 'node:assert/strict';
import test from 'node:test';
import {
  resolveBuildRepositorySlug,
  resolveRepositorySlug,
  resolveSiteBase,
} from '../site.config.mjs';

test('derives the case-sensitive Pages base from the GitHub repository identity', () => {
  assert.equal(resolveRepositorySlug('Infer-Lab/InferLab'), 'Infer-Lab/InferLab');
  assert.equal(resolveSiteBase('Infer-Lab/InferLab'), '/InferLab');
  assert.equal(resolveSiteBase('example/MixedCase'), '/MixedCase');
});

test('uses the canonical repository identity outside GitHub Actions', () => {
  assert.equal(resolveBuildRepositorySlug(false, undefined), 'Infer-Lab/InferLab');
  assert.equal(resolveBuildRepositorySlug(false, 'example/MixedCase'), 'Infer-Lab/InferLab');
});

test('requires and uses the GitHub repository identity in GitHub Actions', () => {
  assert.equal(resolveBuildRepositorySlug(true, 'Infer-Lab/InferLab'), 'Infer-Lab/InferLab');
  assert.throws(() => resolveBuildRepositorySlug(true, undefined), /GITHUB_REPOSITORY/);
});
