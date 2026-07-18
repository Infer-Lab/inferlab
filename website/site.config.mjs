const localRepositorySlug = 'Infer-Lab/InferLab';

export const siteOrigin = 'https://infer-lab.github.io';

export function resolveRepositorySlug(repository) {
  const slug = repository?.trim() || localRepositorySlug;
  const separator = slug.lastIndexOf('/');
  if (separator <= 0 || separator === slug.length - 1) {
    throw new Error(`expected repository identity in OWNER/REPOSITORY form, received ${slug}`);
  }
  return slug;
}

export function resolveSiteBase(repository) {
  const slug = resolveRepositorySlug(repository);
  return `/${slug.slice(slug.lastIndexOf('/') + 1)}`;
}

export function resolveBuildRepositorySlug(githubActions, repository) {
  if (!githubActions) return resolveRepositorySlug(undefined);
  if (!repository?.trim()) {
    throw new Error('GITHUB_REPOSITORY is required in GitHub Actions');
  }
  return resolveRepositorySlug(repository);
}

export const repositorySlug = resolveBuildRepositorySlug(
  process.env.GITHUB_ACTIONS === 'true',
  process.env.GITHUB_REPOSITORY,
);
export const repositoryUrl = `https://github.com/${repositorySlug}`;
export const siteBase = resolveSiteBase(repositorySlug);
export const siteBaseWithSlash = `${siteBase}/`;
