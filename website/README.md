# InferLab website

The website is a static Astro and Starlight project published under the GitHub Pages project base `/InferLab/`.

## Content authority

Product landing pages and category introductions are presentation-only summaries. The build projects the following public repository sources without maintaining editable documentation copies:

- `README.md`
- `docs/workspace-authoring.md`
- `docs/backend-support.md`
- `docs/tui.md`
- `docs/rfc/*.md`
- `docs/adr/*.md`

`scripts/sync-content.mjs` represents each source document's leading H1 as the Starlight page title, adds presentation frontmatter, and mechanically maps relative link destinations to website routes in generated, ignored files. It preserves the remaining authoritative prose, headings, code, tables, and link labels, while omitting the private RFC changelog tail from the public projection. The website consumes only the closed public content set listed above.

Human-facing product text uses `InferLab`. Commands, package distribution names, local storage paths, and independently specified protocol or evidence literals retain their defined spelling. Repository and default project-site URLs use the repository's canonical `InferLab` spelling.

## Local workflow

From the repository root:

```sh
pixi run -e website website-install
pixi run -e website website-dev
pixi run -e website website-verify
```

The development server accepts page routes with or without a trailing slash. Generated page-navigation and canonical URL paths use the trailing-slash form under `/InferLab/`, matching the production static-directory layout.

`site.config.mjs` derives the production base from the repository-name component of `GITHUB_REPOSITORY`. Outside GitHub Actions it uses the explicit canonical fallback `Infer-Lab/InferLab`, so local production-equivalent builds retain `/InferLab/` without depending on Git metadata or a particular remote name.

The production check validates the static output, Pagefind index, canonical base path, page paths, single-H1 projection, and local references through that same configuration authority. `website/package.json` and `website/package-lock.json` own JavaScript requirements; Pixi owns the Node/npm runtime used locally and in CI.

## Deployment

Public pull requests run the locked production build. A push to public `main` runs Rust, Python, and website validation, uploads the site artifact produced by that revision, and deploys it through GitHub Pages. The automatic workflow begins only after a reviewed public snapshot reaches `main`.
