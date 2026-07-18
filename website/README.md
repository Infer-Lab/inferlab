# InferLab website

The website is a static Astro and Starlight project published under the GitHub Pages project base `/inferlab/`.

## Content authority

Product landing pages and category introductions are presentation-only summaries. The build projects the following public repository sources without maintaining editable documentation copies:

- `README.md`
- `docs/workspace-authoring.md`
- `docs/backend-support.md`
- `docs/tui.md`
- `docs/rfc/*.md`
- `docs/adr/*.md`

`scripts/sync-content.mjs` represents each source document's leading H1 as the Starlight page title, adds presentation frontmatter, and mechanically maps relative link destinations to website routes in generated, ignored files. It preserves the remaining authoritative prose, headings, code, tables, and link labels, while omitting the private RFC changelog tail from the public projection. The website consumes only the closed public content set listed above.

Human-facing product text uses `InferLab`. The lowercase `inferlab` spelling is reserved for commands, packages, repository names, paths, URLs, and other machine identifiers; independently specified protocol and evidence literals retain their exact value.

## Local workflow

From the repository root:

```sh
pixi run -e website website-install
pixi run -e website website-dev
pixi run -e website website-verify
```

The development server accepts page routes with or without a trailing slash. Generated page-navigation and canonical URL paths use the trailing-slash form under `/inferlab/`, matching the production static-directory layout.

The production check validates the static output, Pagefind index, `/inferlab/` base path, canonical page paths, single-H1 projection, and local references. `website/package.json` and `website/package-lock.json` own JavaScript requirements; Pixi owns the Node/npm runtime used locally and in CI.

## Deployment

Public pull requests run the locked production build. A push to public `main` runs Rust, Python, and website validation, uploads the site artifact produced by that revision, and deploys it through GitHub Pages. The automatic workflow begins only after a reviewed public snapshot reaches `main`.
