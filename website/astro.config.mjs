import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import starlight from '@astrojs/starlight';
import {
  repositoryUrl,
  siteBase,
  siteBaseWithSlash,
  siteOrigin,
} from './site.config.mjs';

export default defineConfig({
  site: siteOrigin,
  base: siteBase,
  trailingSlash: 'ignore',
  integrations: [
    sitemap({
      filter: (page) => page !== `${siteOrigin}${siteBase}`,
      customPages: [`${siteOrigin}${siteBaseWithSlash}`],
    }),
    starlight({
      title: 'InferLab',
      description: 'Reproducible LLM inference experiments, from declared intent to durable evidence.',
      favicon: '/favicon.svg',
      customCss: ['./src/styles/brand.css', './src/styles/starlight.css'],
      components: {
        SiteTitle: './src/components/StarlightSiteTitle.astro',
      },
      social: [
        {
          icon: 'github',
          label: 'InferLab on GitHub',
          href: repositoryUrl,
        },
      ],
      sidebar: [
        { label: 'Product', link: '/' },
        { label: 'Documentation', slug: 'docs' },
        {
          label: 'Getting Started',
          items: [
            'docs/getting-started',
            'docs/getting-started/installation',
          ],
        },
        {
          label: 'Concepts',
          items: ['docs/concepts'],
        },
        {
          label: 'Guides',
          items: [
            'docs/guides',
            'docs/guides/workspace-authoring',
            'docs/guides/tui',
          ],
        },
        {
          label: 'Reference',
          items: [
            'docs/reference',
            'docs/reference/backend-support',
          ],
        },
        {
          label: 'Architecture & Specification',
          items: [
            'docs/architecture',
            {
              label: 'RFCs',
              items: [
                { autogenerate: { directory: 'docs/architecture/rfc', collapsed: true } },
              ],
            },
            {
              label: 'ADRs',
              items: [
                { autogenerate: { directory: 'docs/architecture/adr', collapsed: true } },
              ],
            },
          ],
        },
      ],
    }),
  ],
});
