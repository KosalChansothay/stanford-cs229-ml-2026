import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const cleanCodeLightTheme = {
  name: 'clean-code-light',
  type: 'light',
  colors: {
    'editor.background': '#F8FAFC',
    'editor.foreground': '#334155',
  },
  tokenColors: [
    {
      scope: ['comment', 'punctuation.definition.comment'],
      settings: { foreground: '#64748B', fontStyle: 'italic' }
    },
    {
      scope: [
        'keyword',
        'keyword.control',
        'storage',
        'storage.type',
        'storage.modifier',
        'keyword.operator.new',
        'keyword.operator.expression',
        'keyword.operator.logical'
      ],
      settings: { foreground: '#2563EB' }
    },
    {
      scope: [
        'entity.name.function',
        'support.function',
        'meta.function-call.generic',
        'entity.name.function.definition'
      ],
      settings: { foreground: '#EC4899' }
    },
    {
      scope: [
        'support.function.builtin',
        'support.function.builtin.python',
        'support.type.python'
      ],
      settings: { foreground: '#10B981' }
    },
    {
      scope: [
        'constant.numeric',
        'constant.language',
        'constant.language.boolean',
        'constant.language.python',
        'constant.character'
      ],
      settings: { foreground: '#EF4444' }
    },
    {
      scope: [
        'keyword.operator',
        'keyword.operator.comparison',
        'keyword.operator.arithmetic',
        'keyword.operator.assignment'
      ],
      settings: { foreground: '#F59E0B' }
    },
    {
      scope: ['string', 'string.quoted', 'string.template'],
      settings: { foreground: '#059669' }
    },
    {
      scope: [
        'variable',
        'variable.other',
        'variable.parameter',
        'variable.language.self'
      ],
      settings: { foreground: '#334155' }
    },
    {
      scope: [
        'entity.name.type',
        'entity.name.class',
        'support.class'
      ],
      settings: { foreground: '#8B5CF6' }
    },
    {
      scope: ['punctuation', 'punctuation.separator', 'punctuation.terminator'],
      settings: { foreground: '#64748B' }
    }
  ]
};

// https://astro.build/config
export default defineConfig({
  site: 'https://kosalchansothay.github.io',
  base: '/stanford-cs229-ml-2026',
  trailingSlash: 'ignore',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    syntaxHighlight: 'shiki',
    shikiConfig: {
      themes: {
        light: cleanCodeLightTheme,
        dark: 'github-dark'
      },
      wrap: true
    }
  }
});
