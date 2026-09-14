import { defineConfig } from 'vitepress'
import type MarkdownIt from 'markdown-it'
import type Token from 'markdown-it/lib/token.mjs'

const repo = 'https://github.com/zwldarren/llm-proxy'
const base = '/llm-proxy/'

const gettingStarted = [
  { text: 'Installation', link: '/getting-started/installation' },
  { text: 'First Setup', link: '/getting-started/first-setup' },
  { text: 'Connect a Client', link: '/getting-started/connect-a-client' },
  { text: 'Connect an AI Agent', link: '/getting-started/connect-an-agent' },
  { text: 'Core Concepts', link: '/getting-started/concepts' },
]

const deployment = [
  { text: 'Docker Compose', link: '/deployment/docker' },
  { text: 'PostgreSQL & Redis', link: '/deployment/databases' },
  { text: 'Reverse Proxy & TLS', link: '/deployment/reverse-proxy' },
  { text: 'Upgrades & Backups', link: '/deployment/upgrades' },
  { text: 'Troubleshooting', link: '/deployment/troubleshooting' },
]

const admin = [
  { text: 'Admin Console Overview', link: '/admin/overview' },
  { text: 'Providers', link: '/admin/providers' },
  { text: 'Models & Pricing', link: '/admin/models' },
  { text: 'API Keys', link: '/admin/api-keys' },
  { text: 'Users & Roles', link: '/admin/users' },
  { text: 'MCP Servers', link: '/admin/mcp' },
  { text: 'Server Settings', link: '/admin/settings' },
  { text: 'Logs, Usage & Tracing', link: '/admin/observability' },
]

const api = [
  {
    text: 'Overview',
    items: [
      { text: 'API Reference', link: '/api/' },
      { text: 'Authentication', link: '/api/authentication' },
      { text: 'Endpoint Index', link: '/api/endpoints' },
      { text: 'Streaming', link: '/api/streaming' },
      { text: 'Errors & Rate Limits', link: '/api/errors' },
    ],
  },
  {
    text: 'Chat',
    items: [
      { text: 'Create chat completion', link: '/api/chat/create-completion' },
      { text: 'Create response', link: '/api/chat/create-response' },
      { text: 'Retrieve response', link: '/api/chat/retrieve-response' },
      { text: 'Delete response', link: '/api/chat/delete-response' },
      { text: 'Cancel response', link: '/api/chat/cancel-response' },
      { text: 'List input items', link: '/api/chat/list-input-items' },
      { text: 'Compact response', link: '/api/chat/compact-response' },
      { text: 'Anthropic Messages', link: '/api/chat/messages' },
      { text: 'Count tokens', link: '/api/chat/count-tokens' },
    ],
  },
  {
    text: 'Images',
    items: [
      { text: 'Create image', link: '/api/images/create' },
      { text: 'Edit image', link: '/api/images/edit' },
    ],
  },
  {
    text: 'Audio',
    items: [
      { text: 'Create speech', link: '/api/audio/speech' },
      { text: 'Create transcription', link: '/api/audio/transcription' },
      { text: 'Create translation', link: '/api/audio/translation' },
    ],
  },
  {
    text: 'Embeddings',
    items: [{ text: 'Create embeddings', link: '/api/embeddings' }],
  },
  {
    text: 'Models',
    items: [{ text: 'List models', link: '/api/models' }],
  },
  {
    text: 'Advanced',
    items: [
      { text: 'Tools, Reasoning & Web Search', link: '/api/tools' },
      { text: 'Virtual Models & Routing', link: '/api/routing' },
    ],
  },
]

const guides = [
  { text: 'Cost Control', link: '/guides/cost-control' },
  { text: 'Monitoring & Alerting', link: '/guides/monitoring' },
  { text: 'Security Hardening', link: '/guides/security' },
]
const reference = [
  { text: 'Environment Variables', link: '/reference/environment' },
  { text: 'FAQ', link: '/reference/faq' },
]

/**
 * API reference container: renders `::: endpoint POST /v1/...  # note` blocks as
 * OpenAI-style endpoint header cards (method badge + monospace path + copy button).
 */
const apiEndpointPlugin = (md: MarkdownIt) => {
  md.block.ruler.before(
    'fence',
    'api_endpoint',
    (state, startLine, endLine, silent) => {
      let pos = state.bMarks[startLine] + state.tShift[startLine]
      const max = state.eMarks[startLine]
      if (pos + 3 > max || state.src.slice(pos, pos + 3) !== ':::') return false
      pos += 3
      const firstLine = state.src.slice(pos, max).trim()
      if (!firstLine.startsWith('endpoint')) return false
      if (silent) return true

      // find the closing ::: (auto-close like markdown-it-container)
      let nextLine = startLine
      let autoClosed = false
      for (;;) {
        nextLine++
        if (nextLine >= endLine) break
        pos = state.bMarks[nextLine] + state.tShift[nextLine]
        const lineMax = state.eMarks[nextLine]
        if (pos < lineMax && state.sCount[nextLine] < state.blkIndent) break
        if (
          state.src.slice(pos, pos + 3) === ':::' &&
          state.skipSpaces(pos + 3) >= lineMax
        ) {
          autoClosed = true
          break
        }
      }

      // "endpoint POST /v1/images/generations  # JSON only" → method, path, note
      const rest = firstLine.slice('endpoint'.length).trim()
      const hashIdx = rest.indexOf('#')
      const head = (hashIdx === -1 ? rest : rest.slice(0, hashIdx)).trim()
      const note = hashIdx === -1 ? '' : rest.slice(hashIdx + 1).trim()
      const [method = '', path = ''] = head.split(/\s+/)
      if (!method || !path) return false

      const oldParent = state.parentType
      const oldLineMax = state.lineMax
      state.parentType = 'api_endpoint'
      state.lineMax = nextLine

      const open = state.push('api_endpoint_open', 'div', 1)
      open.meta = { method: method.toUpperCase(), path, note }
      state.md.block.tokenize(state, startLine + 1, nextLine)
      state.push('api_endpoint_close', 'div', -1)

      state.parentType = oldParent
      state.lineMax = oldLineMax
      state.line = nextLine + (autoClosed ? 1 : 0)
      return true
    },
    { alt: ['paragraph', 'reference', 'blockquote', 'list'] },
  )

  md.renderer.rules.api_endpoint_open = (tokens, idx) => {
    const { method, path, note } = tokens[idx].meta as {
      method: string
      path: string
      note: string
    }
    return (
      `<div class="api-endpoint" data-method="${method}">` +
      '<div class="api-endpoint__header">' +
      `<span class="api-endpoint__method">${method}</span>` +
      `<code class="api-endpoint__path">${md.utils.escapeHtml(path)}</code>` +
      (note ? `<span class="api-endpoint__note">${md.utils.escapeHtml(note)}</span>` : '') +
      '<button class="api-endpoint__copy" type="button" ' +
      'aria-label="Copy endpoint path">Copy</button>' +
      '</div>' +
      '<div class="api-endpoint__body">\n'
    )
  }
  md.renderer.rules.api_endpoint_close = () => '</div>\n</div>\n'
}

/**
 * API reference code rail, resolved at build time instead of via client-side DOM
 * surgery: on `pageClass: api-reference` pages, every top-level fenced block is
 * wrapped in a labeled `.api-code-panel` — one copy moves into a trailing
 * `.api-code-rail` container (CSS places it into the right-hand sticky column on
 * wide screens), and a `.api-code-panel--inline` twin stays in the reading flow
 * (hidden above the breakpoint) so narrow viewports keep the panels too — all
 * CSS, zero client JS.
 *
 * A fence info string may carry a display label after the language:
 * ` ```python [Request — OpenAI SDK] ` — the label becomes the panel head and
 * is stripped from the info before highlighting.
 */
const apiCodeRailPlugin = (md: MarkdownIt) => {
  md.core.ruler.push('api_code_rail', (state) => {
    const env = state.env as { frontmatter?: { pageClass?: string } }
    const pageClass = env.frontmatter?.pageClass ?? ''
    if (!pageClass.split(/\s+/).includes('api-reference')) return

    // Track container nesting so only top-level fences are paneled — code inside
    // lists, blockquotes, or ::: endpoint bodies stays a plain code block.
    const topLevel: number[] = []
    let depth = 0
    for (let i = 0; i < state.tokens.length; i++) {
      const token = state.tokens[i]
      if (depth === 0 && token.type === 'fence') topLevel.push(i)
      if (token.nesting === 1) depth++
      else if (token.nesting === -1) depth--
    }
    if (topLevel.length === 0) return

    const html = (content: string): Token => {
      const token = new state.Token('html_block', '', 0)
      token.content = content
      return token
    }

    // "python [Request — OpenAI SDK]" → panel head "Request — OpenAI SDK", info
    // trimmed to "python". Without a bracketed label the head is the uppercased
    // language.
    const parseInfo = (raw: string): { head: string; cleanInfo: string } => {
      const info = raw.trim()
      const labeled = info.match(/^(\S+)\s+\[([^\]]+)\]\s*$/)
      if (labeled) return { head: labeled[2], cleanInfo: labeled[1] }
      const lang = info.split(/\s+/)[0] || 'code'
      return { head: lang.toUpperCase(), cleanInfo: info }
    }

    const rail: Token[] = [html('<div class="api-code-rail">\n')]
    for (const index of topLevel) {
      const fence = state.tokens[index]
      const { head, cleanInfo } = parseInfo(fence.info)
      fence.info = cleanInfo
      const headHtml = (inline: boolean): string =>
        `<div class="api-code-panel${inline ? ' api-code-panel--inline' : ''}">` +
        `<div class="api-code-panel__head">${md.utils.escapeHtml(head)}</div>\n`

      rail.push(html(headHtml(false)), fence, html('</div>\n'))

      // Inline twin (same panel markup), hidden on wide screens where the rail
      // takes over. The same fence token renders in both places.
      state.tokens.splice(index, 0, html(headHtml(true)))
      state.tokens.splice(index + 2, 0, html('</div>\n'))
      for (let k = 0; k < topLevel.length; k++) {
        if (topLevel[k] > index) topLevel[k] += 2
      }
    }
    rail.push(html('</div>\n'))

    state.tokens.push(...rail)
  })
}

export default defineConfig({
  // The site is published to https://zwldarren.github.io/llm-proxy/ — keep `base`
  // in sync with any absolute asset paths in `head`.
  base,
  lang: 'en-US',
  title: 'LLM Proxy',
  titleTemplate: '%s · LLM Proxy',
  description: 'Self-hostable LLM API gateway — any client protocol in, any provider out.',
  head: [['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }]],
  // Internal, non-user-facing docs stay in the repo but out of the published site.
  srcExclude: ['**/adr/**', '**/agents/**'],
  // Example URLs that point at a running proxy (http://localhost:8080) are
  // intentional, not site-internal references.
  ignoreDeadLinks: [/^https?:\/\/localhost/],

  vite: {
    server: {
      watch: {
        // VitePress sets `emptyOutDir: false`, so Vite does not add its output
        // directory to the watcher's ignore list. Without this, a symlink inside
        // `dist/` (e.g. a `llm-proxy -> .` alias for serving the production build
        // at the site's `base` path) sends chokidar into an infinite recursion
        // until it dies with ELOOP and takes `docs:dev` down with it. Nothing in
        // the build output is ever a source file, so never watch it.
        ignored: ['**/.vitepress/dist/**'],
      },
    },
  },

  markdown: {
    config(md) {
      apiEndpointPlugin(md)
      apiCodeRailPlugin(md)
    },
  },
  themeConfig: {
    logo: { light: '/logo.svg', dark: '/logo-dark.svg' },
    siteTitle: 'LLM Proxy',
    outline: [2, 3],
    search: {
      provider: 'local',
      options: {
        detailedView: false,
      },
    },
    socialLinks: [{ icon: 'github', link: repo }],
    editLink: {
      pattern: `${repo}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub',
    },
    footer: {
      message: 'MIT licensed',
      copyright: 'LLM Proxy',
    },
    nav: [
      { text: 'Getting Started', link: '/getting-started/installation' },
      { text: 'Deployment', link: '/deployment/docker' },
      { text: 'Admin Console', link: '/admin/overview' },
      { text: 'API', link: '/api/' },
      { text: 'Guides', link: '/guides/cost-control' },
      {
        text: 'More',
        items: [
          ...reference,
          { text: 'Changelog', link: `${repo}/blob/main/CHANGELOG.md` },
          { text: 'GitHub', link: repo },
        ],
      },
    ],
    sidebar: {
      '/getting-started/': gettingStarted,
      '/deployment/': deployment,
      '/admin/': admin,
      '/api/': api,
      '/guides/': guides,
      '/reference/': reference,
    },
  },
})