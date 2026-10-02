import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { theme, useOpenapi } from 'vitepress-openapi/client'
import 'vitepress-openapi/dist/style.css'
import spec from '../../public/v1-openapi.json' with { type: 'json' }
import Layout from './Layout.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    // The published /v1 contract, generated from the FastAPI app by
    // `uv run llm-proxy-openapi`. See llm_proxy/api/openapi_docs.py.
    useOpenapi({ spec })
    theme.enhanceApp({ app })
  },
} satisfies Theme
