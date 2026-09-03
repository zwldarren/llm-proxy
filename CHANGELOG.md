# Changelog

## [0.2.2](https://github.com/zwldarren/llm-proxy/compare/v0.2.1...v0.2.2) (2026-09-03)


### Features

* **frontend:** replace loading spinners with geometry-mirroring skeletons ([e84a723](https://github.com/zwldarren/llm-proxy/commit/e84a7239e40cbfd8c60c5e5a0bb92d50cc053e77))
* **streaming:** tolerate no-space SSE field framing from upstreams ([b5afe3f](https://github.com/zwldarren/llm-proxy/commit/b5afe3f093ddab8479dca68fd02904dc5ee3bb8c))


### Bug Fixes

* **gemini:** align serialization with the live API (thoughtSignature, tools array, code execution) ([aa2c9ca](https://github.com/zwldarren/llm-proxy/commit/aa2c9cab595b98ee71df805881661f8a6731240a))
* **gemini:** promote STOP finish reason to tool_calls when tool calls produced ([ad0ac50](https://github.com/zwldarren/llm-proxy/commit/ad0ac501e3ec33a1dfc15be3c23659e94ffa6972))

## [0.2.1](https://github.com/zwldarren/llm-proxy/compare/v0.2.0...v0.2.1) (2026-09-01)


### Features

* **anthropic:** support 2026 tool types, beta usage, and diagnostics ([059c144](https://github.com/zwldarren/llm-proxy/commit/059c144cd4c8d69080942f8bcab228a08337cb5b))
* enable keepalive by default, SSE comment heartbeats, 499 on client disconnect ([c623070](https://github.com/zwldarren/llm-proxy/commit/c623070bd0803563635e6158e2914cd74ccd8864))
* harden Ollama provider serialization, streaming, and error handling ([1264c6b](https://github.com/zwldarren/llm-proxy/commit/1264c6ba3a0d95b99acd899fe2f539da560d1c93))
* **openresponses:** spec compliance for cancel, input_items, tiers, and Anthropic bridging ([a2c4763](https://github.com/zwldarren/llm-proxy/commit/a2c4763f1562f315d40a4094df8b7a78db22c5eb))
* **serialization:** cache-stable tool-call args and session-derived prompt_cache_key ([e39d0a3](https://github.com/zwldarren/llm-proxy/commit/e39d0a3fb6978482fd76ab3d55c4174ff848e7af))


### Bug Fixes

* address v0.2.0..HEAD review findings (keepalive, layering, billing) ([4f1ffaf](https://github.com/zwldarren/llm-proxy/commit/4f1ffaf62bdbb6b09f9a444e27ceba3a11ecffb3))
* **anthropic:** official-API-compliant lossless passthrough for native wire fields ([c3b04f2](https://github.com/zwldarren/llm-proxy/commit/c3b04f204238b50b2843d773206136b95e8e7a67))
* cascade model renames to model_experience rows ([ec98e5f](https://github.com/zwldarren/llm-proxy/commit/ec98e5fe4dbc7bf4e0d7f6b57e36362811e72f56))
* **frontend:** pricing-sync source recall, sidebar rail breakpoint, scrollbar styling ([4eb1f94](https://github.com/zwldarren/llm-proxy/commit/4eb1f94be8153233f0a5cce8c1b840c2298287ad))
* harden Ollama extras handling and dedupe stream header stashes ([60e9cc8](https://github.com/zwldarren/llm-proxy/commit/60e9cc810e87a79ddf43d3566dea7edadefd377b))

## [0.2.0](https://github.com/zwldarren/llm-proxy/compare/v0.1.0...v0.2.0) (2026-08-25)


### Features

* harden API key security and make smart-routing optional ([63cc0a9](https://github.com/zwldarren/llm-proxy/commit/63cc0a92a5657919b814d4a4851c22a5e9720df8))
* version display and update check in admin UI ([#1](https://github.com/zwldarren/llm-proxy/issues/1)) ([40e081d](https://github.com/zwldarren/llm-proxy/commit/40e081dbca9979679785044777df9309b1ea38cd))


### Bug Fixes

* make provider-types catalog tests hermetic ([a0c1b86](https://github.com/zwldarren/llm-proxy/commit/a0c1b86e7cf7147857653ba6491805f155332eaa))
