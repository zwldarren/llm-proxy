# Changelog

## [0.2.8](https://github.com/zwldarren/llm-proxy/compare/v0.2.7...v0.2.8) (2026-09-26)


### Bug Fixes

* **reasoning:** keep thinking and encrypted reasoning faithful end to end ([5b84b42](https://github.com/zwldarren/llm-proxy/commit/5b84b4263ba194c25f71cf8180644cfb40a37f4e))

## [0.2.7](https://github.com/zwldarren/llm-proxy/compare/v0.2.6...v0.2.7) (2026-09-25)


### Features

* **serialization:** map files and media across provider wires ([43b4542](https://github.com/zwldarren/llm-proxy/commit/43b454278aed6ff29a34ac7d896e63c101b64384))


### Bug Fixes

* **config:** add "default" sentinel so per-adapter policy defaults apply ([81ec72d](https://github.com/zwldarren/llm-proxy/commit/81ec72d69818ca4b00c0e67b9bee4b8141b21bbe))
* **config:** refresh worker config when a peer process changes it ([28c5cda](https://github.com/zwldarren/llm-proxy/commit/28c5cdae0b575647443db22c8f150797590e2bcd))
* **openai:** align rebuilt Responses requests with the API reference ([4141b51](https://github.com/zwldarren/llm-proxy/commit/4141b51893babfd1681f1403bc5cecc2f722598d))
* **openai:** degrade audio to text and expand document content chunks ([ca1690b](https://github.com/zwldarren/llm-proxy/commit/ca1690b56e3e4977d2e9f0b617b50cdd39b962e4))
* **openai:** rebuild Responses content parts from Chat Completions shapes ([f967f18](https://github.com/zwldarren/llm-proxy/commit/f967f188b1f2f3a2a90ac2ac9dd23e279de8275e))
* **openresponses:** preserve input_image file_id on requests ([9fd6beb](https://github.com/zwldarren/llm-proxy/commit/9fd6beba0d7f5619ab529e1d981d86db59be1568))
* **serialization:** accept loose image payloads and reject unrepresentable ones ([beea060](https://github.com/zwldarren/llm-proxy/commit/beea060269d9a6651041a02a23bcbe647c01aa3d))
* **serialization:** map files and media onto blocks providers accept ([4a4902e](https://github.com/zwldarren/llm-proxy/commit/4a4902e8651a6fa7d34053c25104f21d2f8850eb))
* **serialization:** stop dropping files and media when routing across providers ([dff9d0a](https://github.com/zwldarren/llm-proxy/commit/dff9d0ac670e91efb3d4839f802b62638b6089ec))


### Documentation

* an FAQ section on how files/media map across providers (file_id is ([dff9d0a](https://github.com/zwldarren/llm-proxy/commit/dff9d0ac670e91efb3d4839f802b62638b6089ec))

## [0.2.6](https://github.com/zwldarren/llm-proxy/compare/v0.2.5...v0.2.6) (2026-09-23)


### Features

* **logging:** prune usage records on the log retention window ([35fde51](https://github.com/zwldarren/llm-proxy/commit/35fde511b44648c0d6ce6cd5a0d54336aad5c5dd))
* **logging:** reassemble streamed response bodies for request logs ([eaaf47f](https://github.com/zwldarren/llm-proxy/commit/eaaf47f5fa194e9fa635e231e23ff69388989abf))
* **mcp:** guide operators to the command allowlist when adding servers ([3db468d](https://github.com/zwldarren/llm-proxy/commit/3db468db3e9622bee15d6aa7833ffaf629f86799))
* **openresponses:** make tool_search-discovered tools callable ([d27b010](https://github.com/zwldarren/llm-proxy/commit/d27b010591e1f9e96d94b698c3b22b42233e6549))
* **openrouter:** extend native passthrough to Messages and media endpoints ([e926871](https://github.com/zwldarren/llm-proxy/commit/e926871d9bf0eded47a6ff8b97fd3b7c0df76666))
* **openrouter:** forward native Responses and app attribution ([b4e5a7c](https://github.com/zwldarren/llm-proxy/commit/b4e5a7c9b89090716aabe88898e52a77e4869a02))


### Bug Fixes

* **anthropic:** keep safeguard_results on the converted response path ([f4e0fe7](https://github.com/zwldarren/llm-proxy/commit/f4e0fe78bd62abf7c03ad28cc8f717080ae9557f))
* **frontend:** align page-header icon and polish filter/search surfaces ([2673f9b](https://github.com/zwldarren/llm-proxy/commit/2673f9bb49d087c2100284e38230bd1d919f9a4d))
* **frontend:** close a11y gaps and align data surfaces with the design system ([c568435](https://github.com/zwldarren/llm-proxy/commit/c5684355253827ec648c64b4423239ab3944cd52))
* **frontend:** resolve audit findings across performance, a11y, and docs ([f9cb70a](https://github.com/zwldarren/llm-proxy/commit/f9cb70aa96f3c3e3a607d2bcee6343a74eb86d43))
* **logging:** classify /v1/models as an endpoint, not an audit event ([9d86f4f](https://github.com/zwldarren/llm-proxy/commit/9d86f4f95015db90acb136c432ce2aa63b89ec41))
* **openai-compatible:** force include_usage on every conversion tier ([0183e4b](https://github.com/zwldarren/llm-proxy/commit/0183e4b05a5f0cd014b29bb481e5ad7317c83a55))

## [0.2.5](https://github.com/zwldarren/llm-proxy/compare/v0.2.4...v0.2.5) (2026-09-17)


### Features

* **chat:** made the console protocol-aware per endpoint ([e9217fe](https://github.com/zwldarren/llm-proxy/commit/e9217fed90423b4ce5007e757d7003dce96e1216))


### Bug Fixes

* **security:** disable login lockout by default and stream-measure body limits ([ada53a5](https://github.com/zwldarren/llm-proxy/commit/ada53a5c82de99dd22f59ff38d04d4c7ffba0e78))
* **usage:** dedup cache-read tokens across dialect columns ([8999150](https://github.com/zwldarren/llm-proxy/commit/8999150cf94abdee7974bbccdf0e9a59b32b4838))


### Performance Improvements

* **api:** skip idle middleware and trim per-request allocations ([90f1881](https://github.com/zwldarren/llm-proxy/commit/90f188198df8d57f1274a7c7d1d53bc23348eb84))
* **deploy:** multi-worker defaults and a bounded database pool ([8836ddc](https://github.com/zwldarren/llm-proxy/commit/8836ddc23b6780324f01ed849a7475ea1c3ecd9c))
* **http:** added an aiohttp outbound backend ([902d76e](https://github.com/zwldarren/llm-proxy/commit/902d76e88ea799980287614d5255a1da72a3eccb))
* **proxy:** cut per-request CPU on the gateway hot path ([17f7322](https://github.com/zwldarren/llm-proxy/commit/17f732254db1aa53e9108555bd2b6b013351431b))

## [0.2.4](https://github.com/zwldarren/llm-proxy/compare/v0.2.3...v0.2.4) (2026-09-15)


### Features

* **api-keys:** added deny-all states for api key model and mcp server lists ([b53dfdb](https://github.com/zwldarren/llm-proxy/commit/b53dfdb611837a0e244e2e0eb57994b4e7e85c34))
* **api-keys:** allowed members to restrict MCP servers on their own keys ([8d5284f](https://github.com/zwldarren/llm-proxy/commit/8d5284fd9e42b57896f84f9076f8778c3a84b658))
* **cors:** exposed trace headers to browser clients ([8474694](https://github.com/zwldarren/llm-proxy/commit/84746945d383dd711788f193b6b8e147aeb2df67))
* **observability:** kept log rows with scrubbed bodies when body logging disabled ([621a0d0](https://github.com/zwldarren/llm-proxy/commit/621a0d09c50bc6e9a76a65f3eda3bf1d4a60964c))
* **pricing:** added context-based pricing tiers ([495cd75](https://github.com/zwldarren/llm-proxy/commit/495cd7562fd9a253c7c5db022b27b2c1da505ebd))
* **providers:** add vLLM and SGLang adapters ([7d790c5](https://github.com/zwldarren/llm-proxy/commit/7d790c5e6fa7b561ec958c12c0b52e199b6e0c05))


### Bug Fixes

* **docker:** allowed docker compose to run without a .env file ([967b192](https://github.com/zwldarren/llm-proxy/commit/967b192ed8add3242d35d84eb6483aa979009585))
* **frontend:** let the provider dialog create keyless providers ([b4d3285](https://github.com/zwldarren/llm-proxy/commit/b4d3285e09c8e3ca53612860b4897ddbabbbc9c8))
* **protocols:** accept base_url aliases and trailing slashes on API paths ([8edd3ae](https://github.com/zwldarren/llm-proxy/commit/8edd3aefc760a509997aec2fc3ca69787f72f69e))
* **router:** used hasOwn for admin route prefetch filter ([5fa5e79](https://github.com/zwldarren/llm-proxy/commit/5fa5e7967e85454f85ed80186c180a829e2baaf9))
* **serialization:** drop proxy-internal markers at the outbound chokepoint ([a137392](https://github.com/zwldarren/llm-proxy/commit/a137392b71e54a220d958b0e821ce257066ed5d1))


### Performance Improvements

* **api:** replaced BaseHTTPMiddleware stack with pure ASGI middlewares ([56c9400](https://github.com/zwldarren/llm-proxy/commit/56c94007428ce3216f4f7ed37b54be6305870b10))


### Documentation

* added VitePress documentation site with GitHub Pages deployment ([9599a89](https://github.com/zwldarren/llm-proxy/commit/9599a895eee68214170dde01020162737ee94931))
* **adr:** accepted context pricing tiers design ([79bb119](https://github.com/zwldarren/llm-proxy/commit/79bb1194bdf9b8bc0f79b406110413ed237b6af9))
* **docs:** polished docs theme, guides, and markdown-it dependency ([6d3fa64](https://github.com/zwldarren/llm-proxy/commit/6d3fa64fdda76dcde37af829d6a1a5feb002f147))

## [0.2.3](https://github.com/zwldarren/llm-proxy/compare/v0.2.2...v0.2.3) (2026-09-12)


### Features

* surfaced provider cache-read tokens in usage records ([cb7a030](https://github.com/zwldarren/llm-proxy/commit/cb7a03081e539c71de81d5726c39e20a3c5744a3))
* **version:** show exact git commit when checkout is not a clean tagged release ([b468078](https://github.com/zwldarren/llm-proxy/commit/b46807873ae721338e7b41626d6e622f5cd0c7f9))


### Bug Fixes

* **billing:** report unknown cost instead of fake $0.00 for unpriced models ([3f9b6a6](https://github.com/zwldarren/llm-proxy/commit/3f9b6a62b617fa904d5fb3592391fb3cab983d59))
* **images:** support GPT Image 2.5 quality tiers and custom resolutions ([8311645](https://github.com/zwldarren/llm-proxy/commit/83116450cb0f303884ce60274fc332021e0d4c0f))

## [0.2.2](https://github.com/zwldarren/llm-proxy/compare/v0.2.1...v0.2.2) (2026-09-08)


### Features

* **frontend:** replace loading spinners with geometry-mirroring skeletons ([e84a723](https://github.com/zwldarren/llm-proxy/commit/e84a7239e40cbfd8c60c5e5a0bb92d50cc053e77))
* **models:** add models.dev-aligned display attributes to catalog and management ([00ecbea](https://github.com/zwldarren/llm-proxy/commit/00ecbeaf65534bfb01d4351cb2d9510f98fb1191))
* **models:** sync capability/metadata fields from models.dev alongside pricing ([d0a1196](https://github.com/zwldarren/llm-proxy/commit/d0a11962397c5de2f37b32547eab94786c06efbd))
* **models:** unify catalog and management behind a single role-switched /models route ([a4b4e6c](https://github.com/zwldarren/llm-proxy/commit/a4b4e6c31e4dee72d058dabda39eea88e72893ac))
* **streaming:** tolerate no-space SSE field framing from upstreams ([b5afe3f](https://github.com/zwldarren/llm-proxy/commit/b5afe3f093ddab8479dca68fd02904dc5ee3bb8c))


### Bug Fixes

* **ci:** type-annotate anthropic pending usage dict; sort test imports ([2ca26a8](https://github.com/zwldarren/llm-proxy/commit/2ca26a8a89c8fd5e1d3e5f18439b45bf403a0ce4))
* **ci:** use valid GitHub expression for stable tag enable in Docker workflow ([d64afc2](https://github.com/zwldarren/llm-proxy/commit/d64afc23dfecd005c939d844b781b008ec206555))
* **gemini:** align serialization with the live API (thoughtSignature, tools array, code execution) ([aa2c9ca](https://github.com/zwldarren/llm-proxy/commit/aa2c9cab595b98ee71df805881661f8a6731240a))
* **gemini:** promote STOP finish reason to tool_calls when tool calls produced ([ad0ac50](https://github.com/zwldarren/llm-proxy/commit/ad0ac501e3ec33a1dfc15be3c23659e94ffa6972))
* **migrations:** use sa.false() for boolean server defaults ([01209c3](https://github.com/zwldarren/llm-proxy/commit/01209c370595b5d5f7b7d2b9ff3a719cc6cab090))

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
