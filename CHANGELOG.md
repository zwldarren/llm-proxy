# Changelog

## [0.5.1](https://github.com/zwldarren/llm-proxy/compare/v0.5.0...v0.5.1) (2026-10-23)


### Features

* **logs:** record fallbacks and reported zeros as queryable log facts ([3b362fe](https://github.com/zwldarren/llm-proxy/commit/3b362fe275540a6fb8cab2624945896155e1776c))
* **observability:** one observation per request, projected to each sink ([e49c1a2](https://github.com/zwldarren/llm-proxy/commit/e49c1a2a146c907f9ffb70cb7f41d7c419a4a1dd))
* **rerank:** a retrieval protocol, and a provider that speaks its own dialect (ADR-0040, ADR-0041) ([8507e40](https://github.com/zwldarren/llm-proxy/commit/8507e40d0d8428228aa62780a277100cae8a3ddf))
* **tracing:** trace every request type with its own Langfuse content ([605dc43](https://github.com/zwldarren/llm-proxy/commit/605dc43657ace42abc6aad40520d3b73d3f5381c))


### Bug Fixes

* **cohere:** read the double-typed ApiMeta counters as counts ([d4b19bf](https://github.com/zwldarren/llm-proxy/commit/d4b19bf5811d77b3bbe9e3ef834b8929ba773d7a))
* **frontend:** keep LogsView's lazy children out of the test's imports ([dbaca80](https://github.com/zwldarren/llm-proxy/commit/dbaca80aaf303685949f57bc5bcf8107e6eadbeb))
* **logging:** keep the reason a provider failed when bodies are off ([f084677](https://github.com/zwldarren/llm-proxy/commit/f08467766b3123a9ffc3cde01cd3f483b6abda70))
* **openresponses:** carry cache_write_tokens in every usage payload ([a78e2da](https://github.com/zwldarren/llm-proxy/commit/a78e2da2358c241690514db0eab22fb5ad3cdcf0))
* **openresponses:** carry every detail counter into the terminal snapshot ([52e6ec9](https://github.com/zwldarren/llm-proxy/commit/52e6ec9b98f5a6fa74cdf798c9cf4a4d40ed3aa8))
* **openresponses:** fold every provider's usage shape into the terminal snapshot ([01d7c31](https://github.com/zwldarren/llm-proxy/commit/01d7c31829118f01d5125afc8d28f828dfa5f850))
* **openresponses:** hold the terminal snapshot until the trailing usage lands ([26c5c66](https://github.com/zwldarren/llm-proxy/commit/26c5c665fd181e7031a1b5ca718abce19bc25f13))
* **rerank:** carry relevant_spans back and keep document objects open ([9a3cbe9](https://github.com/zwldarren/llm-proxy/commit/9a3cbe92ec3423680fcc5ad8f55ee449c858c9c2))


### Documentation

* **openresponses:** correct the access_programs rationale ([68512c6](https://github.com/zwldarren/llm-proxy/commit/68512c6e4cea1f7397b72258e9b53d36cf21704b))

## [0.5.0](https://github.com/zwldarren/llm-proxy/compare/v0.4.1...v0.5.0) (2026-10-20)


### ⚠ BREAKING CHANGES

* **auth:** one key gate owns the auth sequence
* **streaming:** one segment accumulator owns canonical block assembly

### Features

* **gemini:** gate request shapes on the model family, cache what the wire cannot carry (ADR-0039) ([bde5fca](https://github.com/zwldarren/llm-proxy/commit/bde5fcaa8d93081e7dbd6543259695edf66c72e2))
* **minimax:** speak MiniMax's vocabulary on every wire ([fd45080](https://github.com/zwldarren/llm-proxy/commit/fd45080b7bffbe7afc52f06af5c4de0b22c17de3))
* **providers:** add OpenCode gateway adapters with per-model dialect routing ([24bb8cd](https://github.com/zwldarren/llm-proxy/commit/24bb8cdb822cb648c7eb6f5e8b80f2ec9e6e1a1a))
* **routing:** a fallback ladder, a corpus export, a cost-quality readout ([e673eee](https://github.com/zwldarren/llm-proxy/commit/e673eeea249d4ea59a47e16761dbafb6078f5a95))
* **routing:** warm experience store, ask-shaped classifier input, label flywheel ([e060776](https://github.com/zwldarren/llm-proxy/commit/e060776e2b2b45b9fee2579993bc70881b8f64e5))
* **tools:** a hosted tool is a dialect fact, not a function (ADR-0038) ([7775d82](https://github.com/zwldarren/llm-proxy/commit/7775d827cd8b1d63ca87304a900d8e5520cfe1d5))


### Bug Fixes

* address open-code-review findings from the routing/streaming/auth work ([552ff65](https://github.com/zwldarren/llm-proxy/commit/552ff6554530b66464b4bea94a058f00ccbcc400))
* **anthropic:** clamp the wire onto the official Messages shapes ([2df34ae](https://github.com/zwldarren/llm-proxy/commit/2df34ae57336d1c0c6f0a3b63b7180c26a9a6507))
* **streaming:** take the 499 verdict before the stop token can forge it ([420754b](https://github.com/zwldarren/llm-proxy/commit/420754b2b7987fe41dda3978034b96855ba55017))


### Code Refactoring

* **auth:** one key gate owns the auth sequence ([76e1366](https://github.com/zwldarren/llm-proxy/commit/76e13667b4accbc380fbe44add61b8e4f4fde1d8))
* **streaming:** one segment accumulator owns canonical block assembly ([1f63d74](https://github.com/zwldarren/llm-proxy/commit/1f63d740fec9cf73724392c830beb892d24a4cf0))

## [0.4.1](https://github.com/zwldarren/llm-proxy/compare/v0.4.0...v0.4.1) (2026-10-11)


### Bug Fixes

* **deps:** require sqlalchemy[asyncio] so greenlet ships at runtime ([43e9110](https://github.com/zwldarren/llm-proxy/commit/43e9110fdbf5ad3d1c71fb7a2cc8bec40aa65bca))
* **docker:** bypass uv run re-sync at container start ([2f53628](https://github.com/zwldarren/llm-proxy/commit/2f5362886198d842e465ea5e536e63c1f3da4a5e))
* **test:** compare the committed OpenAPI spec structurally ([6ff8ac7](https://github.com/zwldarren/llm-proxy/commit/6ff8ac76b2b37f296a2e5afed4f778d59affc2d0))

## [0.4.0](https://github.com/zwldarren/llm-proxy/compare/v0.3.1...v0.4.0) (2026-10-11)


### ⚠ BREAKING CHANGES

* **python:** adopt 3.15 idioms — PEP 810 lazy imports, frozendict, sentinel
* **routing:** `not_low_anchor` now defaults to False and an unset gate no longer consults every turn; console and docs carry the same posture.
* **security:** deployments that relied on the always-on API-key IP lockout lose it on upgrade. Public instances must enable Settings -> Security -> API Key IP Lockout.

### Features

* **health:** report which store actually holds the rate-limit windows ([c49c466](https://github.com/zwldarren/llm-proxy/commit/c49c466d6df98194ac25d8acf0722c4bbf776916))
* **routing:** drop the embedding signal (Signal C) and its ML stack ([c8ab589](https://github.com/zwldarren/llm-proxy/commit/c8ab589c443e8763edba26c18fb7031ef4f60c64))
* **routing:** repair the ensemble, close the judge gate, bar verdict consumption ([c212846](https://github.com/zwldarren/llm-proxy/commit/c21284656679be0bd0ed09ee5160a4b33adf4217))
* **routing:** ship the not_low_anchor judge gate and the clarified tier rubric ([50efb71](https://github.com/zwldarren/llm-proxy/commit/50efb719fcc30b260806bd651b95c56011e422e3))
* upgrade to Python 3.15 and adopt 3.15 language features ([1315d00](https://github.com/zwldarren/llm-proxy/commit/1315d003d1367dd7c46bb2c86f449caf5468feae))


### Bug Fixes

* address the second review pass on the merged refactor series ([20216eb](https://github.com/zwldarren/llm-proxy/commit/20216eb65ce475572913f0e7f64df87e537d3d47))
* **billing:** bill every web-search continuation hop ([046cb73](https://github.com/zwldarren/llm-proxy/commit/046cb73fe78ed2dad9123ff024226b55d8e4af20))
* **ci:** drop removed smart-routing extra; use python:3.15-rc-slim image ([597419b](https://github.com/zwldarren/llm-proxy/commit/597419ba3829398721eff55ce5b1cf298c5f5a5d))
* **config:** address review findings on the server settings registry ([dff1859](https://github.com/zwldarren/llm-proxy/commit/dff1859417a9ea768d5f2a04a007210558dd63b5))
* **docker:** skip dev group in production image ([aa47cb0](https://github.com/zwldarren/llm-proxy/commit/aa47cb06edd2192eeef62520e093a5905ce7b70e))
* **observability:** keep the provider-info search count monotonic ([5bafaec](https://github.com/zwldarren/llm-proxy/commit/5bafaec9e880304a415ed8c87385e21bbcc49bd1))
* **observability:** reach the Langfuse generation across the streaming task boundary ([d153521](https://github.com/zwldarren/llm-proxy/commit/d153521bb7c220d2bfe55e798900fd0ebb58a986))
* **ollama:** report tool_calls when the model produced them ([ab186e9](https://github.com/zwldarren/llm-proxy/commit/ab186e9661a70a9621e8197109c1a6e7cfcd1a57))
* **ratelimit:** give every bucket its own window and harden the limiter ([18a0992](https://github.com/zwldarren/llm-proxy/commit/18a0992ee874d53cc8c3a331101d41e8a3d9d777))
* **ratelimit:** make the Redis check-and-record one atomic script ([7e5542c](https://github.com/zwldarren/llm-proxy/commit/7e5542cf1341a52fe6bf262ac4d2867de776099e))
* **release:** bump the published OpenAPI version with the release ([497eb6d](https://github.com/zwldarren/llm-proxy/commit/497eb6d886d04306b3a2735ffc037bb54499e79e))
* **security:** make "disable rate limiting" cover every auth-path throttle ([85f8d4b](https://github.com/zwldarren/llm-proxy/commit/85f8d4b61e198ceb215c772a8587aa1b74e0c65e))
* **streaming:** keep the transformer module importable on its own ([442282d](https://github.com/zwldarren/llm-proxy/commit/442282d524f591dcee02c48f6edd4cbf08c3b105))
* **streaming:** tell a clean close at stream end from an abandonment ([a75a4f9](https://github.com/zwldarren/llm-proxy/commit/a75a4f9b1dd4972aaf57b80a17d02c4fefe7a394))
* **usage:** repair the facts the seam's first review found lost ([c1f9cc5](https://github.com/zwldarren/llm-proxy/commit/c1f9cc5a0459771ccf3d2e081e24595c59a877cc))
* **web-search:** attribute streamed searches to the request's user ([5f81f7f](https://github.com/zwldarren/llm-proxy/commit/5f81f7f5ef04720cfa1879f2f6fead7661ee4395))


### Documentation

* **adr:** give the server settings registry its own number ([a8ddb99](https://github.com/zwldarren/llm-proxy/commit/a8ddb99f27584820a883c41af206b9522f1dae5d))
* **adr:** point the superseded usage decisions at the intake seam ([c96671a](https://github.com/zwldarren/llm-proxy/commit/c96671a1dc376a1bf9923a0e428f7dccd1714270))
* **adr:** record the declared web-search transformer seam ([cfd4d92](https://github.com/zwldarren/llm-proxy/commit/cfd4d9253a8b4cce5050d61aa124c96672bb9d5c))
* **adr:** record the server settings registry decision ([5550bdf](https://github.com/zwldarren/llm-proxy/commit/5550bdf3ac1eaf1502c813415bdc9fdb2718b19e))
* **config:** finish repointing the server settings registry at ADR-0026 ([8b2aea4](https://github.com/zwldarren/llm-proxy/commit/8b2aea4a9a381f44317399f665741fa6148d1dfb))
* **context:** name the provider declaration and where it is asserted ([5d05adc](https://github.com/zwldarren/llm-proxy/commit/5d05adc8a00097471332c1dd324c46c7c1d7375f))


### Code Refactoring

* **python:** adopt 3.15 idioms — PEP 810 lazy imports, frozendict, sentinel ([d836ec2](https://github.com/zwldarren/llm-proxy/commit/d836ec2e466c60e08622278dca367cb17d452205))

## [0.3.1](https://github.com/zwldarren/llm-proxy/compare/v0.3.0...v0.3.1) (2026-10-08)


### Features

* **decisions:** serve /v1/decisions through a provider-side bridge ([bb313bf](https://github.com/zwldarren/llm-proxy/commit/bb313bf5371959bc9f97efcd5cc7b2cca0cae855))
* **frontend:** add a sticky settings section index with anchor scrolling ([13d5b64](https://github.com/zwldarren/llm-proxy/commit/13d5b64ec0b7522f85c6080ae7c8625e7b769090))
* **routing:** add a routing judge for the ambiguity band ([2aabea4](https://github.com/zwldarren/llm-proxy/commit/2aabea4ace951ca155726c5b234c03a1867e6489))


### Bug Fixes

* address the review findings on the seam refactor series ([85d61eb](https://github.com/zwldarren/llm-proxy/commit/85d61eb05f8e6073ba8c35ac1273c2f8840b7de0))
* **chat:** surface failed generations and stop cutting long streams ([ee330c8](https://github.com/zwldarren/llm-proxy/commit/ee330c851fa29c3a344afd91f4dec7a9e656fd1f))
* **frontend:** keep the kept-alive logs view out of other pages' ?tab= ([c67b5f3](https://github.com/zwldarren/llm-proxy/commit/c67b5f35847c20c4d229cd47399dcf798d742b44))
* **frontend:** make the model pickers retryable and retire the run tray ([0c5698c](https://github.com/zwldarren/llm-proxy/commit/0c5698c8902ad7cca630b23b81e218aea4112367))
* **observability:** stop losing Langfuse traces and double-counting tokens ([5ab1c53](https://github.com/zwldarren/llm-proxy/commit/5ab1c53cdf15ff7d32232c7cca445e485d5fed02))
* **web-search:** default a blank Ollama base URL to the hosted endpoint ([baea5cc](https://github.com/zwldarren/llm-proxy/commit/baea5ccf90f26a0257c4fd6073adc272a9b7b746))
* **web-search:** keep search results when a conversation is replayed ([6854d42](https://github.com/zwldarren/llm-proxy/commit/6854d425109306ee587973071edbeca61a36b845))


### Documentation

* **frontend:** sync DESIGN.md with the implemented nav rail ([665af13](https://github.com/zwldarren/llm-proxy/commit/665af134f7fa7acb5591af1a783ff6a0abae7fa8))

## [0.3.0](https://github.com/zwldarren/llm-proxy/compare/v0.2.9...v0.3.0) (2026-10-06)


### ⚠ BREAKING CHANGES

* **systemone:** adds the `systemone` request type, the `typesafe` provider type and `POST /v1/systemone` to the `/v1` contract. The shared OpenRouter cost predicate now rejects a boolean or zero `usage.cost` instead of billing it as USD 1.0. No existing endpoint, provider config or payload changes shape.

### Features

* **openapi:** publish a generated /v1 contract in the docs ([1795736](https://github.com/zwldarren/llm-proxy/commit/1795736630a4aa5b7b611a124b11797ba0068335))
* **systemone:** add System One evaluation across TypeSafe, OpenRouter and Ollama ([08f0387](https://github.com/zwldarren/llm-proxy/commit/08f03879a51fed8b4a10453de297e81589041bea))
* **systemone:** gate /v1/systemone on a capability flag and log its payloads ([4e97763](https://github.com/zwldarren/llm-proxy/commit/4e97763a11d913ebea5965004bf84a63c0a7e2ef))


### Bug Fixes

* **anthropic:** keep beta blocks, tool results and token counts faithful ([886f575](https://github.com/zwldarren/llm-proxy/commit/886f5754fd2d35befc574a35193544679f7dd631))
* **config:** apply settings and derived caches on every worker ([06281c5](https://github.com/zwldarren/llm-proxy/commit/06281c5c5c13ad6e8f8ae2105e17ab941796ba77))
* **logging:** enforce the body switch at the store and log rejections ([40b71fb](https://github.com/zwldarren/llm-proxy/commit/40b71fb4ffe34f538537d014093d6f49d7552f2f))
* **openai:** keep Responses fields, tool controls and annotations on the converted path ([8ae1616](https://github.com/zwldarren/llm-proxy/commit/8ae1616e8eb0db4028037e01536502e202831e78))
* **openapi:** pin the published spec version to the release version ([acc768d](https://github.com/zwldarren/llm-proxy/commit/acc768d1799f827c310f146c70c14f13964d8d94))
* **qwen:** forward image extension fields and drop phantom audio routes ([c760a37](https://github.com/zwldarren/llm-proxy/commit/c760a3745e4bf34a68749c05ebde6f8894a3a4ea))
* **web-search:** report the real failure reason, provider and query in logs ([6584415](https://github.com/zwldarren/llm-proxy/commit/658441560966a4d17f14265043f610226c6a6b0b))

## [0.2.9](https://github.com/zwldarren/llm-proxy/compare/v0.2.8...v0.2.9) (2026-09-30)


### Features

* **logs:** surface cache token counts on the log list ([a06e019](https://github.com/zwldarren/llm-proxy/commit/a06e019b3839b9d5b2bac058504d8b61044b655b))
* **models:** give list mode the table's column grammar ([f27a549](https://github.com/zwldarren/llm-proxy/commit/f27a549c84e27e4351e78d9e50f44a276ba537ab))
* **usage:** break hourly buckets down by model in the trends chart ([ba2a978](https://github.com/zwldarren/llm-proxy/commit/ba2a9783266f1a3bd0875333263e20f2e6d4f786))


### Bug Fixes

* **config:** decode bytes generation token so peer workers refresh ([a4f99fd](https://github.com/zwldarren/llm-proxy/commit/a4f99fd663c9c60943cf8cf67e941a7b6aea0488))
* **ui:** keep tooltips closed when focus moves programmatically ([8665f30](https://github.com/zwldarren/llm-proxy/commit/8665f30125d85c7dc5a6fe0255810fd61af512f3))


### Performance Improvements

* **logging:** cap stored bodies and reclaim space on retention sweeps ([b4f7c3b](https://github.com/zwldarren/llm-proxy/commit/b4f7c3bc85c53622970df8c9bd4dae22bf7b7862))

## [0.2.8](https://github.com/zwldarren/llm-proxy/compare/v0.2.7...v0.2.8) (2026-09-28)


### Features

* **usage:** add hourly token breakdown and model drill-down ([ad584f7](https://github.com/zwldarren/llm-proxy/commit/ad584f79028acc56a3d5f167ac854139a472f0f3))


### Bug Fixes

* **anthropic:** emit input_tokens exclusive of cache counters ([0acd0b4](https://github.com/zwldarren/llm-proxy/commit/0acd0b4ea45c96bc8f50c4e48a1832e19aa27b45))
* **reasoning:** keep thinking and encrypted reasoning faithful end to end ([5b84b42](https://github.com/zwldarren/llm-proxy/commit/5b84b4263ba194c25f71cf8180644cfb40a37f4e))
* **routing:** fall back on any provider error while one remains ([e937b4e](https://github.com/zwldarren/llm-proxy/commit/e937b4ea07820243b41ae5109c623c66125d6be7))

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
