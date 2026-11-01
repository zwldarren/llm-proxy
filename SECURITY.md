# Security Policy

LLM Proxy sits in front of paid upstream APIs and stores provider credentials, API-key
hashes, and console secrets. Reports about authentication, authorization, secret storage,
SSRF, and request handling are taken seriously.

## Supported versions

Security fixes land on `main` and in the latest release. Use the latest `stable` image
tag or the newest tagged release before reporting an issue; older releases are not
patched.

## Reporting a vulnerability

**Do not open a public issue, discussion, or pull request for a vulnerability.**

Report it privately through GitHub Security Advisories:

<https://github.com/zwldarren/llm-proxy/security/advisories/new>

Please include:

- the affected version or commit,
- the configuration needed to reproduce it (provider type, auth mode, relevant settings),
- a minimal reproduction — a request, a script, or exact steps,
- the impact you believe it has, and any proof-of-concept you are comfortable sharing.

This is a single-maintainer project, so reports are handled on a best-effort basis rather
than against a contractual SLA. You will get an acknowledgement, an assessment, and an
update when a fix ships. Please give us a reasonable window to release a fix before
disclosing publicly.

## Scope

In scope:

- authentication and authorization bypass on `/v1/*`, `/servers/*`, or `/api/*`,
- exposure of provider keys, API-key plaintext, JWT secrets, or encrypted values,
- SSRF through provider base URLs, media downloads, MCP servers, or web search,
- request smuggling, header injection, or protocol-confusion bugs,
- persistence or privilege escalation through the admin console,
- denial of service that is cheap for the attacker and not merely "send a large request".

Out of scope:

- findings that require an already-compromised host, database, or admin account,
- missing hardening headers with no demonstrated impact,
- rate-limit tuning opinions,
- vulnerabilities in upstream providers or third-party dependencies that have no
  exploitable path through this project (report those upstream; a notice is still
  welcome).

## Safe harbor

Good-faith research that stays within your own deployment, does not access other users'
data, and does not degrade service is welcome. Do not test against instances you do not
own or operate.
