"""OpenCode gateway provider (Console, Zen and Go lanes).

Modules:

- ``dialect`` -- the gateway's wire dialects, id heuristics and URL derivation.
- ``catalog`` -- the models.dev per-model dialect/pricing catalog (TTL-cached).
- ``identity`` -- the OpenCode CLI client-identity headers the edge checks.
- ``adapter`` -- the multi-dialect adapter itself.
"""
