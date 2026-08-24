"""Selection submodules for model routing.

Decomposed from ``llm_proxy.routing.selector`` into focused concerns:

- :mod:`constraints` — candidate filtering and infeasibility reporting.
- :mod:`cost` — pricing, cost estimation, and cost-aware guards.
- :mod:`scoring` — candidate scoring, experience snapshots, bandit bonuses,
  quality priors, and dynamic weight helpers.

The public selection API (``select_from_pool``) remains in
:mod:`llm_proxy.routing.selector`.
"""
