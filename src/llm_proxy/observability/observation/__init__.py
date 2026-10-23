"""One normalized observation, projected to each observability sink.

The pipeline captures a request once, the collector turns that capture into an
immutable :class:`~llm_proxy.observability.observation.model.RequestObservation`,
and each sink receives only what its policy allows. See ADR-0044.

The public entry points are the modules, not re-exports, because they form a
small pipeline with an order the reader should see: ``collector`` (capture) →
``policy`` / ``projection`` (narrow) → ``bus`` (fan out) → ``sinks`` (write),
with ``database`` composing the two stores' bus for the endpoint lifecycle.
"""
