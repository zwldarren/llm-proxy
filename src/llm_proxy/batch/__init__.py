"""Batch-mode package: the OpenAI-compatible batch surface (ADR-0058).

``service`` owns files, job state and translation; ``wire`` renders OpenAI's
objects. Provider dialects live with their adapters under
``llm_proxy/providers/*/batch.py``.
"""
