"""Model allowlist helpers shared by auth middleware and API-key management.

Semantics:
* ``None`` — unrestricted (all models allowed).
* ``[]`` — explicit deny-all (no model is allowed).
* non-empty list — only the listed models are allowed.
"""

__all__ = ["intersect_model_lists"]


def intersect_model_lists(
    key_models: list[str] | None,
    user_models: list[str] | None,
) -> list[str] | None:
    """Intersect an API key's allowlist with its owner's user-level allowlist.

    The user-level constraint always wins: a key can never exceed what its
    owning user is permitted to use. Returns ``[]`` (deny-all) when both sides
    are restricted but disjoint.
    """
    if user_models is None:
        return key_models
    if key_models is None:
        return user_models
    return [model for model in key_models if model in user_models]
