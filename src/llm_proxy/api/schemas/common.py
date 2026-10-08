"""Field validators shared by more than one admin write schema.

Used by the model, provider and MCP server write schemas. Kept out of any
single resource module so that reusing a validator never means importing
another resource's schemas.
"""

from pydantic import field_validator


class ValidatorMixin:
    """Optional-field coercion for the admin write schemas.

    The config types want an empty dict or list where the UI sends ``None`` for
    an unset section, and ``None`` where the UI clears ``base_url`` to ``""``.
    The validators name their fields (``check_fields=False``), so a schema
    applies only the coercions it declares.
    """

    @field_validator(
        "model_metadata",
        "parameter_overrides",
        "server_metadata",
        "custom_headers",
        "provider_metadata",
        "endpoint_base_urls",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def convert_none_to_empty_dict(cls, v):
        return v if v is not None else {}

    @field_validator(
        "provider_models",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def convert_none_to_empty_list(cls, v):
        return v if v is not None else []

    @field_validator("base_url", mode="before", check_fields=False)
    @classmethod
    def convert_empty_base_url_to_none(cls, v):
        return None if v == "" else v
