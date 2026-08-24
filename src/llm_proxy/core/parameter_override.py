"""Simple key-value parameter overrides for model and provider configurations."""

import copy
import re
from typing import Any

# Pattern to match variable placeholders like {model}, {original_model}
VARIABLE_PATTERN = re.compile(r"\{(\w+)\}")


def apply_parameter_overrides(
    request_data: dict[str, Any],
    overrides: dict[str, Any],
    variables: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply simple key-value parameter overrides to raw request data.

    Args:
        request_data: Raw request body as dict
        overrides: Parameter overrides as key-value pairs
        variables: Built-in variables (model, upstream_model, original_model)

    Returns:
        Modified request data
    """
    if not overrides:
        return request_data

    result = copy.deepcopy(request_data)
    vars_ = variables or {}

    for key, value in overrides.items():
        substituted_value = _substitute_variables(value, vars_)
        result[key] = substituted_value

    return result


def _substitute_variables(value: Any, variables: dict[str, str]) -> Any:
    """Substitute built-in variables in string values; other types pass through."""
    if not isinstance(value, str):
        return value

    return VARIABLE_PATTERN.sub(lambda match: variables.get(match.group(1), match.group(0)), value)


def create_variables(
    model: str,
    provider_model_name: str | None = None,
    original_model: str | None = None,
) -> dict[str, str]:
    """Create variables dict for parameter overrides.

    Args:
        model: The model name (after redirection)
        provider_model_name: The provider-specific model name
        original_model: The original model name from request

    Returns:
        Dict with model, upstream_model, original_model variables
    """
    return {
        "model": provider_model_name or model,
        "upstream_model": provider_model_name or model,
        "original_model": original_model or model,
    }


__all__ = [
    "apply_parameter_overrides",
    "create_variables",
]
