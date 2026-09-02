"""CodeExecution part conversion (shared).

Converts Gemini ``executableCode`` / ``codeExecutionResult`` parts into
display text. Used by both the non-streaming response parser and the
streaming transformer so the two paths stay consistent.
"""

from typing import Any


def extract_code_execution_text(part: dict[str, Any]) -> str | None:
    """Convert a CodeExecution part into display text, or None.

    ``executableCode`` surfaces the generated code as a fenced block (the
    language enum is PYTHON); ``codeExecutionResult`` surfaces stdout on
    success, or stderr/a description otherwise.
    """
    if "executableCode" in part:
        ec = part["executableCode"]
        if not isinstance(ec, dict):
            return None
        code = ec.get("code", "")
        if not code:
            return None
        lang = (ec.get("language") or "PYTHON").lower()
        if lang in ("language_unspecified", ""):
            lang = "python"
        return f"```{lang}\n{code}\n```"
    if "codeExecutionResult" in part:
        cer = part["codeExecutionResult"]
        if not isinstance(cer, dict):
            return None
        result_text = cer.get("output", "")
        outcome = cer.get("outcome", "")
        if outcome not in ("", "OUTCOME_UNSPECIFIED", "OUTCOME_OK"):
            result_text = f"[{outcome}] {result_text}".rstrip()
        return result_text or None
    return None
