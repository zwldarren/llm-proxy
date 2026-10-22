"""Trim a published OpenAPI document down to the schemas one endpoint emits.

The contract tests validate what the proxy writes against the *published*
schema, so the fixture must be a verbatim copy of that schema — never
hand-edited. This script is the regeneration path: it parses the published
document (YAML or JSON), walks ``$ref`` from the requested root schemas, and
writes a small JSON file holding only those reachable schemas.

    uv run python tests/protocols/fixtures/trim_schema.py \\
        --spec /tmp/openai-openapi.yml \\
        --source https://raw.githubusercontent.com/openai/openai-openapi/... \\
        --source-commit "$(git ls-remote https://github.com/openai/openai-openapi.git HEAD)" \\
        --root CreateChatCompletionResponse \\
        --root CreateChatCompletionStreamResponse \\
        --out tests/protocols/fixtures/openai_schema_contract.json

Re-run it after a spec release; the tests then tell us what the new spec
requires.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:  # pragma: no cover - dev-only dependency
            sys.exit("PyYAML is required to read a YAML spec: uv add --dev pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)


def refs(node: Any) -> set[str]:
    """Every ``$ref`` target in ``node``, as schema names."""
    found: set[str] = set()
    if isinstance(node, dict):
        target = node.get("$ref")
        if isinstance(target, str):
            found.add(target.rsplit("/", 1)[-1])
        for value in node.values():
            found |= refs(value)
    elif isinstance(node, list):
        for item in node:
            found |= refs(item)
    return found


def closure(schemas: dict[str, Any], roots: list[str]) -> dict[str, Any]:
    keep: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in keep:
            continue
        if name not in schemas:
            sys.exit(f"root or referenced schema {name!r} is not in the document")
        keep.add(name)
        pending.extend(refs(schemas[name]) - keep)
    return {name: schemas[name] for name in sorted(keep)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="published OpenAPI document")
    parser.add_argument("--source", required=True, help="URL the document came from")
    parser.add_argument("--source-commit", default="", help="commit/version of that URL")
    parser.add_argument(
        "--root", action="append", required=True, help="schema to keep (repeatable)"
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    document = load(args.spec)
    schemas = document["components"]["schemas"]
    kept = closure(schemas, args.root)
    fixture = {
        "source": args.source,
        "source_commit": args.source_commit,
        "retrieved": datetime.now(UTC).date().isoformat(),
        "note": (
            "Trimmed closure of the schemas this endpoint emits, copied verbatim from the "
            "published OpenAPI. Do not hand-edit: re-download the source URL and re-run "
            "tests/protocols/fixtures/trim_schema.py."
        ),
        "roots": list(args.root),
        "schemas": kept,
    }
    args.out.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(kept)} schemas, {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
