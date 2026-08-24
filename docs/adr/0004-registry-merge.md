# Merge the serializer registries; collapse one-file packages

Three registries triplicated the same pattern (`ThreadSafeRegistry` + decorator registration), and two of them (`serialization/handlers/registry.py`, `serialization/providers/registry.py`) additionally triplicated the instance-cache + double-checked-locking + miss-error machinery. We introduced a generic `CachedRegistry` in `core/registry_base.py` owning the class store, singleton instance cache, `RLock`, and on-demand import loop; the two serializer registry modules became thin facades keeping their public API names (47 consumer files untouched). The protocols registry was left on plain `ThreadSafeRegistry` — it registers endpoint *objects*, not classes, so it shares no instance-cache semantics.

In the same sweep we collapsed the two one-file packages that duplicated concept names: `core/parameter_overrides/__init__.py` → `core/parameter_override.py` (the stage in `stages/parameter_override.py` now imports the singular engine module), and `providers/utils/reasoning.py` → `providers/reasoning.py`. `providers/components/` and `providers/capabilities/` were verified coherent (docstrings match contents) and left alone.

## Considered Options

- **Single namespaced registry + rewrite 47 consumers**: rejected — churn without payoff; the facade approach gets the deduplication with zero consumer edits.
- **Extract only the instance-cache pattern** (leave decorators per-module): rejected — the get-or-create + on-demand import + error shaping is the duplicated substance; decorators are the cheap part and stay in facades.
- **Unify all three registries onto CachedRegistry**: rejected — the protocols registry has no instances to cache; forcing it would be over-general.
- **Rename `providers/components` ↔ `capabilities`** (suspected swap): rejected after inspection — both directories match their docstrings.
- **Push the re-registration warning into all registries**: rejected — tests legitimately re-register serializers; the warning stays protocols-only where re-registration signals a real conflict.

## Consequences

- Lowercase canonicalization now applies uniformly (previously providers-only; all existing keys are lowercase literals, so no behavior change).
- The providers registry's dual-location on-demand import (providers/<name>/serializer + serialization/<name>/serializer, ADR-0003) is preserved as the facade's import-location list; the `RLock` protection against adapter module-level `get_provider_serializer` recursion is inherited from `CachedRegistry`.
- Registry semantics live in exactly one place; future registries start from `CachedRegistry`.
