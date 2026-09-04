/**
 * Shared model sorting comparators. The admin management view and the
 * read-only browse view at /models order the same model table, so the
 * ordering rules live here once.
 */

interface ModelSortable {
  name: string;
  context_length?: number | null;
}

/** Alphabetical by name, flipped by direction. */
export function compareModelsByName(dir: 1 | -1) {
  return (a: ModelSortable, b: ModelSortable): number => a.name.localeCompare(b.name) * dir;
}

/**
 * By context length, flipped by direction. Models without a context length
 * always sort last, regardless of direction; ties fall back to name order.
 */
export function compareModelsByContextLength(dir: 1 | -1) {
  return (a: ModelSortable, b: ModelSortable): number => {
    const aMissing = a.context_length == null ? 1 : 0;
    const bMissing = b.context_length == null ? 1 : 0;
    if (aMissing !== bMissing) return aMissing - bMissing;
    return (
      ((a.context_length ?? 0) - (b.context_length ?? 0)) * dir || a.name.localeCompare(b.name)
    );
  };
}
