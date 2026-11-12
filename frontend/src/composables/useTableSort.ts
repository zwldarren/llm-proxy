import { useSorted } from "@vueuse/core";
import { ref, type MaybeRefOrGetter, type Ref } from "vue";

export type SortDirection = "asc" | "desc";

interface UseTableSortOptions<T, F extends string> {
  /** Rows to sort, usually the already-filtered list. */
  items: MaybeRefOrGetter<T[]>;
  /** Column key active on first render. */
  initialField: F;
  /** Direction used on first render; defaults to A→Z / smallest-first. */
  initialDirection?: SortDirection;
  /**
   * Direction used the first time a column becomes active. Date columns
   * default to newest-first, labels to A→Z.
   */
  defaultDirectionFor?: (field: F) => SortDirection;
  /**
   * Compare two rows for `field`. `dir` is passed in so a comparator can keep
   * empty values last in both directions instead of letting them flip.
   */
  compare: (field: F, a: T, b: T, dir: 1 | -1) => number;
}

interface UseTableSortReturn<T, F extends string> {
  sortField: Ref<F>;
  sortDir: Ref<SortDirection>;
  onSort: (field: string) => void;
  sortedItems: Ref<T[]>;
  resetSort: () => void;
}

/**
 * Sort state shared by the console's sortable tables.
 *
 * `SortableHead` only needs `sortField`, `sortDir` and `onSort`; the sorted
 * rows come from VueUse's `useSorted`, so the comparison work stays in the
 * consuming view (each table knows its own columns) while the state machine —
 * toggle the active column, reset to its default direction, restore on
 * "clear filters" — lives once.
 */
export function useTableSort<T, F extends string>(
  options: UseTableSortOptions<T, F>
): UseTableSortReturn<T, F> {
  const sortField = ref(options.initialField) as Ref<F>;
  const initialDirection =
    options.initialDirection ?? options.defaultDirectionFor?.(options.initialField) ?? "asc";
  const sortDir = ref(initialDirection) as Ref<SortDirection>;

  const sortedItems = useSorted(options.items, (a: T, b: T) => {
    const dir: 1 | -1 = sortDir.value === "asc" ? 1 : -1;
    return options.compare(sortField.value, a, b, dir);
  });

  function onSort(field: string) {
    const next = field as F;
    if (next === sortField.value) {
      sortDir.value = sortDir.value === "asc" ? "desc" : "asc";
      return;
    }
    sortField.value = next;
    sortDir.value = options.defaultDirectionFor?.(next) ?? "asc";
  }

  function resetSort() {
    sortField.value = options.initialField;
    sortDir.value = initialDirection;
  }

  return { sortField, sortDir, onSort, sortedItems, resetSort };
}
