<script setup lang="ts">
import { ArrowUpDown, ChevronDown, ChevronUp } from "@lucide/vue";
import { useI18n } from "vue-i18n";

/**
 * Pinned column header for the models list view.
 *
 * List rows use fixed-width data columns (the .model-col-* utilities) so they
 * align and scan like the table; this header labels those columns and carries
 * the same sort controls as the table's SortableHead. It sticks to the top of
 * .config-scroll exactly like .config-thead does in table mode — one column
 * grammar shared by both view modes.
 *
 * Cells mirror the rows' geometry: same horizontal gutter (px-4 sm:px-6), same
 * gap, same width classes. The name column is indented by the row icon's
 * footprint (32px icon + 12px gap = pl-11) so its label sits above the model
 * names, not above the icons.
 */

export interface ModelListColumn {
  /** Already-translated column label. */
  label: string;
  /**
   * Full layout classes for the cell: "flex min-w-0 flex-1 pl-11" for the name
   * column, a fixed "model-col-* hidden justify-end sm:flex" for data columns.
   */
  cellClass: string;
  /** When set, the cell renders a sort button for this field. */
  sortKey?: string;
}

const props = withDefaults(
  defineProps<{
    columns: ModelListColumn[];
    activeField?: string;
    activeDir?: "asc" | "desc";
    /** Layout classes for the trailing spacer matching the rows' actions cluster. */
    tailClass?: string;
  }>(),
  { activeField: "", activeDir: "asc", tailClass: "" }
);

const emit = defineEmits<{ sort: [field: string] }>();
const { t } = useI18n();

const isActive = (sortKey?: string) => Boolean(sortKey) && props.activeField === sortKey;

function ariaLabel(col: ModelListColumn): string {
  const base = t("common.sortByColumn", { column: col.label });
  if (!isActive(col.sortKey)) return base;
  return `${base} · ${
    props.activeDir === "asc" ? t("common.sortedAscending") : t("common.sortedDescending")
  }`;
}
</script>

<template>
  <div
    class="sticky top-0 z-10 flex h-9 items-center gap-3 border-b-2 border-border/70 bg-muted/50 px-4 sm:px-6"
  >
    <div v-for="col in columns" :key="col.sortKey ?? col.label" :class="col.cellClass">
      <button
        v-if="col.sortKey"
        type="button"
        :data-testid="`sort-${col.sortKey}`"
        :data-sort-key="col.sortKey"
        :aria-label="ariaLabel(col)"
        :class="[
          '-ml-1.5 flex h-full items-center gap-1 rounded-sm px-1.5 text-xs font-semibold uppercase tracking-wider transition-colors duration-150',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/60',
          isActive(col.sortKey) ? 'text-foreground' : 'text-muted-foreground hover:text-foreground',
        ]"
        @click="emit('sort', col.sortKey!)"
      >
        <span>{{ col.label }}</span>
        <ChevronUp
          v-if="isActive(col.sortKey) && activeDir === 'asc'"
          class="size-3.5 shrink-0"
          aria-hidden="true"
        />
        <ChevronDown
          v-else-if="isActive(col.sortKey) && activeDir === 'desc'"
          class="size-3.5 shrink-0"
          aria-hidden="true"
        />
        <ArrowUpDown v-else class="size-3 shrink-0 text-muted-foreground/40" aria-hidden="true" />
      </button>
      <span
        v-else
        class="block px-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground"
      >
        {{ col.label }}
      </span>
    </div>
    <div v-if="tailClass" :class="tailClass" aria-hidden="true" />
  </div>
</template>
