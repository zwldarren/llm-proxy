<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { ArrowUpDown, ChevronDown, ChevronUp } from "@lucide/vue";
import { TableHead } from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface Props {
  /** Already-translated column label. */
  label: string;
  /** Sort field identifier handled by the parent comparator. */
  sortKey: string;
  activeField?: string;
  activeDir?: "asc" | "desc";
  align?: "left" | "right" | "center";
  /** Optional width utility(s), e.g. "w-32". */
  widthClass?: string;
}

const props = withDefaults(defineProps<Props>(), {
  activeField: "",
  activeDir: "asc",
  align: "left",
  widthClass: "",
});

const emit = defineEmits<{ sort: [field: string] }>();
const { t } = useI18n();

const isActive = computed(() => props.activeField === props.sortKey);

const ariaSort = computed<"none" | "ascending" | "descending">(() => {
  if (!isActive.value) return "none";
  return props.activeDir === "asc" ? "ascending" : "descending";
});

const ariaLabel = computed(() => {
  const base = t("common.sortByColumn", { column: props.label });
  if (!isActive.value) return base;
  return `${base} · ${
    props.activeDir === "asc" ? t("common.sortedAscending") : t("common.sortedDescending")
  }`;
});

const justifyClass = computed(
  () => ({ left: "justify-start", right: "justify-end", center: "justify-center" })[props.align]
);

function onClick() {
  emit("sort", props.sortKey);
}
</script>

<template>
  <TableHead :class="[widthClass, 'p-0']" :aria-sort="ariaSort">
    <button
      type="button"
      :data-testid="`sort-${sortKey}`"
      :data-sort-key="sortKey"
      :aria-label="ariaLabel"
      :class="
        cn(
          'flex w-full h-full items-center gap-1 px-3 py-1.5',
          'text-xs font-semibold uppercase tracking-wider',
          'transition-colors duration-150 hover:bg-muted/70',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/60',
          justifyClass,
          isActive ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'
        )
      "
      @click="onClick"
    >
      <span>{{ label }}</span>
      <ChevronUp
        v-if="isActive && activeDir === 'asc'"
        class="size-3.5 shrink-0"
        aria-hidden="true"
      />
      <ChevronDown
        v-else-if="isActive && activeDir === 'desc'"
        class="size-3.5 shrink-0"
        aria-hidden="true"
      />
      <ArrowUpDown v-else class="size-3 text-muted-foreground/40 shrink-0" aria-hidden="true" />
    </button>
  </TableHead>
</template>
