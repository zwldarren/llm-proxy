import { computed, ref, toValue, type MaybeRefOrGetter } from "vue";
import { useTableFilter } from "@/composables/useTableFilter";
import { useTableSort } from "@/composables/useTableSort";
import { compareModelsByCost, type ModelCostKey } from "@/utils/modelPricing";
import { compareModelsByContextLength, compareModelsByName } from "@/utils/modelSort";
import type { ModelRead } from "@/types/schemas";

/**
 * Table state for the models admin view: search, provider quick-filter, and
 * sortable columns. Cost columns order by the shared pricing rule
 * (`compareModelsByCost`), so the sort order always matches the range the
 * pricing cell displays.
 */

export type ModelSortField =
  "name" | "input_cost" | "output_cost" | "cached_read" | "context_length";

type CostSortField = Exclude<ModelSortField, "name" | "context_length">;

const COST_SORT_KEY: Record<CostSortField, ModelCostKey> = {
  input_cost: "input_cost_per_1m",
  output_cost: "output_cost_per_1m",
  cached_read: "cached_read_cost_per_1m",
};

function isCostSortField(field: ModelSortField): field is CostSortField {
  return field in COST_SORT_KEY;
}

export function useModelTable(models: MaybeRefOrGetter<ModelRead[]>) {
  const {
    searchQuery,
    filteredItems: baseFilteredModels,
    clearFilters: clearBaseFilters,
  } = useTableFilter(models, {
    searchFields: ["name", "description"],
  });

  const selectedProviderFilter = ref<string>("");

  /** All unique provider names across the listed models, for the filter chips. */
  const availableProviders = computed(() => {
    const providerSet = new Set<string>();
    for (const model of toValue(models)) {
      for (const p of model.providers ?? []) {
        providerSet.add(p.provider_name);
      }
    }
    return Array.from(providerSet).sort();
  });

  /** Base filters plus the provider quick-filter; sorting happens below. */
  const providerFilteredModels = computed(() => {
    const items = [...baseFilteredModels.value];
    if (!selectedProviderFilter.value) return items;
    return items.filter((model) =>
      model.providers?.some((p) => p.provider_name === selectedProviderFilter.value)
    );
  });

  const {
    sortField,
    sortDir,
    onSort,
    sortedItems: filteredAndSortedModels,
    resetSort,
  } = useTableSort<ModelRead, ModelSortField>({
    items: providerFilteredModels,
    initialField: "name",
    compare: (field, a, b, dir) => {
      if (isCostSortField(field)) {
        return compareModelsByCost(COST_SORT_KEY[field], dir)(a, b);
      }
      if (field === "context_length") return compareModelsByContextLength(dir)(a, b);
      return compareModelsByName(dir)(a, b);
    },
  });

  const clearFilters = () => {
    clearBaseFilters();
    selectedProviderFilter.value = "";
    resetSort();
  };

  const handleProviderFilter = (provider: string) => {
    selectedProviderFilter.value = selectedProviderFilter.value === provider ? "" : provider;
  };

  return {
    searchQuery,
    sortField,
    sortDir,
    onSort,
    selectedProviderFilter,
    availableProviders,
    filteredAndSortedModels,
    clearFilters,
    handleProviderFilter,
  };
}
