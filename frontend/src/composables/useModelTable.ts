import { computed, ref, toValue, type MaybeRefOrGetter } from "vue";
import { useTableFilter } from "@/composables/useTableFilter";
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
  const sortField = ref<ModelSortField>("name");
  const sortDir = ref<"asc" | "desc">("asc");

  function onSort(field: string) {
    if (field === sortField.value) {
      sortDir.value = sortDir.value === "asc" ? "desc" : "asc";
    } else {
      sortField.value = field as ModelSortField;
      sortDir.value = "asc";
    }
  }

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

  /** Combined pipeline: base filters + provider filter + sorting. */
  const filteredAndSortedModels = computed(() => {
    let items = [...baseFilteredModels.value];

    if (selectedProviderFilter.value) {
      items = items.filter((model) =>
        model.providers?.some((p) => p.provider_name === selectedProviderFilter.value)
      );
    }

    const dir: 1 | -1 = sortDir.value === "asc" ? 1 : -1;
    const field = sortField.value;
    if (isCostSortField(field)) {
      items.sort(compareModelsByCost(COST_SORT_KEY[field], dir));
    } else if (field === "context_length") {
      items.sort(compareModelsByContextLength(dir));
    } else {
      items.sort(compareModelsByName(dir));
    }

    return items;
  });

  const clearFilters = () => {
    clearBaseFilters();
    selectedProviderFilter.value = "";
    sortField.value = "name";
    sortDir.value = "asc";
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
