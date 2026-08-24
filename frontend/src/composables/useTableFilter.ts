import { type ComputedRef, computed, type MaybeRefOrGetter, type Ref, ref, toValue } from "vue";

interface UseTableFilterOptions {
  initialSearch?: string;
  initialTypeFilter?: string;
  searchFields?: string[];
  typeField?: string;
}

export function useTableFilter<T>(
  items: MaybeRefOrGetter<T[]>,
  options: UseTableFilterOptions = {}
) {
  const {
    initialSearch = "",
    initialTypeFilter = "all",
    searchFields = ["name"],
    typeField = "type",
  } = options;

  const searchQuery: Ref<string> = ref(initialSearch);
  const typeFilter: Ref<string> = ref(initialTypeFilter);

  const filteredItems: ComputedRef<T[]> = computed(() => {
    const itemsValue = toValue(items);
    let result = [...itemsValue];

    if (searchQuery.value.trim()) {
      const query = searchQuery.value.toLowerCase();
      result = result.filter((item) =>
        searchFields.some((field) => {
          const value = (item as Record<string, unknown>)[field];
          return String(value ?? "")
            .toLowerCase()
            .includes(query);
        })
      );
    }

    if (typeFilter.value !== "all" && typeField) {
      result = result.filter(
        (item) => String((item as Record<string, unknown>)[typeField] ?? "") === typeFilter.value
      );
    }

    return result;
  });

  const hasActiveFilters: ComputedRef<boolean> = computed(
    () => searchQuery.value.trim() !== "" || typeFilter.value !== "all"
  );

  const clearFilters = () => {
    searchQuery.value = "";
    typeFilter.value = "all";
  };

  return {
    searchQuery,
    typeFilter,
    filteredItems,
    hasActiveFilters,
    clearFilters,
  };
}
