<script setup lang="ts">
import { useDebounceFn } from "@vueuse/core";
import { Search, X } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface FilterOption {
  label: string;
  value: string;
}

interface Props {
  searchPlaceholder?: string;
  typeFilterOptions?: FilterOption[];
  typeFilterLabel?: string;
  showFilters?: boolean;
  searchQuery?: string;
  resultCount?: number;
  totalCount?: number;
}

const emit = defineEmits<{
  search: [value: string];
  typeFilter: [value: string];
  clearFilters: [];
  "update:searchQuery": [value: string];
}>();

const { t } = useI18n();

const props = withDefaults(defineProps<Props>(), {
  searchPlaceholder: "",
  typeFilterOptions: () => [],
  typeFilterLabel: "",
  showFilters: true,
  searchQuery: "",
});

const internalSearchQuery = ref("");
const typeFilter = ref("all");

// Use prop value if provided, otherwise use internal state
const searchQueryValue = computed({
  get: () => (props.searchQuery !== undefined ? props.searchQuery : internalSearchQuery.value),
  set: (value) => {
    if (props.searchQuery !== undefined) {
      emit("update:searchQuery", value);
    } else {
      internalSearchQuery.value = value;
    }
  },
});

const hasActiveFilters = computed(() => {
  return searchQueryValue.value.trim() !== "" || typeFilter.value !== "all";
});

const handleSearch = useDebounceFn(() => {
  emit("search", searchQueryValue.value);
}, 300);

const handleTypeFilterChange = (value: string) => {
  typeFilter.value = value;
  emit("typeFilter", value);
};

const clearFilters = () => {
  searchQueryValue.value = "";
  typeFilter.value = "all";
  emit("clearFilters");
};

watch(searchQueryValue, () => handleSearch());
watch(typeFilter, () => handleTypeFilterChange(typeFilter.value));
</script>

<template>
  <div
    v-if="showFilters"
    class="flex w-full flex-col items-start gap-3 sm:flex-row sm:items-center sm:gap-2.5"
  >
    <div
      class="flex w-full min-w-0 flex-col items-start gap-3 sm:flex-1 sm:flex-row sm:items-center sm:gap-2.5"
    >
      <!-- Search input. The wrapper is a stacking context (isolate) so the icon's
           z-10 is scoped here — needed because the Input's backdrop-blur-sm
           establishes its own stacking context and would otherwise paint above
           (and blur) the icon. -->
      <div class="relative flex-1 min-w-0 w-full sm:w-auto group isolate">
        <Search
          class="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground transition-colors group-focus-within:text-foreground pointer-events-none z-10"
        />
        <Input
          v-model="searchQueryValue"
          :placeholder="searchPlaceholder || t('common.searchPlaceholder')"
          class="min-h-11 w-full border-border/40 bg-muted/15 hover:bg-muted/25 focus:bg-background pl-9 transition-colors duration-200"
        />
      </div>

      <!-- Type filter -->
      <Select v-if="typeFilterOptions && typeFilterOptions.length > 0" v-model="typeFilter">
        <SelectTrigger
          class="min-h-11 w-full shrink-0 border-border/40 bg-muted/15 hover:bg-muted/25 transition-colors duration-200 sm:w-40 focus:bg-background"
        >
          <SelectValue :placeholder="typeFilterLabel || t('common.filter')" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">{{ t("common.allTypes") }}</SelectItem>
          <SelectItem v-for="option in typeFilterOptions" :key="option.value" :value="option.value">
            {{ option.label }}
          </SelectItem>
        </SelectContent>
      </Select>

      <!-- Clear filters button -->
      <Button
        v-if="hasActiveFilters"
        variant="ghost"
        size="sm"
        @click="clearFilters"
        class="h-10 shrink-0"
      >
        <X class="w-4 h-4 mr-2" />
        {{ t("common.clearFilters") }}
      </Button>
    </div>

    <!-- Results count indicator -->
    <div
      v-if="resultCount !== undefined && totalCount !== undefined && resultCount !== totalCount"
      class="hidden sm:block text-xs text-muted-foreground shrink-0 tabular-nums"
      role="status"
      aria-live="polite"
    >
      {{ t("common.showingResults", { count: resultCount, total: totalCount }) }}
    </div>

    <!-- Slot for additional controls (e.g., view toggle) -->
    <div v-if="$slots.default" class="flex w-full justify-end sm:w-auto sm:shrink-0 sm:pl-1">
      <slot />
    </div>
  </div>
</template>
