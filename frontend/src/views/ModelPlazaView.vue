<script setup lang="ts">
import { Boxes, RefreshCw, X } from "@lucide/vue";
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import EmptyFilterResults from "@/components/common/EmptyFilterResults.vue";
import EmptyState from "@/components/common/EmptyState.vue";
import FilterBar from "@/components/common/FilterBar.vue";
import LoadingState from "@/components/common/LoadingState.vue";
import PageHeader from "@/components/common/PageHeader.vue";
import { CAPABILITY_META, CAPABILITY_ORDER } from "@/components/plaza/capabilities";
import PlazaModelListItem from "@/components/plaza/PlazaModelListItem.vue";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTableFilter } from "@/composables/useTableFilter";
import { catalogApi } from "@/services/api/catalog";
import type { ModelCapability, ModelCatalogEntry } from "@/types/schemas";

defineOptions({ name: "ModelPlazaView" });

const { t } = useI18n();

const models = ref<ModelCatalogEntry[]>([]);
const isLoading = ref(false);
const loadError = ref<string | null>(null);
const capabilityFilters = ref<ModelCapability[]>([]);

async function fetchCatalog() {
  isLoading.value = true;
  loadError.value = null;
  try {
    models.value = await catalogApi.getModels();
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : t("plaza.loadError");
  } finally {
    isLoading.value = false;
  }
}

onMounted(fetchCatalog);

const {
  searchQuery,
  typeFilter: tierFilter,
  filteredItems: filteredModels,
  clearFilters: clearBaseFilters,
} = useTableFilter(models, {
  searchFields: ["name", "description", "provider_names"],
  typeField: "quality_tier",
});

const clearFilters = () => {
  clearBaseFilters();
  capabilityFilters.value = [];
};

const availableTiers = computed(() => {
  const tiers = new Set<string>();
  for (const m of models.value) {
    if (m.quality_tier) tiers.add(m.quality_tier);
  }
  return Array.from(tiers).sort();
});

// Capability tags present in the catalog, in stable display order.
const availableCapabilities = computed(() => {
  const present = new Set<ModelCapability>();
  for (const m of models.value) {
    for (const cap of m.capabilities ?? []) present.add(cap);
  }
  return CAPABILITY_ORDER.filter((cap) => present.has(cap));
});

const hasActiveChipFilters = computed(
  () => tierFilter.value !== "all" || capabilityFilters.value.length > 0
);

function toggleCapability(cap: ModelCapability) {
  const idx = capabilityFilters.value.indexOf(cap);
  if (idx >= 0) {
    capabilityFilters.value.splice(idx, 1);
  } else {
    capabilityFilters.value.push(cap);
  }
}

// A model matches when it carries at least one of the selected capabilities
// (union semantics: "show me vision OR image-gen models").
const visibleModels = computed(() => {
  if (capabilityFilters.value.length === 0) return filteredModels.value;
  return filteredModels.value.filter((m) =>
    capabilityFilters.value.some((cap) => m.capabilities?.includes(cap))
  );
});

type SortKey = "name-asc" | "name-desc" | "context-desc" | "context-asc";
const sortKey = ref<SortKey>("name-asc");

const sortedModels = computed(() => {
  const items = [...visibleModels.value];
  switch (sortKey.value) {
    case "name-desc":
      items.sort((a, b) => b.name.localeCompare(a.name));
      break;
    case "context-desc":
    case "context-asc": {
      // Models without a context length always sort last, regardless of direction.
      const dir: 1 | -1 = sortKey.value === "context-asc" ? 1 : -1;
      items.sort((a, b) => {
        const aMissing = a.context_length == null ? 1 : 0;
        const bMissing = b.context_length == null ? 1 : 0;
        if (aMissing !== bMissing) return aMissing - bMissing;
        return (
          ((a.context_length ?? 0) - (b.context_length ?? 0)) * dir || a.name.localeCompare(b.name)
        );
      });
      break;
    }
    case "name-asc":
    default:
      items.sort((a, b) => a.name.localeCompare(b.name));
      break;
  }
  return items;
});
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar px-4 sm:px-6 py-4">
        <PageHeader :title="t('plaza.title')" :description="t('plaza.description')" :icon="Boxes">
          <template #actions>
            <Button
              variant="ghost"
              size="icon"
              :disabled="isLoading"
              :aria-label="t('common.refresh')"
              :title="t('common.refresh')"
              @click="fetchCatalog"
            >
              <RefreshCw class="h-4 w-4" :class="{ 'animate-spin': isLoading }" />
            </Button>
          </template>
        </PageHeader>
      </header>
    </template>

    <!-- Toolbar band (flush): search + sort -->
    <div v-if="models.length > 0 || isLoading" class="config-toolbar px-4 sm:px-6 py-3">
      <FilterBar
        v-model:search-query="searchQuery"
        :search-placeholder="t('plaza.searchPlaceholder')"
        :result-count="sortedModels.length"
        :total-count="models.length"
        @clear-filters="clearFilters"
      >
        <Select v-model="sortKey">
          <SelectTrigger
            class="min-h-11 w-full sm:w-52 shrink-0 border-border/40 bg-muted/15 hover:bg-muted/25 transition-colors duration-200 focus:bg-background"
            :aria-label="t('plaza.sortLabel')"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="name-asc">{{ t("plaza.sortNameAsc") }}</SelectItem>
            <SelectItem value="name-desc">{{ t("plaza.sortNameDesc") }}</SelectItem>
            <SelectItem value="context-desc">{{ t("plaza.sortContextDesc") }}</SelectItem>
            <SelectItem value="context-asc">{{ t("plaza.sortContextAsc") }}</SelectItem>
          </SelectContent>
        </Select>
      </FilterBar>
    </div>

    <!-- Tier + capability quick-filter chips (flush sub-row) -->
    <div
      v-if="models.length > 0 && (availableTiers.length > 0 || availableCapabilities.length > 0)"
      class="flex-none bg-background border-b border-border/60 px-4 sm:px-6 py-2.5"
    >
      <div class="flex flex-wrap items-center gap-2 overflow-x-auto scrollbar-none">
        <template v-if="availableTiers.length > 0">
          <span class="mr-1 text-[11px] text-muted-foreground shrink-0">
            {{ t("plaza.tierFilterLabel") }}:
          </span>
          <Button
            :variant="tierFilter === 'all' ? 'default' : 'outline'"
            size="sm"
            class="h-7 rounded-full px-2.5 text-[11px]"
            :aria-pressed="tierFilter === 'all'"
            @click="tierFilter = 'all'"
          >
            {{ t("plaza.filterAll") }}
          </Button>
          <Button
            v-for="tier in availableTiers"
            :key="tier"
            :variant="tierFilter === tier ? 'default' : 'outline'"
            size="sm"
            class="h-7 rounded-full px-2.5 text-[11px]"
            :aria-pressed="tierFilter === tier"
            @click="tierFilter = tier"
          >
            {{ tier }}
          </Button>
        </template>
        <template v-if="availableCapabilities.length > 0">
          <span
            v-if="availableTiers.length > 0"
            class="mx-1 h-4 w-px bg-border/60"
            aria-hidden="true"
          />
          <span class="mr-1 text-[11px] text-muted-foreground shrink-0">
            {{ t("plaza.capabilityFilterLabel") }}:
          </span>
          <Button
            v-for="cap in availableCapabilities"
            :key="cap"
            :variant="capabilityFilters.includes(cap) ? 'default' : 'outline'"
            size="sm"
            class="h-7 rounded-full px-2.5 text-[11px]"
            :aria-pressed="capabilityFilters.includes(cap)"
            @click="toggleCapability(cap)"
          >
            <component :is="CAPABILITY_META[cap].icon" class="size-3 mr-1" />
            {{ t(CAPABILITY_META[cap].labelKey) }}
          </Button>
        </template>
        <Button
          v-if="hasActiveChipFilters"
          variant="ghost"
          size="sm"
          class="h-7 px-2 text-[11px] text-muted-foreground hover:text-foreground"
          @click="clearFilters"
        >
          <X class="size-3 mr-1" />
          {{ t("common.clearFilters") }}
        </Button>
      </div>
    </div>

    <!-- Content area -->
    <div class="config-content">
      <div
        v-if="isLoading && models.length === 0"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <LoadingState />
      </div>
      <div
        v-else-if="loadError"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <EmptyState :text="loadError" :show-retry="true" @retry="fetchCatalog" />
      </div>
      <div
        v-else-if="models.length === 0"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <EmptyState :icon="Boxes" :text="t('plaza.emptyTitle')">
          <template #description>
            <p class="text-muted-foreground text-sm max-w-md">
              {{ t("plaza.emptyDescription") }}
            </p>
          </template>
        </EmptyState>
      </div>
      <div v-else class="config-scroll">
        <EmptyFilterResults v-if="sortedModels.length === 0" @clear="clearFilters" />
        <div v-else class="config-list list-stagger">
          <PlazaModelListItem v-for="model in sortedModels" :key="model.name" :model="model" />
        </div>
      </div>
    </div>
  </AppLayout>
</template>
