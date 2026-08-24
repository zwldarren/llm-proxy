<script setup lang="ts">
import { Database, Edit, Plus, Server, Trash2 } from "@lucide/vue";
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import ConfirmDialog from "@/components/common/ConfirmDialog.vue";
import EmptyState from "@/components/common/EmptyState.vue";
import EmptyFilterResults from "@/components/common/EmptyFilterResults.vue";
import FilterBar from "@/components/common/FilterBar.vue";
import ContentSkeleton from "@/components/common/ContentSkeleton.vue";
import EmptyTableRow from "@/components/common/EmptyTableRow.vue";
import PageHeader from "@/components/common/PageHeader.vue";
import SortableHead from "@/components/common/SortableHead.vue";
import ProviderDialog from "@/components/config/ProviderDialog.vue";
import ViewToggle from "@/components/common/ViewToggle.vue";
import { ProviderListItem } from "@/components/providers";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useErrorHandler } from "@/composables/useErrorHandler";
import { useProviderTypes } from "@/composables/useProviderTypes";
import { useTableFilter } from "@/composables/useTableFilter";
import { useViewMode } from "@/composables/useViewMode";
import { useProviderStore } from "@/stores/providers";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import type { ProviderRead } from "@/types/schemas";

defineOptions({ name: "ProvidersView" });

const { t } = useI18n();
const { handleDeleteError } = useErrorHandler();
const providerStore = useProviderStore();
const {
  typeOptions,
  providerIconUrl,
  providerIsMono,
  ensureLoaded: ensureProviderTypesLoaded,
} = useProviderTypes();

const failedIcons = ref(new Set<number>());

const showCreateDialog = ref(false);
const showDeleteDialog = ref(false);
const deletingProviderName = ref("");
const selectedProvider = ref<ProviderRead | null>(null);
const viewMode = useViewMode(STORAGE_KEYS.PROVIDERS_VIEW_MODE);

const providers = computed(() => providerStore.providers);
const isLoading = computed(() => providerStore.loading && !providerStore.ready);

// The list filter deliberately omits nanogpt, which only the create/edit
// dialog offers (pre-existing behavior, preserved by the shared list).
const providerTypes = computed(() => typeOptions.value.filter(({ value }) => value !== "nanogpt"));

type SortField = "name" | "type";
const sortField = ref<SortField>("name");
const sortDir = ref<"asc" | "desc">("asc");

function onSort(field: string) {
  if (field === sortField.value) {
    sortDir.value = sortDir.value === "asc" ? "desc" : "asc";
  } else {
    sortField.value = field as SortField;
    sortDir.value = "asc";
  }
}

const sortedProviders = computed(() => {
  const items = [...(filteredProviders.value as unknown as ProviderRead[])];
  const dir: 1 | -1 = sortDir.value === "asc" ? 1 : -1;
  switch (sortField.value) {
    case "type":
      items.sort((a, b) => (a.type.localeCompare(b.type) || a.name.localeCompare(b.name)) * dir);
      break;
    case "name":
    default:
      items.sort((a, b) => a.name.localeCompare(b.name) * dir);
      break;
  }
  return items;
});

const {
  searchQuery,
  typeFilter,
  filteredItems: filteredProviders,
  clearFilters: clearBaseFilters,
} = useTableFilter(providers, {
  searchFields: ["name", "type", "base_url"],
  typeField: "type",
});

const clearFilters = () => {
  clearBaseFilters();
  sortField.value = "name";
  sortDir.value = "asc";
};

const openCreateDialog = () => {
  selectedProvider.value = null;
  showCreateDialog.value = true;
};

const openEditDialog = (provider: ProviderRead) => {
  selectedProvider.value = provider;
  showCreateDialog.value = true;
};

const onSaved = async () => {
  await providerStore.fetchProviders(true);
};

const openDeleteDialog = (name: string) => {
  deletingProviderName.value = name;
  showDeleteDialog.value = true;
};

const confirmDelete = async () => {
  const name = deletingProviderName.value;
  showDeleteDialog.value = false;
  try {
    await providerStore.deleteProvider(name);
    toast.success(t("common.success"), {
      description: t("providers.deleteSuccess"),
    });
  } catch (e) {
    handleDeleteError(e);
  }
};

onMounted(() => {
  providerStore.fetchProviders();
  void ensureProviderTypesLoaded();
});
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar px-4 sm:px-6 py-4">
        <PageHeader
          :title="t('providers.title')"
          :description="t('providers.description')"
          :icon="Server"
        >
          <template #actions>
            <Button @click="openCreateDialog" class="btn-action">
              <Plus class="w-4 h-4 mr-2" />
              {{ t("providers.addProvider") }}
            </Button>
          </template>
        </PageHeader>
      </header>
    </template>

    <!-- Toolbar band (flush) -->
    <div v-if="providers.length > 0 || isLoading" class="config-toolbar px-4 sm:px-6 py-3">
      <FilterBar
        :search-placeholder="t('common.searchPlaceholder')"
        :type-filter-options="providerTypes"
        :type-filter-label="t('providers.type')"
        :result-count="sortedProviders.length"
        :total-count="providers.length"
        @search="searchQuery = $event"
        @type-filter="typeFilter = $event"
        @clear-filters="clearFilters"
      >
        <ViewToggle v-model="viewMode" />
      </FilterBar>
    </div>

    <!-- Content area -->
    <div class="config-content">
      <div
        v-if="isLoading && providers.length === 0"
        class="h-full flex items-start justify-center animate-fade-in px-6"
      >
        <ContentSkeleton />
      </div>
      <div
        v-else-if="providers.length === 0"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <EmptyState
          :text="t('providers.noProviders')"
          :show-cta="true"
          :cta-text="t('providers.addProvider')"
          @click="openCreateDialog"
        />
      </div>
      <template v-else>
        <!-- Table view (default) -->
        <Table
          v-if="viewMode === 'table'"
          class="table-modern"
          container-class="h-full border-0 bg-transparent rounded-none overflow-x-auto"
        >
          <TableHeader class="config-thead">
            <TableRow class="bg-transparent hover:bg-transparent hover:border-l-transparent">
              <TableHead class="w-12"></TableHead>
              <SortableHead
                :label="t('providers.name')"
                sort-key="name"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <SortableHead
                :label="t('providers.type')"
                sort-key="type"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <TableHead>{{ t("providers.baseUrl") }}</TableHead>
              <TableHead class="w-24 text-right">{{ t("common.actions") }}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody class="row-stagger">
            <TableRow v-for="provider in sortedProviders" :key="provider.id" class="group">
              <TableCell>
                <div
                  :class="[
                    'w-9 h-9 rounded-lg flex items-center justify-center overflow-hidden',
                    providerIconUrl(provider) ? 'bg-card border border-border' : 'bg-muted',
                  ]"
                >
                  <img
                    v-if="providerIconUrl(provider) && !failedIcons.has(provider.id)"
                    :src="providerIconUrl(provider)!"
                    :alt="provider.name"
                    :class="[
                      providerIsMono(provider) && !provider.icon_url ? 'icon-mono' : null,
                      'w-5 h-5 object-contain',
                    ]"
                    loading="lazy"
                    @error="failedIcons.add(provider.id)"
                  />
                  <Database v-else class="w-4 h-4 text-muted-foreground" />
                </div>
              </TableCell>
              <TableCell class="font-medium">
                {{ provider.name }}
              </TableCell>
              <TableCell>
                <div class="flex items-center gap-1.5">
                  <Badge variant="secondary" class="text-[11px] uppercase font-medium">
                    {{ provider.type }}
                  </Badge>
                  <Badge
                    v-if="
                      provider.type === 'gemini' &&
                      provider.provider_metadata?.api_variant === 'interactions'
                    "
                    variant="outline"
                    class="text-[11px] font-medium text-muted-foreground"
                  >
                    {{ t("providers.interactionsBadge") }}
                  </Badge>
                </div>
              </TableCell>
              <TableCell class="max-w-[280px] overflow-hidden">
                <span
                  class="font-mono text-xs text-muted-foreground truncate block break-all"
                  :title="provider.base_url || t('labels.default')"
                >
                  {{ provider.base_url || t("labels.default") }}
                </span>
              </TableCell>
              <TableCell class="text-right">
                <div
                  class="flex items-center justify-end gap-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity"
                >
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9"
                    :disabled="isLoading"
                    :aria-label="t('common.edit')"
                    @click="openEditDialog(provider)"
                  >
                    <Edit class="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
                    :disabled="isLoading"
                    :aria-label="t('common.delete')"
                    @click="openDeleteDialog(provider.name)"
                  >
                    <Trash2 class="w-4 h-4" />
                  </Button>
                </div>
              </TableCell>
            </TableRow>
            <EmptyTableRow v-if="sortedProviders.length === 0" :colspan="5" @clear="clearFilters" />
          </TableBody>
        </Table>

        <!-- List view -->
        <div v-else class="config-scroll">
          <EmptyFilterResults v-if="sortedProviders.length === 0" @clear="clearFilters" />
          <div v-if="sortedProviders.length > 0" class="config-list list-stagger">
            <ProviderListItem
              v-for="provider in sortedProviders"
              :key="provider.id"
              :provider="provider"
              :is-loading="isLoading"
              @edit="openEditDialog(provider)"
              @delete="openDeleteDialog(provider.name)"
            />
          </div>
        </div>
      </template>
    </div>

    <!-- Delete Confirmation Dialog -->
    <ConfirmDialog
      v-model:open="showDeleteDialog"
      :title="t('dialogs.confirmDeleteTitle')"
      :description="t('dialogs.confirmDelete', { name: deletingProviderName })"
      :confirm-text="t('common.delete')"
      :cancel-text="t('common.cancel')"
      :loading="isLoading"
      @confirm="confirmDelete"
    />

    <!-- Create/Edit Dialog -->
    <ProviderDialog v-model:open="showCreateDialog" :provider="selectedProvider" @saved="onSaved" />
  </AppLayout>
</template>
