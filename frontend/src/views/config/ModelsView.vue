<script setup lang="ts">
import {
  ArrowUpDown,
  Box,
  Check,
  ChevronDown,
  ChevronUp,
  Edit,
  ImageOff,
  Plus,
  RefreshCw,
  Settings,
  Trash2,
  X,
} from "@lucide/vue";
import { computed, onMounted, ref, watch } from "vue";
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
import ViewToggle from "@/components/common/ViewToggle.vue";
import { ModelListItem, ModelPricingCell, CapabilityToggle } from "@/components/models";
import PricingSyncDialog from "@/components/models/PricingSyncDialog.vue";
import { CAPABILITY_META, deriveModelCapabilities } from "@/components/plaza/capabilities";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import SheetDescription from "@/components/ui/sheet/SheetDescription.vue";
import SheetTitle from "@/components/ui/sheet/SheetTitle.vue";
import { NumberInput } from "@/components/ui/number-input";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useErrorHandler } from "@/composables/useErrorHandler";
import { useTableFilter } from "@/composables/useTableFilter";
import { useViewMode } from "@/composables/useViewMode";
import { useModelStore } from "@/stores/models";
import { useProviderStore } from "@/stores/providers";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import { ROUTING_MODES } from "@/constants/model";

import type { ModelCreate, ModelRead, ModelProviderMapping } from "@/types/schemas";
import type { ParameterOverridesConfig } from "@/types/parameterOverrides";
import ParameterOverridesBuilder from "@/components/config/ParameterOverridesBuilder.vue";
import ProviderModelSelector from "@/components/common/ProviderModelSelector.vue";
import { getIconUrl, isMonoIcon } from "@/utils/icons";
import { formatContextLength } from "@/utils/format";

defineOptions({ name: "ModelsView" });

const { t } = useI18n();
const { handleSaveError, handleDeleteError } = useErrorHandler();
const modelStore = useModelStore();
const providerStore = useProviderStore();

const showCreateDialog = ref(false);
const showPricingSyncDialog = ref(false);
const showDeleteDialog = ref(false);
const deletingModelName = ref("");
const isEditing = ref(false);
const editingModelName = ref("");
const isSaving = ref(false);
const failedIcons = ref(new Set<string>());
const viewMode = useViewMode(STORAGE_KEYS.MODELS_VIEW_MODE);

const models = computed(() => modelStore.models);
const providers = computed(() => providerStore.providers);
const isLoading = computed(() => modelStore.loading && !modelStore.ready);

type SortField = "name" | "input_cost" | "output_cost";
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

const {
  searchQuery,
  filteredItems: baseFilteredModels,
  clearFilters: clearBaseFilters,
} = useTableFilter(models, {
  searchFields: ["name"],
});

const selectedProviderFilter = ref<string>("");

// Get all unique provider names from models
const availableProviders = computed(() => {
  const providerSet = new Set<string>();
  for (const model of models.value) {
    for (const p of model.providers ?? []) {
      providerSet.add(p.provider_name);
    }
  }
  return Array.from(providerSet).sort();
});

// Combined filter: base filters + provider filter + sorting
const filteredAndSortedModels = computed(() => {
  let items = [...baseFilteredModels.value] as unknown as ModelRead[];

  // Apply provider filter
  if (selectedProviderFilter.value) {
    items = items.filter((model) =>
      model.providers?.some((p) => p.provider_name === selectedProviderFilter.value)
    );
  }

  // Apply sorting
  const dir: 1 | -1 = sortDir.value === "asc" ? 1 : -1;
  switch (sortField.value) {
    case "input_cost":
    case "output_cost": {
      const key = sortField.value === "input_cost" ? "input_cost_per_1m" : "output_cost_per_1m";
      items.sort((a, b) => {
        const va = effectiveCost(a, key);
        const vb = effectiveCost(b, key);
        if (va == null && vb == null) return 0;
        if (va == null) return 1; // models without pricing always sort last
        if (vb == null) return -1;
        return (va - vb) * dir;
      });
      break;
    }
    case "name":
    default:
      items.sort((a, b) => a.name.localeCompare(b.name) * dir);
      break;
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

const hasRoutingInfo = (model: ModelRead): boolean => Boolean(model.auto_eligible);

/** Effective price: lowest configured value across provider overrides + model default. */
function effectiveCost(
  model: ModelRead,
  key: "input_cost_per_1m" | "output_cost_per_1m"
): number | null {
  const vals: number[] = [];
  for (const p of model.providers ?? []) {
    const v = p[key];
    if (v != null) vals.push(v);
  }
  const d = model[key];
  if (d != null) vals.push(d);
  return vals.length > 0 ? Math.min(...vals) : null;
}

/** In/out quick-sort buttons rendered inside the pricing column header. */
const pricingSortOptions = computed(() => [
  { key: "input_cost" as SortField, label: t("models.inputShort") },
  { key: "output_cost" as SortField, label: t("models.outputShort") },
]);

const newModel = ref<ModelCreate>({
  name: "",
  providers: [],
  input_cost_per_1m: null,
  output_cost_per_1m: null,
  cached_read_cost_per_1m: null,
  cached_write_cost_per_1m: null,
  audio_input_cost_per_1m: null,
  audio_output_cost_per_1m: null,
  image_input_cost_per_1m: null,
  cost_per_image: null,
  audio_cost_per_minute: null,
  tts_cost_per_1m_chars: null,
  web_search_cost_per_1k: null,
  icon_url: null,
  parameter_overrides: null,
  auto_eligible: false,
  quality_tier: null,
  routing_assignments: null,
  supports_images: false,
  supports_image_generation: false,
  supports_tts: false,
  supports_stt: false,
  supports_embedding: false,
  supports_realtime: false,
  description: null,
  homepage_url: null,
  context_length: null,
});

const parameterOverrides = ref<ParameterOverridesConfig>({});
const showProviderEditDialog = ref(false);
const editingProviderIndex = ref<number | null>(null);
const editingProviderData = ref<ModelProviderMapping>({
  provider_name: "",
  priority: 0,
  provider_model_name: "",
});

onMounted(() => {
  modelStore.fetchModels();
  providerStore.fetchProviders();
});

watch(
  () => newModel.value.auto_eligible,
  (val) => {
    if (!val) {
      newModel.value.quality_tier = null;
      newModel.value.routing_assignments = null;
    }
  }
);

const iconPreviewFailed = ref(false);

// Textarea binds a non-null string; proxy null <-> empty for the description field.
const descriptionModel = computed({
  get: () => newModel.value.description ?? "",
  set: (v: string) => {
    newModel.value.description = v || null;
  },
});

watch([() => newModel.value.icon_url, () => newModel.value.name], () => {
  iconPreviewFailed.value = false;
});

const openCreateDialog = () => {
  isEditing.value = false;
  editingModelName.value = "";
  iconPreviewFailed.value = false;
  newModel.value = {
    name: "",
    providers: [
      {
        provider_name: "",
        priority: 0,
        provider_model_name: "",
      },
    ],
    input_cost_per_1m: null,
    output_cost_per_1m: null,
    cached_read_cost_per_1m: null,
    cached_write_cost_per_1m: null,
    audio_input_cost_per_1m: null,
    audio_output_cost_per_1m: null,
    image_input_cost_per_1m: null,
    cost_per_image: null,
    audio_cost_per_minute: null,
    tts_cost_per_1m_chars: null,
    web_search_cost_per_1k: null,
    icon_url: null,
    parameter_overrides: null,
    auto_eligible: false,
    quality_tier: null,
    routing_assignments: null,
    supports_images: false,
    supports_image_generation: false,
    supports_tts: false,
    supports_stt: false,
    supports_embedding: false,
    supports_realtime: false,
    description: null,
    homepage_url: null,
    context_length: null,
  };
  parameterOverrides.value = {};
  showCreateDialog.value = true;
};

const openEditDialog = (model: ModelRead) => {
  isEditing.value = true;
  editingModelName.value = model.name;
  iconPreviewFailed.value = false;
  newModel.value = {
    name: model.name,
    providers: model.providers?.length
      ? [...model.providers]
          .sort((a, b) => (b.priority || 0) - (a.priority || 0))
          .map((p) => ({
            provider_name: p.provider_name,
            priority: p.priority || 0,
            provider_model_name: p.provider_model_name,
            input_cost_per_1m: p.input_cost_per_1m ?? null,
            output_cost_per_1m: p.output_cost_per_1m ?? null,
            cached_read_cost_per_1m: p.cached_read_cost_per_1m ?? null,
            cached_write_cost_per_1m: p.cached_write_cost_per_1m ?? null,
            audio_input_cost_per_1m: p.audio_input_cost_per_1m ?? null,
            audio_output_cost_per_1m: p.audio_output_cost_per_1m ?? null,
            image_input_cost_per_1m: p.image_input_cost_per_1m ?? null,
            cost_per_image: p.cost_per_image ?? null,
            audio_cost_per_minute: p.audio_cost_per_minute ?? null,
            tts_cost_per_1m_chars: p.tts_cost_per_1m_chars ?? null,
            web_search_cost_per_1k: p.web_search_cost_per_1k ?? null,
            parameter_overrides: p.parameter_overrides ?? {},
          }))
      : [],
    input_cost_per_1m: model.input_cost_per_1m,
    output_cost_per_1m: model.output_cost_per_1m,
    cached_read_cost_per_1m: model.cached_read_cost_per_1m ?? null,
    cached_write_cost_per_1m: model.cached_write_cost_per_1m ?? null,
    audio_input_cost_per_1m: model.audio_input_cost_per_1m ?? null,
    audio_output_cost_per_1m: model.audio_output_cost_per_1m ?? null,
    image_input_cost_per_1m: model.image_input_cost_per_1m ?? null,
    cost_per_image: model.cost_per_image ?? null,
    audio_cost_per_minute: model.audio_cost_per_minute ?? null,
    tts_cost_per_1m_chars: model.tts_cost_per_1m_chars ?? null,
    web_search_cost_per_1k: model.web_search_cost_per_1k ?? null,
    icon_url: model.icon_url,
    parameter_overrides: model.parameter_overrides ?? null,
    auto_eligible: model.auto_eligible ?? false,
    quality_tier: model.quality_tier ?? null,
    supports_images: model.supports_images ?? false,
    supports_image_generation: model.supports_image_generation ?? false,
    supports_tts: model.supports_tts ?? false,
    supports_stt: model.supports_stt ?? false,
    supports_embedding: model.supports_embedding ?? false,
    supports_realtime: model.supports_realtime ?? false,
    routing_assignments: model.routing_assignments ?? null,
    description: model.description ?? null,
    homepage_url: model.homepage_url ?? null,
    context_length: model.context_length ?? null,
  };
  parameterOverrides.value = model.parameter_overrides ?? {};
  showCreateDialog.value = true;
};

const addProvider = () => {
  newModel.value.providers.push({
    provider_name: "",
    priority: 0,
    provider_model_name: "",
    input_cost_per_1m: null,
    output_cost_per_1m: null,
    cached_read_cost_per_1m: null,
    cached_write_cost_per_1m: null,
    audio_input_cost_per_1m: null,
    audio_output_cost_per_1m: null,
    image_input_cost_per_1m: null,
    cost_per_image: null,
    audio_cost_per_minute: null,
    tts_cost_per_1m_chars: null,
    web_search_cost_per_1k: null,
    parameter_overrides: {},
  });
};

const removeProvider = (index: number) => {
  newModel.value.providers.splice(index, 1);
};

const openProviderEditDialog = (index: number) => {
  editingProviderIndex.value = index;
  editingProviderData.value = { ...newModel.value.providers[index]! };
  showProviderEditDialog.value = true;
};

const saveProviderEdit = () => {
  if (editingProviderIndex.value !== null) {
    newModel.value.providers[editingProviderIndex.value] = { ...editingProviderData.value };
  }
  showProviderEditDialog.value = false;
  editingProviderIndex.value = null;
};

const saveModel = async () => {
  if (!newModel.value.name.trim()) {
    toast.error(t("models.nameRequired"));
    return;
  }

  if (newModel.value.providers.length === 0) {
    toast.error(t("models.atLeastOneProvider"));
    return;
  }

  if (newModel.value.providers.some((p) => !p.provider_name)) {
    toast.error(t("models.providerNameRequired"));
    return;
  }

  if (newModel.value.providers.some((p) => !p.provider_model_name?.trim())) {
    toast.error(t("models.providerModelNameRequired"));
    return;
  }

  if (newModel.value.auto_eligible && !newModel.value.quality_tier) {
    toast.error(t("models.qualityTierRequired"));
    return;
  }

  isSaving.value = true;
  try {
    const modelData = { ...newModel.value };
    if (!modelData.auto_eligible) {
      modelData.quality_tier = null;
      modelData.routing_assignments = null;
    }

    // Clean up provider data - preserve all fields
    const cleanedProviders = modelData.providers.map((p) => ({
      provider_name: p.provider_name,
      priority: p.priority,
      provider_model_name: p.provider_model_name,
      input_cost_per_1m: p.input_cost_per_1m ?? null,
      output_cost_per_1m: p.output_cost_per_1m ?? null,
      cached_read_cost_per_1m: p.cached_read_cost_per_1m ?? null,
      cached_write_cost_per_1m: p.cached_write_cost_per_1m ?? null,
      audio_input_cost_per_1m: p.audio_input_cost_per_1m ?? null,
      audio_output_cost_per_1m: p.audio_output_cost_per_1m ?? null,
      image_input_cost_per_1m: p.image_input_cost_per_1m ?? null,
      cost_per_image: p.cost_per_image ?? null,
      audio_cost_per_minute: p.audio_cost_per_minute ?? null,
      tts_cost_per_1m_chars: p.tts_cost_per_1m_chars ?? null,
      web_search_cost_per_1k: p.web_search_cost_per_1k ?? null,
      parameter_overrides: p.parameter_overrides ?? {},
    }));

    // Determine parameter overrides - use empty object if no overrides configured
    const overrides =
      Object.keys(parameterOverrides.value).length > 0 ? parameterOverrides.value : null;

    if (isEditing.value) {
      const newName = modelData.name.trim();
      await modelStore.updateModel(editingModelName.value, {
        name: newName,
        providers: cleanedProviders,
        input_cost_per_1m: modelData.input_cost_per_1m,
        output_cost_per_1m: modelData.output_cost_per_1m,
        cached_read_cost_per_1m: modelData.cached_read_cost_per_1m,
        cached_write_cost_per_1m: modelData.cached_write_cost_per_1m,
        audio_input_cost_per_1m: modelData.audio_input_cost_per_1m,
        audio_output_cost_per_1m: modelData.audio_output_cost_per_1m,
        image_input_cost_per_1m: modelData.image_input_cost_per_1m,
        cost_per_image: modelData.cost_per_image,
        audio_cost_per_minute: modelData.audio_cost_per_minute,
        tts_cost_per_1m_chars: modelData.tts_cost_per_1m_chars,
        web_search_cost_per_1k: modelData.web_search_cost_per_1k,
        icon_url: modelData.icon_url,
        parameter_overrides: overrides,
        auto_eligible: modelData.auto_eligible,
        quality_tier: modelData.quality_tier || null,
        routing_assignments: modelData.routing_assignments,
        supports_images: modelData.supports_images ?? false,
        supports_image_generation: modelData.supports_image_generation ?? false,
        supports_tts: modelData.supports_tts ?? false,
        supports_stt: modelData.supports_stt ?? false,
        supports_embedding: modelData.supports_embedding ?? false,
        supports_realtime: modelData.supports_realtime ?? false,
        description: modelData.description ?? null,
        homepage_url: modelData.homepage_url ?? null,
        context_length: modelData.context_length ?? null,
      });
    } else {
      await modelStore.createModel({
        ...modelData,
        providers: cleanedProviders,
        parameter_overrides: overrides,
      });
    }

    showCreateDialog.value = false;
    toast.success(t("common.success"), {
      description: isEditing.value ? t("models.updateSuccess") : t("models.createSuccess"),
    });
  } catch (e) {
    handleSaveError(e);
  } finally {
    isSaving.value = false;
  }
};

const openDeleteDialog = (name: string) => {
  deletingModelName.value = name;
  showDeleteDialog.value = true;
};

const confirmDelete = async () => {
  const name = deletingModelName.value;
  showDeleteDialog.value = false;
  try {
    await modelStore.deleteModel(name);
    toast.success(t("common.success"), {
      description: t("models.deleteSuccess"),
    });
  } catch (e) {
    handleDeleteError(e);
  }
};
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar px-4 sm:px-6 py-4">
        <PageHeader :title="t('models.title')" :description="t('models.description')" :icon="Box">
          <template #actions>
            <Button variant="outline" @click="showPricingSyncDialog = true">
              <RefreshCw class="w-4 h-4 mr-2" />
              {{ t("models.pricingSync.trigger") }}
            </Button>
            <Button @click="openCreateDialog" class="btn-action">
              <Plus class="w-4 h-4 mr-2" />
              {{ t("models.addModel") }}
            </Button>
          </template>
        </PageHeader>
      </header>
    </template>

    <!-- Toolbar band (flush) -->
    <div v-if="models.length > 0 || isLoading" class="config-toolbar px-4 sm:px-6 py-3">
      <FilterBar
        v-model:search-query="searchQuery"
        :search-placeholder="t('common.searchPlaceholder')"
        :result-count="filteredAndSortedModels.length"
        :total-count="models.length"
        @clear-filters="clearFilters"
      >
        <ViewToggle v-model="viewMode" />
      </FilterBar>
    </div>

    <!-- Provider quick-filter chips (flush sub-row) -->
    <div
      v-if="(models.length > 0 || isLoading) && availableProviders.length > 0"
      class="flex-none bg-background border-b border-border/60 px-4 sm:px-6 py-2.5"
    >
      <div class="flex flex-wrap items-center gap-2 overflow-x-auto scrollbar-none">
        <span class="mr-1 text-[11px] text-muted-foreground shrink-0"
          >{{ t("models.filterByProvider") }}:</span
        >
        <Button
          :variant="selectedProviderFilter === '' ? 'default' : 'outline'"
          size="sm"
          class="h-7 rounded-full px-2.5 text-[11px]"
          @click="selectedProviderFilter = ''"
        >
          {{ t("models.allProviders") }}
        </Button>
        <Button
          v-for="provider in availableProviders"
          :key="provider"
          :variant="selectedProviderFilter === provider ? 'default' : 'outline'"
          size="sm"
          class="h-7 rounded-full px-2.5 text-[11px]"
          @click="handleProviderFilter(provider)"
        >
          {{ provider }}
        </Button>
      </div>
    </div>

    <!-- Content area -->
    <div class="config-content">
      <div
        v-if="isLoading && models.length === 0"
        class="h-full flex items-start justify-center animate-fade-in px-6"
      >
        <ContentSkeleton />
      </div>
      <div
        v-else-if="models.length === 0"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <EmptyState
          :text="t('models.noModels')"
          :show-cta="true"
          :cta-text="t('models.addModel')"
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
              <TableHead class="w-10"></TableHead>
              <SortableHead
                :label="t('models.name')"
                sort-key="name"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />

              <TableHead>{{ t("models.providers") }}</TableHead>
              <TableHead>{{ t("common.routing") }}</TableHead>
              <TableHead class="p-0">
                <div class="flex items-center justify-end gap-1 px-3 py-1.5">
                  <span
                    class="mr-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                  >
                    {{ t("common.pricing") }}
                  </span>
                  <button
                    v-for="opt in pricingSortOptions"
                    :key="opt.key"
                    type="button"
                    :data-testid="`sort-${opt.key}`"
                    :aria-label="t('common.sortByColumn', { column: opt.label })"
                    :class="[
                      'inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wider transition-colors',
                      'hover:bg-muted/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/60',
                      sortField === opt.key
                        ? 'text-foreground'
                        : 'text-muted-foreground hover:text-foreground',
                    ]"
                    @click="onSort(opt.key)"
                  >
                    {{ opt.label }}
                    <ChevronUp
                      v-if="sortField === opt.key && sortDir === 'asc'"
                      class="size-3 shrink-0"
                      aria-hidden="true"
                    />
                    <ChevronDown
                      v-else-if="sortField === opt.key && sortDir === 'desc'"
                      class="size-3 shrink-0"
                      aria-hidden="true"
                    />
                    <ArrowUpDown
                      v-else
                      class="size-2.5 text-muted-foreground/40 shrink-0"
                      aria-hidden="true"
                    />
                  </button>
                </div>
              </TableHead>
              <TableHead class="w-24 text-right">{{ t("common.actions") }}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody class="row-stagger">
            <TableRow v-for="model in filteredAndSortedModels" :key="model.id" class="group">
              <TableCell>
                <div
                  :class="[
                    'w-8 h-8 rounded-lg flex items-center justify-center shrink-0 overflow-hidden',
                    getIconUrl(model.icon_url, model.name)
                      ? 'bg-card border border-border'
                      : 'bg-primary/10',
                  ]"
                >
                  <img
                    v-if="getIconUrl(model.icon_url, model.name) && !failedIcons.has(model.name)"
                    :src="getIconUrl(model.icon_url, model.name)!"
                    :alt="model.name"
                    :class="[isMonoIcon(model.name) ? 'icon-mono' : null, 'w-5 h-5 object-contain']"
                    loading="lazy"
                    @error="failedIcons.add(model.name)"
                  />
                  <ImageOff v-else class="w-4 h-4 text-muted-foreground" />
                </div>
              </TableCell>
              <TableCell class="font-medium">
                <div class="flex items-center gap-1.5 flex-wrap">
                  <span class="truncate" :title="model.name">{{ model.name }}</span>
                  <Badge
                    v-for="cap in deriveModelCapabilities(model)"
                    :key="cap"
                    variant="outline"
                    :class="['text-[11px] px-1.5 py-0 shrink-0', CAPABILITY_META[cap].badgeClass]"
                  >
                    <component :is="CAPABILITY_META[cap].icon" class="size-3 mr-0.5" />
                    {{ t(CAPABILITY_META[cap].labelKey) }}
                  </Badge>
                </div>
              </TableCell>
              <TableCell>
                <div class="flex items-center gap-1.5 flex-wrap">
                  <Badge
                    v-for="p in (model.providers || []).slice(0, 3)"
                    :key="p.provider_name"
                    variant="outline"
                    class="cursor-pointer border-border/60 bg-background/55 px-1.5 py-0 font-mono text-[11px] transition-colors hover:bg-accent"
                    @click.stop="handleProviderFilter(p.provider_name)"
                  >
                    {{ p.provider_name }}
                  </Badge>
                  <span
                    v-if="(model.providers || []).length > 3"
                    class="text-[11px] text-muted-foreground font-medium"
                  >
                    +{{ (model.providers || []).length - 3 }}
                  </span>
                  <span
                    v-if="!model.providers || model.providers.length === 0"
                    class="text-xs text-muted-foreground italic"
                  >
                    -
                  </span>
                </div>
              </TableCell>
              <!-- Smart Routing -->
              <TableCell>
                <div class="flex items-center gap-1 flex-wrap max-w-[240px]">
                  <template v-if="hasRoutingInfo(model)">
                    <span
                      v-if="model.auto_eligible"
                      class="inline-flex items-center justify-center size-4 rounded-full bg-status-success/15 text-status-success shrink-0"
                      :title="t('models.autoEligible')"
                    >
                      <Check class="size-2.5" />
                    </span>
                    <Badge
                      v-if="model.quality_tier"
                      :variant="
                        model.quality_tier === 'PREMIUM'
                          ? 'default'
                          : model.quality_tier === 'BALANCED'
                            ? 'secondary'
                            : 'outline'
                      "
                      class="text-[11px] uppercase font-medium px-1.5 py-0"
                    >
                      {{ model.quality_tier }}
                    </Badge>
                    <Badge
                      v-for="mode in model.routing_assignments || []"
                      :key="mode"
                      variant="outline"
                      class="text-[11px] px-1.5 py-0 border-action-blue/30 bg-action-blue/5 text-action-blue uppercase"
                    >
                      {{ mode }}
                    </Badge>
                    <span
                      v-if="model.context_length"
                      class="inline-flex items-center font-mono text-[11px] tabular-nums text-muted-foreground"
                      :title="t('models.contextLength')"
                    >
                      {{ formatContextLength(model.context_length) }}
                    </span>
                  </template>
                  <span v-else class="text-xs text-muted-foreground">–</span>
                </div>
              </TableCell>
              <TableCell class="text-right">
                <ModelPricingCell :model="model" />
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
                    @click="openEditDialog(model)"
                  >
                    <Edit class="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
                    :disabled="isLoading"
                    :aria-label="t('common.delete')"
                    @click="openDeleteDialog(model.name)"
                  >
                    <Trash2 class="w-4 h-4" />
                  </Button>
                </div>
              </TableCell>
            </TableRow>
            <EmptyTableRow
              v-if="filteredAndSortedModels.length === 0"
              :colspan="7"
              @clear="clearFilters"
            />
          </TableBody>
        </Table>

        <!-- List view -->
        <div v-else class="config-scroll">
          <EmptyFilterResults v-if="filteredAndSortedModels.length === 0" @clear="clearFilters" />
          <div v-if="filteredAndSortedModels.length > 0" class="config-list list-stagger">
            <ModelListItem
              v-for="model in filteredAndSortedModels"
              :key="model.id"
              :model="model"
              :is-loading="isLoading"
              @edit="openEditDialog(model)"
              @delete="openDeleteDialog(model.name)"
              @filter-provider="handleProviderFilter"
            />
          </div>
        </div>
      </template>
    </div>

    <!-- Delete Confirmation Dialog -->
    <ConfirmDialog
      v-model:open="showDeleteDialog"
      :title="t('dialogs.confirmDeleteTitle')"
      :description="t('dialogs.confirmDelete', { name: deletingModelName })"
      :confirm-text="t('common.delete')"
      :cancel-text="t('common.cancel')"
      :loading="isLoading"
      @confirm="confirmDelete"
    />

    <!-- Pricing sync dialog (models.dev) -->
    <PricingSyncDialog v-model:open="showPricingSyncDialog" @applied="modelStore.fetchModels()" />

    <!-- Create/Edit Sheet (slides in from the right) -->
    <Sheet v-model:open="showCreateDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[640px] lg:max-w-[760px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card transition-colors duration-300 pb-[env(safe-area-inset-bottom\,0px)]"
      >
        <!-- Header band -->
        <div
          class="px-4 sm:px-6 py-4 sm:py-5 border-b border-border/60 bg-muted/10 shrink-0 relative pr-16"
        >
          <div class="flex items-center gap-3">
            <div
              class="p-2.5 rounded-md bg-muted border border-border/40 text-muted-foreground shrink-0 flex items-center justify-center"
            >
              <Box class="size-5" />
            </div>
            <div class="flex flex-col min-w-0">
              <SheetTitle class="text-sm sm:text-base font-semibold text-foreground">
                {{ isEditing ? t("models.editModel") : t("models.addModel") }}
              </SheetTitle>
              <SheetDescription class="text-xs text-muted-foreground mt-1 truncate">
                {{ isEditing ? editingModelName : t("models.description") }}
              </SheetDescription>
            </div>
          </div>
        </div>

        <Tabs default-value="general" class="flex-1 flex flex-col min-h-0 w-full">
          <div class="px-4 sm:px-6 pt-4 shrink-0">
            <TabsList class="grid w-full grid-cols-3 sm:grid-cols-5 h-auto">
              <TabsTrigger value="general">{{ t("common.general") }}</TabsTrigger>
              <TabsTrigger value="pricing">{{ t("common.pricing") }}</TabsTrigger>
              <TabsTrigger value="routing">{{ t("common.routing") }}</TabsTrigger>
              <TabsTrigger value="overrides">{{ t("common.overrides") }}</TabsTrigger>
              <TabsTrigger value="advanced">{{ t("common.advanced") }}</TabsTrigger>
            </TabsList>
          </div>

          <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4">
            <TabsContent value="general" class="space-y-4 mt-0">
              <div class="grid gap-2">
                <Label for="modelName"
                  >{{ t("models.modelName") }} <span class="text-destructive">*</span></Label
                >
                <Input
                  id="modelName"
                  v-model="newModel.name"
                  :placeholder="t('placeholders.modelName')"
                />
                <p class="text-[11px] text-muted-foreground">
                  {{ t("models.modelNameHelperText") }}
                </p>
              </div>

              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <Label class="text-sm font-medium"
                    >{{ t("models.providers") }} <span class="text-destructive">*</span></Label
                  >
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    @click="addProvider"
                    class="h-7 text-[11px]"
                  >
                    <Plus class="w-3 h-3 mr-1" />
                    {{ t("models.addProvider") }}
                  </Button>
                </div>

                <div class="border-t border-border/60">
                  <div
                    v-if="newModel.providers.length > 0"
                    class="hidden sm:grid grid-cols-[1.5fr_2fr_80px_90px] gap-3 py-2 border-b border-border/60 text-[11px] uppercase tracking-wider text-muted-foreground font-medium"
                  >
                    <div>{{ t("models.provider") }}</div>
                    <div>{{ t("models.providerModelName") }}</div>
                    <div class="text-center">{{ t("models.priority") }}</div>
                    <div class="text-right">{{ t("common.actions") }}</div>
                  </div>

                  <div class="divide-y divide-border/60 border-b border-border/60">
                    <div
                      v-for="(p, index) in newModel.providers"
                      :key="index"
                      class="grid grid-cols-1 sm:grid-cols-[1.5fr_2fr_80px_90px] gap-3 py-3 items-center hover:bg-muted/10 transition-colors"
                    >
                      <div class="space-y-1 sm:space-y-0">
                        <span
                          class="text-[11px] text-muted-foreground uppercase font-medium sm:hidden"
                        >
                          {{ t("models.provider") }}
                        </span>
                        <Select
                          v-model="p.provider_name"
                          @update:model-value="p.provider_model_name = ''"
                        >
                          <SelectTrigger class="h-8 text-xs bg-background">
                            <SelectValue :placeholder="t('placeholders.selectProvider')" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem
                              v-for="providerOption in providers"
                              :key="providerOption.name"
                              :value="providerOption.name"
                            >
                              {{ providerOption.name }}
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div class="space-y-1 sm:space-y-0">
                        <span
                          class="text-[11px] text-muted-foreground uppercase font-medium sm:hidden"
                        >
                          {{ t("models.providerModelName") }}
                        </span>
                        <ProviderModelSelector
                          v-model="p.provider_model_name"
                          :provider-name="p.provider_name"
                          :placeholder="t('models.selectProviderModel')"
                          :disabled="!p.provider_name"
                        />
                      </div>

                      <div class="space-y-1 sm:space-y-0">
                        <span
                          class="text-[11px] text-muted-foreground uppercase font-medium sm:hidden block"
                        >
                          {{ t("models.priority") }}
                        </span>
                        <NumberInput
                          v-model.number="p.priority"
                          class="h-8 text-xs text-center"
                          :placeholder="t('placeholders.priority')"
                          min="0"
                        />
                      </div>

                      <div
                        class="flex items-center justify-end gap-1.5 pt-2 sm:pt-0 border-t sm:border-t-0 border-border/50"
                      >
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          class="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-muted/80"
                          :title="t('models.editProvider')"
                          @click="openProviderEditDialog(index)"
                        >
                          <Settings class="w-4 h-4" />
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          class="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
                          @click="removeProvider(index)"
                        >
                          <X class="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>

                  <div
                    v-if="newModel.providers.length === 0"
                    class="border-b border-border/60 py-8 text-center text-xs text-muted-foreground"
                  >
                    {{ t("models.noProvidersConfigured") }}
                  </div>
                </div>
                <p class="text-[11px] text-muted-foreground">
                  {{ t("models.priorityHelp") }}
                </p>
              </div>
            </TabsContent>

            <TabsContent value="pricing" class="space-y-4 mt-0">
              <div
                class="rounded-lg border border-action-amber/30 bg-action-amber/10 px-3.5 py-2.5 mb-4"
              >
                <p class="text-xs text-action-amber leading-relaxed">
                  {{ t("models.defaultPricingNote") }}
                </p>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div class="grid gap-2">
                  <Label for="inputCost">
                    {{ t("models.costDimInput") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perMillionTokens") }}</span
                    >
                  </Label>
                  <div class="relative">
                    <div
                      class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                    >
                      <span class="text-muted-foreground sm:text-sm">$</span>
                    </div>

                    <NumberInput
                      id="inputCost"
                      v-model.number="newModel.input_cost_per_1m"
                      step="0.01"
                      placeholder="0.0"
                      class="pl-7"
                    />
                  </div>
                </div>
                <div class="grid gap-2">
                  <Label for="outputCost">
                    {{ t("models.costDimOutput") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perMillionTokens") }}</span
                    >
                  </Label>
                  <div class="relative">
                    <div
                      class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                    >
                      <span class="text-muted-foreground sm:text-sm">$</span>
                    </div>

                    <NumberInput
                      id="outputCost"
                      v-model.number="newModel.output_cost_per_1m"
                      step="0.01"
                      placeholder="0.0"
                      class="pl-7"
                    />
                  </div>
                </div>
                <div class="grid gap-2">
                  <Label for="cachedReadCost">
                    {{ t("models.costDimCachedRead") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perMillionTokens") }}</span
                    >
                  </Label>
                  <div class="relative">
                    <div
                      class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                    >
                      <span class="text-muted-foreground sm:text-sm">$</span>
                    </div>

                    <NumberInput
                      id="cachedReadCost"
                      v-model.number="newModel.cached_read_cost_per_1m"
                      step="0.01"
                      placeholder="0.0"
                      class="pl-7"
                    />
                  </div>
                </div>
                <div class="grid gap-2">
                  <Label for="cachedWriteCost">
                    {{ t("models.costDimCachedWrite") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perMillionTokens") }}</span
                    >
                  </Label>
                  <div class="relative">
                    <div
                      class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                    >
                      <span class="text-muted-foreground sm:text-sm">$</span>
                    </div>

                    <NumberInput
                      id="cachedWriteCost"
                      v-model.number="newModel.cached_write_cost_per_1m"
                      step="0.01"
                      placeholder="0.0"
                      class="pl-7"
                    />
                  </div>
                </div>
                <div class="grid gap-2">
                  <Label for="audioInputCost">
                    {{ t("models.costDimAudioInput") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perMillionTokens") }}</span
                    >
                  </Label>
                  <div class="relative">
                    <div
                      class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                    >
                      <span class="text-muted-foreground sm:text-sm">$</span>
                    </div>

                    <NumberInput
                      id="audioInputCost"
                      v-model.number="newModel.audio_input_cost_per_1m"
                      step="0.01"
                      placeholder="0.0"
                      class="pl-7"
                    />
                  </div>
                </div>
                <div class="grid gap-2">
                  <Label for="audioOutputCost">
                    {{ t("models.costDimAudioOutput") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perMillionTokens") }}</span
                    >
                  </Label>
                  <div class="relative">
                    <div
                      class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                    >
                      <span class="text-muted-foreground sm:text-sm">$</span>
                    </div>

                    <NumberInput
                      id="audioOutputCost"
                      v-model.number="newModel.audio_output_cost_per_1m"
                      step="0.01"
                      placeholder="0.0"
                      class="pl-7"
                    />
                  </div>
                </div>
              </div>

              <!-- Other pricing dimensions (non-token modalities) -->
              <div class="border-t border-border pt-4 mt-2">
                <div class="mb-3">
                  <p class="text-sm font-medium">
                    {{ t("models.otherPricingDimensions") }}
                  </p>
                  <p class="text-[11px] text-muted-foreground">
                    {{ t("models.otherPricingDimensionsHelp") }}
                  </p>
                </div>
                <div class="grid grid-cols-2 gap-4">
                  <div class="grid gap-2">
                    <Label for="imageInputCost">
                      {{ t("models.costDimImageInput") }}
                      <span class="ml-1 font-normal text-muted-foreground"
                        >· {{ t("models.perMillionTokens") }}</span
                      >
                    </Label>
                    <div class="relative">
                      <div
                        class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                      >
                        <span class="text-muted-foreground sm:text-sm">$</span>
                      </div>
                      <NumberInput
                        id="imageInputCost"
                        v-model.number="newModel.image_input_cost_per_1m"
                        :min="0"
                        step="0.01"
                        placeholder="0.0"
                        class="pl-7"
                      />
                    </div>
                  </div>
                  <div class="grid gap-2">
                    <Label for="costPerImage">
                      {{ t("models.costDimImageGen") }}
                      <span class="ml-1 font-normal text-muted-foreground"
                        >· {{ t("models.perImage") }}</span
                      >
                    </Label>
                    <div class="relative">
                      <div
                        class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                      >
                        <span class="text-muted-foreground sm:text-sm">$</span>
                      </div>
                      <NumberInput
                        id="costPerImage"
                        v-model.number="newModel.cost_per_image"
                        :min="0"
                        step="0.01"
                        placeholder="0.0"
                        class="pl-7"
                      />
                    </div>
                  </div>
                  <div class="grid gap-2">
                    <Label for="audioCostPerMinute">
                      {{ t("models.costDimAudioStt") }}
                      <span class="ml-1 font-normal text-muted-foreground"
                        >· {{ t("models.perMinute") }}</span
                      >
                    </Label>
                    <div class="relative">
                      <div
                        class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                      >
                        <span class="text-muted-foreground sm:text-sm">$</span>
                      </div>
                      <NumberInput
                        id="audioCostPerMinute"
                        v-model.number="newModel.audio_cost_per_minute"
                        :min="0"
                        step="0.01"
                        placeholder="0.0"
                        class="pl-7"
                      />
                    </div>
                  </div>
                  <div class="grid gap-2">
                    <Label for="ttsCostPer1mChars">
                      {{ t("models.costDimTts") }}
                      <span class="ml-1 font-normal text-muted-foreground"
                        >· {{ t("models.perMillionChars") }}</span
                      >
                    </Label>
                    <div class="relative">
                      <div
                        class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                      >
                        <span class="text-muted-foreground sm:text-sm">$</span>
                      </div>
                      <NumberInput
                        id="ttsCostPer1mChars"
                        v-model.number="newModel.tts_cost_per_1m_chars"
                        :min="0"
                        step="0.01"
                        placeholder="0.0"
                        class="pl-7"
                      />
                    </div>
                  </div>
                  <div class="grid gap-2">
                    <Label for="webSearchCostPer1k">
                      {{ t("models.costDimWebSearch") }}
                      <span class="ml-1 font-normal text-muted-foreground"
                        >· {{ t("models.per1kSearches") }}</span
                      >
                    </Label>
                    <div class="relative">
                      <div
                        class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                      >
                        <span class="text-muted-foreground sm:text-sm">$</span>
                      </div>
                      <NumberInput
                        id="webSearchCostPer1k"
                        v-model.number="newModel.web_search_cost_per_1k"
                        :min="0"
                        step="0.01"
                        placeholder="0.0"
                        class="pl-7"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="overrides" class="space-y-4 mt-0">
              <div class="grid gap-2">
                <Label class="text-sm font-medium">{{ t("models.parameterOverrides") }}</Label>
                <p class="text-[11px] text-muted-foreground">
                  {{ t("models.parameterOverridesHelp") }}
                </p>
                <ParameterOverridesBuilder v-model="parameterOverrides" />
              </div>
            </TabsContent>

            <!-- Smart Routing -->
            <TabsContent value="routing" class="space-y-5 mt-0">
              <p class="text-xs text-muted-foreground leading-relaxed">
                {{ t("models.routingTabHelp") }}
              </p>

              <h4 class="text-sm font-semibold text-foreground border-b border-border/60 pb-2">
                {{ t("models.routingEligibilityHeader") }}
              </h4>

              <!-- Auto-Eligible -->
              <div class="flex items-start justify-between gap-4">
                <div class="space-y-0.5">
                  <Label class="text-sm font-medium">{{ t("models.autoEligible") }}</Label>
                  <p class="text-xs text-muted-foreground leading-relaxed">
                    {{ t("models.autoEligibleHelp") }}
                  </p>
                </div>
                <Switch v-model="newModel.auto_eligible" class="mt-0.5" />
              </div>

              <h4
                class="text-sm font-semibold text-foreground border-b border-border/60 pb-2 pt-1"
                :class="{ 'opacity-50': !newModel.auto_eligible }"
              >
                {{ t("models.routingProfileHeader") }}
              </h4>

              <!-- Quality Tier -->
              <div class="grid gap-2" :class="{ 'opacity-50': !newModel.auto_eligible }">
                <Label
                  for="qualityTier"
                  class="after:content-['*'] after:ml-0.5 after:text-destructive"
                >
                  {{ t("models.qualityTier") }}
                </Label>
                <Select v-model="newModel.quality_tier" :disabled="!newModel.auto_eligible">
                  <SelectTrigger class="h-9">
                    <SelectValue :placeholder="t('models.qualityTierSelect')" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ECONOMY">{{ t("models.qualityTierEconomy") }}</SelectItem>
                    <SelectItem value="BALANCED">{{ t("models.qualityTierBalanced") }}</SelectItem>
                    <SelectItem value="PREMIUM">{{ t("models.qualityTierPremium") }}</SelectItem>
                  </SelectContent>
                </Select>
                <p class="text-xs text-muted-foreground leading-relaxed">
                  {{ t("models.qualityTierHelp") }}
                </p>
              </div>

              <!-- Routing Assignments -->
              <div class="grid gap-2" :class="{ 'opacity-50': !newModel.auto_eligible }">
                <Label>{{ t("models.routingAssignments") }}</Label>
                <p class="text-xs text-muted-foreground leading-relaxed">
                  {{ t("models.routingAssignmentsHelp") }}
                </p>
                <div class="flex flex-wrap gap-2 pt-0.5">
                  <button
                    type="button"
                    v-for="mode in ROUTING_MODES"
                    :key="mode"
                    :disabled="!newModel.auto_eligible"
                    :class="[
                      'inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors cursor-pointer focus-visible-ring',
                      newModel.routing_assignments?.includes(mode)
                        ? 'border-action-blue/40 bg-action-blue/10 text-action-blue'
                        : 'border-border bg-background/50 text-muted-foreground hover:bg-accent hover:text-foreground',
                      !newModel.auto_eligible
                        ? 'opacity-50 cursor-not-allowed pointer-events-none'
                        : '',
                    ]"
                    @click="
                      newModel.routing_assignments = (newModel.routing_assignments || []).includes(
                        mode
                      )
                        ? (newModel.routing_assignments || []).filter((m) => m !== mode)
                        : [...(newModel.routing_assignments || []), mode]
                    "
                  >
                    <Check v-if="newModel.routing_assignments?.includes(mode)" class="size-3" />
                    {{
                      {
                        fast: t("smartRouting.modeFast"),
                        auto: t("smartRouting.modeAuto"),
                        best: t("smartRouting.modeBest"),
                      }[mode]
                    }}
                  </button>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="advanced" class="space-y-6 mt-0">
              <!-- Section 1: Model Profile (description, links, specs) -->
              <div class="space-y-4">
                <h4 class="text-sm font-semibold text-foreground border-b border-border/60 pb-2">
                  {{ t("models.tabProfileHeader") }}
                </h4>

                <!-- Description -->
                <div class="grid gap-2">
                  <Label for="modelDescription" class="text-xs font-medium text-foreground">
                    {{ t("models.modelDescription") }}
                  </Label>
                  <Textarea
                    id="modelDescription"
                    v-model="descriptionModel"
                    :placeholder="t('models.descriptionPlaceholder')"
                    rows="3"
                    class="resize-y text-sm"
                  />
                  <p class="text-[11px] text-muted-foreground leading-normal">
                    {{ t("models.descriptionHelp") }}
                  </p>
                </div>

                <!-- Homepage URL + Context Length -->
                <div class="grid gap-4 sm:grid-cols-2 items-start">
                  <div class="grid gap-2">
                    <Label for="modelHomepageUrl" class="text-xs font-medium text-foreground">
                      {{ t("models.homepageUrl") }}
                    </Label>
                    <Input
                      id="modelHomepageUrl"
                      v-model="newModel.homepage_url"
                      :placeholder="t('models.homepageUrlPlaceholder')"
                      class="h-9"
                    />
                    <p class="text-[11px] text-muted-foreground leading-normal">
                      {{ t("models.homepageUrlHelp") }}
                    </p>
                  </div>
                  <div class="grid gap-2">
                    <Label for="modelContextLength" class="text-xs font-medium text-foreground">
                      {{ t("models.contextLength") }}
                    </Label>
                    <NumberInput
                      id="modelContextLength"
                      v-model.number="newModel.context_length"
                      :placeholder="t('models.contextLengthPlaceholder')"
                      :min="0"
                      class="h-9"
                    />
                    <p class="text-[11px] text-muted-foreground leading-normal">
                      {{ t("models.contextLengthHelp") }}
                    </p>
                  </div>
                </div>
              </div>

              <!-- Section 2: Capabilities & Behavior -->
              <div class="space-y-4">
                <h4 class="text-sm font-semibold text-foreground border-b border-border/60 pb-2">
                  {{ t("models.tabCapabilitiesHeader") }}
                </h4>

                <div class="divide-y divide-border/60">
                  <CapabilityToggle
                    v-model="newModel.supports_images"
                    :label="t('models.supportsImages')"
                    :help-text="t('models.supportsImagesHelp')"
                  />
                  <CapabilityToggle
                    v-model="newModel.supports_image_generation"
                    :label="t('models.supportsImageGeneration')"
                    :help-text="t('models.supportsImageGenerationHelp')"
                  />
                  <CapabilityToggle
                    v-model="newModel.supports_tts"
                    :label="t('models.supportsTts')"
                    :help-text="t('models.supportsTtsHelp')"
                  />
                  <CapabilityToggle
                    v-model="newModel.supports_stt"
                    :label="t('models.supportsStt')"
                    :help-text="t('models.supportsSttHelp')"
                  />
                  <CapabilityToggle
                    v-model="newModel.supports_embedding"
                    :label="t('models.supportsEmbedding')"
                    :help-text="t('models.supportsEmbeddingHelp')"
                  />
                  <CapabilityToggle
                    v-model="newModel.supports_realtime"
                    :label="t('models.supportsRealtime')"
                    :help-text="t('models.supportsRealtimeHelp')"
                  />
                </div>
              </div>

              <!-- Section 3: Visual Settings -->
              <div class="space-y-4">
                <h4 class="text-sm font-semibold text-foreground border-b border-border/60 pb-2">
                  {{ t("models.tabVisualsHeader") }}
                </h4>

                <div class="flex items-start gap-4">
                  <!-- Icon Preview -->
                  <div class="flex flex-col items-center gap-1.5 shrink-0">
                    <span class="text-xs text-muted-foreground">Preview</span>
                    <div
                      class="w-12 h-12 rounded-xl flex items-center justify-center border border-border bg-background overflow-hidden"
                    >
                      <img
                        v-if="getIconUrl(newModel.icon_url, newModel.name) && !iconPreviewFailed"
                        :src="getIconUrl(newModel.icon_url, newModel.name)!"
                        :alt="newModel.name"
                        :class="[
                          isMonoIcon(newModel.name) ? 'icon-mono' : null,
                          'w-8 h-8 object-contain',
                        ]"
                        @error="iconPreviewFailed = true"
                      />
                      <Box v-else class="w-5 h-5 text-muted-foreground" />
                    </div>
                  </div>

                  <!-- Icon URL Input -->
                  <div class="grid gap-2 flex-1">
                    <Label for="modelIconUrl" class="text-xs font-medium text-foreground">
                      {{ t("models.iconUrl") }}
                    </Label>
                    <Input
                      id="modelIconUrl"
                      v-model="newModel.icon_url"
                      :placeholder="t('models.iconUrlPlaceholder')"
                      class="h-9"
                    />
                    <p class="text-[11px] text-muted-foreground leading-normal">
                      {{ t("models.iconUrlHelp") }}
                    </p>
                  </div>
                </div>
              </div>
            </TabsContent>
          </div>
        </Tabs>

        <!-- Footer -->
        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" @click="showCreateDialog = false">{{
            t("common.cancel")
          }}</Button>
          <Button @click="saveModel" :disabled="isSaving || isLoading">{{
            t("common.save")
          }}</Button>
        </div>
      </SheetContent>
    </Sheet>

    <!-- Provider Edit Sheet (nested, slides in from the right above the model sheet) -->
    <Sheet v-model:open="showProviderEditDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[520px] lg:max-w-[600px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card transition-colors duration-300 pb-[env(safe-area-inset-bottom\,0px)]"
      >
        <!-- Header band -->
        <div
          class="px-4 sm:px-6 py-4 sm:py-5 border-b border-border/60 bg-muted/10 shrink-0 relative pr-16"
        >
          <div class="flex items-center gap-3">
            <div
              class="p-2.5 rounded-md bg-muted border border-border/40 text-muted-foreground shrink-0 flex items-center justify-center"
            >
              <Settings class="size-5" />
            </div>
            <div class="flex flex-col min-w-0">
              <SheetTitle class="text-sm sm:text-base font-semibold text-foreground">
                {{ t("models.editProvider") }}
              </SheetTitle>
              <SheetDescription class="text-xs text-muted-foreground mt-1 truncate font-mono">
                {{ editingProviderData.provider_name || t("models.provider") }}
              </SheetDescription>
            </div>
          </div>
        </div>

        <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
          <div class="grid grid-cols-2 gap-4">
            <div class="grid gap-2">
              <Label>{{ t("models.provider") }} <span class="text-destructive">*</span></Label>
              <Select v-model="editingProviderData.provider_name">
                <SelectTrigger class="h-9">
                  <SelectValue :placeholder="t('placeholders.selectProvider')" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem
                    v-for="providerOption in providers"
                    :key="providerOption.name"
                    :value="providerOption.name"
                  >
                    {{ providerOption.name }}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div class="grid gap-2">
              <Label
                >{{ t("models.providerModelName") }} <span class="text-destructive">*</span></Label
              >
              <ProviderModelSelector
                v-model="editingProviderData.provider_model_name"
                :provider-name="editingProviderData.provider_name"
                :placeholder="t('models.selectProviderModel')"
                :disabled="!editingProviderData.provider_name"
              />
            </div>
          </div>

          <div class="grid gap-2 w-32">
            <Label>{{ t("models.priority") }}</Label>
            <NumberInput
              v-model.number="editingProviderData.priority"
              class="h-9"
              :placeholder="t('placeholders.priority')"
            />
          </div>

          <div class="border-t pt-4">
            <Label class="text-sm font-medium mb-3 block">{{ t("models.providerCosts") }}</Label>
            <div class="grid grid-cols-2 gap-3">
              <div class="space-y-1">
                <Label class="text-xs">
                  {{ t("models.costDimInput") }}
                  <span class="ml-1 font-normal text-muted-foreground"
                    >· {{ t("models.perMillionTokens") }}</span
                  >
                </Label>
                <NumberInput
                  v-model.number="editingProviderData.input_cost_per_1m"
                  step="0.01"
                  placeholder="0.00"
                  class="h-9"
                />
              </div>
              <div class="space-y-1">
                <Label class="text-xs">
                  {{ t("models.costDimOutput") }}
                  <span class="ml-1 font-normal text-muted-foreground"
                    >· {{ t("models.perMillionTokens") }}</span
                  >
                </Label>
                <NumberInput
                  v-model.number="editingProviderData.output_cost_per_1m"
                  step="0.01"
                  placeholder="0.00"
                  class="h-9"
                />
              </div>
              <div class="space-y-1">
                <Label class="text-xs">
                  {{ t("models.costDimCachedRead") }}
                  <span class="ml-1 font-normal text-muted-foreground"
                    >· {{ t("models.perMillionTokens") }}</span
                  >
                </Label>
                <NumberInput
                  v-model.number="editingProviderData.cached_read_cost_per_1m"
                  step="0.01"
                  placeholder="0.00"
                  class="h-9"
                />
              </div>
              <div class="space-y-1">
                <Label class="text-xs">
                  {{ t("models.costDimCachedWrite") }}
                  <span class="ml-1 font-normal text-muted-foreground"
                    >· {{ t("models.perMillionTokens") }}</span
                  >
                </Label>
                <NumberInput
                  v-model.number="editingProviderData.cached_write_cost_per_1m"
                  step="0.01"
                  placeholder="0.00"
                  class="h-9"
                />
              </div>
              <div class="space-y-1">
                <Label class="text-xs">
                  {{ t("models.costDimAudioInput") }}
                  <span class="ml-1 font-normal text-muted-foreground"
                    >· {{ t("models.perMillionTokens") }}</span
                  >
                </Label>
                <NumberInput
                  v-model.number="editingProviderData.audio_input_cost_per_1m"
                  step="0.01"
                  placeholder="0.00"
                  class="h-9"
                />
              </div>
              <div class="space-y-1">
                <Label class="text-xs">
                  {{ t("models.costDimAudioOutput") }}
                  <span class="ml-1 font-normal text-muted-foreground"
                    >· {{ t("models.perMillionTokens") }}</span
                  >
                </Label>
                <NumberInput
                  v-model.number="editingProviderData.audio_output_cost_per_1m"
                  step="0.01"
                  placeholder="0.00"
                  class="h-9"
                />
              </div>
            </div>
            <!-- Other pricing dimensions (non-token modalities) -->
            <div class="mt-3 pt-3 border-t border-border/60">
              <p class="text-xs font-medium text-muted-foreground mb-2">
                {{ t("models.otherPricingDimensions") }}
              </p>
              <div class="grid grid-cols-2 gap-3">
                <div class="space-y-1">
                  <Label class="text-xs">
                    {{ t("models.costDimImageInput") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perMillionTokens") }}</span
                    >
                  </Label>
                  <NumberInput
                    v-model.number="editingProviderData.image_input_cost_per_1m"
                    step="0.01"
                    placeholder="0.00"
                    class="h-9"
                  />
                </div>
                <div class="space-y-1">
                  <Label class="text-xs">
                    {{ t("models.costDimImageGen") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perImage") }}</span
                    >
                  </Label>
                  <NumberInput
                    v-model.number="editingProviderData.cost_per_image"
                    step="0.01"
                    placeholder="0.00"
                    class="h-9"
                  />
                </div>
                <div class="space-y-1">
                  <Label class="text-xs">
                    {{ t("models.costDimAudioStt") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perMinute") }}</span
                    >
                  </Label>
                  <NumberInput
                    v-model.number="editingProviderData.audio_cost_per_minute"
                    step="0.01"
                    placeholder="0.00"
                    class="h-9"
                  />
                </div>
                <div class="space-y-1">
                  <Label class="text-xs">
                    {{ t("models.costDimTts") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.perMillionChars") }}</span
                    >
                  </Label>
                  <NumberInput
                    v-model.number="editingProviderData.tts_cost_per_1m_chars"
                    step="0.01"
                    placeholder="0.00"
                    class="h-9"
                  />
                </div>
                <div class="space-y-1">
                  <Label class="text-xs">
                    {{ t("models.costDimWebSearch") }}
                    <span class="ml-1 font-normal text-muted-foreground"
                      >· {{ t("models.per1kSearches") }}</span
                    >
                  </Label>
                  <NumberInput
                    v-model.number="editingProviderData.web_search_cost_per_1k"
                    step="0.01"
                    placeholder="0.00"
                    class="h-9"
                  />
                </div>
              </div>
            </div>
          </div>

          <div class="border-t pt-4">
            <Label class="text-sm font-medium mb-3 block">{{
              t("models.parameterOverrides")
            }}</Label>
            <ParameterOverridesBuilder
              :model-value="editingProviderData.parameter_overrides || {}"
              @update:model-value="editingProviderData.parameter_overrides = $event"
            />
          </div>
        </div>

        <!-- Footer -->
        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button
            variant="outline"
            @click="
              showProviderEditDialog = false;
              editingProviderIndex = null;
            "
            >{{ t("common.cancel") }}</Button
          >
          <Button @click="saveProviderEdit">{{ t("common.save") }}</Button>
        </div>
      </SheetContent>
    </Sheet>
  </AppLayout>
</template>
