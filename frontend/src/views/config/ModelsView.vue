<script setup lang="ts">
import { Box, Check, Edit, Plus, RefreshCw, Settings, Trash2, X } from "@lucide/vue";
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import ConfirmDialog from "@/components/common/ConfirmDialog.vue";
import EmptyState from "@/components/common/EmptyState.vue";
import EmptyFilterResults from "@/components/common/EmptyFilterResults.vue";
import FilterBar from "@/components/common/FilterBar.vue";
import TableSkeleton from "@/components/common/TableSkeleton.vue";
import EmptyTableRow from "@/components/common/EmptyTableRow.vue";
import PageHeader from "@/components/common/PageHeader.vue";
import SortableHead from "@/components/common/SortableHead.vue";
import ViewToggle from "@/components/common/ViewToggle.vue";
import {
  ModelListItem,
  ModelIcon,
  ModelPricingCell,
  CapabilityToggle,
  ModelStatusChip,
  ModelProviderList,
  ModelContextCell,
} from "@/components/models";
import ModelsSyncDialog from "@/components/models/ModelsSyncDialog.vue";
import {
  BOUND_CAPABILITIES,
  INFO_CAPABILITIES,
  CAPABILITY_META,
} from "@/components/plaza/capabilities";
import CapabilityIcons from "@/components/plaza/CapabilityIcons.vue";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
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
import { useModelForm } from "@/composables/useModelForm";
import { useModelTable } from "@/composables/useModelTable";
import { useViewMode } from "@/composables/useViewMode";
import { useModelStore } from "@/stores/models";
import { useProviderStore } from "@/stores/providers";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import { ROUTING_MODES } from "@/constants/model";

import ParameterOverridesBuilder from "@/components/config/ParameterOverridesBuilder.vue";
import ProviderModelSelector from "@/components/common/ProviderModelSelector.vue";
import { getIconUrl, isMonoIcon } from "@/utils/icons";

import type { ModelRead } from "@/types/schemas";

// Rendered inside views/ModelsView.vue (the role dispatcher), which owns the
// "ModelsView" name that App.vue's KeepAlive include list matches on.
defineOptions({ name: "ModelsAdminView" });

const { t } = useI18n();
const { handleSaveError, handleDeleteError } = useErrorHandler();
const modelStore = useModelStore();
const providerStore = useProviderStore();

const showCreateDialog = ref(false);
const showSyncDialog = ref(false);
const showDeleteDialog = ref(false);
const deletingModelName = ref("");
const isEditing = ref(false);
const editingModelName = ref("");
const isSaving = ref(false);
const viewMode = useViewMode(STORAGE_KEYS.MODELS_VIEW_MODE);

const models = computed(() => modelStore.models);
const providers = computed(() => providerStore.providers);
const isLoading = computed(() => modelStore.loading && !modelStore.ready);

const {
  searchQuery,
  sortField,
  sortDir,
  onSort,
  selectedProviderFilter,
  availableProviders,
  filteredAndSortedModels,
  clearFilters,
  handleProviderFilter,
} = useModelTable(models);

const {
  form: newModel,
  parameterOverrides,
  iconPreviewFailed,
  getCapability,
  setCapability,
  showProviderEditDialog,
  editingProviderIndex,
  editingProviderData,
  addProvider,
  removeProvider,
  openProviderEditDialog,
  saveProviderEdit,
  descriptionModel,
  statusModel,
  familyModel,
  knowledgeModel,
  releaseDateModel,
  openForCreate,
  openForEdit,
  validate,
  buildPayload,
} = useModelForm();

onMounted(() => {
  modelStore.fetchModels();
  providerStore.fetchProviders();
});

const hasRoutingInfo = (model: ModelRead): boolean => Boolean(model.auto_eligible);

const openCreateDialog = () => {
  isEditing.value = false;
  editingModelName.value = "";
  openForCreate();
  showCreateDialog.value = true;
};

const openEditDialog = (model: ModelRead) => {
  isEditing.value = true;
  editingModelName.value = model.name;
  openForEdit(model);
  showCreateDialog.value = true;
};

const saveModel = async () => {
  const errorKey = validate();
  if (errorKey) {
    toast.error(t(errorKey));
    return;
  }

  isSaving.value = true;
  try {
    const payload = buildPayload();
    if (isEditing.value) {
      await modelStore.updateModel(editingModelName.value, payload);
    } else {
      await modelStore.createModel(payload);
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
            <Button variant="outline" @click="showSyncDialog = true">
              <RefreshCw class="w-4 h-4 mr-2" />
              {{ t("models.sync.trigger") }}
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
      <div v-if="isLoading && models.length === 0" class="h-full animate-fade-in">
        <TableSkeleton />
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
              <SortableHead
                :label="t('models.name')"
                sort-key="name"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <TableHead>{{ t("models.providers") }}</TableHead>
              <SortableHead
                :label="t('models.contextShort')"
                sort-key="context_length"
                align="right"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <TableHead>{{ t("common.routing") }}</TableHead>
              <SortableHead
                :label="t('models.inputCostShort')"
                sort-key="input_cost"
                align="right"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <SortableHead
                :label="t('models.outputCostShort')"
                sort-key="output_cost"
                align="right"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <SortableHead
                :label="t('models.cachedReadCostShort')"
                sort-key="cached_read"
                align="right"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <TableHead class="w-24 text-right">{{ t("common.actions") }}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody class="row-stagger">
            <TableRow v-for="model in filteredAndSortedModels" :key="model.id" class="group">
              <!-- Name: icon + name + status + capability icons -->
              <TableCell class="font-medium">
                <div class="flex items-center gap-2.5 min-w-0">
                  <ModelIcon :name="model.name" :icon-url="model.icon_url" size="sm" />
                  <Tooltip>
                    <TooltipTrigger as-child>
                      <span class="truncate">{{ model.name }}</span>
                    </TooltipTrigger>
                    <TooltipContent>{{ model.name }}</TooltipContent>
                  </Tooltip>
                  <ModelStatusChip v-if="model.status" :status="model.status" />
                  <CapabilityIcons :capabilities="model.capabilities ?? []" />
                </div>
              </TableCell>
              <!-- Providers: quiet mono text, click to filter -->
              <TableCell>
                <ModelProviderList
                  class="text-xs"
                  :providers="model.providers || []"
                  @filter="handleProviderFilter"
                />
              </TableCell>
              <!-- Context / max output -->
              <TableCell class="text-right">
                <ModelContextCell
                  :context-length="model.context_length"
                  :max-output-tokens="model.max_output_tokens"
                />
              </TableCell>
              <!-- Smart Routing -->
              <TableCell>
                <div
                  v-if="hasRoutingInfo(model)"
                  class="flex items-center gap-1.5 whitespace-nowrap"
                >
                  <Tooltip>
                    <TooltipTrigger as-child>
                      <span
                        class="inline-flex items-center justify-center size-4 rounded-full bg-status-success/15 text-status-success shrink-0"
                      >
                        <Check class="size-2.5" />
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>{{ t("models.autoEligible") }}</TooltipContent>
                  </Tooltip>
                  <span v-if="model.quality_tier" class="text-xs font-medium capitalize">
                    {{ model.quality_tier.toLowerCase() }}
                  </span>
                  <span
                    v-if="model.routing_assignments?.length"
                    class="font-mono text-[11px] text-muted-foreground"
                  >
                    {{ model.routing_assignments.join(", ") }}
                  </span>
                </div>
                <span v-else class="text-xs text-muted-foreground">–</span>
              </TableCell>
              <TableCell class="text-right">
                <ModelPricingCell :model="model" field="input" />
              </TableCell>
              <TableCell class="text-right">
                <ModelPricingCell :model="model" field="output" />
              </TableCell>
              <TableCell class="text-right">
                <ModelPricingCell :model="model" field="cached" />
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
              :colspan="8"
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

    <!-- Sync dialog (models.dev: pricing + capabilities) -->
    <ModelsSyncDialog v-model:open="showSyncDialog" @applied="modelStore.fetchModels()" />

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
            <TabsList class="grid w-full grid-cols-3 sm:grid-cols-6 h-auto">
              <TabsTrigger value="general">{{ t("common.general") }}</TabsTrigger>
              <TabsTrigger value="capabilities">{{ t("common.capabilities") }}</TabsTrigger>
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
                        <Tooltip>
                          <TooltipTrigger as-child>
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              class="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-muted/80"
                              :aria-label="t('models.editProvider')"
                              @click="openProviderEditDialog(index)"
                            >
                              <Settings class="w-4 h-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>{{ t("models.editProvider") }}</TooltipContent>
                        </Tooltip>
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

            <TabsContent value="capabilities" class="space-y-6 mt-0">
              <!-- Feature-bound capabilities: these gate proxy behavior -->
              <div class="space-y-4">
                <div>
                  <h4 class="text-sm font-semibold text-foreground border-b border-border/60 pb-2">
                    {{ t("models.capBoundHeader") }}
                  </h4>
                  <p class="text-[11px] text-muted-foreground leading-normal mt-1.5">
                    {{ t("models.capBoundHint") }}
                  </p>
                </div>
                <div class="divide-y divide-border/60">
                  <CapabilityToggle
                    v-for="cap in BOUND_CAPABILITIES"
                    :key="cap"
                    :model-value="getCapability(cap)"
                    :label="t(CAPABILITY_META[cap].labelKey)"
                    :icon="CAPABILITY_META[cap].icon"
                    :help-text="t('models.capBoundHelp.' + cap)"
                    @update:model-value="setCapability(cap, $event)"
                  />
                </div>
              </div>

              <!-- Informational models.dev attributes: display-only -->
              <div class="space-y-4">
                <div>
                  <h4 class="text-sm font-semibold text-foreground border-b border-border/60 pb-2">
                    {{ t("models.capInfoHeader") }}
                  </h4>
                  <p class="text-[11px] text-muted-foreground leading-normal mt-1.5">
                    {{ t("models.capInfoHint") }}
                  </p>
                </div>
                <div class="divide-y divide-border/60">
                  <CapabilityToggle
                    v-for="cap in INFO_CAPABILITIES"
                    :key="cap"
                    :model-value="getCapability(cap)"
                    :label="t(CAPABILITY_META[cap].labelKey)"
                    :icon="CAPABILITY_META[cap].icon"
                    :field="cap"
                    :help-text="t('models.capInfoHelp.' + cap)"
                    @update:model-value="setCapability(cap, $event)"
                  />
                </div>
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
                      class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                      class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                      class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                      class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                      class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                      class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                        class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                        class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                        class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                        class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
                        class="absolute inset-y-0 left-0 z-10 flex items-center pl-3 pointer-events-none"
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
              <!-- Section 1: Model Profile (description, links) -->
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

                <!-- Homepage URL -->
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
              </div>

              <!-- Section 2: Limits (models.dev limit.*) -->
              <div class="space-y-4">
                <div>
                  <h4 class="text-sm font-semibold text-foreground border-b border-border/60 pb-2">
                    {{ t("models.limitsHeader") }}
                  </h4>
                  <p class="text-[11px] text-muted-foreground leading-normal mt-1.5">
                    {{ t("models.limitsHint") }}
                  </p>
                </div>
                <div class="grid gap-4 sm:grid-cols-2 items-start">
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
                  <div class="grid gap-2">
                    <Label for="modelMaxOutputTokens" class="text-xs font-medium text-foreground">
                      {{ t("models.maxOutputTokens") }}
                    </Label>
                    <NumberInput
                      id="modelMaxOutputTokens"
                      v-model.number="newModel.max_output_tokens"
                      :placeholder="t('models.maxOutputTokensPlaceholder')"
                      :min="0"
                      class="h-9"
                    />
                    <p class="text-[11px] text-muted-foreground leading-normal">
                      {{ t("models.maxOutputTokensHelp") }}
                    </p>
                  </div>
                </div>
              </div>

              <!-- Section 3: Classification (models.dev metadata) -->
              <div class="space-y-4">
                <div>
                  <h4 class="text-sm font-semibold text-foreground border-b border-border/60 pb-2">
                    {{ t("models.classificationHeader") }}
                  </h4>
                  <p class="text-[11px] text-muted-foreground leading-normal mt-1.5">
                    {{ t("models.classificationHint") }}
                  </p>
                </div>
                <div class="grid gap-4 sm:grid-cols-2 items-start">
                  <div class="grid gap-2">
                    <Label for="modelStatus" class="text-xs font-medium text-foreground">
                      {{ t("models.modelStatus") }}
                    </Label>
                    <Select v-model="statusModel">
                      <SelectTrigger id="modelStatus" class="h-9 bg-background">
                        <SelectValue :placeholder="t('models.modelStatusNone')" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">{{ t("models.modelStatusNone") }}</SelectItem>
                        <SelectItem value="beta">beta</SelectItem>
                        <SelectItem value="deprecated">deprecated</SelectItem>
                      </SelectContent>
                    </Select>
                    <p class="text-[11px] text-muted-foreground leading-normal">
                      {{ t("models.modelStatusHelp") }}
                    </p>
                  </div>
                  <div class="grid gap-2">
                    <Label for="modelFamily" class="text-xs font-medium text-foreground">
                      {{ t("models.modelFamily") }}
                    </Label>
                    <Input
                      id="modelFamily"
                      v-model="familyModel"
                      placeholder="claude-sonnet"
                      class="h-9"
                    />
                    <p class="text-[11px] text-muted-foreground leading-normal">
                      {{ t("models.modelFamilyHelp") }}
                    </p>
                  </div>
                  <div class="grid gap-2">
                    <Label for="modelKnowledge" class="text-xs font-medium text-foreground">
                      {{ t("models.knowledgeCutoff") }}
                    </Label>
                    <Input id="modelKnowledge" v-model="knowledgeModel" type="date" class="h-9" />
                    <p class="text-[11px] text-muted-foreground leading-normal">
                      {{ t("models.knowledgeHelp") }}
                    </p>
                  </div>
                  <div class="grid gap-2">
                    <Label for="modelReleaseDate" class="text-xs font-medium text-foreground">
                      {{ t("models.releaseDate") }}
                    </Label>
                    <Input
                      id="modelReleaseDate"
                      v-model="releaseDateModel"
                      type="date"
                      class="h-9"
                    />
                    <p class="text-[11px] text-muted-foreground leading-normal">
                      {{ t("models.releaseDateHelp") }}
                    </p>
                  </div>
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
