<script setup lang="ts">
import { AlertCircle, ChevronDown, Loader2, RefreshCw, Search } from "@lucide/vue";
import { computed, nextTick, ref, useId, watch } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";

import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { configApi } from "@/services/api/config";
import type { ProviderModelInfo } from "@/types/schemas";
import { getErrorMessage } from "@/utils/error";
import { formatContextLength } from "@/utils/format";

const props = defineProps<{
  providerName: string;
  modelValue: string;
  placeholder?: string;
  disabled?: boolean;
}>();

const emit = defineEmits<(e: "update:modelValue", value: string) => void>();

const { t } = useI18n();

const isOpen = ref(false);
const isLoading = ref(false);
const error = ref<string | null>(null);
const models = ref<ProviderModelInfo[]>([]);
const hasFetched = ref(false);
const searchQuery = ref("");

// ARIA wiring: the search field owns the combobox role (it is the element that
// holds DOM focus while the popup is open), and points at the listbox through
// aria-activedescendant so arrow keys move a highlight instead of focus.
const listboxId = useId();
const optionId = (index: number) => `${listboxId}-option-${index}`;
const highlightedIndex = ref(-1);

// Filter models based on search query
const filteredModels = computed(() => {
  if (!searchQuery.value) return models.value;
  const query = searchQuery.value.toLowerCase();
  return models.value.filter(
    (model) =>
      model.id.toLowerCase().includes(query) ||
      model.name.toLowerCase().includes(query) ||
      model.description?.toLowerCase().includes(query) ||
      // Modalities and parameters are shown as badges, so they must be
      // searchable: "image" has to find the image models.
      model.architecture?.output_modalities.some((m) => m.toLowerCase().includes(query)) ||
      model.architecture?.input_modalities.some((m) => m.toLowerCase().includes(query)) ||
      model.supported_parameters?.some((p) => p.toLowerCase().includes(query))
  );
});

/** Compact per-1M price from the upstream's per-token figure. */
const formatPerMillion = (perToken?: number | null): string | null =>
  typeof perToken === "number" ? `$${(perToken * 1_000_000).toFixed(2)}` : null;

const priceHint = (model: ProviderModelInfo): string | null => {
  const input = formatPerMillion(model.pricing?.prompt);
  const output = formatPerMillion(model.pricing?.completion);
  const parts = [
    input ? `${t("models.inputShort")} ${input}` : null,
    output ? `${t("models.outputShort")} ${output}` : null,
  ].filter(Boolean);
  return parts.length ? `${parts.join(" · ")} ${t("models.perMillionSuffix")}` : null;
};

/** ``text→image`` for models that are not plain text-to-text. */
const modalityHint = (model: ProviderModelInfo): string | null => {
  const input = model.architecture?.input_modalities ?? [];
  const output = model.architecture?.output_modalities ?? [];
  if (!input.length && !output.length) return null;
  const isPlainText = output.every((m) => m === "text") && input.every((m) => m === "text");
  if (isPlainText) return null;
  return `${input.join("/") || "?"}→${output.join("/") || "?"}`;
};

// Fetch models when provider changes or popover opens
const fetchModels = async () => {
  if (!props.providerName) {
    models.value = [];
    error.value = null;
    return;
  }

  isLoading.value = true;
  error.value = null;

  try {
    const response = await configApi.getProviderModels(props.providerName);
    models.value = response.models;
    hasFetched.value = true;
  } catch (e) {
    error.value = getErrorMessage(e);
    models.value = [];
  } finally {
    isLoading.value = false;
  }
};

// Fetch models when popover opens (lazy loading); reset the keyboard highlight
// whenever the popup closes, including an Escape or outside-click dismissal.
watch(isOpen, (open) => {
  if (open && !hasFetched.value && props.providerName) {
    fetchModels();
  }
  if (!open) {
    highlightedIndex.value = -1;
    searchQuery.value = "";
  }
});

// Keep the highlight inside the filtered list as the query changes it.
watch(
  () => filteredModels.value.length,
  (count) => {
    if (count === 0) highlightedIndex.value = -1;
    else if (highlightedIndex.value >= count) highlightedIndex.value = 0;
  }
);

// Reset when provider changes
watch(
  () => props.providerName,
  () => {
    models.value = [];
    hasFetched.value = false;
    error.value = null;
    searchQuery.value = "";
  }
);

const selectModel = (modelId: string) => {
  emit("update:modelValue", modelId);
  isOpen.value = false;
  searchQuery.value = "";
};

const setHighlight = (index: number) => {
  highlightedIndex.value = index;
  void nextTick(() => {
    document.getElementById(optionId(index))?.scrollIntoView({ block: "nearest" });
  });
};

const moveHighlight = (delta: number) => {
  const count = filteredModels.value.length;
  if (count === 0) return;
  const current = highlightedIndex.value;
  if (current < 0) {
    setHighlight(delta > 0 ? 0 : count - 1);
    return;
  }
  setHighlight((current + delta + count) % count);
};

const handleListKeydown = (event: KeyboardEvent) => {
  // The manual-entry field below the list is a plain text input: leave Enter
  // and the movement keys to its native behaviour.
  if (event.target instanceof HTMLElement && event.target.closest("[data-manual-entry]")) {
    return;
  }
  switch (event.key) {
    case "ArrowDown":
      event.preventDefault();
      moveHighlight(1);
      break;
    case "ArrowUp":
      event.preventDefault();
      moveHighlight(-1);
      break;
    case "Home":
      event.preventDefault();
      setHighlight(0);
      break;
    case "End":
      event.preventDefault();
      setHighlight(filteredModels.value.length - 1);
      break;
    case "Enter": {
      const model = filteredModels.value[highlightedIndex.value];
      if (!model) return;
      event.preventDefault();
      selectModel(model.id);
      break;
    }
  }
};

const handleRefresh = () => {
  hasFetched.value = false;
  fetchModels();
};

// Get display value for the trigger
const displayValue = computed(() => {
  if (props.modelValue) {
    const model = models.value.find((m) => m.id === props.modelValue);
    return model ? model.name : props.modelValue;
  }
  return "";
});
</script>

<template>
  <Popover v-model:open="isOpen">
    <PopoverTrigger as-child>
      <Button
        variant="outline"
        aria-haspopup="listbox"
        :aria-expanded="isOpen"
        :aria-controls="isOpen ? listboxId : undefined"
        :disabled="disabled || !providerName"
        class="w-full justify-between h-9 text-xs font-normal"
        @keydown.down.prevent="isOpen = true"
        @keydown.up.prevent="isOpen = true"
      >
        <span :class="{ 'text-muted-foreground': !modelValue }">
          {{ displayValue || placeholder || t("models.selectProviderModel") }}
        </span>
        <ChevronDown class="ml-2 h-3 w-3 shrink-0 opacity-50" />
      </Button>
    </PopoverTrigger>
    <PopoverContent class="w-80 p-0" align="start" @keydown="handleListKeydown">
      <div class="flex flex-col">
        <!-- Search header -->
        <div class="flex items-center border-b h-9">
          <div class="flex items-center flex-1 gap-2 px-2">
            <Search class="size-4 shrink-0 opacity-50" />
            <Input
              v-model="searchQuery"
              role="combobox"
              aria-autocomplete="list"
              :aria-expanded="isOpen"
              :aria-controls="listboxId"
              :aria-activedescendant="
                highlightedIndex >= 0 ? optionId(highlightedIndex) : undefined
              "
              :aria-label="t('models.searchModels')"
              :placeholder="t('models.searchModels')"
              class="h-8 text-xs border-0 focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent shadow-none pl-1.5 pr-2"
            />
          </div>
          <Tooltip>
            <TooltipTrigger as-child>
              <Button
                variant="ghost"
                size="icon"
                class="h-9 w-9 shrink-0 rounded-l-none"
                :disabled="isLoading"
                @click="handleRefresh"
                :aria-label="t('common.refresh')"
              >
                <Loader2 v-if="isLoading" class="h-4 w-4 animate-spin" />
                <RefreshCw v-else class="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{{ t("common.refresh") }}</TooltipContent>
          </Tooltip>
        </div>

        <!-- Loading state (only on first fetch) -->
        <div v-if="isLoading && models.length === 0" class="flex items-center justify-center py-6">
          <Loader2 class="h-5 w-5 animate-spin text-muted-foreground" />
          <span class="ml-2 text-sm text-muted-foreground">{{ t("models.loadingModels") }}</span>
        </div>

        <!-- Error state -->
        <div v-else-if="error" class="flex flex-col items-center justify-center py-6 px-4">
          <AlertCircle class="h-5 w-5 text-destructive mb-2" />
          <p class="text-sm text-destructive text-center">{{ error }}</p>
          <Button variant="outline" size="sm" class="mt-3" @click="handleRefresh">
            {{ t("common.refresh") }}
          </Button>
        </div>

        <!-- Model items (scrollable) -->
        <div
          v-else-if="models.length > 0"
          :id="listboxId"
          role="listbox"
          :aria-label="t('models.selectProviderModel')"
          class="max-h-[250px] overflow-y-auto"
        >
          <div
            v-for="(model, index) in filteredModels"
            :key="model.id"
            :id="optionId(index)"
            role="option"
            :aria-selected="model.id === modelValue"
            class="flex flex-col items-start px-3 py-2 cursor-pointer hover:bg-accent"
            :class="{
              'bg-accent': model.id === modelValue,
              'bg-accent/60': index === highlightedIndex && model.id !== modelValue,
            }"
            @mouseenter="highlightedIndex = index"
            @click="selectModel(model.id)"
          >
            <div class="flex items-center justify-between gap-2 w-full">
              <span class="text-sm font-medium truncate flex-1" :title="model.name">{{
                model.name
              }}</span>
              <span v-if="model.owned_by" class="text-[11px] text-muted-foreground shrink-0">
                {{ model.owned_by }}
              </span>
            </div>
            <span
              class="text-[11px] text-muted-foreground font-mono truncate w-full"
              :title="model.id"
              >{{ model.id }}</span
            >
            <div class="flex items-center gap-1.5 mt-0.5 text-[11px] text-muted-foreground">
              <span v-if="model.context_length" class="shrink-0 font-mono">
                {{ formatContextLength(model.context_length) }}
                {{ t("models.contextShort") }}
              </span>
              <span v-if="modalityHint(model)" class="shrink-0 font-mono">{{
                modalityHint(model)
              }}</span>
              <span v-if="priceHint(model)" class="shrink-0 font-mono">{{ priceHint(model) }}</span>
            </div>
            <span
              v-if="model.description"
              class="text-[11px] text-muted-foreground mt-0.5 line-clamp-2 wrap-break-word"
              :title="model.description"
            >
              {{ model.description }}
            </span>
          </div>

          <!-- No results after filtering -->
          <div v-if="filteredModels.length === 0 && searchQuery" class="py-6 text-center">
            <p class="text-sm text-muted-foreground">{{ t("models.noModelsFound") }}</p>
          </div>
        </div>

        <!-- Empty state: no models available -->
        <div v-else-if="hasFetched && models.length === 0" class="py-6 text-center">
          <p class="text-sm text-muted-foreground">{{ t("models.noModelsAvailable") }}</p>
        </div>

        <!-- Manual input option -->
        <div class="border-t px-3 py-2" data-manual-entry>
          <p class="text-[11px] text-muted-foreground mb-1.5">
            {{ t("models.orEnterManually") }}
          </p>
          <Input
            :model-value="modelValue"
            @update:model-value="emit('update:modelValue', $event as string)"
            :placeholder="t('placeholders.providerModelName')"
            class="h-7 text-xs"
          />
        </div>
      </div>
    </PopoverContent>
  </Popover>
</template>
