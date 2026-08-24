<script setup lang="ts">
import { AlertCircle, ChevronDown, Loader2, RefreshCw, Search } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";

import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { configApi } from "@/services/api/config";
import type { ProviderModelInfo } from "@/types/schemas";
import { getErrorMessage } from "@/utils/error";

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

// Filter models based on search query
const filteredModels = computed(() => {
  if (!searchQuery.value) return models.value;
  const query = searchQuery.value.toLowerCase();
  return models.value.filter(
    (model) =>
      model.id.toLowerCase().includes(query) ||
      model.name.toLowerCase().includes(query) ||
      model.description?.toLowerCase().includes(query)
  );
});

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

// Fetch models when popover opens (lazy loading)
watch(isOpen, (open) => {
  if (open && !hasFetched.value && props.providerName) {
    fetchModels();
  }
});

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
        role="combobox"
        :aria-expanded="isOpen"
        :disabled="disabled || !providerName"
        class="w-full justify-between h-9 text-xs font-normal"
      >
        <span :class="{ 'text-muted-foreground': !modelValue }">
          {{ displayValue || placeholder || t("models.selectProviderModel") }}
        </span>
        <ChevronDown class="ml-2 h-3 w-3 shrink-0 opacity-50" />
      </Button>
    </PopoverTrigger>
    <PopoverContent class="w-80 p-0" align="start">
      <div class="flex flex-col">
        <!-- Search header -->
        <div class="flex items-center border-b h-9">
          <div class="flex items-center flex-1 gap-2 px-2">
            <Search class="size-4 shrink-0 opacity-50" />
            <Input
              v-model="searchQuery"
              :placeholder="t('models.searchModels')"
              class="h-8 text-xs border-0 focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent shadow-none pl-1.5 pr-2"
            />
          </div>
          <Button
            variant="ghost"
            size="icon"
            class="h-9 w-9 shrink-0 rounded-l-none"
            :disabled="isLoading"
            @click="handleRefresh"
            :title="t('common.refresh')"
          >
            <Loader2 v-if="isLoading" class="h-4 w-4 animate-spin" />
            <RefreshCw v-else class="h-4 w-4" />
          </Button>
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
        <div v-else-if="models.length > 0" class="max-h-[250px] overflow-y-auto">
          <div
            v-for="model in filteredModels"
            :key="model.id"
            class="flex flex-col items-start px-3 py-2 cursor-pointer hover:bg-accent"
            :class="{ 'bg-accent': model.id === modelValue }"
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
        <div class="border-t px-3 py-2">
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
