<script setup lang="ts">
import { ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Plus, Trash2, Code2, ListTree } from "@lucide/vue";
import type { ParameterOverridesConfig } from "@/types/parameterOverrides";

const props = defineProps<{
  modelValue: ParameterOverridesConfig;
}>();

const emit = defineEmits<{
  "update:modelValue": [value: ParameterOverridesConfig];
}>();

const { t } = useI18n();

const viewMode = ref<"json" | "visual">("visual");
const simpleEntries = ref<{ key: string; value: unknown }[]>([]);
const jsonValue = ref("");

// Serialized config of the last local emit. The watcher below skips echoes of
// our own emits so it does not rebuild the entry list mid-edit (which would
// drop rows whose key is still empty).
let lastLocalEmit: string | null = null;

const syncEntriesToModel = () => {
  const newConfig: ParameterOverridesConfig = {};
  for (const entry of simpleEntries.value) {
    if (entry.key.trim()) {
      newConfig[entry.key] = entry.value;
    }
  }
  lastLocalEmit = JSON.stringify(newConfig);
  jsonValue.value = JSON.stringify(newConfig, null, 2);
  emit("update:modelValue", newConfig);
};

// Sync modelValue to both visual and JSON view modes in a single watcher.
watch(
  () => props.modelValue,
  (newVal) => {
    if (lastLocalEmit !== null && JSON.stringify(newVal) === lastLocalEmit) {
      return;
    }
    // Visual mode entries
    if ("operations" in newVal) {
      simpleEntries.value = [];
    } else {
      simpleEntries.value = Object.entries(newVal)
        .filter(([key]) => key !== "operations")
        .map(([key, value]) => ({ key, value }));
    }
    // JSON view string
    jsonValue.value = JSON.stringify(newVal, null, 2);
  },
  { immediate: true, deep: true }
);

const syncJsonToModel = (jsonStr: string) => {
  try {
    const parsed = JSON.parse(jsonStr || "{}");
    lastLocalEmit = JSON.stringify(parsed);
    emit("update:modelValue", parsed);
  } catch {
    toast.error(t("common.invalidJson"));
  }
};

const addSimpleEntry = () => {
  simpleEntries.value.push({ key: "", value: "" });
};

const removeSimpleEntry = (index: number) => {
  simpleEntries.value.splice(index, 1);
  syncEntriesToModel();
};

const updateSimpleEntry = (
  index: number,
  field: "key" | "value",
  newValue: string | number | null
) => {
  const entry = simpleEntries.value[index];
  if (!entry) return;
  if (field === "key") {
    entry.key = String(newValue ?? "");
  } else {
    const strValue = String(newValue ?? "");
    try {
      entry.value = strValue ? JSON.parse(strValue) : "";
    } catch {
      entry.value = strValue;
    }
  }
  syncEntriesToModel();
};

const getSimpleValueDisplay = (value: unknown): string => {
  if (typeof value === "string") return value;
  return JSON.stringify(value);
};
</script>

<template>
  <div class="space-y-3">
    <!-- View Mode Toggle -->
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-1 bg-muted rounded-md p-1">
        <Button
          variant="ghost"
          size="sm"
          :class="['h-7 px-2 text-xs', viewMode === 'visual' && 'bg-background shadow-sm']"
          @click="viewMode = 'visual'"
        >
          <ListTree class="w-3.5 h-3.5 mr-1" />
          {{ t("parameterOverrides.visualView") }}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          :class="['h-7 px-2 text-xs', viewMode === 'json' && 'bg-background shadow-sm']"
          @click="viewMode = 'json'"
        >
          <Code2 class="w-3.5 h-3.5 mr-1" />
          {{ t("parameterOverrides.jsonView") }}
        </Button>
      </div>
      <Button
        v-if="viewMode === 'visual'"
        variant="outline"
        size="sm"
        class="h-7"
        @click="addSimpleEntry"
      >
        <Plus class="w-3.5 h-3.5 mr-1" />
        {{ t("parameterOverrides.addSimpleEntry") }}
      </Button>
    </div>

    <!-- Visual Mode -->
    <div v-if="viewMode === 'visual'" class="space-y-2">
      <div v-if="simpleEntries.length === 0" class="py-6 text-center text-sm text-muted-foreground">
        {{ t("parameterOverrides.noSimpleEntries") }}
      </div>
      <div v-else class="divide-y divide-border/60 border-y border-border/60">
        <div
          v-for="(entry, index) in simpleEntries"
          :key="index"
          class="flex items-start gap-2 py-3"
        >
          <div class="flex-1 grid grid-cols-2 gap-2">
            <div class="space-y-1">
              <Label class="text-[11px] text-muted-foreground" :class="{ 'sr-only': index > 0 }">{{
                t("parameterOverrides.key")
              }}</Label>
              <Input
                :model-value="entry.key"
                :placeholder="t('parameterOverrides.keyPlaceholder')"
                class="h-8 text-xs"
                @update:model-value="updateSimpleEntry(index, 'key', $event)"
              />
            </div>
            <div class="space-y-1">
              <Label class="text-[11px] text-muted-foreground" :class="{ 'sr-only': index > 0 }">{{
                t("parameterOverrides.value")
              }}</Label>
              <Input
                :model-value="getSimpleValueDisplay(entry.value)"
                :placeholder="t('parameterOverrides.valuePlaceholder')"
                class="h-8 text-xs font-mono"
                @update:model-value="updateSimpleEntry(index, 'value', $event)"
              />
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            class="h-8 w-8 p-0 text-muted-foreground hover:text-destructive shrink-0 self-center"
            @click="removeSimpleEntry(index)"
          >
            <Trash2 class="w-4 h-4" />
          </Button>
        </div>
      </div>
      <p class="text-[11px] text-muted-foreground">
        {{ t("parameterOverrides.simpleModeHelp") }}
      </p>
    </div>

    <!-- JSON Mode: Textarea -->
    <div v-else class="space-y-2">
      <Textarea
        v-model="jsonValue"
        class="code-textarea min-h-32 text-xs font-mono"
        rows="8"
        @blur="syncJsonToModel(jsonValue)"
      />
      <p class="text-[11px] text-muted-foreground">
        {{ t("parameterOverrides.jsonModeHelp") }}
      </p>
    </div>
  </div>
</template>
