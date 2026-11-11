<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useDebounceFn } from "@vueuse/core";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { SegmentedControl, SegmentedControlItem } from "@/components/ui/segmented-control";
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
/** Inline parse error for the JSON view; null when the text is valid. */
const jsonError = ref<string | null>(null);

/**
 * Operations mode. The key/value list can only express the simple shape of the
 * config, so once the value carries an `operations` array, the visual tab is not
 * an editor for it — it would render "nothing configured" and then overwrite the
 * operations on the next keystroke. JSON is the only view for that value.
 */
const isOperationsMode = computed(() => "operations" in props.modelValue);

// Serialized config of the last local emit. The watcher below skips echoes of
// our own emits so it does not rebuild the entry list mid-edit (which would
// drop rows whose key is still empty).
let lastLocalEmit: string | null = null;

const syncEntriesToModel = () => {
  const newConfig: ParameterOverridesConfig = {};
  // Carry the keys the visual list does not model. Rebuilding from the rows
  // alone silently deleted an `operations` array the operator never touched.
  if (isOperationsMode.value) {
    newConfig.operations = props.modelValue.operations;
  }
  for (const entry of simpleEntries.value) {
    if (entry.key.trim()) {
      newConfig[entry.key] = entry.value;
    }
  }
  lastLocalEmit = JSON.stringify(newConfig);
  jsonValue.value = JSON.stringify(newConfig, null, 2);
  jsonError.value = null;
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
      viewMode.value = "json";
    } else {
      simpleEntries.value = Object.entries(newVal)
        .filter(([key]) => key !== "operations")
        .map(([key, value]) => ({ key, value }));
    }
    // JSON view string
    jsonValue.value = JSON.stringify(newVal, null, 2);
    jsonError.value = null;
  },
  { immediate: true, deep: true }
);

/** Parse the JSON view's text into a config object, or null when unusable. */
const parseConfig = (jsonStr: string): ParameterOverridesConfig | null => {
  try {
    const parsed: unknown = JSON.parse(jsonStr.trim() || "{}");
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    return parsed as ParameterOverridesConfig;
  } catch {
    return null;
  }
};

/**
 * Commit the JSON view to the model. Errors are shown in place rather than as a
 * toast that outlives the field: the old value stayed in the parent while the
 * textarea held something else, and a disappearing toast left an operator
 * saving a payload without the content he had just typed.
 */
const commitJson = (jsonStr: string) => {
  const parsed = parseConfig(jsonStr);
  if (parsed === null) {
    jsonError.value = t("common.invalidJson");
    return;
  }
  jsonError.value = null;
  lastLocalEmit = JSON.stringify(parsed);
  emit("update:modelValue", parsed);
};

// Parse while typing (debounced) so the error appears next to the field, and
// flush immediately on blur instead of waiting out the debounce.
const commitJsonDebounced = useDebounceFn(commitJson, 300);

const onJsonBlur = (jsonStr: string) => {
  commitJsonDebounced.cancel();
  commitJson(jsonStr);
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
      <SegmentedControl :aria-label="t('parameterOverrides.viewMode')" size="sm">
        <SegmentedControlItem
          :active="viewMode === 'visual'"
          :disabled="isOperationsMode"
          @click="viewMode = 'visual'"
        >
          <ListTree class="w-3.5 h-3.5" />
          {{ t("parameterOverrides.visualView") }}
        </SegmentedControlItem>
        <SegmentedControlItem :active="viewMode === 'json'" @click="viewMode = 'json'">
          <Code2 class="w-3.5 h-3.5" />
          {{ t("parameterOverrides.jsonView") }}
        </SegmentedControlItem>
      </SegmentedControl>
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

    <!-- Operations mode: say why the visual editor is unavailable instead of
         showing an empty list that implies nothing is configured. -->
    <p
      v-if="isOperationsMode"
      class="rounded-lg border border-border/60 bg-muted/40 px-3 py-2 text-[11px] text-muted-foreground"
    >
      {{ t("parameterOverrides.operationsJsonOnly") }}
    </p>

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
            :aria-label="`${t('common.delete')} ${entry.key || index + 1}`"
            :title="t('common.delete')"
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
        :class="{ 'border-destructive': jsonError }"
        rows="8"
        :aria-invalid="jsonError ? 'true' : undefined"
        :aria-describedby="jsonError ? 'parameter-overrides-json-error' : undefined"
        @input="commitJsonDebounced(jsonValue)"
        @blur="onJsonBlur(jsonValue)"
      />
      <p
        v-if="jsonError"
        id="parameter-overrides-json-error"
        role="alert"
        class="text-[11px] text-destructive"
      >
        {{ jsonError }}
      </p>
      <p class="text-[11px] text-muted-foreground">
        {{ t("parameterOverrides.jsonModeHelp") }}
      </p>
    </div>
  </div>
</template>
