<script setup lang="ts">
/**
 * Formatted/Raw view-mode switch for the Logs I/O columns.
 *
 * Fully separates the structured rendering from raw payload inspection:
 * each column (request/response) flips independently, so operators can
 * compare e.g. a formatted request against the raw response stream.
 * One mode is always on — clicking the active item does not deselect it.
 */
import type { AcceptableValue } from "reka-ui";
import { useI18n } from "vue-i18n";
import { SegmentedControl, SegmentedControlItem } from "@/components/ui/segmented-control";

defineProps<{
  modelValue: "formatted" | "raw";
}>();

const emit = defineEmits<{
  "update:modelValue": [value: "formatted" | "raw"];
}>();

const { t } = useI18n();

// reka-ui single toggle-groups emit a nullish value when the active item is
// clicked again — ignore that so one mode always stays selected.
function onUpdate(value: AcceptableValue | AcceptableValue[]) {
  if (value === "formatted" || value === "raw") emit("update:modelValue", value);
}
</script>

<template>
  <SegmentedControl :aria-label="t('common.viewMode')" size="sm">
    <SegmentedControlItem
      :active="modelValue === 'formatted'"
      :aria-label="t('logs.formatted')"
      caps
      @click="onUpdate('formatted')"
    >
      {{ t("logs.formatted") }}
    </SegmentedControlItem>
    <SegmentedControlItem
      :active="modelValue === 'raw'"
      :aria-label="t('logs.raw')"
      caps
      @click="onUpdate('raw')"
    >
      {{ t("logs.raw") }}
    </SegmentedControlItem>
  </SegmentedControl>
</template>
