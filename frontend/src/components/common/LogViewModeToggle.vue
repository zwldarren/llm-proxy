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
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

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
  <ToggleGroup
    :model-value="modelValue"
    type="single"
    :spacing="1"
    class="shrink-0 rounded-lg border border-border/60 bg-background/70 p-0.5 shadow-[inset_0_1px_0_hsl(var(--background)/0.7)]"
    @update:model-value="onUpdate"
  >
    <ToggleGroupItem
      value="formatted"
      :aria-label="t('logs.formatted')"
      class="h-7 rounded-md px-2.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground data-[state=on]:bg-primary/12 data-[state=on]:text-primary data-[state=on]:hover:text-primary"
    >
      {{ t("logs.formatted") }}
    </ToggleGroupItem>
    <ToggleGroupItem
      value="raw"
      :aria-label="t('logs.raw')"
      class="h-7 rounded-md px-2.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground data-[state=on]:bg-primary/12 data-[state=on]:text-primary data-[state=on]:hover:text-primary"
    >
      {{ t("logs.raw") }}
    </ToggleGroupItem>
  </ToggleGroup>
</template>
