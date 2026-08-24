<script setup lang="ts">
import { List, Table2 } from "@lucide/vue";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { ViewMode } from "@/composables/useViewMode";

interface Props {
  modelValue: ViewMode;
}

defineProps<Props>();
defineEmits<{
  "update:modelValue": [value: ViewMode];
}>();
</script>

<template>
  <ToggleGroup
    :model-value="modelValue"
    type="single"
    :spacing="1"
    class="shrink-0 rounded-lg border border-border/60 bg-background/70 p-0.5 shadow-[inset_0_1px_0_hsl(var(--background)/0.7)]"
    @update:model-value="$emit('update:modelValue', $event as ViewMode)"
  >
    <ToggleGroupItem
      value="list"
      :aria-label="$t('common.listView')"
      class="h-9 min-w-9 rounded-md px-2 data-[state=on]:bg-primary/12 data-[state=on]:text-primary"
    >
      <List class="w-4 h-4" />
    </ToggleGroupItem>
    <ToggleGroupItem
      value="table"
      :aria-label="$t('common.tableView')"
      class="h-9 min-w-9 rounded-md px-2 data-[state=on]:bg-primary/12 data-[state=on]:text-primary"
    >
      <Table2 class="w-4 h-4" />
    </ToggleGroupItem>
  </ToggleGroup>
</template>
