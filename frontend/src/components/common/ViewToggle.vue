<script setup lang="ts">
import { List, Table2 } from "@lucide/vue";
import { SegmentedControl, SegmentedControlItem } from "@/components/ui/segmented-control";
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
  <SegmentedControl class="max-lg:hidden" :aria-label="$t('common.viewMode')" size="icon">
    <SegmentedControlItem
      :active="modelValue === 'list'"
      :aria-label="$t('common.listView')"
      @click="$emit('update:modelValue', 'list')"
    >
      <List class="size-4" />
    </SegmentedControlItem>
    <SegmentedControlItem
      :active="modelValue === 'table'"
      :aria-label="$t('common.tableView')"
      @click="$emit('update:modelValue', 'table')"
    >
      <Table2 class="size-4" />
    </SegmentedControlItem>
  </SegmentedControl>
</template>
