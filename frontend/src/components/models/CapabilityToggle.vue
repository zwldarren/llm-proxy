<script setup lang="ts">
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

/**
 * Reusable label + help-text + Switch row for model capability toggles.
 * Keeps the advanced-capabilities grid consistent and makes adding a new
 * capability a one-line change in the parent config array.
 */
interface Props {
  modelValue?: boolean;
  label: string;
  helpText: string;
}

defineProps<Props>();
defineEmits<{ "update:modelValue": [value: boolean] }>();
</script>

<template>
  <div class="flex items-start justify-between gap-4 py-3">
    <div class="space-y-0.5">
      <Label class="text-xs font-medium text-foreground">{{ label }}</Label>
      <p class="text-[11px] text-muted-foreground leading-relaxed">{{ helpText }}</p>
    </div>
    <Switch
      :model-value="modelValue"
      class="mt-0.5"
      @update:model-value="$emit('update:modelValue', $event)"
    />
  </div>
</template>
