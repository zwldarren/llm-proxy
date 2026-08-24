<script setup lang="ts">
import { Label } from "@/components/ui/label";

/**
 * One Smart Routing mode-weight row: label + mono value badge + the
 * design-system signature range slider (`.range-thumb`). Commits on release
 * (`change`) so auto-save fires once per adjustment, not per pixel.
 */
defineProps<{
  label: string;
  description: string;
  modelValue: number;
}>();

const emit = defineEmits<{
  (e: "commit", value: number): void;
}>();

function onChange(event: Event) {
  emit("commit", Number((event.target as HTMLInputElement).value));
}
</script>

<template>
  <div
    class="px-5.5 py-4.5 border-t border-border/40 bg-muted/5 hover:bg-muted/10 transition-colors duration-150"
  >
    <div class="flex items-center justify-between gap-3">
      <Label class="text-sm font-semibold text-foreground">{{ label }}</Label>
      <span
        class="text-data-xs text-foreground/90 bg-muted/40 px-2 py-0.5 rounded border border-border/30"
      >
        {{ modelValue.toFixed(2) }}
      </span>
    </div>
    <input
      type="range"
      min="0"
      max="1"
      step="0.05"
      :value="modelValue"
      :aria-label="label"
      :aria-valuetext="modelValue.toFixed(2)"
      class="range-thumb appearance-none w-full mt-3 h-2 bg-muted rounded-full cursor-pointer focus-visible:outline-none"
      @change="onChange"
    />
    <p class="text-xs text-muted-foreground mt-2.5 leading-normal max-w-[65ch]">
      {{ description }}
    </p>
  </div>
</template>
