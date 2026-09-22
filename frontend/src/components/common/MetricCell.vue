<script setup lang="ts">
/**
 * One cell of the log-details spec sheet: a hairline-divided panel with a
 * small uppercase label row and a trailing icon. Extracted so the cells in
 * LogDetailsMetrics share one definition of the chrome instead of repeating
 * the same class strings in every cell.
 */
defineProps<{
  label: string;
}>();
</script>

<template>
  <div class="bg-card p-3 sm:p-4 flex flex-col gap-1.5 sm:gap-2 min-w-0">
    <div
      class="flex items-center justify-between text-[11px] text-muted-foreground font-semibold uppercase tracking-wider"
    >
      <span>{{ label }}</span>
      <slot name="icon" />
    </div>
    <slot />
  </div>
</template>
