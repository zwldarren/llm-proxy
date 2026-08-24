<script setup lang="ts">
/**
 * ValueSwap — crossfade-rise transition for a metric value that changes in
 * place (e.g. dashboard numbers after a date-range change). The old value
 * lifts out quickly, the new one rises in with the system's exponential
 * ease-out, so a data update reads as the console settling, not a hard cut.
 *
 * Transition classes live in main.css (`.value-swap-*`) and are neutralized
 * by the global prefers-reduced-motion rule. Initial render never animates
 * (no `appear`), so first paint is instant.
 */
defineProps<{
  /** The rendered value; the transition re-keys on every change. */
  value: string | number;
}>();
</script>

<template>
  <Transition name="value-swap" mode="out-in">
    <span :key="value" class="inline-block">{{ value }}</span>
  </Transition>
</template>
