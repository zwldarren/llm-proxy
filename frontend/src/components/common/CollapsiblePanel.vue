<script setup lang="ts">
/**
 * Grid-rows collapsible: animates between 0fr and 1fr so the panel's height
 * transitions smoothly without measuring content. Used across settings
 * sections to reveal sub-options when a parent toggle is on.
 */
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    /** Render a top border on the expanded state (matches section dividers). */
    bordered?: boolean;
  }>(),
  { bordered: false }
);

const panelClass = computed(() =>
  props.open
    ? props.bordered
      ? "grid-rows-[1fr] opacity-100 border-t border-border/40"
      : "grid-rows-[1fr] opacity-100"
    : "grid-rows-[0fr] opacity-0 pointer-events-none"
);
</script>

<template>
  <div class="grid transition-all duration-300 ease-in-out" :class="panelClass">
    <div class="overflow-hidden">
      <slot />
    </div>
  </div>
</template>
