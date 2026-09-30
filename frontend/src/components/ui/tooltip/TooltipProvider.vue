<script setup lang="ts">
import { TooltipProvider, type TooltipProviderProps } from "reka-ui";

// `ignoreNonKeyboardFocus` keeps tooltips closed when an ancestor element
// moves focus programmatically — most notably a Dialog/Sheet auto-focusing
// its first focusable child on open, which otherwise leaves the tooltip
// stuck open until the user clicks elsewhere.
const props = withDefaults(defineProps<TooltipProviderProps>(), {
  delayDuration: 200,
  ignoreNonKeyboardFocus: true,
});
</script>

<template>
  <TooltipProvider v-bind="props">
    <slot />
  </TooltipProvider>
</template>
