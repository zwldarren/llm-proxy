<script setup lang="ts">
/**
 * A single run specimen in the tray: status dot + mono content slot.
 * Color is redundant — the dot is always paired with the status text/content.
 */
import { computed } from "vue";
import type { ChatRunStatus } from "@/types/runs";

const props = withDefaults(
  defineProps<{
    status: ChatRunStatus | "idle";
    selected?: boolean;
  }>(),
  { selected: false }
);

const dotClass = computed(() => {
  switch (props.status) {
    case "streaming":
      return "bg-foreground animate-pulse";
    case "ok":
      return "bg-status-success";
    case "error":
      return "bg-status-error";
    case "stopped":
      return "bg-status-warning";
    default:
      return "bg-status-unknown";
  }
});
</script>

<template>
  <button
    type="button"
    class="flex items-center gap-2 h-8 px-2.5 rounded-md border text-data-xs shrink-0 transition-[color,background-color,border-color] duration-150 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
    :class="
      selected
        ? 'border-foreground/30 bg-muted/70 text-foreground'
        : 'border-border/50 bg-transparent text-muted-foreground hover:bg-muted/40 hover:text-foreground'
    "
    :aria-pressed="selected"
  >
    <span class="size-1.5 rounded-full shrink-0" :class="dotClass" aria-hidden="true" />
    <slot />
  </button>
</template>
