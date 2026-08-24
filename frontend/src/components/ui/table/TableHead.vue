<script setup lang="ts">
import type { HTMLAttributes } from "vue";
import { cn } from "@/lib/utils";

const props = defineProps<{
  class?: HTMLAttributes["class"];
}>();
</script>

<template>
  <th
    data-slot="table-head"
    :class="
      cn(
        'h-8 px-3 py-1.5 text-left align-middle font-semibold text-xs',
        'text-muted-foreground uppercase tracking-wider',
        'bg-muted/30',
        'border-b border-border',
        'whitespace-nowrap',
        '[&:has([role=checkbox])]:pr-0 *:[[role=checkbox]]:translate-y-0.5',
        props.class
      )
    "
  >
    <slot />
  </th>
</template>
