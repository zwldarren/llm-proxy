<script setup lang="ts">
import type { HTMLAttributes } from "vue";
import { cn } from "@/lib/utils";

const props = defineProps<{
  class?: HTMLAttributes["class"];
}>();
</script>

<template>
  <td
    data-slot="table-cell"
    :class="
      cn(
        'py-2 px-3 align-middle whitespace-nowrap',
        'text-sm text-foreground',
        '[&:has([role=checkbox])]:pr-0 *:[[role=checkbox]]:translate-y-0.5',
        props.class
      )
    "
  >
    <slot />
  </td>
</template>
