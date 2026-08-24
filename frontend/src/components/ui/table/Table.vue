<script setup lang="ts">
import type { HTMLAttributes } from "vue";
import { cn } from "@/lib/utils";

const props = defineProps<{
  containerClass?: HTMLAttributes["class"];
  class?: HTMLAttributes["class"];
}>();
</script>

<template>
  <div
    data-slot="table-container"
    :class="
      cn(
        'relative w-full overflow-auto rounded-md border border-border bg-card',
        props.containerClass
      )
    "
  >
    <table data-slot="table" :class="cn('w-full caption-bottom text-sm', props.class)">
      <slot />
    </table>
  </div>
</template>
