<script setup lang="ts">
import type { HTMLAttributes } from "vue";
import { cn } from "@/lib/utils";

const props = withDefaults(
  defineProps<{
    class?: HTMLAttributes["class"];
    variant?: "plain" | "bar";
    justify?: "end" | "between";
  }>(),
  {
    variant: "plain",
    justify: "end",
  }
);

const VARIANTS = {
  plain: "flex flex-col-reverse gap-2 sm:flex-row",
  bar: "flex flex-wrap items-center gap-2 border-t border-border/60 bg-muted/10 px-4 py-4 sm:px-6 shrink-0",
} as const;
</script>

<template>
  <div
    data-slot="dialog-footer"
    :class="
      cn(
        VARIANTS[variant],
        justify === 'between' ? 'justify-between' : 'sm:justify-end',
        props.class
      )
    "
  >
    <slot />
  </div>
</template>
