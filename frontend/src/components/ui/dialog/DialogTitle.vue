<script setup lang="ts">
import type { DialogTitleProps } from "reka-ui";
import type { HTMLAttributes } from "vue";
import { reactiveOmit } from "@vueuse/core";
import { DialogTitle, useForwardProps } from "reka-ui";
import { cn } from "@/lib/utils";

const props = withDefaults(
  defineProps<
    DialogTitleProps & {
      class?: HTMLAttributes["class"];
      size?: "lg" | "base";
    }
  >(),
  {
    size: "lg",
  }
);

const delegatedProps = reactiveOmit(props, "class", "size");

const forwardedProps = useForwardProps(delegatedProps);
</script>

<template>
  <DialogTitle
    data-slot="dialog-title"
    v-bind="forwardedProps"
    :class="
      cn(
        'flex min-w-0 items-center gap-2 leading-none font-semibold',
        props.size === 'base' ? 'text-base' : 'text-lg',
        props.class
      )
    "
  >
    <slot />
  </DialogTitle>
</template>
