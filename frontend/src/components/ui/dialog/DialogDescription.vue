<script setup lang="ts">
import type { DialogDescriptionProps } from "reka-ui";
import type { HTMLAttributes } from "vue";
import { reactiveOmit } from "@vueuse/core";
import { DialogDescription, useForwardProps } from "reka-ui";
import { cn } from "@/lib/utils";

const props = withDefaults(
  defineProps<
    DialogDescriptionProps & {
      class?: HTMLAttributes["class"];
      size?: "sm" | "xs";
    }
  >(),
  {
    size: "sm",
  }
);

const delegatedProps = reactiveOmit(props, "class", "size");

const forwardedProps = useForwardProps(delegatedProps);
</script>

<template>
  <DialogDescription
    data-slot="dialog-description"
    v-bind="forwardedProps"
    :class="
      cn(
        'text-muted-foreground',
        props.size === 'xs' ? 'text-xs leading-normal' : 'text-sm',
        props.class
      )
    "
  >
    <slot />
  </DialogDescription>
</template>
