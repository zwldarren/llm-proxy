<script setup lang="ts">
import type { PrimitiveProps } from "reka-ui";
import type { HTMLAttributes } from "vue";
import { Primitive } from "reka-ui";
import { cn } from "@/lib/utils";

/**
 * In the icon rail the label slides up under the first item of its group and
 * fades out, but it used to keep its 32px box and its pointer events — so the
 * element on top of that first item was the invisible label, and clicking
 * Usage / Chat / API Keys / Providers did nothing until the sidebar was
 * expanded again.
 */
const ICON_RAIL_CLASSES =
  "group-data-[collapsible=icon]:pointer-events-none group-data-[collapsible=icon]:-mt-8 group-data-[collapsible=icon]:opacity-0";

const props = defineProps<
  PrimitiveProps & {
    class?: HTMLAttributes["class"];
  }
>();
</script>

<template>
  <Primitive
    data-slot="sidebar-group-label"
    data-sidebar="group-label"
    :as="as"
    :as-child="asChild"
    :class="
      cn(
        'text-sidebar-foreground/70 ring-sidebar-ring flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium outline-hidden transition-opacity duration-200 ease-linear focus-visible:ring-2 [&>svg]:size-4 [&>svg]:shrink-0',
        ICON_RAIL_CLASSES,
        props.class
      )
    "
  >
    <slot />
  </Primitive>
</template>
