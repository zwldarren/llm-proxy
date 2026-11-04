<script setup lang="ts">
import type { TagsInputItemProps } from "reka-ui";
import type { HTMLAttributes } from "vue";

import { reactiveOmit } from "@vueuse/core";
import { TagsInputItem, useForwardProps } from "reka-ui";
import { cn } from "@/lib/utils";
import { vDropEmptyAriaLabelledby } from "./dropEmptyAriaLabelledby";

const props = defineProps<TagsInputItemProps & { class?: HTMLAttributes["class"] }>();

const delegatedProps = reactiveOmit(props, "class");

const forwardedProps = useForwardProps(delegatedProps);
</script>

<template>
  <TagsInputItem
    v-drop-empty-aria-labelledby
    v-bind="forwardedProps"
    :class="
      cn(
        'flex h-5 min-w-0 max-w-full items-center rounded-md bg-secondary data-[state=active]:ring-ring data-[state=active]:ring-2 data-[state=active]:ring-offset-2 ring-offset-background',
        props.class
      )
    "
  >
    <slot />
  </TagsInputItem>
</template>
