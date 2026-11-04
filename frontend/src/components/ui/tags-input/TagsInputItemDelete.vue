<script setup lang="ts">
import type { TagsInputItemDeleteProps } from "reka-ui";
import type { HTMLAttributes } from "vue";
import { X } from "@lucide/vue";
import { reactiveOmit } from "@vueuse/core";
import { TagsInputItemDelete, useForwardProps } from "reka-ui";
import { cn } from "@/lib/utils";
import { vDropEmptyAriaLabelledby } from "./dropEmptyAriaLabelledby";

const props = withDefaults(
  defineProps<
    TagsInputItemDeleteProps & {
      class?: HTMLAttributes["class"];
      /**
       * Accessible name for the remove control. Reka's `aria-labelledby`
       * reference to the token text is dropped (see the directive), so a direct
       * usage without this prop would ship an unnamed button.
       */
      label?: string;
    }
  >(),
  { label: undefined }
);

const delegatedProps = reactiveOmit(props, "class", "label");

const forwardedProps = useForwardProps(delegatedProps);
</script>

<template>
  <TagsInputItemDelete
    v-drop-empty-aria-labelledby
    :aria-label="props.label"
    v-bind="forwardedProps"
    :class="cn('flex shrink-0 rounded bg-transparent mr-1', props.class)"
  >
    <slot>
      <X class="w-4 h-4" />
    </slot>
  </TagsInputItemDelete>
</template>
