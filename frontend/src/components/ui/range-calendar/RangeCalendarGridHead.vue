<script lang="ts" setup>
import type { RangeCalendarGridHeadProps } from "reka-ui";
import type { HTMLAttributes } from "vue";
import { reactiveOmit } from "@vueuse/core";
import { RangeCalendarGridHead, useForwardProps } from "reka-ui";
import { cn } from "@/lib/utils";

const props = defineProps<RangeCalendarGridHeadProps & { class?: HTMLAttributes["class"] }>();

const delegatedProps = reactiveOmit(props, "class");

const forwardedProps = useForwardProps(delegatedProps);
</script>

<template>
  <RangeCalendarGridHead
    data-slot="range-calendar-grid-head"
    :class="cn(props.class)"
    v-bind="forwardedProps"
  >
    <slot />
  </RangeCalendarGridHead>
</template>
