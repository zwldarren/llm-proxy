<script setup lang="ts">
import type { Component, HTMLAttributes } from "vue";
import { computed, useSlots } from "vue";
import { cn } from "@/lib/utils";
import DialogDescription from "./DialogDescription.vue";
import DialogTitle from "./DialogTitle.vue";

type DialogHeaderVariant = "plain" | "band" | "divided" | "sr-only";

const props = withDefaults(
  defineProps<{
    class?: HTMLAttributes["class"];
    variant?: DialogHeaderVariant;
    icon?: Component;
    title?: string;
    description?: string;
  }>(),
  {
    variant: "plain",
  }
);

const slots = useSlots();

// `pr-16` reserves the close button's column (4px inset + 44px target); the
// `sm:` step repeats it because `sm:px-6` would otherwise win the media query
// and let the actions run under the close control.
const HEADER_VARIANTS: Record<DialogHeaderVariant, string> = {
  plain: "flex flex-col gap-2 text-center sm:text-left",
  band: "relative shrink-0 border-b border-border/60 bg-muted/10 px-4 py-4 pr-16 sm:px-6 sm:py-5 sm:pr-16",
  divided: "border-b border-border/60 pb-2",
  "sr-only": "sr-only",
};

const usesCompactType = computed(() => props.variant === "band" || props.variant === "divided");

// The icon tile and the actions column share one row with the text block; with
// neither, the wrappers step out of the way so the text stays a direct child of
// the header and keeps its own gap and alignment.
const hasAside = computed(() => Boolean(props.icon) || Boolean(slots.actions));
</script>

<template>
  <div data-slot="dialog-header" :class="cn(HEADER_VARIANTS[variant], props.class)">
    <template v-if="variant === 'sr-only'">
      <DialogTitle>
        <slot name="title">{{ title }}</slot>
      </DialogTitle>
      <DialogDescription>
        <slot name="description">{{ description }}</slot>
      </DialogDescription>
    </template>

    <div v-else :class="hasAside ? 'flex items-start gap-3' : 'contents'">
      <div
        v-if="icon"
        class="flex shrink-0 items-center justify-center rounded-md border border-border/40 bg-muted p-2.5 text-muted-foreground"
      >
        <component :is="icon" class="size-5" />
      </div>
      <div :class="hasAside ? 'flex min-w-0 flex-1 flex-col gap-1' : 'contents'">
        <DialogTitle v-if="title || $slots.title" :size="usesCompactType ? 'base' : 'lg'">
          <slot name="title">{{ title }}</slot>
        </DialogTitle>
        <DialogDescription
          v-if="description || $slots.description"
          :size="usesCompactType ? 'xs' : 'sm'"
        >
          <slot name="description">{{ description }}</slot>
        </DialogDescription>
        <slot />
      </div>
      <div v-if="$slots.actions" class="shrink-0">
        <slot name="actions" />
      </div>
    </div>
  </div>
</template>
