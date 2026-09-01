<script setup lang="ts">
/**
 * One raw-mode section of the Logs I/O columns (headers / body / stream):
 * an icon + title above a bordered payload box, or the shared empty-state
 * note when nothing was recorded for this side.
 */
import type { Component } from "vue";

withDefaults(
  defineProps<{
    /** Section icon (lucide component). */
    icon: Component;
    /** Translated section title. */
    title: string;
    /** False renders the empty-state note instead of the content slot. */
    hasContent: boolean;
    /** Whether the payload box pads itself (off when content pads itself). */
    padded?: boolean;
  }>(),
  { padded: true }
);
</script>

<template>
  <div class="space-y-2">
    <h4 class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2">
      <component :is="icon" class="size-3.5" />
      {{ title }}
    </h4>
    <div
      v-if="hasContent"
      class="border border-border/40 rounded-lg overflow-hidden bg-muted/10"
      :class="padded ? 'p-3' : ''"
    >
      <slot />
    </div>
    <div
      v-else
      class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-lg text-center"
    >
      <slot name="empty" />
    </div>
  </div>
</template>
