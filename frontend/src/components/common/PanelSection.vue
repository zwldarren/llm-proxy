<script setup lang="ts">
import type { Component } from "vue";

/**
 * A stacked personal-settings panel: one topic per card, read top-to-bottom.
 * The shared grammar for the Account workspace's Profile and Sign-in &
 * security tabs (DESIGN.md §Personal-Settings Panels). The page owns the
 * column width and rhythm; the panel only owns its heading, optional status
 * badge, and hairline surface.
 */
interface Props {
  title: string;
  description?: string;
  /** Topic icon, set muted at the trailing edge of the heading row. */
  icon?: Component;
}

withDefaults(defineProps<Props>(), {
  description: "",
  icon: undefined,
});
</script>

<template>
  <section class="rounded-xl border border-border/60 bg-card p-5 scroll-mt-6">
    <div class="flex items-start justify-between gap-4">
      <div class="min-w-0">
        <div class="flex flex-wrap items-center gap-2">
          <h2 class="font-medium">{{ title }}</h2>
          <slot name="badge" />
        </div>
        <p v-if="description" class="mt-1 text-xs text-muted-foreground">{{ description }}</p>
      </div>
      <component
        :is="icon"
        v-if="icon"
        class="size-5 shrink-0 text-muted-foreground"
        aria-hidden="true"
      />
    </div>

    <div class="mt-4">
      <slot />
    </div>
  </section>
</template>
