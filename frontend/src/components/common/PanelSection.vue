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
  /**
   * Anchor target for the in-page links that point at a panel (`/account#access`,
   * `/account/security#passkeys`). Declared rather than inherited so the anchor
   * cannot break when the root element changes.
   */
  id?: string;
}

withDefaults(defineProps<Props>(), {
  description: "",
  icon: undefined,
  id: undefined,
});

/*
 * Layout stays with the page: callers pass their own rhythm class (`mt-6`,
 * `pt-2`) alongside the declared `id`. Turn off attribute inheritance and bind
 * `$attrs` on the surface explicitly, so that class merges with the panel's own
 * surface class instead of replacing it (Vue merges `class`/`style` on a single
 * element) and the contract is visible here rather than implied by fallthrough.
 */
defineOptions({ inheritAttrs: false });
</script>

<template>
  <section
    :id="id"
    v-bind="$attrs"
    class="rounded-xl border border-border/60 bg-card p-5 scroll-mt-6"
  >
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
