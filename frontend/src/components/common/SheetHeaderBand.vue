<script setup lang="ts">
/**
 * Sheet/drawer header band: icon tile + title/description. The standard
 * chrome on top of every right-drawer form (provider, api key, team member,
 * mcp server…). Title and description are slots so callers can pass i18n
 * expressions and conditional copy directly.
 */
import type { Component } from "vue";
import SheetDescription from "@/components/ui/sheet/SheetDescription.vue";
import SheetTitle from "@/components/ui/sheet/SheetTitle.vue";

defineProps<{
  /** Lucide (or any) icon component rendered in the tile. */
  icon?: Component;
}>();
</script>

<template>
  <div
    class="px-4 sm:px-6 py-4 sm:py-5 border-b border-border/60 bg-muted/10 shrink-0 relative pr-16"
  >
    <div class="flex items-center gap-3">
      <div
        v-if="icon"
        class="p-2.5 rounded-md bg-muted border border-border/40 text-muted-foreground shrink-0 flex items-center justify-center"
      >
        <component :is="icon" class="size-5" />
      </div>
      <div class="flex flex-col min-w-0">
        <SheetTitle class="text-sm sm:text-base font-semibold text-foreground">
          <slot name="title" />
        </SheetTitle>
        <SheetDescription class="text-xs text-muted-foreground mt-1 truncate">
          <slot name="description" />
        </SheetDescription>
      </div>
    </div>
  </div>
</template>
