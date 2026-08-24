<script setup lang="ts">
import type { Component } from "vue";

interface Props {
  title: string;
  description?: string;
  icon?: Component;
  iconClass?: string;
  actions?: string;
  actionBorder?: boolean;
}

withDefaults(defineProps<Props>(), {
  description: "",
  iconClass: "",
  actionBorder: false,
});
</script>

<template>
  <div class="flex flex-col gap-2">
    <div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
      <div class="flex items-center gap-2.5">
        <div
          v-if="icon"
          class="flex items-center justify-center h-8 w-8 rounded-md bg-muted text-foreground border border-border/50 shrink-0"
        >
          <component :is="icon" class="w-4 h-4 text-foreground/80" :class="iconClass" />
        </div>
        <div class="space-y-0.5">
          <h1
            class="brand-heading text-xl sm:text-2xl text-foreground leading-tight"
            id="page-title"
          >
            {{ title }}
          </h1>
          <p
            v-if="description"
            class="text-muted-foreground text-xs max-w-2xl leading-normal"
            id="page-description"
          >
            {{ description }}
          </p>
        </div>
      </div>
      <div
        class="flex items-center gap-2 flex-wrap animate-in fade-in duration-300"
        :class="{ 'rounded-xl border border-border/55 bg-card/76 px-2 py-1': actionBorder }"
      >
        <slot name="actions" />
      </div>
    </div>
    <slot />
  </div>
</template>
