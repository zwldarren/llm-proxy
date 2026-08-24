<script setup lang="ts">
import type { Component } from "vue";

interface Props {
  title: string;
  icon?: Component;
  description?: string;
}

withDefaults(defineProps<Props>(), {
  icon: undefined,
  description: "",
});
</script>

<template>
  <div class="space-y-4">
    <!-- Section Header -->
    <div class="flex flex-col gap-1.5 pb-1 px-1">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2.5">
          <div
            v-if="icon"
            class="flex items-center justify-center size-8 rounded-lg bg-muted/65 text-foreground border border-border/40 shrink-0"
          >
            <component :is="icon" class="size-4 text-foreground/80" />
          </div>
          <h2 class="text-base font-semibold tracking-tight text-foreground">
            {{ title }}
          </h2>
        </div>
        <div class="flex items-center gap-2">
          <slot name="actions" />
        </div>
      </div>
      <p
        v-if="description"
        class="text-xs text-muted-foreground max-w-2xl leading-normal sm:pl-10.5"
      >
        {{ description }}
      </p>
    </div>

    <!-- Section Body (Card Container) -->
    <div
      class="border border-border/50 rounded-xl overflow-hidden bg-card/35 backdrop-blur-xs divide-y divide-border/45"
    >
      <slot />
    </div>
  </div>
</template>
