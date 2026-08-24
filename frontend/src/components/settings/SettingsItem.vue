<script setup lang="ts">
import type { Component } from "vue";
import { Loader2 } from "@lucide/vue";

interface Props {
  icon?: Component;
  title: string;
  description?: string;
  variant?: "row" | "full-width" | "nested";
  loading?: boolean;
  error?: string | null;
}

withDefaults(defineProps<Props>(), {
  variant: "row",
  loading: false,
  error: null,
});
</script>

<template>
  <div
    class="group transition-colors duration-200"
    :class="[
      variant === 'row'
        ? 'flex flex-row items-center justify-between gap-6 py-4.5 px-5.5'
        : variant === 'nested'
          ? 'flex flex-row items-center justify-between gap-6 py-3.5 px-5.5 pl-6 sm:pl-11'
          : 'flex flex-col gap-3.5 py-4.5 px-5.5',
      'hover:bg-muted/12 first:rounded-t-xl last:rounded-b-xl',
    ]"
  >
    <div class="flex items-center gap-3.5 min-w-0 flex-1">
      <div v-if="icon && variant !== 'nested'" class="shrink-0">
        <div
          class="flex items-center justify-center size-9 rounded-lg bg-muted text-foreground border border-border/40"
        >
          <component :is="icon" class="size-4.5 text-foreground/80" />
        </div>
      </div>

      <div class="flex-1 min-w-0" :class="variant === 'full-width' ? 'w-full' : ''">
        <div class="text-sm font-semibold text-foreground tracking-tight">{{ title }}</div>
        <div
          v-if="description"
          class="text-xs text-muted-foreground mt-0.5 leading-normal max-w-[60ch]"
        >
          {{ description }}
        </div>
        <div v-if="error" class="text-xs text-destructive mt-1 flex items-center gap-1">
          <span>{{ error }}</span>
        </div>
      </div>
    </div>

    <div
      class="shrink-0 flex items-center gap-2"
      :class="[
        variant === 'full-width' ? 'w-full justify-end' : '',
        loading ? 'opacity-70 pointer-events-none' : '',
      ]"
    >
      <Loader2 v-if="loading" class="size-3.5 animate-spin text-muted-foreground/80 shrink-0" />
      <slot name="action" />
    </div>
  </div>
</template>
