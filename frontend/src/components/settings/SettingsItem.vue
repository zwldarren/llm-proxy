<script setup lang="ts">
import type { Component } from "vue";
import { Loader2 } from "@lucide/vue";

interface Props {
  icon?: Component;
  title: string;
  description?: string;
  variant?: "row" | "nested";
  loading?: boolean;
  error?: string | null;
  /**
   * Announce this row's error as a live alert. Off by default: the section
   * paints its shared save error in every row, and the section owns one live
   * region for it (see SettingsSection). Set it on a row whose error is its own.
   */
  errorLive?: boolean;
}

withDefaults(defineProps<Props>(), {
  variant: "row",
  loading: false,
  error: null,
  errorLive: false,
});
</script>

<template>
  <!--
    Grid instead of a flex row. From sm up this is the desktop row: icon, then
    title with its description underneath, then the control on the right edge.

    On a phone the control moves to its own line instead of sharing one with
    the title. It used to keep the third column, and a control with any
    intrinsic width — a text input, a pair of steppers — then took what it
    needed and left the title column 32px wide, so "Extra Sensitive Keys"
    wrapped one character per line. The control keeps its natural width and the
    text keeps the full row.
  -->
  <!-- Inner padding is px-5 (20px) at every level, the same measure
       PanelSection's p-5 uses, so a settings card and a panel card share a left
       edge. sm:pl-11 here and sm:pl-10.5 on the section description are the one
       deliberate exception: both align to the icon chip's width plus its gap. -->
  <div
    class="group transition-colors duration-200 grid items-start gap-x-3.5 gap-y-1 sm:items-center sm:gap-x-6"
    :class="[
      variant === 'nested'
        ? 'grid-cols-[minmax(0,1fr)] sm:grid-cols-[minmax(0,1fr)_auto] py-3.5 px-5 pl-6 sm:pl-11'
        : 'grid-cols-[auto_minmax(0,1fr)] sm:grid-cols-[auto_minmax(0,1fr)_auto] py-4 px-5',
      'hover:bg-muted/12 first:rounded-t-xl last:rounded-b-xl',
    ]"
  >
    <div v-if="icon && variant !== 'nested'" class="col-start-1 row-start-1 shrink-0">
      <div
        class="flex items-center justify-center size-9 rounded-lg bg-muted text-foreground border border-border/40"
      >
        <component :is="icon" class="size-4.5 text-foreground/80" />
      </div>
    </div>

    <div
      class="min-w-0"
      :class="variant === 'nested' ? 'col-start-1 row-start-1' : 'col-start-2 row-start-1'"
    >
      <div class="text-sm font-semibold text-foreground tracking-tight break-words">
        {{ title }}
      </div>
    </div>

    <div
      v-if="description || error"
      class="min-w-0"
      :class="variant === 'nested' ? 'col-start-1 row-start-2' : 'col-start-2 row-start-2'"
    >
      <div
        v-if="description"
        class="text-xs text-muted-foreground leading-normal max-w-[60ch] break-words"
      >
        {{ description }}
      </div>
      <div
        v-if="error"
        :role="errorLive ? 'alert' : undefined"
        class="text-xs text-destructive mt-1 flex items-center gap-1 min-w-0"
      >
        <span class="break-words min-w-0">{{ error }}</span>
      </div>
    </div>

    <div
      class="flex min-w-0 max-w-full flex-wrap items-center gap-2 justify-end self-start sm:self-center"
      :class="[
        variant === 'nested'
          ? 'col-start-1 row-start-3 sm:col-start-2 sm:row-start-1 sm:row-span-2'
          : 'col-start-1 col-span-2 row-start-3 sm:col-start-3 sm:col-span-1 sm:row-start-1 sm:row-span-2',
        loading ? 'opacity-70 pointer-events-none' : '',
      ]"
    >
      <Loader2 v-if="loading" class="size-3.5 animate-spin text-muted-foreground/80 shrink-0" />
      <slot name="action" />
    </div>
  </div>
</template>
