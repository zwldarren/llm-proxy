<script setup lang="ts">
/**
 * Compact key/value chip for the Logs I/O views — the single styling for
 * scalar request parameters (request column) and usage/response meta tags
 * (response column), so both columns read identically.
 *
 * A quiet annotation, not a mini card: borderless tonal pill (tags/chips are
 * the pill tier of the radius system). Mono label at muted strength,
 * semibold tabular value, value truncates and the full `label: value` pair
 * lives in the title tooltip (pass `title` to override, e.g. to show the
 * untruncated value).
 */
withDefaults(
  defineProps<{
    label: string;
    value: string | number;
    /** Tooltip text; defaults to `label: value`. */
    title?: string;
  }>(),
  { title: undefined }
);
</script>

<template>
  <span
    class="text-data-xs inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-full bg-muted/50 px-2.5 py-0.5"
    :title="title ?? `${label}: ${value}`"
  >
    <span class="shrink-0 lowercase text-muted-foreground/70">{{ label }}</span>
    <span class="min-w-0 truncate font-semibold tabular-nums text-foreground/80">
      {{ value }}
    </span>
  </span>
</template>
