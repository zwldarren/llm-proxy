<script setup lang="ts">
/**
 * The budget bar itself: one track, one fill, one progressbar role.
 *
 * Three surfaces draw this bar (API key spend cell, team member budget cell,
 * account budget banner) and each had its own copy of the markup and of the
 * ARIA wiring. Only the number's source differs between them, so only that
 * stays outside.
 *
 * The fill is a `scaleX` transform rather than a width, so the transition runs
 * on the compositor; the track's `overflow-hidden` clips a ratio above 1 (an
 * overspent budget) instead of letting the fill run past the track.
 */
const props = withDefaults(
  defineProps<{
    /** Spent / budget, unclamped. 0 when there is no budget. */
    ratio: number | null | undefined;
    /** Accessible name for the bar — the same sentence the row shows as title. */
    label: string;
    /** Tone of the fill (e.g. destructive once the budget is exceeded). */
    barClass?: string;
    class?: string;
  }>(),
  { barClass: "", class: "" }
);

const clamped = () => Math.min(1, Math.max(0, props.ratio ?? 0));
</script>

<template>
  <div
    class="h-1.5 w-full rounded-full bg-muted overflow-hidden"
    :class="props.class"
    role="progressbar"
    :aria-label="label"
    :aria-valuenow="Math.round(clamped() * 100)"
    aria-valuemin="0"
    aria-valuemax="100"
  >
    <div
      class="h-full w-full origin-left rounded-full transition-transform"
      :class="barClass"
      :style="{ transform: `scaleX(${clamped()})` }"
    />
  </div>
</template>
