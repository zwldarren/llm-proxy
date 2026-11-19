<script setup lang="ts">
import { computed, ref, useSlots, watch } from "vue";
import { cn } from "@/lib/utils";
import { monogramInitial, type BrandIcon } from "@/utils/icons";

/**
 * One brand tile: the logo when there is one, otherwise the name's initial.
 *
 * The tile owns everything that used to be repeated at every call site — the
 * frame, the mono theming (`.icon-mono`), the load-failure latch and the
 * fallback. Callers pass what they resolved (`modelIcon`/`providerIcon`) and get
 * something legible either way: a model nobody ships a logo for still renders as
 * a deliberate mark, not as a missing icon.
 */
const props = withDefaults(
  defineProps<{
    /** Resolved mark, or null to render the fallback. */
    icon?: BrandIcon | null;
    /** Name the mark belongs to: the alt text and the monogram letter. */
    label: string;
    size?: "xs" | "sm" | "md" | "lg" | "xl";
    /** `pill` is for the round log chips; `square` takes the size's radius. */
    shape?: "square" | "pill";
    /** Decorative marks sit next to a visible name and get no accessible name. */
    decorative?: boolean;
  }>(),
  { icon: null, size: "md", shape: "square", decorative: true }
);

const SIZES = {
  xs: { tile: "size-4", image: "size-3", initial: "text-[9px]", radius: "rounded" },
  sm: { tile: "w-7 h-7", image: "w-4.5 h-4.5", initial: "text-xs", radius: "rounded-md" },
  md: { tile: "w-8 h-8", image: "w-5 h-5", initial: "text-xs", radius: "rounded-lg" },
  lg: { tile: "w-9 h-9", image: "w-5 h-5", initial: "text-sm", radius: "rounded-lg" },
  xl: { tile: "w-12 h-12", image: "w-8 h-8", initial: "text-base", radius: "rounded-xl" },
} as const;

const sizeClasses = computed(() => SIZES[props.size]);

const failed = ref(false);
// Reset the failure latch when the mark changes, so a corrected icon_url shows
// without remounting the whole row.
watch(
  () => props.icon?.url,
  () => {
    failed.value = false;
  }
);

// Callers with their own glyph (a lucide chip in a dense log row) keep it.
const hasCustomFallback = computed(() => Boolean(useSlots().default));

const showImage = computed(() => Boolean(props.icon?.url) && !failed.value);
const showMonogram = computed(() => !showImage.value && !hasCustomFallback.value);
const initial = computed(() => monogramInitial(props.label));

const tileClass = computed(() =>
  cn(
    "flex shrink-0 select-none items-center justify-center overflow-hidden",
    sizeClasses.value.tile,
    props.shape === "pill" ? "rounded-full" : sizeClasses.value.radius,
    // Same two surfaces as the rest of the console: a card for a logo, the
    // muted identity-tile surface for the fallback.
    showImage.value ? "border border-border bg-card" : "border border-border/50 bg-muted"
  )
);
</script>

<template>
  <span
    :class="tileClass"
    :role="decorative ? undefined : 'img'"
    :aria-label="decorative ? undefined : label"
  >
    <img
      v-if="showImage && icon"
      :src="icon.url"
      :alt="decorative ? '' : label"
      :class="[icon.mono ? 'icon-mono' : null, sizeClasses.image, 'object-contain']"
      loading="lazy"
      @error="failed = true"
    />
    <!-- Callers with their own glyph keep it; everyone else gets the initial. -->
    <slot v-if="!showImage" />
    <span
      v-if="showMonogram"
      :class="[sizeClasses.initial, 'font-semibold leading-none text-muted-foreground uppercase']"
      aria-hidden="true"
    >
      {{ initial }}
    </span>
  </span>
</template>
