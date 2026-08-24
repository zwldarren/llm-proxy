<!--
DIRECTION CONTRACT — Chat & Images instrument family (seed 9dffaf95)
THESIS: Both playgrounds are one instrument family — a workbench where every API
run deposits an inspectable specimen in a tray docked at the canvas's bottom
edge. Refuses the consumer chat-app rut (centered bubbles, no telemetry) and
the creative-studio rut.
OWN-WORLD: committed Operator's Console — dark-first cool-neutral, hairline
bands, IBM Plex Mono telemetry, status-tinted specimen dots, right-drawer
grammar for inspection.
STORY: the operator fires a request, watches it land in the tray with status
and latency, selects any specimen to audit its exact payload — and trusts the
wiring.
FIRST VIEWPORT: config header bar (model · endpoint · settings), workbench
canvas, this tray at its bottom edge, composer/prompt console.
FORM: workbench + specimen tray (grounded candidate 5); held-failure staging
fused into error specimens — a failed run freezes as an inspectable tableau.
-->
<script setup lang="ts">
import { nextTick, ref, watch } from "vue";
import { useI18n } from "vue-i18n";

const props = withDefaults(
  defineProps<{
    /** Number of recorded runs, shown next to the label. */
    count: number;
  }>(),
  {}
);

const { t } = useI18n();

const scrollEl = ref<HTMLElement | null>(null);

// New specimens land at the right edge — keep the latest run in view.
watch(
  () => props.count,
  async () => {
    await nextTick();
    scrollEl.value?.scrollTo({ left: scrollEl.value.scrollWidth, behavior: "smooth" });
  }
);
</script>

<template>
  <div
    class="flex-none border-t border-border/60 bg-background/95"
    role="region"
    :aria-label="t('playground.runs')"
  >
    <div ref="scrollEl" class="flex items-center gap-2.5 h-11 px-3 sm:px-4 overflow-x-auto">
      <!-- Tray label: mono instrument readout, not a tracked eyebrow -->
      <span
        class="shrink-0 text-data-xs text-muted-foreground select-none flex items-center gap-1.5"
      >
        {{ t("playground.runs") }}
        <span class="text-foreground/80">{{ String(count).padStart(2, "0") }}</span>
      </span>
      <div class="h-4 w-px bg-border/60 shrink-0" aria-hidden="true" />

      <!-- Specimens (newest last) -->
      <div v-if="count > 0" class="flex items-center gap-1.5 min-w-0">
        <slot />
      </div>
      <span v-else class="text-code-xs text-muted-foreground/60 truncate select-none">
        {{ t("playground.runsEmpty") }}
      </span>

      <!-- Optional right-side affordances -->
      <div v-if="$slots.actions" class="ml-auto flex items-center gap-1 shrink-0">
        <slot name="actions" />
      </div>
    </div>
  </div>
</template>
