<script setup lang="ts">
/**
 * One System One question (request column) or answer (response column) card —
 * the id line, its kind badge and the shared card shell, so both columns present
 * the same payload identically. The kind is the upstream `type` string, shown
 * verbatim: it is not a normalized label.
 *
 * Slot `header-end` sits at the far end of the id line (an answer's confidence);
 * the default slot holds the question criteria or the answer value.
 */
import { Badge } from "@/components/ui/badge";

defineProps<{
  /** Question/answer id — the client's question key, echoed by the answer. */
  id: string;
  /** Upstream kind (noul | choice | score | …); omit to hide the badge. */
  type?: string;
}>();
</script>

<template>
  <div class="bg-muted/10 border border-border/40 rounded-lg p-3 space-y-2.5">
    <div class="flex items-center gap-2 min-w-0">
      <span class="text-[11px] font-mono text-muted-foreground truncate">{{ id }}</span>
      <Badge
        v-if="type"
        variant="outline"
        class="font-mono text-[10px] uppercase font-bold py-0 rounded-full border-action-violet/30 bg-action-violet/5 text-action-violet shrink-0"
      >
        {{ type }}
      </Badge>
      <slot name="header-end" />
    </div>
    <slot />
  </div>
</template>
