<script setup lang="ts">
import type { LucideIcon } from "@lucide/vue";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

/**
 * Reusable icon + label + help-text + Switch row for model capability toggles.
 * Rows sit in bare divide-y groups (no box), matching the routing tab's
 * toggle styling. Icons render in a neutral muted chip — the per-capability
 * color tints stay in the plaza catalog, where they differentiate dense data;
 * in a settings form they would be decoration (DESIGN.md monochrome-first).
 * The optional `field` renders the models.dev field name in quiet mono next
 * to the label — used for attributes mirrored 1:1 from the catalog spec.
 */
interface Props {
  modelValue?: boolean;
  label: string;
  helpText: string;
  /** Capability icon, rendered in a neutral chip for row scanning. */
  icon?: LucideIcon;
  /** models.dev field name, shown in mono after the label. */
  field?: string;
}

defineProps<Props>();
defineEmits<{ "update:modelValue": [value: boolean] }>();
</script>

<template>
  <div class="flex items-start justify-between gap-4 py-3">
    <div class="flex items-start gap-3 min-w-0">
      <span
        v-if="icon"
        class="mt-0.5 inline-flex size-7 shrink-0 items-center justify-center rounded-md border border-border/60 bg-muted/40 text-muted-foreground"
        aria-hidden="true"
      >
        <component :is="icon" class="size-3.5" aria-hidden="true" />
      </span>
      <div class="space-y-1 min-w-0">
        <Label class="text-sm font-medium text-foreground">
          {{ label }}
          <code v-if="field" class="text-data text-[10px] text-muted-foreground/70 font-normal">
            {{ field }}
          </code>
        </Label>
        <p class="text-xs text-muted-foreground leading-relaxed">{{ helpText }}</p>
      </div>
    </div>
    <Switch
      :model-value="modelValue"
      :aria-label="label"
      class="mt-0.5 shrink-0"
      @update:model-value="$emit('update:modelValue', $event)"
    />
  </div>
</template>
