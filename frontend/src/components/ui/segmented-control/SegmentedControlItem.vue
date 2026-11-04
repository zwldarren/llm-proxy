<script setup lang="ts">
import type { PrimitiveProps } from "reka-ui";
import type { ComputedRef, HTMLAttributes } from "vue";
import { Primitive } from "reka-ui";
import { computed, inject } from "vue";
import { cn } from "@/lib/utils";
import { segmentedControlItemVariants } from ".";

interface SegmentedControlContext {
  size: "sm" | "md" | "icon";
  variant: "raised" | "quiet";
  fill: boolean;
  kind: "toggle" | "nav";
}

interface Props extends PrimitiveProps {
  class?: HTMLAttributes["class"];
  active?: boolean;
  disabled?: boolean;
  kind?: "toggle" | "nav";
  size?: SegmentedControlContext["size"];
  variant?: SegmentedControlContext["variant"];
  caps?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  as: "button",
  active: false,
  disabled: false,
  caps: false,
});

const context = inject<ComputedRef<SegmentedControlContext> | null>("segmentedControl", null);

const resolvedSize = computed(() => props.size ?? context?.value.size ?? "md");
const resolvedVariant = computed(() => props.variant ?? context?.value.variant ?? "raised");
const resolvedFill = computed(() => context?.value.fill ?? false);
const resolvedKind = computed(() => props.kind ?? context?.value.kind ?? "toggle");

// A disabled item still renders (so the group keeps its shape) but must not act:
// `disabled` only exists on native buttons, so anchors rendered through `asChild`
// need the click cancelled and removed from the tab order.
function onClick(event: MouseEvent) {
  if (!props.disabled) return;
  event.preventDefault();
}
</script>

<template>
  <Primitive
    data-slot="segmented-control-item"
    :as="as"
    :as-child="asChild"
    :type="!asChild && as === 'button' ? 'button' : undefined"
    :disabled="!asChild && as === 'button' ? disabled : undefined"
    :aria-disabled="disabled || undefined"
    :role="resolvedKind === 'toggle' ? 'radio' : undefined"
    :aria-checked="resolvedKind === 'toggle' ? (active ? 'true' : 'false') : undefined"
    :aria-current="resolvedKind === 'nav' && active ? 'page' : undefined"
    :tabindex="disabled ? -1 : active ? 0 : -1"
    :data-state="active ? 'on' : 'off'"
    :class="
      cn(
        segmentedControlItemVariants({
          size: resolvedSize,
          variant: resolvedVariant,
          active,
          fill: resolvedFill,
          caps,
        }),
        props.class
      )
    "
    @click.capture="onClick"
  >
    <slot />
  </Primitive>
</template>
