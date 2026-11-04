<script setup lang="ts">
import type { HTMLAttributes } from "vue";
import { computed, onMounted, onUpdated, provide, ref } from "vue";
import { cn } from "@/lib/utils";
import { segmentedControlVariants, type SegmentedControlVariants } from ".";

const props = withDefaults(
  defineProps<{
    class?: HTMLAttributes["class"];
    variant?: SegmentedControlVariants["variant"];
    size?: "sm" | "md" | "icon";
    fill?: boolean;
    /**
     * `toggle` is a mutually-exclusive choice (radiogroup semantics, roving
     * tabindex, arrow-key navigation); `nav` is a set of links rendered inside
     * a `<nav>` landmark.
     */
    kind?: "toggle" | "nav";
  }>(),
  {
    variant: "raised",
    size: "md",
    fill: false,
    kind: "toggle",
  }
);

const root = ref<HTMLElement | null>(null);

provide(
  "segmentedControl",
  computed(() => ({
    size: props.size,
    variant: props.variant,
    fill: props.fill,
    kind: props.kind,
  }))
);

const tag = computed(() => (props.kind === "nav" ? "nav" : "div"));
const role = computed(() => (props.kind === "nav" ? undefined : "radiogroup"));

const items = () =>
  Array.from(
    root.value?.querySelectorAll<HTMLElement>('[data-slot="segmented-control-item"]') ?? []
  ).filter((item) => item.getAttribute("aria-disabled") !== "true");

// Roving tabindex: one stop for the whole group. The items ask for the tab stop
// when they are active; if nothing is active yet the first enabled one takes it
// so the group stays reachable.
function ensureTabStop() {
  if (props.kind !== "toggle") return;
  const enabled = items();
  if (!enabled.length || enabled.some((item) => item.tabIndex === 0)) return;
  enabled[0].tabIndex = 0;
}

onMounted(ensureTabStop);
onUpdated(ensureTabStop);

function onKeydown(event: KeyboardEvent) {
  if (props.kind !== "toggle") return;
  const enabled = items();
  if (!enabled.length) return;
  const index = enabled.indexOf(document.activeElement as HTMLElement);
  const last = enabled.length - 1;
  let next: number;
  switch (event.key) {
    case "ArrowRight":
    case "ArrowDown":
      next = index >= last ? 0 : index + 1;
      break;
    case "ArrowLeft":
    case "ArrowUp":
      next = index <= 0 ? last : index - 1;
      break;
    case "Home":
      next = 0;
      break;
    case "End":
      next = last;
      break;
    default:
      return;
  }
  event.preventDefault();
  const target = enabled[next];
  target.focus();
  // Selection follows focus, matching the radio-group pattern the old
  // ToggleGroup provided.
  target.click();
}
</script>

<template>
  <component
    :is="tag"
    ref="root"
    :role="role"
    data-slot="segmented-control"
    :class="cn(segmentedControlVariants({ variant, fill }), props.class)"
    @keydown="onKeydown"
  >
    <slot />
  </component>
</template>
