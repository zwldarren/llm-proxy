import { useResizeObserver, useScroll } from "@vueuse/core";
import { computed, type Ref } from "vue";

/**
 * Reports whether a horizontally scrollable element has content off either
 * edge. Horizontal tab strips hide their native scrollbar so the tab underline
 * stays a single hairline; the view uses these flags to render an edge fade
 * that makes the hidden tabs discoverable.
 *
 * `useScroll` owns the scroll listener and the edge math, including the
 * mutation observer that re-measures when tabs are added or removed; the
 * resize observer catches the strip itself changing width.
 */
export function useHorizontalOverflow(target: Ref<HTMLElement | null>) {
  const { arrivedState, measure } = useScroll(target, {
    throttle: 100,
    observe: { mutation: true },
  });

  useResizeObserver(target, measure);

  return {
    // "arrived at the start/end" is the inverse of "there is hidden content
    // off that edge". With no overflow both arrived flags are true, so both
    // fades stay hidden.
    hasOverflowLeft: computed(() => !arrivedState.left),
    hasOverflowRight: computed(() => !arrivedState.right),
    update: measure,
  };
}
