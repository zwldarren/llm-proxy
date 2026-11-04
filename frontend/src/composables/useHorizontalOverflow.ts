import { onBeforeUnmount, onMounted, ref, watch, type Ref } from "vue";

/**
 * Reports whether a horizontally scrollable element has content off either
 * edge. Horizontal tab strips hide their native scrollbar so the tab underline
 * stays a single hairline; the view uses these flags to render an edge fade
 * that makes the hidden tabs discoverable.
 */
export function useHorizontalOverflow(target: Ref<HTMLElement | null>) {
  const hasOverflowLeft = ref(false);
  const hasOverflowRight = ref(false);
  // The element the listeners were attached to, not whatever the ref holds at
  // teardown time: a `v-if` that recreates the strip would otherwise leak the
  // old listener and observer.
  let attached: HTMLElement | null = null;
  let observer: ResizeObserver | null = null;
  let mutations: MutationObserver | null = null;

  const update = () => {
    const el = attached;
    if (!el) return;
    const maxScroll = el.scrollWidth - el.clientWidth;
    if (maxScroll <= 1) {
      hasOverflowLeft.value = false;
      hasOverflowRight.value = false;
      return;
    }
    // `scrollLeft` is negative in RTL in current browsers, so measure the
    // distance from the inline start instead of assuming a direction, then map
    // it back onto the physical edges the fade is drawn on.
    const offset = Math.abs(el.scrollLeft);
    const atStart = offset <= 1;
    const atEnd = offset >= maxScroll - 1;
    const isRtl = getComputedStyle(el).direction === "rtl";
    hasOverflowLeft.value = isRtl ? !atEnd : !atStart;
    hasOverflowRight.value = isRtl ? !atStart : !atEnd;
  };

  const detach = () => {
    if (!attached) return;
    attached.removeEventListener("scroll", update);
    observer?.disconnect();
    mutations?.disconnect();
    observer = null;
    mutations = null;
    attached = null;
  };

  const observeChildren = (el: HTMLElement) => {
    for (const child of Array.from(el.children)) observer?.observe(child);
  };

  const attach = (el: HTMLElement | null) => {
    if (el === attached) return;
    detach();
    if (!el) return;
    attached = el;
    update();
    el.addEventListener("scroll", update, { passive: true });

    observer = new ResizeObserver(update);
    observer.observe(el);
    observeChildren(el);

    // Tabs come and go without resizing the strip's own box; observing the
    // children only helps if the observer learns about the new ones.
    mutations = new MutationObserver(() => {
      update();
      observeChildren(el);
    });
    mutations.observe(el, { childList: true, subtree: true });
  };

  onMounted(() => {
    attach(target.value);
    window.addEventListener("resize", update, { passive: true });
  });

  watch(target, (el) => attach(el));

  onBeforeUnmount(() => {
    detach();
    window.removeEventListener("resize", update);
  });

  return { hasOverflowLeft, hasOverflowRight, update };
}
