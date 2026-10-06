/**
 * Anchor-scroll helpers for deep links into long, tabbed pages.
 *
 * The settings page renders its sections with v-show, so a hash target can be
 * present in the DOM while still `display:none` (inactive tab) — scrolling to
 * it then is a silent no-op. These helpers wait until the target is actually
 * rendered before scrolling, and optionally flash a highlight ring so the
 * jump destination is unmistakable.
 */

export interface AnchorScrollOptions {
  behavior?: ScrollBehavior;
  /** Total retry budget in ms while the target is missing or hidden. */
  timeoutMs?: number;
  intervalMs?: number;
  highlight?: boolean;
}

/** How long the highlight ring stays on the target before it is removed. */
export const ANCHOR_FLASH_MS = 900;

/** Upper bound on waiting for a smooth scroll to come to rest. */
const SETTLE_TIMEOUT_MS = 1600;

const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

const flashTimers = new WeakMap<HTMLElement, ReturnType<typeof setTimeout>>();

/**
 * Briefly pulse a ring around the element to mark it as the jump target.
 *
 * The class is cleared on a timer rather than on `animationend`: under
 * `prefers-reduced-motion` the ring is held statically (no animation to end),
 * and the same timer keeps both branches bounded.
 */
export function flashAnchorTarget(el: HTMLElement, durationMs = ANCHOR_FLASH_MS): void {
  const pending = flashTimers.get(el);
  if (pending) clearTimeout(pending);

  el.classList.remove("anchor-target-flash");
  // Force a reflow so re-adding the class restarts the CSS animation.
  void el.offsetWidth;
  el.classList.add("anchor-target-flash");
  flashTimers.set(
    el,
    setTimeout(() => {
      el.classList.remove("anchor-target-flash");
      flashTimers.delete(el);
    }, durationMs)
  );
}

/**
 * Run `cb` once the element has stopped moving, i.e. the smooth scroll has
 * arrived. Flashing at the start of the scroll instead would burn the mark
 * before a long jump lands. Returns a cancel function.
 */
function onceSettled(el: HTMLElement, cb: () => void): () => void {
  let frame = 0;
  let lastTop = el.getBoundingClientRect().top;
  let stableFrames = 0;
  const startedAt = Date.now();

  const tick = () => {
    const top = el.getBoundingClientRect().top;
    stableFrames = Math.abs(top - lastTop) < 1 ? stableFrames + 1 : 0;
    lastTop = top;
    // Two frames without movement means the scroll animation is done.
    if (stableFrames >= 2 || Date.now() - startedAt > SETTLE_TIMEOUT_MS) {
      frame = 0;
      cb();
      return;
    }
    frame = requestAnimationFrame(tick);
  };

  frame = requestAnimationFrame(tick);
  return () => {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
  };
}

/**
 * Scroll to the element with the given id once it exists and has a rendered
 * box. Returns a cancel function for the pending retry timer.
 */
export function scrollToAnchor(id: string, options: AnchorScrollOptions = {}): () => void {
  const { behavior = "smooth", timeoutMs = 3000, intervalMs = 100, highlight = false } = options;
  // The reduced-motion block in main.css only neutralizes CSS scroll-behavior;
  // a JS scroll must honour the preference itself.
  const resolvedBehavior: ScrollBehavior = prefersReducedMotion() ? "auto" : behavior;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let cancelSettle: (() => void) | null = null;
  const startedAt = Date.now();

  const attempt = () => {
    const el = document.getElementById(id);
    // getClientRects() is empty for display:none subtrees (v-show'd inactive
    // tabs), so this doubles as a visibility check.
    if (el && el.getClientRects().length > 0) {
      el.scrollIntoView({ behavior: resolvedBehavior, block: "start" });
      if (highlight) {
        cancelSettle?.();
        cancelSettle = onceSettled(el, () => flashAnchorTarget(el));
      }
      return;
    }
    if (Date.now() - startedAt < timeoutMs) {
      timer = setTimeout(attempt, intervalMs);
    }
  };

  attempt();
  return () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    cancelSettle?.();
    cancelSettle = null;
  };
}
