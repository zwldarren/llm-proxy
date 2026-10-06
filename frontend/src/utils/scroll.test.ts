import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import { ANCHOR_FLASH_MS, flashAnchorTarget, scrollToAnchor } from "@/utils/scroll";

/** happy-dom has no layout, so a rendered box has to be faked per element. */
const setBoxes = (el: HTMLElement, count: number) => {
  Object.defineProperty(el, "getClientRects", {
    configurable: true,
    value: () => (count > 0 ? [{}] : []),
  });
};

const addTarget = (id: string, visible = true) => {
  const el = document.createElement("div");
  el.id = id;
  document.body.append(el);
  setBoxes(el, visible ? 1 : 0);
  const scrollIntoView = vi.fn();
  Object.defineProperty(el, "scrollIntoView", { configurable: true, value: scrollIntoView });
  return { el, scrollIntoView };
};

describe("scroll utils", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = "";
    vi.unstubAllGlobals();
  });

  describe("scrollToAnchor", () => {
    it("scrolls straight to an already-rendered target", () => {
      const { scrollIntoView } = addTarget("about");

      scrollToAnchor("about");

      expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth", block: "start" });
    });

    it("retries until a display:none target gets a box", () => {
      const { el, scrollIntoView } = addTarget("about", false);

      scrollToAnchor("about", { intervalMs: 50 });
      expect(scrollIntoView).not.toHaveBeenCalled();

      vi.advanceTimersByTime(200);
      setBoxes(el, 1);
      vi.advanceTimersByTime(50);

      expect(scrollIntoView).toHaveBeenCalledTimes(1);
    });

    it("gives up once the retry budget is spent", () => {
      const { el, scrollIntoView } = addTarget("about", false);

      scrollToAnchor("about", { timeoutMs: 150, intervalMs: 50 });
      vi.advanceTimersByTime(1000);

      setBoxes(el, 1);
      vi.advanceTimersByTime(1000);

      expect(scrollIntoView).not.toHaveBeenCalled();
    });

    it("stops retrying when cancelled", () => {
      const { el, scrollIntoView } = addTarget("about", false);

      const cancel = scrollToAnchor("about", { intervalMs: 50 });
      vi.advanceTimersByTime(50);
      cancel();

      setBoxes(el, 1);
      vi.advanceTimersByTime(1000);

      expect(scrollIntoView).not.toHaveBeenCalled();
    });

    it("ignores reduced motion for the scroll behaviour", () => {
      vi.stubGlobal(
        "matchMedia",
        vi.fn(() => ({ matches: true }))
      );
      const { scrollIntoView } = addTarget("about");

      scrollToAnchor("about");

      expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "auto", block: "start" });
    });

    it("flashes the target only once the scroll has settled", () => {
      const { el } = addTarget("about");
      const frames: FrameRequestCallback[] = [];
      vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
      vi.stubGlobal("cancelAnimationFrame", vi.fn());
      let top = 400;
      Object.defineProperty(el, "getBoundingClientRect", {
        configurable: true,
        value: () => ({ top }),
      });
      const runFrame = () => {
        for (const cb of frames.splice(0)) cb(0);
      };

      scrollToAnchor("about", { highlight: true });

      // Still travelling — a mark here would fade before the jump lands.
      runFrame();
      top = 200;
      runFrame();
      expect(el.classList.contains("anchor-target-flash")).toBe(false);

      // Two frames without movement: arrived.
      top = 0;
      runFrame();
      runFrame();
      runFrame();
      expect(el.classList.contains("anchor-target-flash")).toBe(true);
    });

    it("drops a pending flash when cancelled", () => {
      const { el } = addTarget("about");
      const frames: FrameRequestCallback[] = [];
      vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
      vi.stubGlobal("cancelAnimationFrame", vi.fn());

      const cancel = scrollToAnchor("about", { highlight: true });
      cancel();
      for (const cb of frames.splice(0)) cb(0);

      expect(el.classList.contains("anchor-target-flash")).toBe(false);
    });
  });

  describe("flashAnchorTarget", () => {
    it("marks the target and clears the mark on a timer", () => {
      const { el } = addTarget("about");

      flashAnchorTarget(el);
      expect(el.classList.contains("anchor-target-flash")).toBe(true);

      // Still marked while the reduced-motion branch would hold it statically.
      vi.advanceTimersByTime(ANCHOR_FLASH_MS - 50);
      expect(el.classList.contains("anchor-target-flash")).toBe(true);

      vi.advanceTimersByTime(100);
      expect(el.classList.contains("anchor-target-flash")).toBe(false);
    });

    it("keeps the mark alive across rapid re-flashes", () => {
      const { el } = addTarget("about");

      flashAnchorTarget(el);
      vi.advanceTimersByTime(ANCHOR_FLASH_MS - 100);
      flashAnchorTarget(el);
      vi.advanceTimersByTime(ANCHOR_FLASH_MS - 100);

      expect(el.classList.contains("anchor-target-flash")).toBe(true);
    });
  });
});
