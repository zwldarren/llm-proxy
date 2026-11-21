import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, nextTick, ref } from "vue";

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));

import ConfirmDialog from "./ConfirmDialog.vue";
import Sheet from "@/components/ui/sheet/Sheet.vue";
import SheetContent from "@/components/ui/sheet/SheetContent.vue";

/** A surface's stacking layer as a number, read off its Tailwind utility. */
function layerOf(element: Element | null): number | null {
  if (!element) return null;
  const utility = element.className.match(/(?:^|\s)z-(?:\[(\d+)\]|(\d+))(?:\s|$)/);
  if (!utility) return null;
  return Number(utility[1] ?? utility[2]);
}

/**
 * The shape ModelsView uses: the discard guard is declared *above* the sheet it
 * guards, so DOM order alone would put the confirm underneath the panel. That
 * is the reported freeze — the prompt shows up dimmed behind the sheet's scrim
 * (which swallows the click), while the sheet itself is `pointer-events: none`
 * because the confirm owns the pointer, so neither surface responds.
 */
const Host = defineComponent({
  setup() {
    const sheet = ref(true);
    const confirm = ref(true);
    return { sheet, confirm };
  },
  render: () =>
    h("div", [
      h(ConfirmDialog, {
        open: true,
        "onUpdate:open": () => {},
        title: "Discard unsaved changes?",
        description: "Close without saving?",
        confirmText: "Discard and close",
        cancelText: "Stay",
      }),
      h(Sheet, { open: true }, () =>
        h(SheetContent, { side: "right" }, () => h("p", "unfinished model form"))
      ),
    ]),
});

afterEach(() => {
  document.body.innerHTML = "";
});

describe("ConfirmDialog", () => {
  it("stacks above the surface it is guarding, whichever is declared first", async () => {
    const wrapper = mount(Host, { attachTo: document.body });
    await nextTick();
    await nextTick();

    const confirmContent = document.querySelector('[data-slot="alert-dialog-content"]');
    const confirmScrim = document.querySelector('[data-slot="alert-dialog-overlay"]');
    const sheetContent = document.querySelector('[data-slot="sheet-content"]');

    // Declaration order puts the confirm first in the DOM; the layer is what
    // keeps it paintable and clickable on top.
    const confirmLayer = layerOf(confirmContent);
    expect(confirmLayer).not.toBeNull();
    expect(layerOf(confirmScrim)).toBe(confirmLayer);
    expect(confirmLayer!).toBeGreaterThan(layerOf(sheetContent)!);

    wrapper.unmount();
  });
});
