import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { defineComponent, nextTick } from "vue";

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));

import Dialog from "./Dialog.vue";
import DialogContent from "./DialogContent.vue";
import DialogTitle from "./DialogTitle.vue";
import Sheet from "../sheet/Sheet.vue";
import SheetContent from "../sheet/SheetContent.vue";

const TestDialog = defineComponent({
  components: { Dialog, DialogContent, DialogTitle },
  template: `
    <div>
      <button id="background-action">Background</button>
      <Dialog :open="true">
        <DialogContent>
          <DialogTitle>Test dialog</DialogTitle>
          <button>Inside</button>
        </DialogContent>
      </Dialog>
    </div>
  `,
});

const TestSheet = defineComponent({
  components: { Sheet, SheetContent },
  template: `
    <Sheet :open="true">
      <SheetContent side="right">Panel</SheetContent>
    </Sheet>
  `,
});

/** Read the z-index utility a surface carries, e.g. `z-50` or `z-[100]`. */
const zIndexOf = (element: Element): string | undefined =>
  element.className.match(/(?:^|\s)z-(\[[^\]]+\]|\d+)(?:\s|$)/)?.[1];

afterEach(() => {
  document.body.innerHTML = "";
});

describe("DialogSurface", () => {
  it("declares a modal dialog and inerts the app behind the portal", async () => {
    const app = document.createElement("div");
    app.id = "app";
    document.body.append(app);

    const wrapper = mount(TestDialog, { attachTo: app });
    await nextTick();
    await nextTick();

    const content = document.querySelector<HTMLElement>('[data-slot="dialog-content"]');
    expect(content).not.toBeNull();
    expect(content!.getAttribute("aria-modal")).toBe("true");
    expect(app.hasAttribute("inert")).toBe(true);

    wrapper.unmount();
    expect(app.hasAttribute("inert")).toBe(false);
  });

  it("keeps a sheet panel and its scrim in one stacking layer", async () => {
    const wrapper = mount(TestSheet, { attachTo: document.body });
    await nextTick();
    await nextTick();

    const overlay = document.querySelector<HTMLElement>('[data-slot="dialog-overlay"]');
    const panel = document.querySelector<HTMLElement>('[data-slot="sheet-content"]');
    expect(overlay).not.toBeNull();
    expect(panel).not.toBeNull();

    // A panel that declares no z-index loses to the scrim and the opened surface
    // reads as "the page just went dark", which is the regression this guards.
    const layer = zIndexOf(overlay!);
    expect(layer).toBeDefined();
    expect(zIndexOf(panel!)).toBe(layer);

    wrapper.unmount();
  });
});
