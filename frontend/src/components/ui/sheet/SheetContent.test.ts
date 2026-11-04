import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { defineComponent, nextTick } from "vue";

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));

import Sheet from "./Sheet.vue";
import SheetContent from "./SheetContent.vue";

type SheetProps = {
  side?: "top" | "right" | "bottom" | "left";
  size?: "sm" | "md" | "lg" | "xl" | "2xl" | "none";
  variant?: "default" | "sidebar";
  showCloseButton?: boolean;
};

async function mountSheet(props: SheetProps) {
  const wrapper = mount(
    defineComponent({
      components: { Sheet, SheetContent },
      setup: () => ({ props }),
      template: `
        <Sheet :open="true">
          <SheetContent v-bind="props">Panel</SheetContent>
        </Sheet>
      `,
    }),
    { attachTo: document.body }
  );
  await nextTick();
  await nextTick();
  const panel = document.querySelector<HTMLElement>('[data-slot="sheet-content"]');
  if (!panel) throw new Error("sheet content not rendered");
  return { wrapper, panel };
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("SheetContent", () => {
  it("maps the horizontal size to its width classes", async () => {
    const { wrapper, panel } = await mountSheet({ side: "right", size: "lg" });

    expect(panel.className).toContain("sm:max-w-[520px]");
    expect(panel.className).toContain("lg:max-w-[600px]");
    expect(panel.className).toContain("border-l");

    wrapper.unmount();
  });

  it("ignores the size for top and bottom sheets", async () => {
    const { wrapper, panel } = await mountSheet({ side: "top", size: "lg" });

    expect(panel.className).not.toContain("sm:max-w-[520px]");
    expect(panel.className).toContain("h-auto");
    expect(panel.className).toContain("border-b");

    wrapper.unmount();
  });

  it("lets the sidebar variant supply its own width instead of the size", async () => {
    const { wrapper, panel } = await mountSheet({
      side: "right",
      size: "lg",
      variant: "sidebar",
    });

    expect(panel.className).toContain("w-(--sidebar-width)");
    expect(panel.className).not.toContain("sm:max-w-[520px]");

    wrapper.unmount();
  });

  it("omits the close control when asked to", async () => {
    const withClose = await mountSheet({});
    expect(document.querySelector('[data-slot="dialog-close"]')).not.toBeNull();
    withClose.wrapper.unmount();

    const withoutClose = await mountSheet({ showCloseButton: false });
    expect(document.querySelector('[data-slot="dialog-close"]')).toBeNull();
    withoutClose.wrapper.unmount();
  });
});
