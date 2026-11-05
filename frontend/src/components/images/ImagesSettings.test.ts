import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));

import ImagesSettings from "./ImagesSettings.vue";

/** Every v-model the drawer owns, at its default. */
function render(props: Record<string, unknown> = {}) {
  return mount(ImagesSettings, {
    props: {
      open: true,
      numberOfImages: 3,
      size: "auto",
      quality: "auto",
      background: null,
      moderation: null,
      outputCompression: 80,
      outputFormat: "png",
      partialImages: 2,
      frameCount: null,
      frameDurationMs: null,
      ...props,
    },
  });
}

describe("ImagesSettings envelope formats", () => {
  it("locks the count to 1 for SVG and restores the user's choice afterwards", async () => {
    const wrapper = render({ outputFormat: "png", numberOfImages: 3 });

    await wrapper.setProps({ outputFormat: "svg" });
    expect(wrapper.emitted("update:numberOfImages")?.at(-1)).toEqual([1]);

    await wrapper.setProps({ outputFormat: "webp" });
    expect(wrapper.emitted("update:numberOfImages")?.at(-1)).toEqual([3]);
  });

  it("does not restore over a count the user changed after unlocking", async () => {
    const wrapper = render({ outputFormat: "png", numberOfImages: 3 });

    await wrapper.setProps({ outputFormat: "svg" });
    await wrapper.setProps({ outputFormat: "webp" });
    // The parent's value wins once the field is editable again.
    await wrapper.setProps({ numberOfImages: 2, outputFormat: "svg" });
    expect(wrapper.emitted("update:numberOfImages")?.at(-1)).toEqual([1]);
    await wrapper.setProps({ outputFormat: "png" });
    expect(wrapper.emitted("update:numberOfImages")?.at(-1)).toEqual([2]);
  });

  it("clears compression and partial images for GIF and SVG output", async () => {
    const wrapper = render({ outputFormat: "png", outputCompression: 80, partialImages: 2 });

    for (const outputFormat of ["gif", "svg"]) {
      await wrapper.setProps({ outputFormat });
      expect(wrapper.emitted("update:outputCompression")?.at(-1)).toEqual([null]);
      expect(wrapper.emitted("update:partialImages")?.at(-1)).toEqual([null]);
    }
  });

  it("leaves the still-image settings alone when a raster format is chosen", async () => {
    const wrapper = render({ outputFormat: null, outputCompression: 80, partialImages: 2 });

    await wrapper.setProps({ outputFormat: "webp" });

    expect(wrapper.emitted("update:outputCompression")).toBeUndefined();
    expect(wrapper.emitted("update:partialImages")).toBeUndefined();
  });
});
