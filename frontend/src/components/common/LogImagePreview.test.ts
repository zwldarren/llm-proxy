import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { defineComponent } from "vue";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));

import { TooltipProvider } from "@/components/ui/tooltip";
import type { ImageInfo } from "@/utils/logResponseParser";
import LogImagePreview from "./LogImagePreview.vue";

function render(images: ImageInfo[], outputFormat?: string) {
  const Host = defineComponent({
    components: { LogImagePreview, TooltipProvider },
    setup: () => ({ images, outputFormat }),
    template: `<TooltipProvider>
      <LogImagePreview :images="images" :output-format="outputFormat" />
    </TooltipProvider>`,
  });
  return mount(Host);
}

/** The preview and the download action address the same image; only the
 * open-in-new-tab action is a navigation sink. */
const openButton = (wrapper: ReturnType<typeof render>) =>
  wrapper.find('button[aria-label="logs.openInNewTabN"]');
const downloadButton = (wrapper: ReturnType<typeof render>) =>
  wrapper.find('button[aria-label="logs.downloadImageN"]');

afterEach(() => {
  vi.restoreAllMocks();
});

describe("LogImagePreview open-in-new-tab", () => {
  it("offers the navigation action for a raster base64 payload", async () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null);
    const wrapper = render([{ b64Json: "AAAA", outputFormat: "png" }]);

    expect(downloadButton(wrapper).exists()).toBe(true);
    await openButton(wrapper).trigger("click");

    expect(open).toHaveBeenCalledWith(
      "data:image/png;base64,AAAA",
      "_blank",
      "noopener,noreferrer"
    );
  });

  it("never navigates to base64 SVG, which would run in the dashboard origin", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null);
    const wrapper = render([{ b64Json: "PHN2Zz4", outputFormat: "svg" }]);

    // The preview still renders the data URL as an <img>, and the bytes stay
    // downloadable — only the top-level navigation is withheld.
    expect(wrapper.find("img").attributes("src")).toBe("data:image/svg+xml;base64,PHN2Zz4");
    expect(openButton(wrapper).exists()).toBe(false);
    expect(downloadButton(wrapper).exists()).toBe(true);
    expect(open).not.toHaveBeenCalled();
  });

  it("withholds the action for a provider URL that is not navigable", () => {
    const wrapper = render([{ url: "javascript:alert(1)" }]);
    expect(openButton(wrapper).exists()).toBe(false);
  });
});
