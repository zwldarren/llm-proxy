// frontend/src/components/common/BrandMark.test.ts
import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import BrandMark from "@/components/common/BrandMark.vue";
import type { BrandIcon } from "@/utils/icons";

const COLOR_ICON: BrandIcon = { url: "https://cdn.example/deepseek-color.svg", mono: false };
const MONO_ICON: BrandIcon = { url: "https://cdn.example/openai.svg", mono: true };

describe("BrandMark", () => {
  it("renders the mark and skins a mono mark", () => {
    const wrapper = mount(BrandMark, { props: { icon: MONO_ICON, label: "openai" } });
    const img = wrapper.get("img");

    expect(img.attributes("src")).toBe(MONO_ICON.url);
    expect(img.classes()).toContain("icon-mono");
    expect(img.attributes("alt")).toBe("");
  });

  it("leaves a fixed-palette mark alone", () => {
    const wrapper = mount(BrandMark, { props: { icon: COLOR_ICON, label: "deepseek" } });

    expect(wrapper.get("img").classes()).not.toContain("icon-mono");
  });

  it("falls back to the name's initial when there is no mark", () => {
    const wrapper = mount(BrandMark, { props: { icon: null, label: "fast-model-1" } });

    expect(wrapper.find("img").exists()).toBe(false);
    expect(wrapper.text()).toBe("F");
    expect(wrapper.classes()).toContain("bg-muted");
  });

  it("prefers a caller's own glyph over the initial", () => {
    const wrapper = mount(BrandMark, {
      props: { icon: null, label: "fast-model-1" },
      slots: { default: '<svg class="caller-glyph" />' },
    });

    expect(wrapper.find(".caller-glyph").exists()).toBe(true);
    expect(wrapper.text()).toBe("");
  });

  it("falls back to the initial when the image fails to load", async () => {
    const wrapper = mount(BrandMark, { props: { icon: COLOR_ICON, label: "deepseek" } });

    await wrapper.get("img").trigger("error");

    expect(wrapper.find("img").exists()).toBe(false);
    expect(wrapper.text()).toBe("D");
  });

  it("shows a new URL after the failed one changes", async () => {
    const wrapper = mount(BrandMark, { props: { icon: COLOR_ICON, label: "deepseek" } });
    await wrapper.get("img").trigger("error");

    await wrapper.setProps({ icon: { url: "https://cdn.example/fixed.svg", mono: false } });

    expect(wrapper.get("img").attributes("src")).toBe("https://cdn.example/fixed.svg");
  });

  it("names the tile only when it is not decorative", () => {
    const decorative = mount(BrandMark, { props: { icon: COLOR_ICON, label: "deepseek" } });
    expect(decorative.attributes("role")).toBeUndefined();

    const labelled = mount(BrandMark, {
      props: { icon: COLOR_ICON, label: "deepseek", decorative: false },
    });
    expect(labelled.attributes("role")).toBe("img");
    expect(labelled.attributes("aria-label")).toBe("deepseek");
    expect(labelled.get("img").attributes("alt")).toBe("deepseek");
  });
});
