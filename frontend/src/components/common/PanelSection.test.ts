import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (key: string) => key }) }));

import PanelSection from "./PanelSection.vue";

describe("PanelSection", () => {
  it("keeps the anchor id its callers link to", () => {
    // /account#access and /account/security#passkeys both depend on this.
    const wrapper = mount(PanelSection, {
      props: { id: "access", title: "Access" },
    });

    expect(wrapper.attributes("id")).toBe("access");
  });

  it("merges a caller's rhythm class instead of replacing the panel's surface", () => {
    const wrapper = mount(PanelSection, {
      props: { id: "profile", title: "Profile" },
      attrs: { class: "mt-6" },
    });

    expect(wrapper.classes()).toContain("mt-6");
    expect(wrapper.classes()).toContain("rounded-xl");
  });
});
