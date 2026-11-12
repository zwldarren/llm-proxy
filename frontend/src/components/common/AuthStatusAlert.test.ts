import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import AuthStatusAlert from "./AuthStatusAlert.vue";

function roleFor(tone: "error" | "success" | "info") {
  return mount(AuthStatusAlert, { props: { tone }, slots: { default: "Message" } })
    .find("[data-slot='alert']")
    .attributes("role");
}

describe("AuthStatusAlert", () => {
  it("keeps failures assertive", () => {
    expect(roleFor("error")).toBe("alert");
  });

  it("keeps neutral information assertive", () => {
    expect(roleFor("info")).toBe("alert");
  });

  it("announces success non-interruptively", () => {
    expect(roleFor("success")).toBe("status");
  });
});
