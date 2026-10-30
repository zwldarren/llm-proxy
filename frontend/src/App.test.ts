import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, nextTick, type Component } from "vue";

/** Route/controller shared with the mocked `useRoute` / `useRouter`. */
const ctl = vi.hoisted(() => ({
  path: "/",
  setPath: (_path: string) => {},
  replace: vi.fn(async () => {}),
}));

vi.mock("vue-router", async () => {
  const { reactive: reactiveImpl } = await import("vue");
  const route = reactiveImpl({ fullPath: "/" });
  ctl.setPath = (path: string) => {
    route.fullPath = path;
  };
  return {
    useRoute: () => route,
    useRouter: () => ({ replace: ctl.replace }),
  };
});

vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) };
});

vi.mock("@/components/common/StepUpDialog.vue", () => ({ default: { template: "<div />" } }));
vi.mock("@/components/ui/sonner", () => ({ Toaster: { template: "<div />" } }));
vi.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: { template: "<div><slot /></div>" },
}));
vi.mock("@/i18n", () => ({
  default: { global: { locale: { value: "en" } } },
}));

import App from "./App.vue";

/**
 * RouterView stand-in: renders a component that throws on its first mount and
 * renders normally afterwards, so a second mount (after the route moves on)
 * proves the boundary actually cleared.
 */
let thrown = false;
const Thrower = defineComponent({
  name: "Thrower",
  setup() {
    if (!thrown) {
      thrown = true;
      throw new Error("Cannot read properties of undefined (reading 'map')");
    }
  },
  template: '<div data-test="healthy">healthy</div>',
});

beforeEach(() => {
  thrown = false;
});

const RouterViewStub = defineComponent({
  name: "RouterView",
  setup(_props, { slots }) {
    return () => h("div", slots.default?.({ Component: Thrower as Component }));
  },
});

function mountApp() {
  return mount(App, { global: { components: { RouterView: RouterViewStub } } });
}

describe("App error boundary", () => {
  it("replaces the broken view with the recovery surface", async () => {
    const wrapper = mountApp();
    await nextTick();
    await flushPromises();

    expect(wrapper.text()).toContain("errors.somethingWrong");
    expect(wrapper.text()).toContain("Cannot read properties of undefined (reading 'map')");
    expect(wrapper.find('[data-test="healthy"]').exists()).toBe(false);
  });

  it("clears itself once the route moves on", async () => {
    const wrapper = mountApp();
    await nextTick();
    await flushPromises();
    expect(wrapper.find("main[role='alert']").exists()).toBe(true);

    // A 401 redirect or a store funnelling the operator elsewhere: the screen
    // that threw is gone, so the boundary must not hide where they were sent.
    ctl.setPath("/login");
    await nextTick();
    await flushPromises();

    expect(wrapper.find("main[role='alert']").exists()).toBe(false);
    expect(wrapper.find('[data-test="healthy"]').exists()).toBe(true);
  });

  it("returns to the dashboard from the recovery surface", async () => {
    const wrapper = mountApp();
    await nextTick();
    await flushPromises();

    const home = wrapper
      .findAll("button")
      .find((button) => button.text().includes("errors.backToDashboard"));
    await home!.trigger("click");
    await flushPromises();

    expect(ctl.replace).toHaveBeenCalledWith("/");
    expect(wrapper.find("main[role='alert']").exists()).toBe(false);
  });
});
