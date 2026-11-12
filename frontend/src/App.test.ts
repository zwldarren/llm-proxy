import { flushPromises, mount } from "@vue/test-utils";
import { createPinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, nextTick, type Component } from "vue";

/** Route/controller shared with the mocked `useRoute` / `useRouter`. */
const ctl = vi.hoisted(() => ({
  path: "/",
  setPath: (_path: string) => {},
  setTitleKey: (_key: string) => {},
  locale: { value: "en" },
  replace: vi.fn(async () => {}),
}));

vi.mock("vue-router", async () => {
  const { reactive: reactiveImpl } = await import("vue");
  const route = reactiveImpl({ fullPath: "/", meta: { titleKey: "nav.usage" } });
  ctl.setPath = (path: string) => {
    route.fullPath = path;
  };
  ctl.setTitleKey = (key: string) => {
    route.meta.titleKey = key;
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
vi.mock("@/i18n", async () => {
  const { reactive: reactiveImpl } = await import("vue");
  const locale: { value: string } = reactiveImpl({ value: "en" });
  ctl.locale = locale;
  return {
    default: {
      global: {
        locale,
        t: (key: string) => {
          if (key === "app.name") return "LLM Proxy";
          return locale.value === "en" ? key : `zh:${key}`;
        },
      },
    },
  };
});

import App from "./App.vue";
import CommandPalette from "@/components/layout/CommandPalette.vue";
import { useAuthStore } from "@/stores/auth";

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
  // App.vue reads the auth store to decide whether the console's command
  // palette belongs on screen (it is not part of the public auth screens).
  return mount(App, {
    global: {
      components: { RouterView: RouterViewStub },
      plugins: [createPinia()],
    },
  });
}

describe("App error boundary", () => {
  it("derives the browser title from the route and active locale", async () => {
    document.title = "LLM Proxy";
    mountApp();
    await nextTick();

    expect(document.title).toBe("nav.usage · LLM Proxy");

    ctl.locale.value = "zh";
    await nextTick();

    expect(document.title).toBe("zh:nav.usage · LLM Proxy");

    ctl.setTitleKey("nav.logs");
    await nextTick();

    expect(document.title).toBe("zh:nav.logs · LLM Proxy");
  });

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

/**
 * The palette is app chrome, mounted once here rather than in AppLayout.
 *
 * Every console view renders its own AppLayout and the router keeps several of
 * them alive, so a per-layout copy gave one palette — one teleported dialog, one
 * Cmd/Ctrl+K handler — per live view, all bound to the same module-level open
 * flag. The stacked panels swallowed clicks and wheel scrolling.
 */
describe("command palette mounting", () => {
  it("mounts in the app shell, and only for a signed-in session", async () => {
    const wrapper = mount(App, {
      global: { components: { RouterView: RouterViewStub }, plugins: [createPinia()] },
    });
    await nextTick();

    // Signed out: the public screens get no palette.
    expect(wrapper.findComponent(CommandPalette).exists()).toBe(false);

    useAuthStore().isAuthenticated = true;
    await nextTick();

    expect(wrapper.findAllComponents(CommandPalette)).toHaveLength(1);
  });
});
