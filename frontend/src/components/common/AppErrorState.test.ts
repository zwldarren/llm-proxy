import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import AppErrorState from "./AppErrorState.vue";

vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) };
});

function errorWith(message: string, stack?: string): Error {
  const error = new Error(message);
  if (stack) error.stack = stack;
  return error;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("AppErrorState", () => {
  it("shows the failure message and the recovery actions", () => {
    const wrapper = mount(AppErrorState, {
      props: { error: errorWith("Cannot read properties of undefined (reading 'map')") },
    });

    expect(wrapper.get("h1").text()).toBe("errors.somethingWrong");
    expect(wrapper.text()).toContain("errors.renderFailed");
    // The raw message stays readable: it is what an operator quotes in a report.
    expect(wrapper.text()).toContain("Cannot read properties of undefined (reading 'map')");
    expect(wrapper.text()).toContain("errors.backToDashboard");
    expect(wrapper.text()).toContain("common.reload");
    expect(wrapper.get("main").attributes("role")).toBe("alert");
  });

  it("keeps the stack behind a disclosure, labelled with the throwing component", () => {
    const wrapper = mount(AppErrorState, {
      props: {
        error: errorWith("boom", "Error: boom\n    at UsageTrendsChart.vue:74"),
        info: "render function",
        componentName: "UsageTrendsChart",
      },
    });

    const details = wrapper.get("details");
    expect(details.text()).toContain("errors.errorDetails");
    expect(details.get("pre").text()).toContain("<UsageTrendsChart · render function>");
    expect(details.get("pre").text()).toContain("at UsageTrendsChart.vue:74");
  });

  it("emits the two recovery actions", async () => {
    const wrapper = mount(AppErrorState, { props: { error: errorWith("boom") } });

    const buttonWith = (text: string) =>
      wrapper.findAll("button").find((button) => button.text().includes(text))!;

    await buttonWith("common.reload").trigger("click");
    expect(wrapper.emitted("reload")).toHaveLength(1);

    await buttonWith("errors.backToDashboard").trigger("click");
    expect(wrapper.emitted("home")).toHaveLength(1);
  });

  it("copies the message and stack together", async () => {
    class ClipboardItemStub {
      constructor(readonly items: Record<string, string>) {}
    }
    vi.stubGlobal("ClipboardItem", ClipboardItemStub);
    const write = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { write },
      configurable: true,
    });
    // VueUse only takes the Clipboard API path when the permission is not
    // denied; happy-dom ships no Permissions API, so report it as granted.
    // The status must be an EventTarget because VueUse subscribes to `change`.
    const permissionStatus = Object.assign(new EventTarget(), { state: "granted" });
    Object.defineProperty(navigator, "permissions", {
      value: { query: vi.fn().mockResolvedValue(permissionStatus) },
      configurable: true,
    });

    const wrapper = mount(AppErrorState, {
      props: { error: errorWith("boom", "Error: boom\n    at App.vue:1"), info: "render function" },
    });
    await flushPromises();

    await wrapper.get("summary button").trigger("click");
    expect(write).toHaveBeenCalledTimes(1);
    const [item] = write.mock.calls[0]![0] as unknown as ClipboardItemStub[];
    const copied = item!.items["text/plain"];
    expect(copied).toContain("boom");
    expect(copied).toContain("at App.vue:1");
  });
});
