import { mount } from "@vue/test-utils";
import { nextTick, ref } from "vue";
import type * as VueI18n from "vue-i18n";
import { afterEach, describe, expect, it, vi } from "vitest";
import ChatMessage from "./ChatMessage.vue";

vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof VueI18n>();
  return {
    ...actual,
    useI18n: () => ({
      t: (key: string) => key,
      locale: ref("en"),
    }),
  };
});

vi.mock("@/stores/chat", () => ({
  useChatStore: () => ({
    isLoading: false,
    messages: [],
    currentlyPlayingId: null,
  }),
}));

vi.mock("@/stores/models", () => ({
  useModelStore: () => ({ models: [] }),
}));

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("ChatMessage", () => {
  it("rejects unsafe fence info and escapes fenced code", () => {
    const wrapper = mount(ChatMessage, {
      props: {
        message: {
          role: "assistant",
          content: '```js" onmouseover="alert(1)\n<img src=x onerror="alert(1)">\n```',
        },
      },
    });

    expect(wrapper.get("code").classes()).toContain("language-code");
    expect(wrapper.get("code").html()).toContain('&lt;img src=x onerror="alert(1)"&gt;');
    expect(wrapper.html()).not.toContain('<img src="x"');
    wrapper.unmount();
  });

  it("shows code-copy feedback until the copied window elapses", async () => {
    Object.defineProperty(document, "execCommand", {
      value: vi.fn(() => true),
      configurable: true,
    });
    vi.useFakeTimers();
    const wrapper = mount(ChatMessage, {
      props: {
        message: { role: "assistant", content: "```ts\nconst value = 1;\n```" },
      },
    });

    const setupState: unknown = Reflect.get(wrapper.vm.$, "setupState");
    if (
      typeof setupState !== "object" ||
      setupState === null ||
      !("handleCodeCopy" in setupState) ||
      typeof setupState.handleCodeCopy !== "function"
    ) {
      throw new Error("ChatMessage setup does not expose handleCodeCopy");
    }

    const button = wrapper.get(".code-copy-btn");
    const codeBlock = document.createElement("div");
    codeBlock.className = "code-block-wrapper";
    const code = document.createElement("code");
    code.textContent = "const value = 1;";
    codeBlock.append(button.element, code);
    await setupState.handleCodeCopy(button.element);

    expect(button.get(".copy-icon").classes()).toContain("hidden");
    expect(button.get(".check-icon").classes()).not.toContain("hidden");
    expect(button.get(".copy-text").text()).toBe("common.copied");

    await vi.advanceTimersByTimeAsync(2000);
    await nextTick();

    expect(button.get(".copy-icon").classes()).not.toContain("hidden");
    expect(button.get(".check-icon").classes()).toContain("hidden");
    expect(button.get(".copy-text").text()).toBe("common.copy");
    wrapper.unmount();
  });
});
