import { mount, flushPromises } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, nextTick } from "vue";
import { useSettingAutoSave, type AutoSaveState } from "@/composables/useSettingAutoSave";

const toast = vi.hoisted(() => ({ success: vi.fn() }));

vi.mock("vue-sonner", () => ({ toast }));
vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) };
});

describe("useSettingAutoSave", () => {
  afterEach(() => {
    vi.useRealTimers();
    toast.success.mockClear();
  });

  it("confirms a successful background save", async () => {
    vi.useFakeTimers();
    let autoSave!: AutoSaveState<{ enabled: boolean }>;
    const wrapper = mount(
      defineComponent({
        setup() {
          autoSave = useSettingAutoSave<{ enabled: boolean }>(
            { enabled: false },
            async () => undefined
          );
          return () => h("div");
        },
      })
    );

    autoSave.state.value.enabled = true;
    await nextTick();
    await vi.advanceTimersByTimeAsync(300);
    await flushPromises();

    expect(autoSave.pending.value).toBe(false);
    expect(toast.success).toHaveBeenCalledWith("settings.saved");
    wrapper.unmount();
  });

  it("reports a failed save so callers cannot announce success", async () => {
    let autoSave!: AutoSaveState<{ enabled: boolean }>;
    const wrapper = mount(
      defineComponent({
        setup() {
          autoSave = useSettingAutoSave<{ enabled: boolean }>({ enabled: false }, async () => {
            throw new Error("offline");
          });
          return () => h("div");
        },
      })
    );

    const saved = await autoSave.save();

    expect(saved).toBe(false);
    expect(autoSave.error.value).toBe("offline");
    expect(toast.success).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  it("keeps the operator's edit when the save fails", async () => {
    let autoSave!: AutoSaveState<{ name: string }>;
    const wrapper = mount(
      defineComponent({
        setup() {
          autoSave = useSettingAutoSave<{ name: string }>({ name: "before" }, async () => {
            throw new Error("offline");
          });
          return () => h("div");
        },
      })
    );

    autoSave.state.value.name = "typed-by-the-operator";
    await autoSave.save();

    // Reverting would discard typing the console never said it discarded.
    expect(autoSave.state.value.name).toBe("typed-by-the-operator");
    expect(autoSave.error.value).toBe("offline");
    wrapper.unmount();
  });
});
