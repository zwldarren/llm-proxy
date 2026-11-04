import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, nextTick } from "vue";

const push = vi.fn();

vi.mock("vue-i18n", () => {
  return {
    useI18n: () => ({ t: (key: string) => key }),
  };
});

vi.mock("vue-router", () => {
  return {
    useRouter: () => ({ push }),
  };
});

// The palette opens on Cmd/Ctrl+K; invoking the watcher during setup mounts it
// in its open state, which is the only state worth asserting on. Everything else
// keeps the real implementation.
vi.mock(import("@vueuse/core"), async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    useMagicKeys: (() => ({})) as unknown as typeof actual.useMagicKeys,
    whenever: ((_source: unknown, handler: () => void) =>
      handler()) as unknown as typeof actual.whenever,
  };
});

vi.mock("@/services/api/logs", () => {
  return {
    logsApi: { getLogs: vi.fn(() => Promise.resolve({ items: [] })) },
  };
});

vi.mock("@/stores/auth", () => {
  return {
    useAuthStore: () => ({ isAdmin: true }),
  };
});

// The dialog primitives teleport into a portal that happy-dom never flushes, so
// they are replaced by transparent wrappers. What this file asserts is the
// palette's own combobox/listbox wiring, not reka's portal behaviour.
vi.mock("@/components/ui/dialog", () => {
  const passthrough = (name: string) =>
    defineComponent({
      name,
      inheritAttrs: false,
      setup(_props, { slots, attrs }) {
        return () => h("div", attrs, slots.default?.());
      },
    });
  return {
    Dialog: passthrough("Dialog"),
    DialogContent: passthrough("DialogContent"),
    DialogTitle: passthrough("DialogTitle"),
    DialogDescription: passthrough("DialogDescription"),
  };
});

import CommandPalette from "./CommandPalette.vue";

async function mountOpenPalette() {
  const wrapper = mount(CommandPalette);
  await nextTick();
  await nextTick();
  return wrapper;
}

/**
 * The palette is a combobox over a listbox: focus stays in the input and the
 * selection is published through aria-activedescendant. Both halves are
 * asserted here because a palette that renders options but never names the
 * active one is silent to a screen reader.
 */
describe("CommandPalette semantics", () => {
  beforeEach(() => {
    push.mockClear();
  });

  it("exposes the search field as a combobox owning the listbox", async () => {
    const wrapper = await mountOpenPalette();

    const input = wrapper.get('[role="combobox"]');
    const listbox = wrapper.get('[role="listbox"]');

    expect(input.attributes("aria-expanded")).toBe("true");
    expect(input.attributes("aria-controls")).toBe(listbox.attributes("id"));
    expect(input.attributes("aria-activedescendant")).toBe("command-palette-option-0");
    expect(input.attributes("aria-labelledby")).toBe("command-palette-title");

    const options = wrapper.findAll('[role="option"]');
    expect(options.length).toBeGreaterThan(1);
    expect(options[0]!.attributes("aria-selected")).toBe("true");
  });

  it("moves aria-activedescendant with the arrow keys", async () => {
    const wrapper = await mountOpenPalette();
    const input = wrapper.get('[role="combobox"]');

    await input.trigger("keydown", { key: "ArrowDown" });
    expect(input.attributes("aria-activedescendant")).toBe("command-palette-option-1");
    expect(wrapper.findAll('[role="option"]')[1]!.attributes("aria-selected")).toBe("true");

    await input.trigger("keydown", { key: "End", ctrlKey: true });
    const options = wrapper.findAll('[role="option"]');
    expect(input.attributes("aria-activedescendant")).toBe(
      `command-palette-option-${options.length - 1}`
    );

    await input.trigger("keydown", { key: "Home", ctrlKey: true });
    expect(input.attributes("aria-activedescendant")).toBe("command-palette-option-0");
  });

  it("leaves bare Home/End to the caret inside the query field", async () => {
    const wrapper = await mountOpenPalette();
    const input = wrapper.get('[role="combobox"]');

    await input.trigger("keydown", { key: "ArrowDown" });
    await input.trigger("keydown", { key: "End" });
    expect(input.attributes("aria-activedescendant")).toBe("command-palette-option-1");
  });

  it("runs the row that was clicked, not the row that was active", async () => {
    const wrapper = await mountOpenPalette();

    // Row 2 while row 0 is the active one: a click must follow the pointer.
    const options = wrapper.findAll('[role="option"]');
    expect(options[1]!.attributes("aria-selected")).toBe("false");
    await options[2]!.trigger("click");

    expect(push).toHaveBeenCalledTimes(1);
    expect(push).toHaveBeenCalledWith("/chat");
  });

  it("runs the active row on Enter", async () => {
    const wrapper = await mountOpenPalette();
    const input = wrapper.get('[role="combobox"]');

    await input.trigger("keydown", { key: "ArrowDown" });
    await input.trigger("keydown", { key: "Enter" });

    expect(push).toHaveBeenCalledTimes(1);
    expect(push).toHaveBeenCalledWith("/logs");
  });
});
