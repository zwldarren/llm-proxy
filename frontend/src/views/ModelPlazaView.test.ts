// frontend/src/views/ModelPlazaView.test.ts
import { mount, type VueWrapper } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { defineComponent, h } from "vue";
import type { ModelCatalogEntry } from "@/types/schemas";

vi.mock("@/utils/icons", () => ({
  getProviderIconUrl: vi.fn(() => null),
  getIconUrl: vi.fn(() => null),
  isMonoIcon: vi.fn(() => false),
}));

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));

vi.mock("@/services/api/catalog", () => ({
  catalogApi: {
    getModels: vi.fn(),
  },
}));

vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return {
    ArrowUpDown: Stub,
    AudioLines: Stub,
    Binary: Stub,
    Boxes: Stub,
    Brain: Stub,
    Braces: Stub,
    Check: Stub,
    ChevronDown: Stub,
    ChevronUp: Stub,
    Eye: Stub,
    FlaskConical: Stub,
    ImagePlus: Stub,
    List: Stub,
    LockOpen: Stub,
    Mic: Stub,
    Paperclip: Stub,
    Plus: Stub,
    Radio: Stub,
    RefreshCw: Stub,
    Search: Stub,
    Table2: Stub,
    Thermometer: Stub,
    Wrench: Stub,
    X: Stub,
  };
});

import { catalogApi } from "@/services/api/catalog";
import ModelPlazaView from "./ModelPlazaView.vue";

const AppLayoutStub = defineComponent({
  name: "AppLayout",
  setup(_, { slots }) {
    return () => h("div", [slots.header?.(), slots.default?.()]);
  },
});

// Row stubs: the view's contract is that store entries reach the rows;
// row rendering itself is the row components' own concern.
const PlazaTableRowStub = defineComponent({
  name: "PlazaModelTableRow",
  props: ["model"],
  setup(props) {
    return () => h("div", { class: "plaza-row" }, props.model.name);
  },
});

const PlazaListItemStub = defineComponent({
  name: "PlazaModelListItem",
  props: ["model"],
  setup(props) {
    return () => h("div", { class: "plaza-row" }, props.model.name);
  },
});

// Passthrough stubs: Tooltip parts rely on the app-level TooltipProvider,
// which is absent in isolated view mounts.
const TooltipStub = defineComponent({
  name: "Tooltip",
  setup(_, { slots }) {
    return () => slots.default?.();
  },
});
const TooltipTriggerStub = defineComponent({
  name: "TooltipTrigger",
  props: ["asChild"],
  setup(_, { slots }) {
    return () => slots.default?.();
  },
});
const TooltipContentStub = defineComponent({
  name: "TooltipContent",
  setup() {
    return () => null;
  },
});

// reka-ui ToggleGroup needs app-level providers; the toggle is inert here.
const ViewToggleStub = defineComponent({
  name: "ViewToggle",
  props: ["modelValue"],
  emits: ["update:modelValue"],
  setup() {
    return () => h("div");
  },
});

const entry = (name: string, overrides: Partial<ModelCatalogEntry> = {}): ModelCatalogEntry => ({
  name,
  capabilities: [],
  provider_names: ["openai"],
  ...overrides,
});

function mountPlaza() {
  return mount(ModelPlazaView, {
    global: {
      stubs: {
        AppLayout: AppLayoutStub,
        PlazaModelTableRow: PlazaTableRowStub,
        PlazaModelListItem: PlazaListItemStub,
        Tooltip: TooltipStub,
        TooltipTrigger: TooltipTriggerStub,
        TooltipContent: TooltipContentStub,
        ViewToggle: ViewToggleStub,
      },
    },
  });
}

/** Await the rendered condition instead of a guessed flush duration. */
async function waitForText(wrapper: VueWrapper, text: string) {
  await vi.waitFor(() => {
    expect(wrapper.text()).toContain(text);
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.mocked(catalogApi.getModels).mockReset();
});

describe("ModelPlazaView", () => {
  it("loads the catalog through the store and renders entries", async () => {
    vi.mocked(catalogApi.getModels).mockResolvedValue([entry("gpt-4o"), entry("claude-opus")]);

    const wrapper = mountPlaza();
    await waitForText(wrapper, "gpt-4o");

    expect(catalogApi.getModels).toHaveBeenCalledTimes(1);
    expect(wrapper.text()).toContain("claude-opus");
  });

  it("serves a remount from the store cache without refetching", async () => {
    vi.mocked(catalogApi.getModels).mockResolvedValue([entry("gpt-4o")]);

    const first = mountPlaza();
    await waitForText(first, "gpt-4o");
    first.unmount();

    // The cached entries render synchronously on the second mount.
    const second = mountPlaza();
    expect(second.text()).toContain("gpt-4o");
    expect(catalogApi.getModels).toHaveBeenCalledTimes(1);
    second.unmount();
  });

  it("refresh button forces a refetch", async () => {
    vi.mocked(catalogApi.getModels).mockResolvedValue([entry("gpt-4o")]);

    const wrapper = mountPlaza();
    await waitForText(wrapper, "gpt-4o");

    const refresh = wrapper.find('button[aria-label="common.refresh"]');
    expect(refresh.exists()).toBe(true);
    await refresh.trigger("click");

    await vi.waitFor(() => {
      expect(catalogApi.getModels).toHaveBeenCalledTimes(2);
    });
  });

  it("shows the store error and recovers via retry", async () => {
    vi.mocked(catalogApi.getModels).mockRejectedValue(new Error("boom"));

    const wrapper = mountPlaza();
    await waitForText(wrapper, "boom");

    vi.mocked(catalogApi.getModels).mockResolvedValue([entry("gpt-4o")]);
    // The empty state's retry shares the "refresh" label with the header
    // button; only the header one carries an aria-label.
    const retry = wrapper
      .findAll("button")
      .find((b) => b.text().includes("common.refresh") && !b.attributes("aria-label"));
    expect(retry).toBeDefined();
    await retry!.trigger("click");
    await waitForText(wrapper, "gpt-4o");

    expect(wrapper.text()).not.toContain("boom");
  });
});
