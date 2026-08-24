import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { ArrowUpDown: Stub, ChevronDown: Stub, ChevronUp: Stub };
});

import SortableHead from "./SortableHead.vue";

function mountHead(props: Record<string, unknown> = {}) {
  return mount(SortableHead, {
    props: { label: "Name", sortKey: "name", ...props },
  });
}

describe("SortableHead", () => {
  it("renders aria-sort=none when inactive", () => {
    expect(mountHead().attributes("aria-sort")).toBe("none");
  });

  it("renders aria-sort=ascending when active and dir asc", () => {
    expect(mountHead({ activeField: "name", activeDir: "asc" }).attributes("aria-sort")).toBe(
      "ascending"
    );
  });

  it("renders aria-sort=descending when active and dir desc", () => {
    expect(mountHead({ activeField: "name", activeDir: "desc" }).attributes("aria-sort")).toBe(
      "descending"
    );
  });

  it("only marks the matching column as active", () => {
    expect(mountHead({ activeField: "type", sortKey: "name" }).attributes("aria-sort")).toBe(
      "none"
    );
  });

  it("emits sort with the sortKey when the header button is clicked", async () => {
    const w = mountHead({ sortKey: "type" });
    await w.find("button").trigger("click");
    expect(w.emitted("sort")).toBeTruthy();
    expect(w.emitted("sort")![0]).toEqual(["type"]);
  });

  it("exposes the sort key as a data attribute for targeting", () => {
    expect(mountHead({ sortKey: "type" }).find("button").attributes("data-sort-key")).toBe("type");
  });
});
