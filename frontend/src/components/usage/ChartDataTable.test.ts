// frontend/src/components/usage/ChartDataTable.test.ts
import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ChartDataTable from "./ChartDataTable.vue";

const mountTable = () =>
  mount(ChartDataTable, {
    props: {
      caption: "Token breakdown",
      rowHeader: "Timestamp",
      columns: ["Input", "Output"],
      rows: [
        { key: "2026-11-04T00", label: "Nov 4 00:00", values: ["10", "20"] },
        { key: "2026-11-04T01", label: "Nov 4 01:00", values: ["30", "40"] },
      ],
    },
  });

describe("ChartDataTable", () => {
  it("renders the caption, headers and one row per bucket", () => {
    const wrapper = mountTable();

    expect(wrapper.find("caption").text()).toBe("Token breakdown");
    expect(wrapper.findAll("thead th").map((th) => th.text())).toEqual([
      "Timestamp",
      "Input",
      "Output",
    ]);
    expect(wrapper.findAll("tbody tr")).toHaveLength(2);
    expect(
      wrapper
        .findAll("tbody tr")[0]!
        .findAll("th, td")
        .map((cell) => cell.text())
    ).toEqual(["Nov 4 00:00", "10", "20"]);
  });

  it("hides the wrapper, not the table", async () => {
    // Firefox ignores width/height/clip-path on a table box, so `sr-only` on the
    // <table> leaves the caption and rows painted at full size. The class has to
    // sit on a wrapper element.
    const wrapper = mountTable();
    const table = wrapper.find("table");

    expect(table.classes()).not.toContain("sr-only");
    expect(table.element.parentElement?.classList.contains("sr-only")).toBe(true);
  });
});
