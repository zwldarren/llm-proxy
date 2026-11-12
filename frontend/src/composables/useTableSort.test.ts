import { describe, expect, it } from "vitest";
import { useTableSort } from "@/composables/useTableSort";

interface Row {
  name: string;
  created: string | null;
}

const rows: Row[] = [
  { name: "beta", created: "2024-02-01" },
  { name: "alpha", created: null },
  { name: "gamma", created: "2024-01-01" },
];

/** nulls stay last in both directions, mirroring the API-key table. */
function compareRows(field: "name" | "created", a: Row, b: Row, dir: 1 | -1): number {
  if (field === "created") {
    if (!a.created && !b.created) return 0;
    if (!a.created) return 1;
    if (!b.created) return -1;
    return a.created.localeCompare(b.created) * dir;
  }
  return a.name.localeCompare(b.name) * dir;
}

describe("useTableSort", () => {
  it("starts on the initial field and direction", () => {
    const table = useTableSort<Row, "name" | "created">({
      items: rows,
      initialField: "name",
      compare: compareRows,
    });

    expect(table.sortField.value).toBe("name");
    expect(table.sortDir.value).toBe("asc");
    expect(table.sortedItems.value.map((row) => row.name)).toEqual(["alpha", "beta", "gamma"]);
  });

  it("toggles the direction when the active column is clicked again", () => {
    const table = useTableSort<Row, "name">({
      items: rows,
      initialField: "name",
      compare: compareRows,
    });

    table.onSort("name");
    expect(table.sortDir.value).toBe("desc");
    expect(table.sortedItems.value.map((row) => row.name)).toEqual(["gamma", "beta", "alpha"]);

    table.onSort("name");
    expect(table.sortDir.value).toBe("asc");
  });

  it("uses the per-column default direction for a newly active column", () => {
    const table = useTableSort<Row, "name" | "created">({
      items: rows,
      initialField: "name",
      defaultDirectionFor: (field) => (field === "created" ? "desc" : "asc"),
      compare: compareRows,
    });

    table.onSort("created");
    expect(table.sortField.value).toBe("created");
    expect(table.sortDir.value).toBe("desc");
    // Newest first, and the missing date stays last despite the direction.
    expect(table.sortedItems.value.map((row) => row.name)).toEqual(["beta", "gamma", "alpha"]);
  });

  it("restores the initial state on reset", () => {
    const table = useTableSort<Row, "name" | "created">({
      items: rows,
      initialField: "name",
      compare: compareRows,
    });

    table.onSort("created");
    table.onSort("created");
    table.resetSort();

    expect(table.sortField.value).toBe("name");
    expect(table.sortDir.value).toBe("asc");
  });

  it("honors the initial column's default direction and restores it", () => {
    const table = useTableSort<Row, "name" | "created">({
      items: rows,
      initialField: "created",
      defaultDirectionFor: (field) => (field === "created" ? "desc" : "asc"),
      compare: compareRows,
    });

    expect(table.sortDir.value).toBe("desc");
    expect(table.sortedItems.value.map((row) => row.name)).toEqual(["beta", "gamma", "alpha"]);

    table.onSort("name");
    table.resetSort();

    expect(table.sortField.value).toBe("created");
    expect(table.sortDir.value).toBe("desc");
  });
});
