import { describe, expect, it } from "vitest";
import { safeImageUrl } from "./safeUrl";

describe("safeImageUrl", () => {
  it("keeps ordinary image URLs", () => {
    expect(safeImageUrl("https://cdn.example/a.png")).toBe("https://cdn.example/a.png");
    expect(safeImageUrl("http://cdn.example/a.png")).toBe("http://cdn.example/a.png");
    expect(safeImageUrl("/generated/a.png")).toBe("/generated/a.png");
    expect(safeImageUrl("blob:https://app.example/123")).toBe("blob:https://app.example/123");
    expect(safeImageUrl("data:image/png;base64,AAAA")).toBe("data:image/png;base64,AAAA");
  });

  it("drops script-bearing and non-image schemes", () => {
    expect(safeImageUrl("javascript:alert(1)")).toBeUndefined();
    expect(safeImageUrl("java\nscript:alert(1)")).toBeUndefined();
    expect(safeImageUrl("java\tscript:alert(1)")).toBeUndefined();
    expect(safeImageUrl("data:text/html,<script>alert(1)</script>")).toBeUndefined();
    expect(safeImageUrl("data:image/svg+xml,<svg onload=alert(1)>")).toBeUndefined();
    expect(safeImageUrl("data:image/png,<svg onload=alert(1)>")).toBeUndefined();
    expect(safeImageUrl("file:///etc/passwd")).toBeUndefined();
    expect(safeImageUrl("vbscript:msgbox(1)")).toBeUndefined();
    expect(safeImageUrl(null)).toBeUndefined();
    expect(safeImageUrl(undefined)).toBeUndefined();
    expect(safeImageUrl("")).toBeUndefined();
  });

  it("handles protocol-relative and mixed-case http URLs", () => {
    expect(safeImageUrl("//cdn.example/a.png")).toBe("//cdn.example/a.png");
    expect(safeImageUrl("HTTPS://cdn.example/a.png")).toBe("HTTPS://cdn.example/a.png");
  });
});
