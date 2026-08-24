// frontend/src/utils/sanitize.test.ts
import { describe, expect, it } from "vitest";
import { sanitizeHighlightText } from "@/utils/sanitize";

describe("sanitizeHighlightText", () => {
  it("wraps matching terms in <mark>", () => {
    const out = sanitizeHighlightText("Hello world", "world");
    expect(out).toContain("<mark");
    expect(out).toContain("world</mark>");
  });

  it("matches all occurrences case-insensitively", () => {
    const out = sanitizeHighlightText("Anthropic anthropic", "anthropic");
    expect(out.match(/<mark/g)).toHaveLength(2);
  });

  it("escapes HTML in both text and search term", () => {
    const out = sanitizeHighlightText("<script>alert(1)</script>", "<script>");
    expect(out).not.toContain("<script>alert");
    expect(out).toContain("&lt;script&gt;");
  });

  it("returns escaped text when no term is given", () => {
    expect(sanitizeHighlightText("<b>bold</b>", "")).toBe("&lt;b&gt;bold&lt;/b&gt;");
    expect(sanitizeHighlightText(null, undefined)).toBe("-");
  });

  it("does not serve a cached highlight for a different text with the same length and prefix", () => {
    const prefix = "a".repeat(40);
    const textWithMatch = `${prefix}${"x".repeat(100)}`;
    const textWithoutMatch = `${prefix}${"y".repeat(100)}`;
    const withMatch = sanitizeHighlightText(textWithMatch, "xxx");
    const withoutMatch = sanitizeHighlightText(textWithoutMatch, "xxx");
    expect(withMatch).toContain("<mark");
    // A cache key of term|length|prefix would collide here and return the
    // highlighted version for a text that contains no match at all.
    expect(withoutMatch).toBe(textWithoutMatch);
  });
});
