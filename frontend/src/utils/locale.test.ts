import { describe, expect, it } from "vitest";
import { getActiveLocale, setActiveLocale } from "./locale";

describe("locale holder", () => {
  it("reports the locale that was set", () => {
    setActiveLocale("zh");
    expect(getActiveLocale()).toBe("zh");
    setActiveLocale("en");
    expect(getActiveLocale()).toBe("en");
  });
});
