import { describe, expect, it } from "vitest";
import en from "@/i18n/en";
import router from "@/router";

function hasMessage(messages: Record<string, unknown>, key: string): boolean {
  let value: unknown = messages;
  for (const part of key.split(".")) {
    if (!value || typeof value !== "object" || !(part in value)) return false;
    value = (value as Record<string, unknown>)[part];
  }
  return typeof value === "string";
}

describe("route titles", () => {
  it("gives every named route a resolvable browser title key", () => {
    const failures: string[] = [];

    for (const route of router.getRoutes()) {
      if (typeof route.name !== "string") continue;

      const titleKey = route.meta.titleKey;
      if (!titleKey) {
        failures.push(`${route.name}: missing titleKey`);
      } else if (!hasMessage(en as unknown as Record<string, unknown>, titleKey)) {
        failures.push(`${route.name}: ${titleKey}`);
      }
    }

    expect(failures).toEqual([]);
  });
});
