import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import en from "@/i18n/en";

// Vitest runs from `frontend/`, where the Vite alias maps `@` to `src`.
const SRC_DIR = join(process.cwd(), "src");

function flatten(messages: Record<string, unknown>, prefix = ""): string[] {
  return Object.entries(messages).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object") {
      return flatten(value as Record<string, unknown>, path);
    }
    return [path];
  });
}

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      // Message bundles are the definition, not a consumer.
      return entry.name === "i18n" ? [] : sourceFiles(path);
    }
    if (!/\.(ts|vue)$/.test(entry.name) || entry.name.endsWith(".test.ts")) return [];
    return [path];
  });
}

/**
 * A key that is referenced but missing from the bundle renders as the key
 * itself — "common.saved" in a toast — and the runtime compile check cannot see
 * it, because it only walks keys that exist. This walks the other direction.
 */
describe("i18n key references", () => {
  it("resolves every literal t() key used by the app", () => {
    const keys = new Set(flatten(en as unknown as Record<string, unknown>));
    const missing: string[] = [];

    for (const file of sourceFiles(SRC_DIR)) {
      const source = readFileSync(file, "utf8");
      for (const match of source.matchAll(/\bt\(\s*["']([A-Za-z0-9_.]+)["']/g)) {
        const key = match[1];
        // A trailing dot marks a prefix concatenated at runtime
        // (`t("models.capInfoHelp." + capability)`); its branches are declared
        // statically and checked by their own references.
        if (key.endsWith(".")) continue;
        if (!keys.has(key)) missing.push(`${relative(SRC_DIR, file)}: ${key}`);
      }
    }

    expect(missing).toEqual([]);
  });
});
