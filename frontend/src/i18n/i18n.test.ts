import { describe, expect, it } from "vitest";
import { createI18n } from "vue-i18n";
import en from "@/i18n/en";
import zh from "@/i18n/zh";

/** Every dotted key of a nested message object. */
function collectKeys(messages: Record<string, unknown>, prefix = ""): string[] {
  return Object.entries(messages).flatMap(([key, value]) => {
    const path = prefix ? prefix + "." + key : key;
    if (value && typeof value === "object") {
      return collectKeys(value as Record<string, unknown>, path);
    }
    return typeof value === "string" ? [path] : [];
  });
}

/** Superset of the params used anywhere in the messages, so every key resolves. */
const PARAMS = {
  count: 2,
  n: 2,
  endpoint: "/v1/responses",
  value: 1,
  key: "seed",
  size: "5 MB",
  name: "file.txt",
};

const locales = { en, zh };

describe("i18n messages", () => {
  for (const [locale, messages] of Object.entries(locales)) {
    it("compiles every " + locale + " message", () => {
      const i18n = createI18n({
        legacy: false,
        locale,
        messages: { [locale]: messages },
        missingWarn: false,
        fallbackWarn: false,
      });

      // vue-i18n treats braces as placeholders, so copy containing a literal
      // object (e.g. a JSON snippet) fails to compile. Resolve every key here to
      // catch that at test time instead of inside a render.
      const failures: string[] = [];
      for (const key of collectKeys(messages)) {
        try {
          i18n.global.t(key, PARAMS);
        } catch (error) {
          failures.push(key + ": " + (error as Error).message);
        }
      }
      expect(failures).toEqual([]);
    });
  }

  it("walks nested keys", () => {
    // Guards the helper itself: a broken key builder would make the compile
    // check pass by never resolving any nested message.
    expect(collectKeys(en)).toContain("chat.settingsEffects.notSentByEndpoint");
  });

  it("keeps both locales in sync", () => {
    const enKeys = new Set(collectKeys(en));
    const zhKeys = new Set(collectKeys(zh));
    expect([...enKeys].filter((key) => !zhKeys.has(key))).toEqual([]);
    expect([...zhKeys].filter((key) => !enKeys.has(key))).toEqual([]);
  });
});
