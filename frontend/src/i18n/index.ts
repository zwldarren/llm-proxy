import { createI18n } from "vue-i18n";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import en from "./en";

const savedLocale = localStorage.getItem(STORAGE_KEYS.LOCALE) || "en";

const i18n = createI18n({
  legacy: false,
  locale: savedLocale,
  fallbackLocale: "en",
  messages: { en },
});

/** Locales whose message bundles are already registered. */
const loadedLocales = new Set<string>(["en"]);

/**
 * Register a non-default locale's messages on demand. Chinese is a separate
 * chunk, so an English-only session never downloads it — it used to be bundled
 * eagerly and preloaded on the critical path. Unknown locales fall back to
 * `fallbackLocale`.
 */
export async function loadLocaleMessages(locale: string): Promise<void> {
  if (loadedLocales.has(locale)) return;
  if (locale === "zh") {
    // The CJK fallback face is only needed once a session actually renders
    // Chinese; its 400+ @font-face rules must not ship to English sessions.
    // Loaded in parallel with the messages but not awaited — font-display
    // swap covers the brief fallback-font window.
    void Promise.all([
      import("@fontsource/noto-sans-sc/400.css"),
      import("@fontsource/noto-sans-sc/500.css"),
      import("@fontsource/noto-sans-sc/600.css"),
      import("@fontsource/noto-sans-sc/700.css"),
    ]);
    const { default: zh } = await import("./zh");
    i18n.global.setLocaleMessage("zh", zh);
    loadedLocales.add("zh");
  }
}

/**
 * Load the persisted locale before the app mounts so a Chinese session never
 * paints a frame of English. No-op for the default locale.
 */
export async function initializeLocale(): Promise<void> {
  await loadLocaleMessages(savedLocale);
}

export default i18n;
