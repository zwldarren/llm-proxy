import { STORAGE_KEYS } from "@/constants/storageKeys";

/**
 * The locale the console is currently rendered in.
 *
 * Date and number formatting has to follow the interface language, not the
 * browser's. The i18n instance knows the locale, but importing it here would
 * drag `vue-i18n`'s `createI18n` into every module that formats a date — and
 * the unit tests mock `vue-i18n` partially. This module keeps the two in sync
 * without the dependency: the i18n bootstrap and the language selector write
 * to it, and the formatters read from it.
 */
let activeLocale = "";

/** Read the persisted locale; falls back to English outside a browser. */
function persistedLocale(): string {
  try {
    return localStorage.getItem(STORAGE_KEYS.LOCALE) || "en";
  } catch {
    return "en";
  }
}

/**
 * The active locale.
 *
 * The persisted value is read once and cached: a date column formats one
 * value per row, and a localStorage read per cell is a cost the table pays
 * on every render.
 */
export function getActiveLocale(): string {
  if (!activeLocale) activeLocale = persistedLocale();
  return activeLocale;
}

/** Record the locale the console is now rendering in. */
export function setActiveLocale(locale: string): void {
  activeLocale = locale;
}
