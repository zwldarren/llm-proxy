import { createI18n } from "vue-i18n";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import en from "./en";
import zh from "./zh";

const messages = {
  en,
  zh,
};

const savedLocale = localStorage.getItem(STORAGE_KEYS.LOCALE) || "en";

export default createI18n({
  legacy: false,
  locale: savedLocale,
  fallbackLocale: "en",
  messages,
});
