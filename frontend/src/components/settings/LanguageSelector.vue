<script setup lang="ts">
import { Languages } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { loadLocaleMessages } from "@/i18n";
import { STORAGE_KEYS } from "@/constants/storageKeys";

const { locale, t } = useI18n();

const changeLanguage = async (langCode: string) => {
  // Message bundles for non-default locales are loaded on demand.
  await loadLocaleMessages(langCode);
  locale.value = langCode;
  localStorage.setItem(STORAGE_KEYS.LOCALE, langCode);
};
</script>

<template>
  <Select
    :model-value="locale"
    @update:model-value="
      (v) => {
        if (v && typeof v === 'string') changeLanguage(v);
      }
    "
  >
    <SelectTrigger class="w-[140px] h-9">
      <div class="flex items-center gap-2">
        <Languages class="size-4 text-muted-foreground" />
        <SelectValue :placeholder="t('settings.language')" />
      </div>
    </SelectTrigger>
    <SelectContent>
      <SelectItem value="en">
        {{ t("language.english") }}
      </SelectItem>
      <SelectItem value="zh">
        {{ t("language.chinese") }}
      </SelectItem>
    </SelectContent>
  </Select>
</template>
