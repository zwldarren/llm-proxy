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

interface Props {
  /**
   * Borderless, text-sized form for quiet surroundings such as the auth
   * footer, where the selector is chrome rather than part of the task.
   */
  compact?: boolean;
}

withDefaults(defineProps<Props>(), {
  compact: false,
});

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
    <SelectTrigger
      :class="
        compact
          ? 'h-9 w-auto gap-1.5 border-0 bg-transparent px-2 text-xs shadow-none max-md:h-11'
          : 'w-[140px] h-9'
      "
      :aria-label="t('settings.language')"
      :title="compact ? t('settings.language') : undefined"
    >
      <div class="flex items-center" :class="compact ? 'gap-1.5' : 'gap-2'">
        <Languages
          :class="compact ? 'size-3.5 text-muted-foreground' : 'size-4 text-muted-foreground'"
        />
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
