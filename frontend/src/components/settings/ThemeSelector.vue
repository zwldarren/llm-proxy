<script setup lang="ts">
import { Monitor, Moon, Sun } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SegmentedControl, SegmentedControlItem } from "@/components/ui/segmented-control";
import { useTheme } from "@/composables/useTheme";

interface Props {
  /**
   * Icon-only, unboxed form for quiet surroundings such as the auth footer,
   * where the switcher is chrome rather than part of the task on screen.
   */
  compact?: boolean;
}

withDefaults(defineProps<Props>(), {
  compact: false,
});

const { t } = useI18n();
const { theme, setTheme } = useTheme();

const options = [
  { value: "light" as const, icon: Sun, label: "theme.light" },
  { value: "dark" as const, icon: Moon, label: "theme.dark" },
  { value: "system" as const, icon: Monitor, label: "theme.system" },
];
</script>

<template>
  <SegmentedControl
    :aria-label="t('settings.theme')"
    :variant="compact ? 'quiet' : 'raised'"
    :size="compact ? 'icon' : 'md'"
  >
    <SegmentedControlItem
      v-for="opt in options"
      :key="opt.value"
      :active="theme === opt.value"
      :title="compact ? t(opt.label) : undefined"
      @click="setTheme(opt.value)"
    >
      <component :is="opt.icon" />
      <span :class="compact ? 'sr-only' : undefined">{{ t(opt.label) }}</span>
    </SegmentedControlItem>
  </SegmentedControl>
</template>
