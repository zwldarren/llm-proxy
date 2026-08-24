<script setup lang="ts">
import { Monitor, Moon, Sun } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useTheme, type Theme } from "@/composables/useTheme";

const { t } = useI18n();
const { theme, setTheme } = useTheme();

const options = [
  { value: "light" as const, icon: Sun, label: "theme.light" },
  { value: "dark" as const, icon: Moon, label: "theme.dark" },
  { value: "system" as const, icon: Monitor, label: "theme.system" },
];
</script>

<template>
  <ToggleGroup
    type="single"
    class="inline-flex rounded-lg border border-border p-0.5 bg-muted/40"
    :spacing="1"
    :model-value="theme"
    @update:model-value="(v) => v && setTheme(v as Theme)"
  >
    <ToggleGroupItem
      v-for="opt in options"
      :key="opt.value"
      :value="opt.value"
      class="flex items-center gap-1.5 px-3 py-1.5 h-8 text-xs font-medium rounded-md data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-xs transition-all duration-150"
    >
      <component :is="opt.icon" class="size-3.5" />
      <span>{{ t(opt.label) }}</span>
    </ToggleGroupItem>
  </ToggleGroup>
</template>
