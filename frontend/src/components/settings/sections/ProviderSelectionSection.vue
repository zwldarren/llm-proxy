<script setup lang="ts">
import type { Component } from "vue";
import { Coins, Info, Loader2, Pin, Scale, Shuffle, Waypoints } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SettingsSection } from "@/components/settings";
import { Badge } from "@/components/ui/badge";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { ProviderSelectionConfig, ProviderSelectionStrategy } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<ProviderSelectionConfig>;
}>();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const { t } = useI18n();

interface StrategyOption {
  value: ProviderSelectionStrategy;
  icon: Component;
  nameKey: string;
  descriptionKey: string;
  isDefault?: boolean;
}

/* The options themselves are the control: a radio-card group co-locates each
 * strategy's name, icon, and trade-off description so choices are compared in
 * place instead of cross-referencing a dropdown against separate info rows. */
const options: StrategyOption[] = [
  {
    value: "random",
    icon: Shuffle,
    nameKey: "providerSelection.random",
    descriptionKey: "providerSelection.randomDescription",
    isDefault: true,
  },
  {
    value: "session_sticky",
    icon: Pin,
    nameKey: "providerSelection.sessionSticky",
    descriptionKey: "providerSelection.sessionStickyDescription",
  },
  {
    value: "cost_optimized",
    icon: Coins,
    nameKey: "providerSelection.costOptimized",
    descriptionKey: "providerSelection.costOptimizedDescription",
  },
  {
    value: "balanced",
    icon: Scale,
    nameKey: "providerSelection.balanced",
    descriptionKey: "providerSelection.balancedDescription",
  },
];
</script>

<template>
  <SettingsSection
    :title="t('providerSelection.title')"
    :icon="Waypoints"
    :description="t('providerSelection.description')"
  >
    <template #actions>
      <Loader2 v-if="pending" class="size-3.5 animate-spin text-muted-foreground/80" />
    </template>

    <!-- Save error -->
    <div v-if="error" class="px-5.5 py-3 text-xs text-destructive">
      {{ error }}
    </div>

    <!-- Strategy radio cards -->
    <div class="px-5.5 py-5">
      <div
        role="radiogroup"
        :aria-label="t('providerSelection.title')"
        class="grid grid-cols-1 sm:grid-cols-2 gap-3"
      >
        <label v-for="opt in options" :key="opt.value" class="group cursor-pointer">
          <input
            type="radio"
            class="peer sr-only"
            name="provider-selection-strategy"
            :value="opt.value"
            :checked="state.strategy === opt.value"
            :disabled="pending"
            @change="state.strategy = opt.value"
          />
          <div
            class="flex h-full items-start gap-3.5 rounded-lg border p-4 transition-colors duration-200 peer-focus-visible:ring-2 peer-focus-visible:ring-ring/50 peer-focus-visible:ring-offset-1 peer-focus-visible:ring-offset-background peer-disabled:opacity-60"
            :class="
              state.strategy === opt.value
                ? 'border-foreground/35 bg-muted/35'
                : 'border-border/50 bg-background/25 hover:border-border hover:bg-muted/20'
            "
          >
            <div
              class="flex items-center justify-center size-8 rounded-md border shrink-0 transition-colors duration-200"
              :class="
                state.strategy === opt.value
                  ? 'border-border/60 bg-muted text-foreground'
                  : 'border-border/40 bg-muted/60 text-foreground/70'
              "
            >
              <component :is="opt.icon" class="size-4" />
            </div>

            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2">
                <span class="text-sm font-semibold text-foreground tracking-tight">
                  {{ t(opt.nameKey) }}
                </span>
                <Badge
                  v-if="opt.isDefault"
                  variant="outline"
                  class="px-1.5 py-0 text-[10px] font-normal text-muted-foreground border-border/50"
                >
                  {{ t("labels.default") }}
                </Badge>
              </div>
              <p class="text-xs text-muted-foreground mt-1 leading-normal">
                {{ t(opt.descriptionKey) }}
              </p>
            </div>

            <!-- Radio indicator: filled dot + hairline + tonal step, never color-only -->
            <span
              class="mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full border transition-colors duration-200"
              :class="
                state.strategy === opt.value
                  ? 'border-foreground/70'
                  : 'border-border/70 group-hover:border-muted-foreground/60'
              "
            >
              <span v-if="state.strategy === opt.value" class="size-2 rounded-full bg-foreground" />
            </span>
          </div>
        </label>
      </div>
    </div>

    <!-- Priority-group scoping footnote -->
    <div class="flex items-start gap-2.5 px-5.5 py-4">
      <Info class="size-3.5 mt-px shrink-0 text-muted-foreground/80" />
      <p class="text-xs text-muted-foreground leading-normal max-w-[70ch]">
        {{ t("providerSelection.help") }}
      </p>
    </div>
  </SettingsSection>
</template>
