<script setup lang="ts">
import { Shuffle } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import CollapsiblePanel from "@/components/common/CollapsiblePanel.vue";
import ModeWeightSlider from "@/components/settings/sections/ModeWeightSlider.vue";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { LoggingConfig, SmartRoutingConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<SmartRoutingConfig>;
  loggingAutoSave: AutoSaveState<LoggingConfig>;
}>();

const { t } = useI18n();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);
const { state: loggingState, pending: loggingPending, error: loggingError } = props.loggingAutoSave;

type ModeKey = "fast" | "auto" | "best";

const DEFAULT_WEIGHTS: Record<ModeKey, number> = { fast: 0.35, auto: 0.65, best: 1.0 };

function modeWeight(key: ModeKey): number {
  return state.value.mode_weights?.[key] ?? DEFAULT_WEIGHTS[key];
}

function setModeWeight(key: ModeKey, value: number) {
  if (!state.value.mode_weights) {
    state.value.mode_weights = { ...DEFAULT_WEIGHTS };
  }
  state.value.mode_weights[key] = value;
}
</script>

<template>
  <SettingsSection
    :title="t('smartRouting.title')"
    :icon="Shuffle"
    :description="t('smartRouting.description')"
  >
    <SettingsItem
      :title="t('smartRouting.enable')"
      :description="t('smartRouting.enableDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.enabled" />
      </template>
    </SettingsItem>

    <CollapsiblePanel :open="state.enabled">
      <!-- Mode weights section header -->
      <div class="px-5.5 pt-5 pb-2 border-t border-border/40 bg-muted/5">
        <div class="text-sm font-semibold text-foreground">
          {{ t("smartRouting.modeWeights") }}
        </div>
        <p class="text-xs text-muted-foreground mt-1 leading-normal max-w-[65ch]">
          {{ t("smartRouting.modeWeightsDescription") }}
        </p>
      </div>

      <ModeWeightSlider
        :label="t('smartRouting.modeFast')"
        :description="t('smartRouting.modeFastDescription')"
        :model-value="modeWeight('fast')"
        @commit="setModeWeight('fast', $event)"
      />
      <ModeWeightSlider
        :label="t('smartRouting.modeAuto')"
        :description="t('smartRouting.modeAutoDescription')"
        :model-value="modeWeight('auto')"
        @commit="setModeWeight('auto', $event)"
      />
      <ModeWeightSlider
        :label="t('smartRouting.modeBest')"
        :description="t('smartRouting.modeBestDescription')"
        :model-value="modeWeight('best')"
        @commit="setModeWeight('best', $event)"
      />

      <SettingsItem
        :title="t('smartRouting.routingDiagnostics')"
        :description="t('smartRouting.routingDiagnosticsDescription')"
        :loading="loggingPending"
        :error="loggingError"
      >
        <template #action>
          <Switch v-model="loggingState.verbose_routing_logs" />
        </template>
      </SettingsItem>

      <!-- Virtual Models -->
      <div class="px-5.5 py-5 border-t border-border/40 bg-muted/5">
        <div class="text-sm font-semibold text-foreground">
          {{ t("smartRouting.virtualModels") }}
        </div>
        <p class="text-xs text-muted-foreground mt-1 leading-normal max-w-[65ch]">
          {{ t("smartRouting.virtualModelsDescription") }}
        </p>
        <div class="flex flex-wrap gap-2 mt-3.5">
          <Badge
            variant="outline"
            class="font-mono text-xs px-2.5 py-1 bg-background text-foreground/85 border-border/50 hover:bg-muted/40 transition-colors"
          >
            {{ t("smartRouting.modelAuto") }}
          </Badge>
          <Badge
            variant="outline"
            class="font-mono text-xs px-2.5 py-1 bg-background text-foreground/85 border-border/50 hover:bg-muted/40 transition-colors"
          >
            {{ t("smartRouting.modelFast") }}
          </Badge>
          <Badge
            variant="outline"
            class="font-mono text-xs px-2.5 py-1 bg-background text-foreground/85 border-border/50 hover:bg-muted/40 transition-colors"
          >
            {{ t("smartRouting.modelBest") }}
          </Badge>
        </div>
      </div>
    </CollapsiblePanel>
  </SettingsSection>
</template>
