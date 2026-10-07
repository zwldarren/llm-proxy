<script setup lang="ts">
/**
 * Routing judge settings (ADR-0018).
 *
 * The judge is a System One decision model consulted for the ambiguous first
 * turn of a conversation. Everything here is about *when* it is asked, because
 * that is the latency and cost knob: the gate, the per-mode switch, and shadow
 * mode, which records what the judge would have said without acting on it.
 */
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { SettingsItem } from "@/components/settings";
import CollapsiblePanel from "@/components/common/CollapsiblePanel.vue";
import NumberStepper from "@/components/settings/NumberStepper.vue";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import { configApi } from "@/services/api/config";
import type { RoutingJudgeMode, SmartRoutingConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<SmartRoutingConfig>;
}>();

const { t } = useI18n();
const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const JUDGE_MODES: RoutingJudgeMode[] = ["auto", "best", "fast"];

const judge = computed(() => state.value.judge);

/** Models an operator may point the judge at: only System One models. */
const judgeModels = ref<string[]>([]);
const modelsLoading = ref(false);

onMounted(async () => {
  modelsLoading.value = true;
  try {
    const models = await configApi.getModels();
    judgeModels.value = models.filter((m) => m.supports_systemone).map((m) => m.name);
  } catch {
    // The picker degrades to a hint; the configured name is still shown below.
    judgeModels.value = [];
  } finally {
    modelsLoading.value = false;
  }
});

/** Include whatever is configured, even if it is not (or no longer) System One. */
const modelOptions = computed(() => {
  const configured = judge.value.model;
  return configured && !judgeModels.value.includes(configured)
    ? [configured, ...judgeModels.value]
    : judgeModels.value;
});

const hasNoJudgeModels = computed(() => !modelsLoading.value && modelOptions.value.length === 0);

type GateTrigger = "always" | "confidence" | "band" | "both";

/**
 * The two gate predicates collapse into one choice; unset means "always".
 * ``both`` is reachable: the backend treats the predicates as an OR and does not
 * reject a config that sets both, so the UI has to be able to show and clear it
 * rather than silently hiding one behind the other.
 */
const trigger = computed<GateTrigger>(() => {
  if (judge.value.confidence_below !== null && judge.value.complexity_between !== null)
    return "both";
  if (judge.value.confidence_below !== null) return "confidence";
  if (judge.value.complexity_between !== null) return "band";
  return "always";
});

function setTrigger(value: unknown) {
  const next = value as GateTrigger;
  if (next === "confidence") {
    judge.value.confidence_below ??= 0.7;
    judge.value.complexity_between = null;
  } else if (next === "band") {
    judge.value.complexity_between ??= [0.33, 0.67];
    judge.value.confidence_below = null;
  } else if (next === "both") {
    judge.value.confidence_below ??= 0.7;
    judge.value.complexity_between ??= [0.33, 0.67];
  } else {
    judge.value.confidence_below = null;
    judge.value.complexity_between = null;
  }
}

/** Step-by-tenth fields accumulate float dust; store round numbers. */
function round2(value: number | null, fallback: number): number {
  return Math.round((value ?? fallback) * 100) / 100;
}

/** Keep the band ordered and inside [0, 1] so the backend cannot reject it. */
function setBandLow(value: number | null) {
  const high = judge.value.complexity_between?.[1] ?? 0.67;
  judge.value.complexity_between = [Math.max(0, Math.min(round2(value, 0.33), high - 0.05)), high];
}

function setBandHigh(value: number | null) {
  const low = judge.value.complexity_between?.[0] ?? 0.33;
  judge.value.complexity_between = [low, Math.min(1, Math.max(round2(value, 0.67), low + 0.05))];
}

function toggleMode(mode: RoutingJudgeMode, checked: boolean) {
  const modes = new Set(judge.value.modes);
  if (checked) modes.add(mode);
  else modes.delete(mode);
  // Keep the backend's default order so the list reads the same everywhere.
  judge.value.modes = JUDGE_MODES.filter((m) => modes.has(m));
}
</script>

<template>
  <SettingsItem
    :title="t('smartRouting.judge')"
    :description="t('smartRouting.judgeDescription')"
    :loading="pending"
    :error="error"
    class="border-t border-border/40 bg-muted/5"
  >
    <template #action>
      <Switch
        v-model="state.judge.enabled"
        :aria-label="t('smartRouting.judge')"
        data-testid="judge-enabled"
      />
    </template>
  </SettingsItem>

  <CollapsiblePanel :open="state.judge.enabled">
    <!-- Judge model -->
    <SettingsItem
      variant="nested"
      :title="t('smartRouting.judgeModel')"
      :description="
        hasNoJudgeModels
          ? t('smartRouting.judgeModelNone')
          : t('smartRouting.judgeModelDescription')
      "
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Select v-model="judge.model" :disabled="hasNoJudgeModels">
          <SelectTrigger class="w-56" :aria-label="t('smartRouting.judgeModel')">
            <SelectValue :placeholder="t('smartRouting.judgeModelPlaceholder')" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem v-for="name in modelOptions" :key="name" :value="name">
              {{ name }}
            </SelectItem>
          </SelectContent>
        </Select>
      </template>
    </SettingsItem>

    <!-- Which virtual models may be judged -->
    <SettingsItem
      variant="nested"
      :title="t('smartRouting.judgeModes')"
      :description="t('smartRouting.judgeModesDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <div class="flex flex-wrap items-center gap-4">
          <label
            v-for="mode in JUDGE_MODES"
            :key="mode"
            class="flex items-center gap-2 text-sm text-foreground/90"
          >
            <Checkbox
              :model-value="judge.modes.includes(mode)"
              :aria-label="mode"
              @update:model-value="toggleMode(mode, $event === true)"
            />
            <span class="font-mono text-xs">{{ mode }}</span>
          </label>
        </div>
      </template>
    </SettingsItem>

    <!-- Gate -->
    <SettingsItem
      variant="nested"
      :title="t('smartRouting.judgeGate')"
      :description="t('smartRouting.judgeGateDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Select :model-value="trigger" @update:model-value="setTrigger">
          <SelectTrigger class="w-56" :aria-label="t('smartRouting.judgeGate')">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="always">{{ t("smartRouting.judgeGateAlways") }}</SelectItem>
            <SelectItem value="confidence">{{ t("smartRouting.judgeGateConfidence") }}</SelectItem>
            <SelectItem value="band">{{ t("smartRouting.judgeGateBand") }}</SelectItem>
            <SelectItem value="both">{{ t("smartRouting.judgeGateBoth") }}</SelectItem>
          </SelectContent>
        </Select>
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="trigger === 'confidence' || trigger === 'both'"
      variant="nested"
      :title="t('smartRouting.judgeConfidenceBelow')"
      :description="t('smartRouting.judgeConfidenceBelowDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="judge.confidence_below"
          :min="0"
          :max="1"
          :step="0.05"
          :aria-label="t('smartRouting.judgeConfidenceBelow')"
          @update:model-value="judge.confidence_below = round2($event, 0.7)"
        />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="trigger === 'band' || trigger === 'both'"
      variant="nested"
      :title="t('smartRouting.judgeBand')"
      :description="t('smartRouting.judgeBandDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <div class="flex items-center gap-2">
          <NumberStepper
            :model-value="judge.complexity_between?.[0] ?? 0.33"
            :min="0"
            :max="1"
            :step="0.05"
            :aria-label="t('smartRouting.judgeBandLow')"
            @update:model-value="setBandLow($event)"
          />
          <span class="text-xs text-muted-foreground">–</span>
          <NumberStepper
            :model-value="judge.complexity_between?.[1] ?? 0.67"
            :min="0"
            :max="1"
            :step="0.05"
            :aria-label="t('smartRouting.judgeBandHigh')"
            @update:model-value="setBandHigh($event)"
          />
        </div>
      </template>
    </SettingsItem>

    <!-- Deadline -->
    <SettingsItem
      variant="nested"
      :title="t('smartRouting.judgeDeadline')"
      :description="t('smartRouting.judgeDeadlineDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="judge.deadline_s"
          :min="0.1"
          :max="2"
          :step="0.05"
          :suffix="t('smartRouting.judgeDeadlineUnit')"
          :aria-label="t('smartRouting.judgeDeadline')"
          @update:model-value="judge.deadline_s = round2($event, 0.5)"
        />
      </template>
    </SettingsItem>

    <!-- Shadow mode -->
    <SettingsItem
      variant="nested"
      :title="t('smartRouting.judgeShadow')"
      :description="t('smartRouting.judgeShadowDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="judge.shadow" :aria-label="t('smartRouting.judgeShadow')" />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="judge.shadow"
      variant="nested"
      :title="t('smartRouting.judgeShadowSampleRate')"
      :description="t('smartRouting.judgeShadowSampleRateDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="judge.shadow_sample_rate"
          :min="0"
          :max="1"
          :step="0.05"
          :aria-label="t('smartRouting.judgeShadowSampleRate')"
          @update:model-value="judge.shadow_sample_rate = round2($event, 0.05)"
        />
      </template>
    </SettingsItem>

    <!-- Context budget -->
    <SettingsItem
      variant="nested"
      :title="t('smartRouting.judgeContext')"
      :description="t('smartRouting.judgeContextDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <div class="flex items-center gap-2">
          <NumberStepper
            :model-value="judge.context_turns"
            :min="0"
            :max="10"
            :suffix="t('smartRouting.judgeContextTurnsUnit')"
            :aria-label="t('smartRouting.judgeContextTurns')"
            @update:model-value="judge.context_turns = $event ?? 3"
          />
          <NumberStepper
            :model-value="judge.context_chars"
            :min="500"
            :max="20000"
            :step="500"
            :suffix="t('smartRouting.judgeContextCharsUnit')"
            :aria-label="t('smartRouting.judgeContextChars')"
            @update:model-value="judge.context_chars = $event ?? 4000"
          />
        </div>
      </template>
    </SettingsItem>
  </CollapsiblePanel>
</template>
