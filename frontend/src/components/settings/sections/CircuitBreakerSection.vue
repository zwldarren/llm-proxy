<script setup lang="ts">
import { Loader2, RefreshCw, RotateCcw, Zap } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import CollapsiblePanel from "@/components/common/CollapsiblePanel.vue";
import NumberStepper from "@/components/settings/NumberStepper.vue";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { CircuitBreakerListResponse, ResilienceConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<ResilienceConfig>;
  circuitStates: CircuitBreakerListResponse | null;
  loading: boolean;
  resetting: boolean;
}>();

const emit = defineEmits<{
  (e: "refresh"): void;
  (e: "resetAll"): void;
  (e: "resetOne", key: string): void;
}>();

const { t } = useI18n();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const hasOpenCircuits = computed(
  () => props.circuitStates?.circuits.some((c) => c.state === "open") ?? false
);

const sortedCircuits = computed(() => {
  const circuits = props.circuitStates?.circuits ?? [];
  return [...circuits].sort((a, b) => {
    const stateOrder = { open: 0, half_open: 1, closed: 2 };
    const aOrder = stateOrder[a.state] ?? 3;
    const bOrder = stateOrder[b.state] ?? 3;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.provider.localeCompare(b.provider);
  });
});

function getStateVariant(state: string) {
  switch (state) {
    case "open":
      return "destructive";
    case "half_open":
      return "warning";
    case "closed":
      return "secondary";
    default:
      return "secondary";
  }
}

function formatCooldown(key: string): string {
  const circuit = props.circuitStates?.circuits.find((c) => c.key === key);
  if (!circuit || circuit.state !== "open") return "-";

  // Use the cooldown from the circuit state response (backend-truth) if available,
  // otherwise fall back to the current config value.
  const cooldownSeconds =
    circuit.cooldown_seconds ?? state.value.circuit_breaker.cooldown_seconds ?? 60;
  const elapsed = Date.now() / 1000 - circuit.last_state_change;
  const remaining = Math.max(0, cooldownSeconds - elapsed);

  if (remaining <= 0) return t("circuitBreaker.probing");
  return t("common.secondsShort", { n: Math.ceil(remaining) });
}
</script>

<template>
  <SettingsSection
    :title="t('circuitBreaker.title')"
    :icon="Zap"
    :description="t('circuitBreaker.description')"
  >
    <!-- Enabled -->
    <SettingsItem
      :title="t('circuitBreaker.enabled')"
      :description="t('circuitBreaker.enabledDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.circuit_breaker.enabled" />
      </template>
    </SettingsItem>

    <!-- Collapsible settings for Circuit Breaker configuration -->
    <CollapsiblePanel :open="state.circuit_breaker.enabled">
      <!-- Failure Threshold -->
      <SettingsItem
        :title="t('circuitBreaker.failureThreshold')"
        :description="t('circuitBreaker.failureThresholdDescription')"
        :loading="pending"
        :error="error"
        class="border-t border-border/40 bg-muted/5"
      >
        <template #action>
          <NumberStepper
            :model-value="state.circuit_breaker.failure_threshold"
            :min="1"
            @update:model-value="state.circuit_breaker.failure_threshold = $event ?? 1"
          />
        </template>
      </SettingsItem>

      <!-- Cooldown -->
      <SettingsItem
        :title="t('circuitBreaker.cooldown')"
        :description="t('circuitBreaker.cooldownDescription')"
        :loading="pending"
        :error="error"
        class="border-t border-border/40 bg-muted/5"
      >
        <template #action>
          <NumberStepper
            :model-value="state.circuit_breaker.cooldown_seconds"
            :min="1"
            :step="5"
            suffix="s"
            @update:model-value="state.circuit_breaker.cooldown_seconds = $event ?? 1"
          />
        </template>
      </SettingsItem>

      <!-- Circuit states list -->
      <div class="border-t border-border/40 bg-muted/5">
        <div
          class="px-5.5 py-4 border-b border-border/40 bg-muted/10 flex items-center justify-between gap-4"
        >
          <div class="space-y-0.5">
            <h4 class="text-sm font-semibold text-foreground">
              {{ t("circuitBreaker.circuits") }}
              <Badge variant="outline" class="ml-1.5 font-mono">
                {{ sortedCircuits.length }}
              </Badge>
            </h4>
            <p class="text-xs text-muted-foreground">
              {{ t("circuitBreaker.circuitsDescription") }}
            </p>
          </div>
          <div class="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              :disabled="loading"
              class="h-7 text-xs"
              @click="emit('refresh')"
            >
              <RefreshCw :class="{ 'animate-spin': loading }" class="size-3 mr-1" />
              {{ t("common.refresh") }}
            </Button>
            <Button
              variant="outline"
              size="sm"
              :disabled="resetting || !hasOpenCircuits"
              class="h-7 text-xs text-destructive hover:text-destructive"
              @click="emit('resetAll')"
            >
              <RotateCcw :class="{ 'animate-spin': resetting }" class="size-3 mr-1" />
              {{ t("circuitBreaker.resetAll") }}
            </Button>
          </div>
        </div>

        <!-- Circuits List -->
        <div class="p-5.5 space-y-3">
          <div v-if="loading && !circuitStates" class="flex justify-center py-6">
            <Loader2 class="size-6 animate-spin text-muted-foreground" />
          </div>
          <div
            v-else-if="sortedCircuits.length === 0"
            class="text-center py-6 border border-dashed border-border/60 rounded-lg bg-muted/20 text-xs text-muted-foreground"
          >
            {{ t("circuitBreaker.noCircuits") }}
          </div>
          <div v-else class="space-y-2 max-h-96 overflow-y-auto pr-1">
            <div
              v-for="circuit in sortedCircuits"
              :key="circuit.key"
              class="flex items-center justify-between p-3 rounded-lg border border-border/40 bg-background hover:bg-muted/5 transition-colors duration-150"
            >
              <div class="flex items-center gap-3">
                <Badge :variant="getStateVariant(circuit.state)">
                  {{ circuit.state }}
                </Badge>
                <div>
                  <div class="font-medium font-mono text-xs text-foreground">
                    {{ circuit.provider }}
                    <span v-if="circuit.model" class="text-muted-foreground">
                      :{{ circuit.model }}:{{ circuit.index }}
                    </span>
                  </div>
                  <div class="text-xs text-muted-foreground mt-0.5">
                    {{ t("circuitBreaker.failures") }}: {{ circuit.failure_count }}
                    <span v-if="circuit.state === 'open'" class="ml-2 font-mono text-destructive">
                      ({{ t("circuitBreaker.cooldownRemaining") }}:
                      {{ formatCooldown(circuit.key) }})
                    </span>
                  </div>
                </div>
              </div>
              <Button
                v-if="circuit.state !== 'closed'"
                variant="ghost"
                size="icon"
                class="size-7 text-muted-foreground hover:text-foreground cursor-pointer"
                @click="emit('resetOne', circuit.key)"
              >
                <RotateCcw class="size-3.5" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </CollapsiblePanel>
  </SettingsSection>
</template>
