<script setup lang="ts">
import { BarChart3, Edit, KeyRound, Server, Trash2 } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import RateLimitBadge from "@/components/common/RateLimitBadge.vue";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useBudgetDisplay } from "@/composables/useBudgetDisplay";
import type { ApiKeyRead, ApiKeySpendSummary } from "@/services/api/apiKeys";
import { getApiKeyStatus, isApiKeyExpired } from "@/utils/apiKeys";
import { formatCost, formatDate } from "@/utils/format";

interface Props {
  apiKey: ApiKeyRead;
  spend?: ApiKeySpendSummary;
  isLoading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  spend: undefined,
  isLoading: false,
});

const emit = defineEmits<{
  edit: [];
  delete: [];
  viewUsage: [];
}>();

const { t } = useI18n();

const allowedModels = computed(() => props.apiKey.allowed_models ?? []);
// Keys are two-state: null or empty both mean unrestricted (allow all).
const isAllModels = computed(() => allowedModels.value.length === 0);
const allowedMcpServers = computed(() => props.apiKey.allowed_mcp_servers ?? []);
const isAllMcpServers = computed(() => allowedMcpServers.value.length === 0);

const isExpired = computed(() => isApiKeyExpired(props.apiKey.expires_at));
const status = computed(() => getApiKeyStatus(props.apiKey));

const lastUsedLabel = computed(() =>
  props.apiKey.last_used_at ? formatDate(props.apiKey.last_used_at) : t("apiKeys.never")
);

const spendLabel = computed(() => {
  const spend = props.spend;
  if (!spend) return null;
  if (spend.budget_usd !== null && spend.period_spend_usd !== null) {
    return `${formatCost(spend.period_spend_usd)} / ${formatCost(spend.budget_usd)}`;
  }
  return formatCost(spend.total_spend_usd);
});

const { isBudgetExceeded, budgetRatio, spendTitle, barClass } = useBudgetDisplay(
  {
    budgetUsd: computed(() => props.spend?.budget_usd ?? null),
    budgetPeriod: computed(() => props.spend?.budget_period ?? null),
    budgetResetDay: computed(() => props.spend?.budget_reset_day ?? null),
    periodSpendUsd: computed(() => props.spend?.period_spend_usd ?? null),
  },
  t
);
</script>

<template>
  <article
    class="px-4 sm:px-6 py-2.5 border-b border-border transition-colors duration-150 hover:bg-muted/50"
  >
    <div class="flex items-center gap-3">
      <!-- Icon -->
      <div
        class="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 bg-muted"
        role="img"
        :aria-label="apiKey.name"
      >
        <KeyRound class="w-4 h-4 text-muted-foreground" />
      </div>

      <!-- Name + scope badges -->
      <div class="flex-1 min-w-0">
        <button
          type="button"
          class="text-sm font-medium text-foreground truncate hover:underline underline-offset-4 cursor-pointer block max-w-full text-left"
          :title="t('apiKeys.viewUsage')"
          @click.stop="emit('viewUsage')"
        >
          {{ apiKey.name }}
        </button>
        <div class="mt-1 flex items-center gap-1.5 flex-wrap">
          <!-- Status chip: disabled wins over expired so the badge always
               matches the edit dialog's active toggle. Color is paired with a
               label so status is never color-only. -->
          <Badge
            v-if="status === 'disabled'"
            variant="destructive"
            class="font-medium text-[11px] px-1.5 py-0"
            :title="t('apiKeys.disabled')"
          >
            {{ t("apiKeys.disabled") }}
          </Badge>
          <Badge
            v-else-if="status === 'expired'"
            variant="outline"
            class="font-medium text-[11px] px-1.5 py-0 border-status-warning/60 text-status-warning"
            :title="t('apiKeys.expired')"
          >
            {{ t("apiKeys.expired") }}
          </Badge>
          <Badge
            v-else
            variant="secondary"
            class="font-medium text-[11px] px-1.5 py-0"
            :title="t('apiKeys.active')"
          >
            {{ t("apiKeys.active") }}
          </Badge>
          <Badge
            v-if="isBudgetExceeded"
            variant="destructive"
            class="font-medium text-[11px] px-1.5 py-0"
            :title="t('apiKeys.budgetExceeded')"
          >
            {{ t("apiKeys.budgetExceeded") }}
          </Badge>
          <RateLimitBadge v-if="apiKey.rate_limit_rpm != null" :rpm="apiKey.rate_limit_rpm" />
          <Badge v-if="isAllModels" variant="secondary" class="font-normal text-[11px] px-1.5 py-0">
            {{ t("apiKeys.allModels") }}
          </Badge>
          <template v-else>
            <Badge
              v-for="model in allowedModels.slice(0, 2)"
              :key="model"
              variant="outline"
              class="font-mono text-[11px] px-1.5 py-0"
            >
              {{ model }}
            </Badge>
            <Badge
              v-if="allowedModels.length > 2"
              variant="outline"
              class="font-normal text-[11px] px-1.5 py-0 text-muted-foreground"
            >
              +{{ allowedModels.length - 2 }}
            </Badge>
          </template>
          <Badge
            v-if="isAllMcpServers"
            variant="secondary"
            class="font-normal text-[11px] px-1.5 py-0"
          >
            <Server class="w-3 h-3 mr-1" />
            {{ t("apiKeys.allMcpServers") }}
          </Badge>
          <Badge
            v-else
            variant="outline"
            class="text-[11px] px-1.5 py-0 flex items-center gap-1"
            :title="allowedMcpServers.join(', ')"
          >
            <Server class="w-2.5 h-2.5" />
            {{ allowedMcpServers.length }} {{ t("apiKeys.mcpServersShort") }}
          </Badge>
        </div>
      </div>

      <!-- Timestamps + spend: fixed-width, right-aligned stat columns so
           values line up across rows and never wrap. -->
      <div class="hidden md:flex items-center gap-6 shrink-0 text-right">
        <div v-if="spendLabel" class="flex flex-col w-36" :title="spendTitle">
          <span class="text-[11px] uppercase tracking-wide text-muted-foreground">
            {{ t("apiKeys.spend") }}
          </span>
          <span
            class="text-data text-xs whitespace-nowrap"
            :class="isBudgetExceeded ? 'text-destructive font-medium' : 'text-muted-foreground'"
          >
            {{ spendLabel }}
          </span>
          <div
            v-if="budgetRatio !== null"
            class="mt-1 h-1 w-20 rounded-full bg-muted overflow-hidden"
            role="progressbar"
            :aria-label="spendTitle"
            :aria-valuenow="Math.round(budgetRatio * 100)"
            aria-valuemin="0"
            aria-valuemax="100"
          >
            <div
              class="h-full rounded-full transition-all"
              :class="barClass"
              :style="{ width: `${budgetRatio * 100}%` }"
            />
          </div>
        </div>
        <div class="flex flex-col w-40">
          <span class="text-[11px] uppercase tracking-wide text-muted-foreground">
            {{ t("apiKeys.expires") }}
          </span>
          <span
            class="text-data text-xs whitespace-nowrap"
            :class="isExpired ? 'text-status-warning' : 'text-muted-foreground'"
          >
            {{ apiKey.expires_at ? formatDate(apiKey.expires_at) : t("apiKeys.never") }}
          </span>
        </div>
        <div class="flex flex-col w-40">
          <span class="text-[11px] uppercase tracking-wide text-muted-foreground">
            {{ t("apiKeys.lastUsed") }}
          </span>
          <span
            class="text-data text-xs text-muted-foreground whitespace-nowrap"
            :title="lastUsedLabel"
          >
            {{ lastUsedLabel }}
          </span>
        </div>
      </div>

      <!-- Actions -->
      <div class="flex items-center justify-end gap-1 shrink-0">
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9"
          :aria-label="t('apiKeys.viewUsage')"
          @click.stop="emit('viewUsage')"
        >
          <BarChart3 class="w-4 h-4 icon-btn-muted" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9"
          :disabled="isLoading"
          :aria-label="t('common.edit')"
          @click.stop="emit('edit')"
        >
          <Edit class="w-4 h-4 icon-btn-muted" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
          :disabled="isLoading"
          :aria-label="t('common.delete')"
          @click.stop="emit('delete')"
        >
          <Trash2 class="w-4 h-4" />
        </Button>
      </div>
    </div>
  </article>
</template>
