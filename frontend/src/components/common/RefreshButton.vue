<script setup lang="ts">
/**
 * Refresh + auto-refresh pair for a console header.
 *
 * Two icon buttons that switch the same stream: one fetches now, the other
 * starts or stops the poll. They share the header's icon-button metric so the
 * pair sits on the same baseline as the labelled actions beside it, and the
 * running state is a tonal step rather than a fill that competes with them.
 */
import { Pause, Play, RefreshCw } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HEADER_ACTIVE_CLASS, HEADER_ICON_BUTTON_CLASS } from "@/lib/controlMetric";

interface Props {
  isLoading?: boolean;
  isAutoRefresh?: boolean;
  isEnabled?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  isLoading: false,
  isAutoRefresh: false,
  isEnabled: true,
});

const emit = defineEmits<{
  refresh: [];
  toggleAutoRefresh: [];
}>();

const { t } = useI18n();

const refreshTooltip = computed(() => t("common.refresh"));

const autoRefreshTooltip = computed(() => {
  return props.isAutoRefresh ? t("logs.pauseAutoRefresh") : t("logs.resumeAutoRefresh");
});
</script>

<template>
  <div class="flex items-center gap-2">
    <Tooltip>
      <TooltipTrigger as-child>
        <Button
          variant="ghost"
          size="icon-sm"
          :disabled="isLoading || !isEnabled"
          :class="HEADER_ICON_BUTTON_CLASS"
          :aria-label="refreshTooltip"
          @click="emit('refresh')"
        >
          <RefreshCw class="w-4 h-4" :class="{ 'animate-spin': isLoading }" />
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom">{{ refreshTooltip }}</TooltipContent>
    </Tooltip>

    <Tooltip>
      <TooltipTrigger as-child>
        <Button
          variant="ghost"
          size="icon-sm"
          :disabled="!isEnabled"
          :class="[HEADER_ICON_BUTTON_CLASS, isAutoRefresh && isEnabled ? HEADER_ACTIVE_CLASS : '']"
          :aria-label="autoRefreshTooltip"
          @click="emit('toggleAutoRefresh')"
        >
          <Pause v-if="isAutoRefresh" class="w-4 h-4" />
          <Play v-else class="w-4 h-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom">{{ autoRefreshTooltip }}</TooltipContent>
    </Tooltip>
  </div>
</template>
