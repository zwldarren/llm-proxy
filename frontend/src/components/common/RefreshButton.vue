<script setup lang="ts">
import { Pause, Play, RefreshCw } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

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
  <TooltipProvider>
    <div class="flex items-center gap-1">
      <Tooltip>
        <TooltipTrigger as-child>
          <Button
            variant="ghost"
            size="icon"
            :disabled="isLoading || !isEnabled"
            :title="refreshTooltip"
            :aria-label="refreshTooltip"
            @click="emit('refresh')"
          >
            <RefreshCw class="w-4 h-4" :class="{ 'animate-spin': isLoading }" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>{{ refreshTooltip }}</TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger as-child>
          <Button
            variant="ghost"
            size="icon"
            :disabled="!isEnabled"
            :class="{
              'bg-foreground/10 text-foreground': isAutoRefresh && isEnabled,
            }"
            :title="autoRefreshTooltip"
            :aria-label="autoRefreshTooltip"
            @click="emit('toggleAutoRefresh')"
          >
            <Pause v-if="isAutoRefresh" class="w-4 h-4" />
            <Play v-else class="w-4 h-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>{{ autoRefreshTooltip }}</TooltipContent>
      </Tooltip>
    </div>
  </TooltipProvider>
</template>
