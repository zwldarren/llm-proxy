<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { ModelStatus } from "@/types/schemas";

/**
 * Lifecycle-status chip for models.dev status ('beta' | 'deprecated').
 * Deprecated renders in the error tint, beta in the warning tint; the
 * tooltip spells out what the marker means.
 */
defineProps<{ status: ModelStatus }>();

const { t } = useI18n();
</script>

<template>
  <Tooltip>
    <TooltipTrigger as-child>
      <span
        class="inline-flex items-center rounded-full border px-1.5 py-px text-[10px] font-medium lowercase shrink-0"
        :class="
          status === 'deprecated'
            ? 'border-status-error/30 bg-status-error/10 text-status-error'
            : 'border-status-warning/30 bg-status-warning/10 text-status-warning'
        "
      >
        {{ status }}
      </span>
    </TooltipTrigger>
    <TooltipContent>{{ t("models.modelStatus") }}</TooltipContent>
  </Tooltip>
</template>
