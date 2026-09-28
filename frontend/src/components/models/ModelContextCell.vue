<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatContextLength } from "@/utils/format";

/**
 * Total context-length spec shared by the admin models table and the plaza
 * table row; renders an en dash when the limit is not configured.
 */
defineProps<{
  contextLength?: number | null;
}>();

const { t } = useI18n();
</script>

<template>
  <Tooltip v-if="contextLength != null">
    <TooltipTrigger as-child>
      <span class="text-data text-xs text-muted-foreground whitespace-nowrap">
        {{ formatContextLength(contextLength) }}
      </span>
    </TooltipTrigger>
    <TooltipContent>{{ t("models.contextLength") }}</TooltipContent>
  </Tooltip>
  <span v-else class="text-xs text-muted-foreground">–</span>
</template>
