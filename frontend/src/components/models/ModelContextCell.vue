<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatContextLength } from "@/utils/format";

/**
 * Context · max-output spec shared by the admin models table and the plaza
 * table row; renders an en dash when neither limit is configured.
 */
defineProps<{
  contextLength?: number | null;
  maxOutputTokens?: number | null;
}>();

const { t } = useI18n();
</script>

<template>
  <Tooltip v-if="contextLength != null || maxOutputTokens != null">
    <TooltipTrigger as-child>
      <span class="text-data text-xs text-muted-foreground whitespace-nowrap">
        <template v-if="contextLength != null">{{ formatContextLength(contextLength) }}</template>
        <template v-if="contextLength != null && maxOutputTokens != null">
          <span class="text-border" aria-hidden="true">·</span>
        </template>
        <template v-if="maxOutputTokens != null">
          {{ formatContextLength(maxOutputTokens) }}
          <span class="lowercase tracking-wide text-muted-foreground/70">{{ t("plaza.out") }}</span>
        </template>
      </span>
    </TooltipTrigger>
    <TooltipContent>
      {{
        maxOutputTokens != null ? t("models.contextAndOutputTooltip") : t("models.contextLength")
      }}
    </TooltipContent>
  </Tooltip>
  <span v-else class="text-xs text-muted-foreground">–</span>
</template>
