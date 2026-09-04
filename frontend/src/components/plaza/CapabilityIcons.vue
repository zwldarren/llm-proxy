<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { CAPABILITY_META } from "@/components/plaza/capabilities";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { ModelCapability } from "@/types/schemas";

/**
 * Icon-only capability indicators with tooltips.
 * Replaces text badges where scanning density matters (tables, list rows).
 */
defineProps<{ capabilities: ModelCapability[] }>();

const { t } = useI18n();
</script>

<template>
  <span v-if="capabilities.length" class="inline-flex items-center gap-1 shrink-0">
    <Tooltip v-for="cap in capabilities" :key="cap">
      <TooltipTrigger as-child>
        <span
          class="inline-flex items-center justify-center"
          tabindex="0"
          :aria-label="t(CAPABILITY_META[cap].labelKey)"
        >
          <component
            :is="CAPABILITY_META[cap].icon"
            :class="['size-3.5', CAPABILITY_META[cap].iconClass]"
            aria-hidden="true"
          />
        </span>
      </TooltipTrigger>
      <TooltipContent>{{ t(CAPABILITY_META[cap].labelKey) }}</TooltipContent>
    </Tooltip>
  </span>
</template>
