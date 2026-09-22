<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { cn } from "@/lib/utils";
import { mcpStatusConfig } from "@/composables/useMcpServerMeta";
import type { McpServerStatus } from "@/types/schemas";

const props = defineProps<{
  status?: McpServerStatus;
}>();

const { t } = useI18n();
const statusConfig = computed(() => mcpStatusConfig(props.status, t));
</script>

<template>
  <span
    :class="
      cn(
        'w-2 h-2 rounded-full ring-2 transition-all duration-300 shrink-0',
        statusConfig.bg,
        statusConfig.ring,
        statusConfig.pulse && 'animate-pulse'
      )
    "
  />
  <span :class="cn('text-xs font-medium', statusConfig.color)">
    {{ statusConfig.label }}
  </span>
</template>
