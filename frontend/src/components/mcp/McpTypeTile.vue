<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { cn } from "@/lib/utils";
import { mcpTypeConfig } from "@/composables/useMcpServerMeta";
import type { McpServerRead } from "@/types/schemas";

const props = defineProps<{
  server: McpServerRead;
}>();

const { t } = useI18n();
const typeConfig = computed(() => mcpTypeConfig(props.server, t));
</script>

<template>
  <div
    :class="
      cn(
        'w-9 h-9 rounded-lg flex items-center justify-center shrink-0 border',
        typeConfig.bg,
        typeConfig.border
      )
    "
  >
    <component :is="typeConfig.icon" :class="cn('w-4.5 h-4.5', typeConfig.iconColor)" />
  </div>
</template>
