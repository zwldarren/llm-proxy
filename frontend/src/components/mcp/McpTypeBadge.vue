<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { mcpTypeConfig } from "@/composables/useMcpServerMeta";
import type { McpServerRead } from "@/types/schemas";

const props = withDefaults(
  defineProps<{
    server: McpServerRead;
    /** Tighter padding for the compact list row. */
    compact?: boolean;
  }>(),
  { compact: false }
);

const { t } = useI18n();
const typeConfig = computed(() => mcpTypeConfig(props.server, t));
</script>

<template>
  <Badge
    variant="secondary"
    :class="
      cn(
        'py-0 text-[11px] font-semibold uppercase tracking-wider border shrink-0',
        compact ? 'px-1.5' : 'px-2',
        typeConfig.bg,
        typeConfig.text,
        typeConfig.border
      )
    "
  >
    {{ typeConfig.label }}
  </Badge>
</template>
