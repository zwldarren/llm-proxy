<script setup lang="ts">
import { Gauge } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { Badge } from "@/components/ui/badge";

defineProps<{ rpm: number }>();

const { t } = useI18n();
</script>

<template>
  <Badge
    variant="outline"
    class="font-normal text-[11px] px-1.5 py-0 text-muted-foreground"
    :title="t('apiKeys.rateLimit')"
  >
    <Gauge class="w-3 h-3 mr-1" />
    <!-- Mono + tabular for the numeric readout (DESIGN.md mono-data rule) -->
    <span class="text-data-xs">{{ t("apiKeys.rateLimitPerMin", { n: rpm }) }}</span>
  </Badge>
</template>
