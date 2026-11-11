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
    <!-- Mono + tabular for the number only: the unit is a word, and the
         equivalent cell in MemberBudgetCell keeps its label in the body face. -->
    <span class="text-data-xs font-semibold">{{ rpm }}</span>
    <span class="ml-1 text-[11px]">{{ t("apiKeys.rateLimitUnit") }}</span>
  </Badge>
</template>
