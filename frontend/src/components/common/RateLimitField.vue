<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

defineProps<{ id: string }>();
const model = defineModel<number | string | null>({ required: true });

const { t } = useI18n();
</script>

<template>
  <div class="space-y-2">
    <Label :for="id">{{ t("apiKeys.rateLimit") }}</Label>
    <Input
      :id="id"
      v-model.number="model"
      type="number"
      min="1"
      step="1"
      :placeholder="t('apiKeys.rateLimitPlaceholder')"
    />
    <p class="text-xs text-muted-foreground">{{ t("apiKeys.rateLimitHelp") }}</p>
  </div>
</template>
