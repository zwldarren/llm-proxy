<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { useRouter } from "vue-router";
import AuthShell from "@/components/common/AuthShell.vue";
import { Button } from "@/components/ui/button";

const { t } = useI18n();
const router = useRouter();
</script>

<template>
  <AuthShell :title="t('notFound.title')" :description="t('notFound.description')">
    <div class="flex flex-col items-center gap-4">
      <p class="font-mono text-5xl font-semibold tracking-tight text-muted-foreground">404</p>
      <Button class="w-full" @click="router.push('/')">{{ t("notFound.backHome") }}</Button>
    </div>
  </AuthShell>
</template>
