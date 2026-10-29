<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { Loader2 } from "@lucide/vue";
import AuthShell from "@/components/common/AuthShell.vue";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { HttpError, NetworkError, TimeoutError } from "@/services/http";
import { authApi } from "@/services/api/auth";

const route = useRoute();
const router = useRouter();
const { t } = useI18n();
const loading = ref(true);
const email = ref("");
const error = ref("");

onMounted(async () => {
  const token = String(route.query.token || "");
  if (!token) {
    error.value = t("accountSecurity.emailVerificationInvalid");
    loading.value = false;
    return;
  }
  try {
    const result = await authApi.verifyEmail(token);
    email.value = result.email;
  } catch (e: unknown) {
    // A transport failure says nothing about the link. Reporting it as
    // "invalid or expired" would send a user with a perfectly good link off to
    // request another one.
    error.value =
      e instanceof NetworkError || e instanceof TimeoutError
        ? t("common.error")
        : e instanceof HttpError && e.status < 500
          ? t("accountSecurity.emailVerificationInvalid")
          : t("common.error");
  } finally {
    loading.value = false;
  }
});
</script>

<template>
  <AuthShell
    :title="
      email ? t('accountSecurity.emailVerified') : t('accountSecurity.emailVerificationTitle')
    "
    :description="email"
  >
    <div v-if="loading" class="flex flex-col items-center gap-3 py-10">
      <Loader2 class="h-5 w-5 animate-spin text-muted-foreground" />
    </div>
    <template v-else-if="email">
      <Button class="w-full" @click="router.push('/account')">
        {{ t("accountSecurity.backToAccount") }}
      </Button>
    </template>
    <template v-else>
      <Alert variant="destructive">
        <AlertDescription>{{ error }}</AlertDescription>
      </Alert>
      <Button variant="outline" class="mt-4 w-full" @click="router.push('/')">
        {{ t("common.back") }}
      </Button>
    </template>
  </AuthShell>
</template>
