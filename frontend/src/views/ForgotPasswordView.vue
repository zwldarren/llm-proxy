<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { ArrowLeft, Loader2 } from "@lucide/vue";
import AuthShell from "@/components/common/AuthShell.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { authApi } from "@/services/api/auth";
import { getErrorMessage } from "@/utils/error";

const { t } = useI18n();
const route = useRoute();
const router = useRouter();

// The sign-in page hands over whatever the user already typed, so a failed
// password does not have to be retyped as a username.
const identifier = ref(typeof route.query.identifier === "string" ? route.query.identifier : "");
const error = ref("");
const loading = ref(false);
const submitted = ref(false);
// null while the probe is in flight. The form is withheld until we know the
// instance can actually send mail: promising an email that never arrives is
// the failure this page exists to prevent.
const resetAvailable = ref<boolean | null>(null);

onMounted(async () => {
  try {
    resetAvailable.value = (await authApi.getCapabilities()).password_reset;
  } catch {
    // Unreachable backend is not "reset works": fall back to the administrator
    // path, which is always true, and let the user retry from sign-in.
    resetAvailable.value = false;
  }
});

async function handleSubmit() {
  if (loading.value) return;
  if (!identifier.value.trim()) {
    error.value = t("auth.identifierRequired");
    return;
  }
  error.value = "";
  loading.value = true;
  try {
    // The answer is the same for every account, so there is nothing to inspect
    // here: a rejection would only mean the request itself failed.
    await authApi.requestPasswordReset(identifier.value.trim());
    submitted.value = true;
  } catch (e: unknown) {
    error.value = getErrorMessage(e);
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <AuthShell :title="t('auth.forgotTitle')" :description="t('auth.forgotDescription')">
    <div v-if="resetAvailable === null" class="flex justify-center py-10">
      <Loader2 class="h-5 w-5 animate-spin text-muted-foreground" />
    </div>

    <template v-else-if="resetAvailable === false">
      <Alert id="forgot-unavailable">
        <AlertDescription>{{ t("auth.forgotUnavailable") }}</AlertDescription>
      </Alert>
      <Button variant="outline" class="mt-4 w-full" @click="router.push('/login')">
        <ArrowLeft class="h-4 w-4" />
        {{ t("auth.backToLogin") }}
      </Button>
    </template>

    <template v-else-if="submitted">
      <Alert id="forgot-sent">
        <AlertDescription>{{ t("auth.forgotSent") }}</AlertDescription>
      </Alert>
      <Button class="mt-4 w-full" @click="router.push('/login')">
        {{ t("auth.backToLogin") }}
      </Button>
    </template>

    <form
      v-else
      class="flex flex-col gap-4"
      :aria-describedby="error ? 'forgot-error' : undefined"
      :aria-busy="loading"
      @submit.prevent="handleSubmit"
    >
      <Alert v-if="error" variant="destructive" id="forgot-error">
        <AlertDescription>{{ error }}</AlertDescription>
      </Alert>

      <div class="flex flex-col gap-1.5">
        <Label for="forgot-identifier" class="text-sm font-medium text-foreground/90">
          {{ t("auth.identifier") }}
        </Label>
        <Input
          id="forgot-identifier"
          v-model="identifier"
          type="text"
          autocomplete="username"
          required
          :disabled="loading"
          class="auth-input h-11"
        />
      </div>

      <Button
        type="submit"
        :disabled="loading"
        class="btn-press mt-1 flex h-11 w-full items-center justify-center gap-2"
        size="lg"
      >
        <Loader2 v-if="loading" class="h-4 w-4 animate-spin text-primary-foreground" />
        {{ loading ? t("common.loading") : t("auth.forgotSubmit") }}
      </Button>

      <Button
        type="button"
        variant="ghost"
        :disabled="loading"
        class="w-full text-muted-foreground"
        @click="router.push('/login')"
      >
        <ArrowLeft class="h-4 w-4" />
        {{ t("auth.backToLogin") }}
      </Button>
    </form>
  </AuthShell>
</template>
