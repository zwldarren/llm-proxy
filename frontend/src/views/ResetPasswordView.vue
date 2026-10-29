<script setup lang="ts">
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { ArrowLeft, Loader2 } from "@lucide/vue";
import { toast } from "vue-sonner";
import AuthShell from "@/components/common/AuthShell.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { authApi } from "@/services/api/auth";
import { getErrorMessage } from "@/utils/error";
import { passwordRequirementsText, validatePasswordStrength } from "@/utils/password";

const { t } = useI18n();
const route = useRoute();
const router = useRouter();

// The token is the whole credential here, so a link without one is a dead end:
// say so instead of rendering a form that cannot submit.
const token = computed(() => (typeof route.query.token === "string" ? route.query.token : ""));
const newPassword = ref("");
const confirmPassword = ref("");
const error = ref("");
const loading = ref(false);
const passwordHint = passwordRequirementsText();

async function handleSubmit() {
  if (loading.value || !token.value) return;
  if (newPassword.value.length < 8) {
    error.value = t("auth.passwordTooShort");
    return;
  }
  if (newPassword.value !== confirmPassword.value) {
    error.value = t("auth.passwordMismatch");
    return;
  }
  const strengthError = validatePasswordStrength(newPassword.value);
  if (strengthError) {
    error.value = strengthError;
    return;
  }
  error.value = "";
  loading.value = true;
  try {
    await authApi.resetPassword(token.value, newPassword.value);
    toast.success(t("auth.resetSuccess"));
    // A reset signs nobody in: the backend revoked every session and token the
    // account had, so the new password has to be used to sign in again.
    await router.replace("/login");
  } catch (e: unknown) {
    error.value = getErrorMessage(e) || t("errors.saveFailed");
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <AuthShell :title="t('auth.resetTitle')" :description="t('auth.resetDescription')">
    <template v-if="!token">
      <Alert variant="destructive" id="reset-missing-token">
        <AlertDescription>{{ t("auth.resetMissingToken") }}</AlertDescription>
      </Alert>
      <Button class="mt-4 w-full" @click="router.replace('/forgot-password')">
        {{ t("auth.forgotSubmit") }}
      </Button>
    </template>

    <form
      v-else
      class="flex flex-col gap-4"
      :aria-describedby="error ? 'reset-error' : undefined"
      :aria-busy="loading"
      @submit.prevent="handleSubmit"
    >
      <Alert v-if="error" variant="destructive" id="reset-error">
        <AlertDescription>{{ error }}</AlertDescription>
      </Alert>

      <div class="flex flex-col gap-1.5">
        <Label for="reset-new-password" class="text-sm font-medium text-foreground/90">
          {{ t("profile.newPassword") }}
        </Label>
        <Input
          id="reset-new-password"
          v-model="newPassword"
          type="password"
          autocomplete="new-password"
          required
          :disabled="loading"
          class="auth-input h-11"
        />
        <p v-if="newPassword" id="reset-password-hint" class="mt-1 text-xs text-muted-foreground">
          {{ passwordHint }}
        </p>
      </div>

      <div class="flex flex-col gap-1.5">
        <Label for="reset-confirm-password" class="text-sm font-medium text-foreground/90">
          {{ t("profile.confirmPassword") }}
        </Label>
        <Input
          id="reset-confirm-password"
          v-model="confirmPassword"
          type="password"
          autocomplete="new-password"
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
        {{ loading ? t("common.loading") : t("auth.resetSubmit") }}
      </Button>

      <Button
        type="button"
        variant="ghost"
        :disabled="loading"
        class="w-full text-muted-foreground"
        @click="router.replace('/login')"
      >
        <ArrowLeft class="h-4 w-4" />
        {{ t("auth.backToLogin") }}
      </Button>
    </form>
  </AuthShell>
</template>
