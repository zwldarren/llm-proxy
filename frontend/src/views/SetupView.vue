<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useRouter } from "vue-router";
import { Loader2 } from "@lucide/vue";
import AuthShell from "@/components/common/AuthShell.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { HttpError } from "@/services/http";
import { useAuthStore } from "@/stores/auth";
import { getErrorMessage } from "@/utils/error";
import { passwordRequirementsText, validatePasswordStrength } from "@/utils/password";

const router = useRouter();
const { t } = useI18n();
const authStore = useAuthStore();

const username = ref("");
const password = ref("");
const confirmPassword = ref("");
const error = ref("");
const loading = ref(false);

const passwordError = computed(() => validatePasswordStrength(password.value));
const usernameError = computed(() => {
  const value = username.value.trim();
  if (!value) return t("auth.usernameRequired");
  if (value.length > 64 || !/^[a-zA-Z0-9_-]+$/.test(value)) return t("profile.usernameInvalid");
  return "";
});

const passwordHint = computed(() => passwordRequirementsText());

watch(
  () => authStore.needsSetup,
  (value) => {
    if (value === false) {
      void router.replace(authStore.isAuthenticated ? "/" : "/login");
    }
  },
  { immediate: true }
);

const confirmPasswordError = computed(() => {
  if (confirmPassword.value && password.value !== confirmPassword.value) {
    return t("auth.passwordMismatch");
  }
  return "";
});

function extractErrorMessage(e: unknown): string {
  if (e instanceof HttpError) {
    const code = (e.data as { error?: { code?: string } })?.error?.code;
    if (code === "setup_complete") return t("auth.setupAlreadyDone");
    if (code === "conflict") return t("auth.setupUsernameTaken");
    // Validation / other backend errors carry a descriptive message already;
    // surface it instead of a generic "setup failed" so the user knows what to fix
    // (e.g. weak password).
    const msg = getErrorMessage(e);
    if (msg && msg !== `${e.status}: ${e.statusText}`) return msg;
    return t("auth.setupError");
  }
  return t("auth.setupError");
}

const retryingStatus = ref(false);
const retryStatus = async () => {
  retryingStatus.value = true;
  try {
    await authStore.retrySetupStatus();
  } finally {
    retryingStatus.value = false;
  }
};

const handleSetup = async () => {
  if (loading.value) return; // Prevent multiple submissions

  // Client-side validation
  if (usernameError.value) {
    error.value = usernameError.value;
    return;
  }
  if (password.value.length < 8) {
    error.value = t("auth.passwordTooShort");
    return;
  }
  if (password.value !== confirmPassword.value) {
    error.value = t("auth.passwordMismatch");
    return;
  }
  if (passwordError.value) {
    error.value = passwordError.value;
    return;
  }

  error.value = "";
  loading.value = true;

  try {
    await authStore.setup({
      username: username.value.trim(),
      password: password.value,
    });
    // setup() auto-logs in (token set, needsSetup cleared). Land on the
    // passkey prompt instead of the dashboard: the next sign-in should not
    // depend on the password just typed.
    await router.push("/account/security?from=setup");
  } catch (e: unknown) {
    error.value = extractErrorMessage(e);
    loading.value = false;
  }
};
</script>

<template>
  <AuthShell
    :title="authStore.setupStatusError ? t('auth.setupStatusTitle') : t('auth.setupTitle')"
    :description="authStore.setupStatusError ? '' : t('auth.setupDescription')"
  >
    <div
      v-if="authStore.setupStatusError"
      class="flex flex-col items-center gap-4 text-center"
      role="alert"
    >
      <Alert variant="destructive">
        <AlertDescription>{{ t("auth.setupStatusFailed") }}</AlertDescription>
      </Alert>
      <Button variant="outline" :disabled="retryingStatus" @click="retryStatus">
        <Loader2 v-if="retryingStatus" class="mr-2 size-4 animate-spin" />
        {{ t("common.retry") }}
      </Button>
    </div>

    <!-- Floating form — no enclosing card -->
    <form
      v-else
      @submit.prevent="handleSetup"
      class="flex flex-col gap-4"
      :aria-describedby="error ? 'setup-error' : undefined"
      :aria-busy="loading"
    >
      <Alert
        v-if="error"
        variant="destructive"
        class="animate-in slide-in-from-top-2 fade-in duration-300"
        id="setup-error"
      >
        <AlertDescription>
          {{ error }}
        </AlertDescription>
      </Alert>

      <div class="flex flex-col gap-1.5">
        <Label for="username" class="text-sm font-medium text-foreground/90">
          {{ t("auth.username") }}
        </Label>
        <Input
          id="username"
          v-model="username"
          type="text"
          autocomplete="username"
          autocapitalize="none"
          spellcheck="false"
          required
          :disabled="loading"
          :aria-invalid="!!username && !!usernameError"
          :aria-describedby="usernameError ? 'username-error' : 'username-hint'"
          class="auth-input h-11"
        />
        <p
          v-if="usernameError"
          id="username-error"
          class="mt-1 text-xs"
          :class="username ? 'text-destructive' : 'text-muted-foreground'"
        >
          {{ username ? usernameError : t("profile.usernameFormatHint") }}
        </p>
        <p v-else id="username-hint" class="mt-1 text-xs text-muted-foreground">
          {{ t("profile.usernameFormatHint") }}
        </p>
      </div>

      <div class="flex flex-col gap-1.5">
        <Label for="password" class="text-sm font-medium text-foreground/90">
          {{ t("auth.password") }}
        </Label>
        <Input
          id="password"
          v-model="password"
          type="password"
          autocomplete="new-password"
          required
          :disabled="loading"
          :aria-invalid="!!passwordError"
          :aria-describedby="passwordError ? 'password-error' : undefined"
          class="auth-input h-11"
        />
        <p v-if="passwordError" id="password-error" class="mt-1 text-xs text-destructive">
          {{ passwordError }}
        </p>
        <p v-else-if="password" id="password-hint" class="mt-1 text-xs text-muted-foreground">
          {{ passwordHint }}
        </p>
      </div>

      <div class="flex flex-col gap-1.5">
        <Label for="confirm-password" class="text-sm font-medium text-foreground/90">
          {{ t("auth.confirmPassword") }}
        </Label>
        <Input
          id="confirm-password"
          v-model="confirmPassword"
          type="password"
          autocomplete="new-password"
          required
          :disabled="loading"
          :aria-invalid="!!confirmPasswordError"
          :aria-describedby="confirmPasswordError ? 'confirm-password-error' : undefined"
          class="auth-input h-11"
        />
        <p
          v-if="confirmPasswordError"
          id="confirm-password-error"
          class="mt-1 text-xs text-destructive"
        >
          {{ confirmPasswordError }}
        </p>
      </div>

      <Button
        type="submit"
        :disabled="loading"
        class="btn-press mt-1 flex h-11 w-full items-center justify-center gap-2"
        size="lg"
      >
        <Loader2 v-if="loading" class="h-4 w-4 animate-spin text-primary-foreground" />
        {{ loading ? t("common.loading") : t("auth.setupButton") }}
      </Button>
    </form>
  </AuthShell>
</template>
