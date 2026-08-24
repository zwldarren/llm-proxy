<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useRouter } from "vue-router";
import { Loader2 } from "@lucide/vue";
import LogoIcon from "@/components/common/LogoIcon.vue";
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

const handleSetup = async () => {
  if (loading.value) return; // Prevent multiple submissions

  // Client-side validation
  if (!username.value.trim()) {
    error.value = t("auth.setupError");
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
    // setup() auto-logs in (token set, needsSetup cleared); go to home.
    await router.push("/");
  } catch (e: unknown) {
    error.value = extractErrorMessage(e);
    loading.value = false;
  }
};
</script>

<template>
  <div
    class="login-bg relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-background px-4"
  >
    <main
      class="relative z-10 flex w-full max-w-sm flex-col animate-in fade-in slide-in-from-bottom-4 duration-500"
    >
      <!-- Brand mark -->
      <div class="mb-8 flex flex-col items-center text-center">
        <div
          class="auth-mark ring-1 ring-primary/25 flex h-14 w-14 items-center justify-center rounded-2xl text-foreground"
        >
          <LogoIcon class="h-8 w-8" />
        </div>
        <span class="mt-6 text-xs text-muted-foreground"> LLM Proxy </span>
        <h1
          class="brand-heading mt-2 text-xl sm:text-2xl leading-tight tracking-tight text-foreground"
        >
          {{ t("auth.setupTitle") }}
        </h1>
        <p class="mt-2 max-w-sm text-sm text-muted-foreground">
          {{ t("auth.setupDescription") }}
        </p>
      </div>

      <!-- Floating form — no enclosing card -->
      <form
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
            required
            :disabled="loading"
            class="auth-input h-11"
          />
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
    </main>

    <footer class="absolute bottom-5 left-0 right-0 z-10 text-center">
      <span class="text-xs text-muted-foreground">{{ t("app.name") }}</span>
    </footer>
  </div>
</template>
