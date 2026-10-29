<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { Loader2 } from "@lucide/vue";
import AuthShell from "@/components/common/AuthShell.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { HttpError } from "@/services/http";
import { securityApi } from "@/services/api/security";
import { useAuthStore } from "@/stores/auth";
import { getErrorMessage } from "@/utils/error";
import { passwordRequirementsText, validatePasswordPair } from "@/utils/password";

const route = useRoute();
const router = useRouter();
const { t } = useI18n();
const authStore = useAuthStore();
const loading = ref(true);
const error = ref("");
const challengeToken = ref<string | null>(null);
const code = ref("");
const recoveryMode = ref(false);
const recoveryCode = ref("");
const onboarding = ref(false);
const completionToken = ref("");
const username = ref("");
const password = ref("");
const confirmPassword = ref("");

const pageTitle = computed(() => {
  if (challengeToken.value) return t("auth.mfaTitle");
  if (onboarding.value) return t("auth.oauthOnboardingTitle");
  return t("auth.oauthTitle");
});
const pageDescription = computed(() => {
  if (challengeToken.value) {
    return recoveryMode.value ? t("auth.recoveryDescription") : t("auth.mfaDescription");
  }
  if (onboarding.value) return t("auth.oauthOnboardingDescription");
  return "";
});

function landingPath(): string {
  // A freshly invited OAuth account has no passkey yet. Send it to account
  // security after sign-in, where the passkey prompt is prominent.
  return onboarding.value ? "/account/security?from=oauth" : "/";
}

/**
 * Map the backend's structured error codes to what the user can act on.
 *
 * A bare catch would turn "that username is taken" into "sign-in could not be
 * completed", and a timeout into "that authentication code is not valid" —
 * which pushes people to burn single-use recovery codes for no reason.
 */
function errorMessage(e: unknown): string {
  if (!(e instanceof HttpError)) return t("auth.oauthError");
  // A 409 here is always the chosen username being taken; every other code
  // (expired completion, MFA, unknown username) is mapped by getErrorMessage.
  if (e.status === 409) return t("auth.oauthUsernameTaken");
  return getErrorMessage(e);
}

/**
 * Take the bearer token out of the address bar as soon as it is read: it stays
 * valid for its TTL, so leaving it in the URL leaves it in browser history and
 * in anything that captures the screen.
 */
async function stripTokenFromUrl(key: "token" | "link_token") {
  if (!route.query[key]) return;
  const query = { ...route.query };
  delete query[key];
  await router.replace({ query });
}

async function completeLink(linkToken: string) {
  loading.value = true;
  error.value = "";
  try {
    await securityApi.completeLink(linkToken);
    await router.replace("/account/security?linked=1");
  } catch (e: unknown) {
    error.value = errorMessage(e);
  } finally {
    loading.value = false;
  }
}

async function complete() {
  const token = completionToken.value;
  if (!token) {
    error.value = t("auth.oauthError");
    loading.value = false;
    return;
  }
  loading.value = true;
  error.value = "";
  try {
    const response = await authStore.completeOAuth(
      token,
      username.value.trim() || undefined,
      password.value || undefined
    );
    if (response.mfa_required && response.challenge_token) {
      challengeToken.value = response.challenge_token;
    } else {
      await router.replace(landingPath());
    }
  } catch (e: unknown) {
    error.value = errorMessage(e);
  } finally {
    loading.value = false;
  }
}

onMounted(async () => {
  const linkToken = String(route.query.link_token || "");
  if (linkToken) {
    await stripTokenFromUrl("link_token");
    await completeLink(linkToken);
    return;
  }
  const token = String(route.query.token || "");
  if (!token) {
    error.value = t("auth.oauthError");
    loading.value = false;
    return;
  }
  // Keep the token in memory and drop it from the URL: onboarding posts it
  // later, and the address bar/history must not keep a live bearer token.
  completionToken.value = token;
  await stripTokenFromUrl("token");
  onboarding.value = route.query.onboarding === "1";
  username.value = String(route.query.suggested_username || "");
  if (onboarding.value) {
    loading.value = false;
    return;
  }
  await complete();
});

async function verify() {
  if (!challengeToken.value) return;
  loading.value = true;
  error.value = "";
  try {
    const response = recoveryMode.value
      ? await authStore.recoverMfa(challengeToken.value, recoveryCode.value)
      : await authStore.verifyMfa(challengeToken.value, code.value);
    if (response.mfa_required && response.challenge_token) {
      challengeToken.value = response.challenge_token;
      return;
    }
    await router.replace(landingPath());
  } catch (e: unknown) {
    const data = e instanceof HttpError ? (e.data as { code?: string } | undefined) : undefined;
    if (data?.code === "mfa_challenge_expired" || data?.code === "invalid_challenge") {
      challengeToken.value = null;
      code.value = "";
      recoveryCode.value = "";
      recoveryMode.value = false;
    }
    error.value = errorMessage(e);
  } finally {
    loading.value = false;
  }
}

async function submitOnboarding() {
  if (!username.value.trim()) {
    error.value = t("auth.usernameRequired");
    return;
  }
  const passwordError = validatePasswordPair(password.value, confirmPassword.value);
  if (passwordError) {
    error.value = passwordError;
    return;
  }
  await complete();
}
</script>

<template>
  <AuthShell :title="pageTitle" :description="pageDescription">
    <Alert v-if="error" variant="destructive" class="mb-4">
      <AlertDescription>{{ error }}</AlertDescription>
    </Alert>
    <div v-if="loading" class="flex flex-col items-center gap-3 py-10">
      <Loader2 class="h-5 w-5 animate-spin text-muted-foreground" />
      <p class="text-sm text-muted-foreground">{{ t("auth.oauthCompleting") }}</p>
    </div>
    <form v-else-if="challengeToken" class="flex flex-col gap-4" @submit.prevent="verify">
      <div class="flex flex-col gap-1.5">
        <Label for="oauth-mfa">
          {{ recoveryMode ? t("auth.recoveryCode") : t("auth.mfaCode") }}
        </Label>
        <Input
          v-if="recoveryMode"
          id="oauth-recovery"
          v-model="recoveryCode"
          autocomplete="one-time-code"
          required
          autofocus
          class="auth-input h-11 font-mono"
        />
        <Input
          v-else
          id="oauth-mfa"
          v-model="code"
          inputmode="numeric"
          autocomplete="one-time-code"
          pattern="[0-9]*"
          maxlength="6"
          required
          autofocus
          class="auth-input h-12 text-center font-mono text-lg tracking-[0.35em]"
        />
      </div>
      <Button
        type="button"
        variant="link"
        size="sm"
        class="self-start px-0"
        @click="recoveryMode = !recoveryMode"
      >
        {{ recoveryMode ? t("auth.useTotpInstead") : t("auth.useRecoveryInstead") }}
      </Button>
      <Button type="submit" :disabled="loading" class="h-11 w-full">
        <Loader2 v-if="loading" class="h-4 w-4 animate-spin" />
        {{ t("auth.verifyButton") }}
      </Button>
    </form>
    <form v-else-if="onboarding" class="flex flex-col gap-4" @submit.prevent="submitOnboarding">
      <div class="flex flex-col gap-1.5">
        <Label for="oauth-username">{{ t("auth.username") }}</Label>
        <Input
          id="oauth-username"
          v-model="username"
          autocomplete="username"
          autocapitalize="none"
          spellcheck="false"
          required
          class="auth-input h-11"
        />
      </div>
      <div class="flex flex-col gap-1.5">
        <Label for="oauth-password">{{ t("auth.optionalPassword") }}</Label>
        <Input
          id="oauth-password"
          v-model="password"
          type="password"
          autocomplete="new-password"
          class="auth-input h-11"
        />
        <p class="text-xs text-muted-foreground">{{ passwordRequirementsText() }}</p>
      </div>
      <div class="flex flex-col gap-1.5">
        <Label for="oauth-confirm-password">{{ t("auth.confirmPassword") }}</Label>
        <Input
          id="oauth-confirm-password"
          v-model="confirmPassword"
          type="password"
          autocomplete="new-password"
          class="auth-input h-11"
        />
      </div>
      <p class="text-xs text-muted-foreground">{{ t("auth.oauthPasskeyHint") }}</p>
      <Button type="submit" :disabled="loading" class="h-11 w-full">
        <Loader2 v-if="loading" class="h-4 w-4 animate-spin" />
        {{ t("auth.oauthContinue") }}
      </Button>
    </form>
    <div v-else class="flex flex-col items-center gap-3 py-10">
      <p class="text-sm text-muted-foreground">{{ t("auth.oauthError") }}</p>
      <Button variant="outline" @click="router.replace('/login')">
        {{ t("auth.backToLogin") }}
      </Button>
    </div>
  </AuthShell>
</template>
