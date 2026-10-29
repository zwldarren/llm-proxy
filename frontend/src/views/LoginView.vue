<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { ArrowLeft, Fingerprint, KeyRound, Loader2 } from "@lucide/vue";
import AuthShell from "@/components/common/AuthShell.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { HttpError } from "@/services/http";
import { authApi } from "@/services/api/auth";
import { usePasskeyAccounts } from "@/composables/usePasskeyAccounts";
import { useAuthStore } from "@/stores/auth";
import { base64UrlToString } from "@/utils/base64url";
import { apiErrorCode, getErrorMessage } from "@/utils/error";
import type { OAuthProvider, Token } from "@/types/schemas";

const router = useRouter();
const route = useRoute();
const { t } = useI18n();
const authStore = useAuthStore();

const identifier = ref("");
const password = ref("");
const error = ref("");
const loading = ref(false);
const passkeyLoading = ref(false);
const providers = ref<OAuthProvider[]>([]);
const challengeToken = ref<string | null>(null);
const mfaCode = ref("");
const recoveryMode = ref(false);
const recoveryCode = ref("");
// null = not asked yet. The sign-in page cannot promise an emailed reset on an
// instance without SMTP, so the entry below is chosen once the backend says
// which recovery path actually exists here.
const resetAvailable = ref<boolean | null>(null);

// WebAuthn only exists in a secure context. On a LAN/HTTP deployment the
// button would always fail, so the password form is the whole form there.
const passkeySupported = computed(
  () => typeof window !== "undefined" && !!window.PublicKeyCredential && window.isSecureContext
);
// A device-local hint, not an account lookup: after this browser registers or
// successfully uses a passkey for this identifier, make that the primary
// action. Fresh browsers and unknown users keep the password form primary.
const { remember: rememberPasskey, isKnown: isKnownPasskey } = usePasskeyAccounts();
const passkeyPreferred = computed(() => passkeySupported.value && isKnownPasskey(identifier.value));
const authTitle = computed(() =>
  challengeToken.value ? t("auth.mfaTitle") : t("auth.loginTitle")
);
const authDescription = computed(() => {
  if (!challengeToken.value) return t("auth.loginDescription");
  return recoveryMode.value ? t("auth.recoveryDescription") : t("auth.mfaDescription");
});
const sessionNotice = computed(() =>
  route.query.reason === "session_revoked" ? t("auth.sessionRevoked") : ""
);

onMounted(async () => {
  const oauthError = route.query.oauth_error;
  if (typeof oauthError === "string" && oauthError) {
    try {
      error.value = base64UrlToString(oauthError);
    } catch {
      error.value = t("auth.oauthError");
    }
  }
  try {
    providers.value = (await authApi.listOAuthProviders()).providers;
  } catch {
    providers.value = [];
  }
  try {
    resetAvailable.value = (await authApi.getCapabilities()).password_reset;
  } catch {
    // An unreadable capability is not proof that reset works; the administrator
    // path is the safe default.
    resetAvailable.value = false;
  }
});

function errorMessage(e: unknown): string {
  // A 401 that carries no backend code is a rejected credential; everything
  // else (MFA codes, lockouts, step-up, OIDC) has a mapped message.
  if (e instanceof HttpError && e.status === 401 && apiErrorCode(e) === null) {
    return t("auth.loginError");
  }
  return getErrorMessage(e);
}

/**
 * Dismissing the browser's passkey prompt rejects with a raw DOMException.
 * Without this, the login page shows the untranslated browser error text.
 */
function isPasskeyCancellation(e: unknown): boolean {
  return e instanceof DOMException && (e.name === "NotAllowedError" || e.name === "AbortError");
}

/**
 * Only same-origin paths are valid post-login destinations. A supplied
 * absolute URL (`?redirect=https://example.com`) would otherwise be treated as
 * a router path and land on a blank in-app route; `//host` is protocol-relative
 * and equally unsafe.
 */
function safeRedirect(value: unknown): string {
  if (typeof value !== "string" || !value.startsWith("/")) return "/";
  if (value.startsWith("//") || value.startsWith("/\\")) return "/";
  return value;
}

async function afterAuth(response: Token) {
  if (response.mfa_required && response.challenge_token) {
    challengeToken.value = response.challenge_token;
    return;
  }
  if (!authStore.isAuthenticated) {
    error.value = t("auth.loginError");
    return;
  }
  await router.push(safeRedirect(route.query.redirect));
}

const handleLogin = async () => {
  if (loading.value) return;
  // The single identifier field accepts a username or verified email.
  // Passkey login may leave it empty to use a discoverable credential, so
  // the input itself is not `required`; the password path enforces it here.
  if (!identifier.value.trim()) {
    error.value = t("auth.identifierRequired");
    return;
  }
  error.value = "";
  loading.value = true;
  try {
    await afterAuth(
      await authStore.login({ username: identifier.value, password: password.value })
    );
  } catch (e: unknown) {
    error.value = errorMessage(e);
  } finally {
    loading.value = false;
  }
};

const handlePasskeyLogin = async () => {
  if (passkeyLoading.value) return;
  error.value = "";
  passkeyLoading.value = true;
  try {
    const response = await authStore.loginWithPasskey(identifier.value || undefined);
    rememberPasskey(authStore.username);
    await afterAuth(response);
  } catch (e: unknown) {
    error.value = isPasskeyCancellation(e) ? t("auth.passkeySignInCancelled") : errorMessage(e);
  } finally {
    passkeyLoading.value = false;
  }
};

const handleMfa = async () => {
  if (!challengeToken.value || loading.value) return;
  error.value = "";
  loading.value = true;
  try {
    const response = recoveryMode.value
      ? await authStore.recoverMfa(challengeToken.value, recoveryCode.value)
      : await authStore.verifyMfa(challengeToken.value, mfaCode.value);
    await afterAuth(response);
  } catch (e: unknown) {
    const code = apiErrorCode(e);
    if (code === "mfa_challenge_expired" || code === "invalid_challenge") {
      // The 5-minute challenge token is dead. Returning to the identifier form
      // is the only way forward; leaving the MFA screen up would trap the user
      // in a form that can never succeed.
      challengeToken.value = null;
      mfaCode.value = "";
      recoveryCode.value = "";
      recoveryMode.value = false;
      error.value = t("auth.mfaChallengeExpired");
    } else {
      error.value = errorMessage(e);
    }
  } finally {
    loading.value = false;
  }
};

function startOAuth(providerId: string) {
  window.location.href = `/api/auth/oauth/${providerId}/begin`;
}

function resetChallenge() {
  challengeToken.value = null;
  mfaCode.value = "";
  recoveryCode.value = "";
  recoveryMode.value = false;
  error.value = "";
}

function toggleRecoveryMode() {
  recoveryMode.value = !recoveryMode.value;
  error.value = "";
}
</script>

<template>
  <AuthShell :title="authTitle" :description="authDescription">
    <Alert v-if="sessionNotice" class="mb-4" id="session-notice">
      <AlertDescription>{{ sessionNotice }}</AlertDescription>
    </Alert>
    <Alert v-if="error" variant="destructive" class="mb-4" id="login-error">
      <AlertDescription>{{ error }}</AlertDescription>
    </Alert>

    <form
      v-if="challengeToken"
      class="flex flex-col gap-4"
      :aria-busy="loading"
      :aria-describedby="error ? 'login-error' : undefined"
      @submit.prevent="handleMfa"
    >
      <div class="flex flex-col gap-1.5">
        <Label for="mfa-code" class="text-sm font-medium text-foreground/90">
          {{ recoveryMode ? t("auth.recoveryCode") : t("auth.mfaCode") }}
        </Label>
        <Input
          v-if="!recoveryMode"
          id="mfa-code"
          v-model="mfaCode"
          inputmode="numeric"
          autocomplete="one-time-code"
          pattern="[0-9]*"
          maxlength="6"
          required
          autofocus
          :disabled="loading"
          class="auth-input h-12 text-center font-mono text-lg tracking-[0.35em]"
        />
        <Input
          v-else
          id="mfa-code"
          v-model="recoveryCode"
          autocomplete="one-time-code"
          required
          autofocus
          :disabled="loading"
          class="auth-input h-11 font-mono"
        />
      </div>
      <Button type="submit" :disabled="loading" class="btn-press h-11 w-full" size="lg">
        <Loader2 v-if="loading" class="size-4 animate-spin" />
        {{ loading ? t("common.loading") : t("auth.verifyButton") }}
      </Button>
      <div class="flex flex-wrap items-center justify-between gap-2">
        <Button
          type="button"
          variant="link"
          size="sm"
          class="h-8 px-0 text-xs text-muted-foreground"
          @click="toggleRecoveryMode"
        >
          {{ recoveryMode ? t("auth.useTotpInstead") : t("auth.useRecoveryInstead") }}
        </Button>
        <Button
          type="button"
          variant="link"
          size="sm"
          class="h-8 gap-1.5 px-0 text-xs text-muted-foreground"
          @click="resetChallenge"
        >
          <ArrowLeft class="size-3.5" />
          {{ t("auth.backToLogin") }}
        </Button>
      </div>
    </form>

    <form
      v-else
      class="flex flex-col gap-4"
      :aria-describedby="error ? 'login-error' : undefined"
      :aria-busy="loading || passkeyLoading"
      @submit.prevent="handleLogin"
    >
      <div class="flex flex-col gap-1.5">
        <Label for="username" class="text-sm font-medium text-foreground/90">
          {{ t("auth.identifier") }}
        </Label>
        <Input
          id="username"
          v-model="identifier"
          type="text"
          autocomplete="username"
          autocapitalize="none"
          spellcheck="false"
          :disabled="loading || passkeyLoading"
          class="auth-input h-11"
        />
      </div>

      <Button
        v-if="passkeySupported"
        type="button"
        :variant="passkeyPreferred ? 'default' : 'outline'"
        :disabled="loading || passkeyLoading"
        class="btn-press h-11 w-full gap-2"
        size="lg"
        @click="handlePasskeyLogin"
      >
        <Loader2 v-if="passkeyLoading" class="size-4 animate-spin" />
        <Fingerprint v-else class="size-4" />
        {{ passkeyLoading ? t("common.loading") : t("auth.passkeyLogin") }}
      </Button>

      <div
        v-if="passkeySupported"
        class="flex items-center gap-3 py-1 text-xs text-muted-foreground"
      >
        <span class="h-px flex-1 bg-border" />
        {{ t("auth.orUsePassword") }}
        <span class="h-px flex-1 bg-border" />
      </div>

      <div class="flex flex-col gap-1.5">
        <Label for="password" class="text-sm font-medium text-foreground/90">
          {{ t("auth.password") }}
        </Label>
        <Input
          id="password"
          v-model="password"
          type="password"
          autocomplete="current-password"
          required
          :disabled="loading || passkeyLoading"
          class="auth-input h-11"
        />
      </div>

      <Button
        type="submit"
        :variant="passkeyPreferred ? 'outline' : 'default'"
        :disabled="loading || passkeyLoading"
        class="btn-press h-11 w-full gap-2"
        size="lg"
      >
        <Loader2 v-if="loading" class="size-4 animate-spin" />
        {{ loading ? t("common.loading") : t("auth.loginButton") }}
      </Button>
    </form>

    <!-- The recovery entry belongs on the identifier form only: an MFA screen is
         already a sign-in in progress, and leaving it must not drop the
         challenge. Where SMTP is missing the link becomes an instruction,
         because sending mail is exactly what this deployment cannot do. -->
    <div v-if="!challengeToken" class="mt-4 flex flex-col items-center gap-2">
      <RouterLink
        v-if="resetAvailable"
        id="forgot-password-link"
        :to="{ path: '/forgot-password', query: identifier ? { identifier } : {} }"
        class="text-xs text-muted-foreground underline-offset-4 transition-colors hover:text-foreground hover:underline"
      >
        {{ t("auth.forgotPassword") }}
      </RouterLink>
      <p
        v-else-if="resetAvailable === false"
        id="forgot-password-contact"
        class="text-center text-xs text-muted-foreground"
      >
        {{ t("auth.forgotContactAdmin") }}
      </p>
    </div>

    <div v-if="!challengeToken && providers.length" class="mt-6 flex flex-col gap-2">
      <div class="flex items-center gap-3 text-xs text-muted-foreground">
        <span class="h-px flex-1 bg-border" />
        {{ t("auth.orContinueWith") }}
        <span class="h-px flex-1 bg-border" />
      </div>
      <Button
        v-for="provider in providers"
        :key="provider.id"
        type="button"
        variant="outline"
        class="h-11 w-full gap-2"
        :disabled="loading || passkeyLoading"
        @click="startOAuth(provider.id)"
      >
        <KeyRound class="h-4 w-4" />
        {{ provider.name }}
      </Button>
    </div>
  </AuthShell>
</template>
