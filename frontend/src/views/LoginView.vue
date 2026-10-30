<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import {
  ArrowLeft,
  CircleAlert,
  Eye,
  EyeOff,
  Fingerprint,
  Info,
  KeyRound,
  Loader2,
} from "@lucide/vue";
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
const revealPassword = ref(false);
const capsLock = ref(false);
// Which field the operator still has to fill in. Field-level feedback points at
// the control that needs attention; the alert is reserved for the server's
// verdict on what was actually sent.
const fieldError = ref<"identifier" | "password" | "mfa" | null>(null);
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
  focusFirstField();
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

/**
 * Focusing the identifier on arrival saves a click for the operator who opened
 * the console to sign in. A touch-first device is excluded: there, the focus
 * would raise the software keyboard over a screen that is still being read.
 */
function focusFirstField() {
  if (typeof window.matchMedia !== "function") return;
  if (window.matchMedia("(pointer: coarse)").matches) return;
  focusField("username");
}

function focusField(id: "username" | "password" | "mfa-code") {
  document.getElementById(id)?.focus({ preventScroll: true });
}

/**
 * Editing retires the previous verdict, so a stale "invalid username or
 * password" never sits under text the operator has already corrected.
 */
function clearFeedback() {
  error.value = "";
  fieldError.value = null;
}

/**
 * The password field is the one place a wrong-looking password is usually not
 * wrong: report Caps Lock while the hardware key is down, and drop the note on
 * blur so it cannot outlive the state it describes.
 */
function trackCapsLock(event: KeyboardEvent) {
  if (typeof event.getModifierState !== "function") return;
  capsLock.value = event.getModifierState("CapsLock");
}

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
    fieldError.value = "identifier";
    focusField("username");
    return;
  }
  if (!password.value) {
    fieldError.value = "password";
    focusField("password");
    return;
  }
  error.value = "";
  fieldError.value = null;
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
  fieldError.value = null;
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
  const submitted = recoveryMode.value ? recoveryCode.value.trim() : mfaCode.value.trim();
  if (!submitted) {
    fieldError.value = "mfa";
    focusField("mfa-code");
    return;
  }
  error.value = "";
  fieldError.value = null;
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
      resetChallenge();
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
  fieldError.value = null;
}

function toggleRecoveryMode() {
  recoveryMode.value = !recoveryMode.value;
  error.value = "";
  fieldError.value = null;
}
</script>

<template>
  <AuthShell :title="authTitle" :description="authDescription">
    <Alert
      v-if="sessionNotice"
      id="session-notice"
      class="mb-4 animate-in fade-in slide-in-from-top-2 duration-300"
    >
      <AlertDescription>
        <span class="flex items-start gap-2">
          <Info class="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <span>{{ sessionNotice }}</span>
        </span>
      </AlertDescription>
    </Alert>
    <Alert
      v-if="error"
      id="login-error"
      variant="destructive"
      class="mb-4 animate-in fade-in slide-in-from-top-2 duration-300"
    >
      <AlertDescription>
        <span class="flex items-start gap-2">
          <CircleAlert class="mt-0.5 size-4 shrink-0" aria-hidden="true" />
          <span>{{ error }}</span>
        </span>
      </AlertDescription>
    </Alert>

    <form
      v-if="challengeToken"
      class="flex flex-col gap-4"
      novalidate
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
          :aria-invalid="fieldError === 'mfa'"
          :aria-describedby="fieldError === 'mfa' ? 'mfa-error' : undefined"
          class="auth-input h-12 text-center font-mono text-lg tracking-[0.35em]"
          @input="clearFeedback"
        />
        <Input
          v-else
          id="mfa-code"
          v-model="recoveryCode"
          autocomplete="one-time-code"
          required
          autofocus
          :disabled="loading"
          :aria-invalid="fieldError === 'mfa'"
          :aria-describedby="fieldError === 'mfa' ? 'mfa-error' : undefined"
          class="auth-input h-11 font-mono"
          @input="clearFeedback"
        />
        <p v-if="fieldError === 'mfa'" id="mfa-error" class="text-xs text-destructive">
          {{ recoveryMode ? t("auth.enterRecoveryCode") : t("auth.enterMfaCode") }}
        </p>
      </div>
      <Button type="submit" :disabled="loading" class="btn-press h-11 w-full" size="lg">
        <Loader2 v-if="loading" class="size-4 animate-spin" />
        {{ loading ? t("auth.verifying") : t("auth.verifyButton") }}
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
      novalidate
      :aria-describedby="error ? 'login-error' : undefined"
      :aria-busy="loading || passkeyLoading"
      @submit.prevent="handleLogin"
    >
      <!-- The passkey action leads, and the hairline that follows it opens the
           whole password path. Leaving the identifier above that hairline split
           one form across two "or" branches and made the field look like part
           of the passkey flow. -->
      <template v-if="passkeySupported">
        <Button
          type="button"
          :variant="passkeyPreferred ? 'default' : 'outline'"
          :disabled="loading || passkeyLoading"
          class="btn-press h-11 w-full gap-2"
          size="lg"
          @click="handlePasskeyLogin"
        >
          <Loader2 v-if="passkeyLoading" class="size-4 animate-spin" />
          <Fingerprint v-else class="size-4" />
          {{ passkeyLoading ? t("auth.waitingForPasskey") : t("auth.passkeyLogin") }}
        </Button>
        <div class="flex items-center gap-3 py-1 text-xs text-muted-foreground">
          <span class="h-px flex-1 bg-border" />
          {{ t("auth.orUsePassword") }}
          <span class="h-px flex-1 bg-border" />
        </div>
      </template>

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
          :aria-invalid="fieldError === 'identifier'"
          :aria-describedby="fieldError === 'identifier' ? 'identifier-error' : undefined"
          class="auth-input h-11"
          @input="clearFeedback"
        />
        <p
          v-if="fieldError === 'identifier'"
          id="identifier-error"
          class="text-xs text-destructive"
        >
          {{ t("auth.identifierRequired") }}
        </p>
      </div>

      <div class="flex flex-col gap-1.5">
        <!-- The recovery entry sits with the field it repairs, not at the foot
             of the column where it read as a footnote to the whole page. -->
        <div class="flex items-baseline justify-between gap-3">
          <Label for="password" class="text-sm font-medium text-foreground/90">
            {{ t("auth.password") }}
          </Label>
          <RouterLink
            v-if="resetAvailable"
            id="forgot-password-link"
            :to="{ path: '/forgot-password', query: identifier ? { identifier } : {} }"
            class="-my-1 rounded-sm py-1 text-xs text-muted-foreground underline-offset-4 transition-colors outline-none hover:text-foreground hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/50"
          >
            {{ t("auth.forgotPassword") }}
          </RouterLink>
        </div>
        <div class="relative">
          <Input
            id="password"
            v-model="password"
            :type="revealPassword ? 'text' : 'password'"
            autocomplete="current-password"
            required
            :disabled="loading || passkeyLoading"
            :aria-invalid="fieldError === 'password'"
            :aria-describedby="fieldError === 'password' ? 'password-error' : undefined"
            class="auth-input h-11 pr-12"
            @input="clearFeedback"
            @keydown="trackCapsLock"
            @keyup="trackCapsLock"
            @blur="capsLock = false"
          />
          <!-- Revealing the value is local to this screen and never widens what
               autocomplete or the browser remembers. The focus ring is inset so
               it cannot bleed past the field's own rounded edge. -->
          <button
            type="button"
            class="absolute inset-y-0 right-0 flex w-11 items-center justify-center rounded-r-md text-muted-foreground transition-colors outline-none hover:text-foreground focus-visible:bg-muted focus-visible:text-foreground focus-visible:ring-1 focus-visible:ring-ring/40 focus-visible:ring-inset"
            :aria-label="revealPassword ? t('auth.hidePassword') : t('auth.showPassword')"
            :aria-pressed="revealPassword"
            @click="revealPassword = !revealPassword"
          >
            <component :is="revealPassword ? EyeOff : Eye" class="size-4" aria-hidden="true" />
          </button>
        </div>
        <p v-if="fieldError === 'password'" id="password-error" class="text-xs text-destructive">
          {{ t("auth.enterPassword") }}
        </p>
        <p v-else-if="capsLock" id="password-caps" class="text-xs text-muted-foreground">
          {{ t("auth.passwordCapsLock") }}
        </p>
      </div>

      <Button
        type="submit"
        :variant="passkeyPreferred ? 'outline' : 'default'"
        :disabled="loading || passkeyLoading"
        class="btn-press h-11 w-full gap-2"
        size="lg"
      >
        <Loader2 v-if="loading" class="size-4 animate-spin" />
        {{ loading ? t("auth.signingIn") : t("auth.loginButton") }}
      </Button>

      <!-- Where SMTP is missing the recovery entry becomes an instruction,
           because sending mail is exactly what this deployment cannot do. -->
      <p
        v-if="resetAvailable === false"
        id="forgot-password-contact"
        class="text-center text-xs text-muted-foreground"
      >
        {{ t("auth.forgotContactAdmin") }}
      </p>
    </form>

    <div v-if="!challengeToken && providers.length" class="mt-8 flex flex-col gap-2">
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
