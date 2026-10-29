<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import {
  Check,
  Copy,
  Download,
  Fingerprint,
  KeyRound,
  Link2,
  Loader2,
  LockKeyhole,
  Pencil,
  ShieldCheck,
  Trash2,
  Unlink,
  X,
} from "@lucide/vue";
import { useClipboard } from "@vueuse/core";
import { toast } from "vue-sonner";
import AccountShell from "@/components/layout/AccountShell.vue";
import LoadingState from "@/components/common/LoadingState.vue";
import PanelSection from "@/components/common/PanelSection.vue";
import StatusBadge from "@/components/common/StatusBadge.vue";
import ConfirmDialog from "@/components/common/ConfirmDialog.vue";
import QrCode from "@/components/common/QrCode.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useChangePassword } from "@/composables/useChangePassword";
import { authApi } from "@/services/api/auth";
import { meApi, type MeProfile } from "@/services/api/me";
import { securityApi } from "@/services/api/security";
import { usePasskeyAccounts } from "@/composables/usePasskeyAccounts";
import { useAuthStore } from "@/stores/auth";
import { getErrorMessage } from "@/utils/error";
import { createPasskey } from "@/utils/webauthn";
import type { OAuthIdentity, OAuthProvider, Passkey, TwoFactorStatus } from "@/types/schemas";

const { t, locale } = useI18n();
const route = useRoute();
const router = useRouter();
const authStore = useAuthStore();
const { copy } = useClipboard({ legacy: true, copiedDuring: 1500 });

const passkeys = ref<Passkey[]>([]);
const twoFactor = ref<TwoFactorStatus | null>(null);
const identities = ref<OAuthIdentity[]>([]);
const providers = ref<OAuthProvider[]>([]);
const profile = ref<MeProfile | null>(null);
const loading = ref(true);
const loadError = ref("");

const passkeyName = ref("");
const addingPasskey = ref(false);
const otpauthUri = ref("");
const totpSecret = ref("");
const totpCode = ref("");
const recoveryCodes = ref<string[]>([]);
const disableCode = ref("");
const regenerateCode = ref("");
const pendingAction = ref(false);

const removePasskeyTarget = ref<Passkey | null>(null);
const showRemovePasskey = ref(false);
// Inline rename: one row at a time, so the list never grows a second input.
const renamingPasskeyId = ref<number | null>(null);
const renamePasskeyName = ref("");
const confirmDisableTwoFactor = ref(false);
const unlinkIdentityTarget = ref<OAuthIdentity | null>(null);
const showUnlinkIdentity = ref(false);
const confirmSignOut = ref(false);

const passkeySupported = computed(
  () => typeof window !== "undefined" && !!window.PublicKeyCredential && window.isSecureContext
);
const { remember: rememberPasskey, forget: forgetPasskey } = usePasskeyAccounts();
const showOnboardingHint = ["setup", "invite", "oauth"].includes(String(route.query.from));
// First-run onboarding is about the same thing as an invite: add a passkey so
// daily sign-in does not depend on the password typed a minute ago.
const onboardingHint = computed(() =>
  route.query.from === "setup"
    ? t("accountSecurity.onboardingHintSetup")
    : t("accountSecurity.onboardingHint")
);

// Passwordless accounts (passkey/OIDC only) can still gain a password: the
// backend takes a null current password and requires a recent step-up instead.
const hasPassword = computed(() => profile.value?.has_password ?? true);
const {
  currentPassword,
  newPassword,
  confirmPassword,
  error: passwordError,
  loading: savingPassword,
  passwordHint,
  submit: submitPasswordForm,
} = useChangePassword({ requireCurrentPassword: () => hasPassword.value });

async function fetchSecurityState() {
  const [passkeyList, status, identityList, providerList, me] = await Promise.all([
    authApi.listPasskeys(),
    securityApi.getTwoFactor(),
    securityApi.listIdentities(),
    authApi.listOAuthProviders(),
    meApi.getProfile(),
  ]);
  passkeys.value = passkeyList;
  twoFactor.value = status;
  identities.value = identityList.identities;
  providers.value = providerList.providers;
  profile.value = me;
}

async function load() {
  loading.value = true;
  loadError.value = "";
  try {
    await fetchSecurityState();
  } catch (e) {
    loadError.value = getErrorMessage(e) || t("errors.fetchFailed");
  } finally {
    loading.value = false;
  }
}

/**
 * Re-read the panels after a mutation without blanking the page behind a
 * spinner: the action already reported success, and the operator is reading
 * the card that just changed.
 */
async function refresh() {
  try {
    await fetchSecurityState();
  } catch (e) {
    console.error("Failed to refresh security state:", e);
    toast.error(t("errors.fetchFailedBackground"));
  }
}

onMounted(async () => {
  await load();
  if (loadError.value) return;

  // The OAuth link flow lands here with ?linked=1; confirm it and drop the
  // flag so a refresh does not replay the toast.
  if (route.query.linked === "1") {
    toast.success(t("accountSecurity.identityLinked"));
    await scrollToSection("linked-accounts");
    const query = { ...route.query };
    delete query.linked;
    await router.replace({ query });
  }
});

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

async function scrollToSection(id: string) {
  await nextTick();
  document.getElementById(id)?.scrollIntoView({
    block: "center",
    behavior: prefersReducedMotion() ? "auto" : "smooth",
  });
}

/** Onboarding CTA: put the cursor in the passkey form rather than only scrolling. */
async function focusPasskeyForm() {
  await scrollToSection("passkeys");
  document.getElementById("passkey-name")?.focus({ preventScroll: true });
}

async function run(action: () => Promise<void>) {
  // Destructive actions share one in-flight guard: a double click would
  // otherwise issue the call twice (rotating the recovery codes twice
  // invalidates the codes the user is in the middle of copying).
  if (pendingAction.value) return;
  pendingAction.value = true;
  try {
    await action();
  } catch (e) {
    toast.error(getErrorMessage(e) || t("common.error"));
  } finally {
    pendingAction.value = false;
  }
}

async function addPasskey() {
  if (!passkeySupported.value) {
    toast.error(t("accountSecurity.passkeysUnsupported"));
    return;
  }
  await run(async () => {
    addingPasskey.value = true;
    try {
      const begin = await authApi.passkeyRegisterBegin();
      const credential = await createPasskey(begin.options);
      await authApi.passkeyRegisterFinish(
        begin.challenge_token,
        credential,
        passkeyName.value.trim() || t("accountSecurity.defaultPasskeyName")
      );
      passkeyName.value = "";
      rememberPasskey(authStore.username);
      toast.success(t("accountSecurity.passkeyAdded"));
      await refresh();
    } finally {
      addingPasskey.value = false;
    }
  });
}

async function removePasskey() {
  const passkey = removePasskeyTarget.value;
  if (!passkey) return;
  await run(async () => {
    await authApi.deletePasskey(passkey.id);
    showRemovePasskey.value = false;
    toast.success(t("accountSecurity.passkeyRemoved"));
    await refresh();
    if (passkeys.value.length === 0) forgetPasskey(authStore.username);
  });
}

function startPasskeyRename(passkey: Passkey) {
  renamingPasskeyId.value = passkey.id;
  renamePasskeyName.value = passkey.name;
}

function cancelPasskeyRename() {
  renamingPasskeyId.value = null;
  renamePasskeyName.value = "";
}

async function savePasskeyName() {
  const id = renamingPasskeyId.value;
  const name = renamePasskeyName.value.trim();
  if (id === null) return;
  if (!name) {
    toast.error(t("accountSecurity.passkeyNameRequired"));
    return;
  }
  await run(async () => {
    const updated = await authApi.renamePasskey(id, name);
    passkeys.value = passkeys.value.map((passkey) =>
      passkey.id === updated.id ? updated : passkey
    );
    cancelPasskeyRename();
    toast.success(t("accountSecurity.passkeyRenamed"));
  });
}

async function beginTwoFactor() {
  await run(async () => {
    const response = await securityApi.beginTwoFactor();
    totpSecret.value = response.secret;
    otpauthUri.value = response.otpauth_uri;
  });
}

function cancelTwoFactorSetup() {
  totpSecret.value = "";
  otpauthUri.value = "";
  totpCode.value = "";
}

async function confirmTwoFactor() {
  await run(async () => {
    const response = await securityApi.confirmTwoFactor(totpCode.value);
    recoveryCodes.value = response.recovery_codes;
    totpCode.value = "";
    cancelTwoFactorSetup();
    toast.success(t("accountSecurity.twoFactorEnabled"));
    await refresh();
  });
}

async function disableTwoFactor() {
  if (!disableCode.value) {
    toast.error(t("accountSecurity.codeRequired"));
    return;
  }
  await run(async () => {
    await securityApi.disableTwoFactor(disableCode.value);
    disableCode.value = "";
    recoveryCodes.value = [];
    confirmDisableTwoFactor.value = false;
    toast.success(t("accountSecurity.twoFactorDisabled"));
    await refresh();
  });
}

async function regenerateRecoveryCodes() {
  if (!regenerateCode.value) {
    toast.error(t("accountSecurity.codeRequired"));
    return;
  }
  await run(async () => {
    // A dedicated field: the code is not the one bound to the destructive
    // "disable" button, so neither action can replay the other's input.
    const response = await securityApi.regenerateRecoveryCodes(regenerateCode.value);
    recoveryCodes.value = response.recovery_codes;
    regenerateCode.value = "";
    toast.success(t("accountSecurity.recoveryCodesRegenerated"));
    await refresh();
  });
}

async function changePassword() {
  if (!(await submitPasswordForm())) return;
  toast.success(t("profile.passwordChanged"));
  // The backend revokes every session when the password changes.
  authStore.clearLocalSession();
  await router.push("/login");
}

async function linkProvider(providerId: string) {
  await run(async () => {
    const response = await securityApi.beginLink(providerId);
    window.location.href = response.authorization_url;
  });
}

async function unlinkProvider() {
  const identity = unlinkIdentityTarget.value;
  if (!identity) return;
  await run(async () => {
    await securityApi.unlinkIdentity(identity.id);
    showUnlinkIdentity.value = false;
    toast.success(t("accountSecurity.identityUnlinked"));
    await refresh();
  });
}

async function signOutCurrent() {
  confirmSignOut.value = false;
  await authStore.logout();
  await router.push("/login");
}

function copyText(value: string) {
  void copy(value);
  toast.success(t("common.copied"));
}

/**
 * Save the codes as a file. The copy button loses them on the next clipboard
 * write and a screenshot lands in the camera roll; a text file is what people
 * actually keep until they have moved them into a password manager.
 */
function downloadRecoveryCodes() {
  const header = [
    "LLM Proxy recovery codes",
    `Account: ${authStore.username ?? ""}`,
    `Generated: ${new Date().toISOString()}`,
    "Each code works once.",
    "",
  ].join("\n");
  const blob = new Blob([`${header}${recoveryCodes.value.join("\n")}\n`], {
    type: "text/plain;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "llm-proxy-recovery-codes.txt";
  link.click();
  URL.revokeObjectURL(url);
  toast.success(t("accountSecurity.recoveryCodesDownloaded"));
}

/** Drop the codes from the DOM once the user confirms they are stored. */
function dismissRecoveryCodes() {
  recoveryCodes.value = [];
}

function formatDate(value: string): string {
  return new Date(value).toLocaleDateString(locale.value, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}
</script>

<template>
  <AccountShell>
    <Alert v-if="showOnboardingHint" class="mb-6">
      <ShieldCheck class="size-4" />
      <AlertDescription class="flex flex-wrap items-center justify-between gap-3">
        <span>{{ onboardingHint }}</span>
        <span class="flex gap-2">
          <Button size="sm" variant="outline" @click="router.push('/account')">
            {{ t("account.viewProfile") }}
          </Button>
          <Button size="sm" :disabled="!passkeySupported" @click="focusPasskeyForm">
            {{ t("accountSecurity.addPasskey") }}
          </Button>
        </span>
      </AlertDescription>
    </Alert>

    <LoadingState v-if="loading" />

    <Alert v-else-if="loadError" variant="destructive">
      <AlertDescription class="flex items-center justify-between gap-4">
        <span>{{ loadError }}</span>
        <Button variant="outline" size="sm" @click="load">{{ t("common.retry") }}</Button>
      </AlertDescription>
    </Alert>

    <!-- Credential panels in strength order: primary factor, fallback,
           second factor, alternative providers, then session control. -->
    <div v-else class="space-y-6">
      <PanelSection
        id="passkeys"
        :title="t('accountSecurity.passkeys')"
        :description="t('accountSecurity.passkeysDescription')"
        :icon="Fingerprint"
      >
        <template #badge>
          <Badge variant="secondary">{{ passkeys.length }}</Badge>
        </template>

        <div class="space-y-2">
          <div
            v-for="passkey in passkeys"
            :key="passkey.id"
            class="flex items-center justify-between gap-3 rounded-lg border border-border/50 px-3 py-2.5"
          >
            <div class="min-w-0 flex-1">
              <form
                v-if="renamingPasskeyId === passkey.id"
                class="flex items-center gap-2"
                @submit.prevent="savePasskeyName"
              >
                <Label :for="`passkey-rename-${passkey.id}`" class="sr-only">
                  {{ t("accountSecurity.passkeyName") }}
                </Label>
                <Input
                  :id="`passkey-rename-${passkey.id}`"
                  v-model="renamePasskeyName"
                  class="h-8"
                  maxlength="100"
                  :disabled="pendingAction"
                />
                <Button
                  type="submit"
                  variant="ghost"
                  size="icon"
                  :aria-label="t('common.save')"
                  :disabled="pendingAction"
                >
                  <Check class="size-4" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  :aria-label="t('common.cancel')"
                  :disabled="pendingAction"
                  @click="cancelPasskeyRename"
                >
                  <X class="size-4" />
                </Button>
              </form>
              <template v-else>
                <p class="truncate text-sm font-medium">{{ passkey.name }}</p>
                <p class="text-xs text-muted-foreground">
                  {{
                    t("accountSecurity.passkeyCreated", {
                      date: formatDate(passkey.created_at),
                    })
                  }}
                  <template v-if="passkey.last_used_at">
                    ·
                    {{
                      t("accountSecurity.passkeyLastUsed", {
                        date: formatDate(passkey.last_used_at),
                      })
                    }}
                  </template>
                </p>
              </template>
            </div>
            <div v-if="renamingPasskeyId !== passkey.id" class="flex shrink-0 items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                :aria-label="t('accountSecurity.renamePasskey')"
                :disabled="pendingAction"
                @click="startPasskeyRename(passkey)"
              >
                <Pencil class="size-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                :aria-label="t('accountSecurity.deletePasskey')"
                :disabled="pendingAction"
                @click="
                  removePasskeyTarget = passkey;
                  showRemovePasskey = true;
                "
              >
                <Trash2 class="size-4" />
              </Button>
            </div>
          </div>
          <p
            v-if="!passkeys.length"
            class="rounded-lg border border-border/50 bg-muted/15 px-3 py-6 text-center text-sm text-muted-foreground"
          >
            {{ t("accountSecurity.noPasskeys") }}
          </p>
        </div>

        <form class="mt-4 flex flex-col gap-2 sm:flex-row" @submit.prevent="addPasskey">
          <Label for="passkey-name" class="sr-only">{{ t("accountSecurity.passkeyName") }}</Label>
          <Input
            id="passkey-name"
            v-model="passkeyName"
            :placeholder="t('accountSecurity.passkeyName')"
            :disabled="!passkeySupported || addingPasskey"
          />
          <Button
            type="submit"
            class="shrink-0 gap-2"
            :disabled="!passkeySupported || addingPasskey"
          >
            <Loader2 v-if="addingPasskey" class="size-4 animate-spin" />
            <Fingerprint v-else class="size-4" />
            {{ t("accountSecurity.addPasskey") }}
          </Button>
        </form>
        <p v-if="!passkeySupported" class="mt-2 text-xs text-muted-foreground">
          {{ t("accountSecurity.passkeysUnsupported") }}
        </p>
      </PanelSection>

      <PanelSection
        id="password"
        :title="t('accountSecurity.password')"
        :description="
          hasPassword
            ? t('accountSecurity.passwordDescription')
            : t('accountSecurity.setPasswordDescription')
        "
        :icon="LockKeyhole"
      >
        <template #badge>
          <Badge variant="secondary">
            {{
              hasPassword ? t("accountSecurity.passwordSet") : t("accountSecurity.passwordNotSet")
            }}
          </Badge>
        </template>

        <form class="flex flex-col gap-3" @submit.prevent="changePassword">
          <div class="grid gap-3 sm:grid-cols-2">
            <div v-if="hasPassword" class="flex flex-col gap-2 sm:col-span-2">
              <Label for="security-current-password">{{ t("profile.currentPassword") }}</Label>
              <Input
                id="security-current-password"
                v-model="currentPassword"
                type="password"
                autocomplete="current-password"
              />
            </div>
            <div class="flex flex-col gap-2">
              <Label for="security-new-password">{{ t("profile.newPassword") }}</Label>
              <Input
                id="security-new-password"
                v-model="newPassword"
                type="password"
                autocomplete="new-password"
              />
            </div>
            <div class="flex flex-col gap-2">
              <Label for="security-confirm-password">{{ t("profile.confirmPassword") }}</Label>
              <Input
                id="security-confirm-password"
                v-model="confirmPassword"
                type="password"
                autocomplete="new-password"
              />
            </div>
          </div>
          <p class="text-xs text-muted-foreground">{{ passwordHint }}</p>
          <p v-if="passwordError" class="text-xs text-destructive">{{ passwordError }}</p>
          <p v-else-if="!hasPassword" class="text-xs text-muted-foreground">
            {{ t("accountSecurity.stepUpRequired") }}
          </p>
          <div class="flex justify-end">
            <Button type="submit" :disabled="savingPassword || !newPassword || !confirmPassword">
              <Loader2 v-if="savingPassword" class="mr-2 size-4 animate-spin" />
              {{ hasPassword ? t("profile.changePassword") : t("accountSecurity.setPassword") }}
            </Button>
          </div>
        </form>
      </PanelSection>

      <PanelSection
        id="two-factor"
        :title="t('accountSecurity.twoFactor')"
        :description="t('accountSecurity.twoFactorDescription')"
        :icon="KeyRound"
      >
        <template #badge>
          <StatusBadge variant="status" :status="twoFactor?.enabled ? 'success' : 'unknown'">
            {{ twoFactor?.enabled ? t("accountSecurity.enabled") : t("accountSecurity.disabled") }}
          </StatusBadge>
        </template>

        <div class="space-y-3">
          <!-- Instance policy: an admin blocked by ``mfa_enrollment_required``
               is redirected to this page, so the panel has to say why and what
               unblocks the console. -->
          <Alert v-if="authStore.isAdmin && twoFactor?.required_for_admins && !twoFactor.enabled">
            <ShieldCheck class="size-4" />
            <AlertDescription>{{ t("accountSecurity.twoFactorRequired") }}</AlertDescription>
          </Alert>
          <p v-if="twoFactor?.enabled" class="text-sm">
            {{
              t("accountSecurity.recoveryCodesRemaining", {
                count: twoFactor.recovery_codes_remaining,
              })
            }}
          </p>

          <template v-if="!twoFactor?.enabled">
            <Button
              v-if="!totpSecret"
              variant="outline"
              class="gap-2"
              :disabled="pendingAction"
              @click="beginTwoFactor"
            >
              <KeyRound class="size-4" />
              {{ t("accountSecurity.enableTwoFactor") }}
            </Button>
            <div v-else class="space-y-4">
              <ol class="list-decimal space-y-3 pl-4 text-sm">
                <li>
                  <p class="text-xs text-muted-foreground">
                    {{ t("accountSecurity.totpSetupStepOne") }}
                  </p>
                  <div class="mt-3 flex flex-wrap items-start gap-4">
                    <!-- Scanning is the primary path; typing the secret is the
                         fallback for authenticators that cannot read a code
                         from the screen. The symbol needs a light quiet zone
                         in both themes. -->
                    <div class="rounded-lg border border-border/50 bg-white p-2">
                      <QrCode
                        :value="otpauthUri"
                        :size="152"
                        :label="t('accountSecurity.qrCodeLabel')"
                      />
                    </div>
                    <div class="min-w-0 flex-1 space-y-2">
                      <p class="text-xs text-muted-foreground">
                        {{ t("accountSecurity.totpScanHint") }}
                      </p>
                      <p class="text-xs font-medium">
                        {{ t("accountSecurity.totpSecretLabel") }}
                      </p>
                      <div class="flex items-center gap-2">
                        <code
                          class="min-w-0 flex-1 break-all rounded-md bg-muted px-3 py-2 font-mono text-xs"
                        >
                          {{ totpSecret }}
                        </code>
                        <Button
                          type="button"
                          variant="outline"
                          size="icon"
                          :aria-label="t('accountSecurity.copySecret')"
                          @click="copyText(totpSecret)"
                        >
                          <Copy class="size-4" />
                        </Button>
                      </div>
                      <details class="mt-2">
                        <summary class="cursor-pointer text-xs text-muted-foreground">
                          {{ t("accountSecurity.manualSetup") }}
                        </summary>
                        <code
                          class="mt-2 block break-all rounded-md bg-muted px-3 py-2 font-mono text-[11px]"
                        >
                          {{ otpauthUri }}
                        </code>
                      </details>
                    </div>
                  </div>
                </li>
                <li>
                  <Label for="totp-confirm" class="text-xs text-muted-foreground">
                    {{ t("accountSecurity.totpSetupStepTwo") }}
                  </Label>
                  <form class="mt-2 flex gap-2" @submit.prevent="confirmTwoFactor">
                    <Input
                      id="totp-confirm"
                      v-model="totpCode"
                      inputmode="numeric"
                      autocomplete="one-time-code"
                      maxlength="6"
                      :aria-label="t('accountSecurity.code')"
                    />
                    <Button
                      type="submit"
                      class="shrink-0"
                      :disabled="pendingAction || totpCode.length < 6"
                    >
                      {{ t("accountSecurity.confirm") }}
                    </Button>
                  </form>
                </li>
              </ol>
              <Button variant="ghost" size="sm" @click="cancelTwoFactorSetup">
                {{ t("common.cancel") }}
              </Button>
            </div>
          </template>

          <template v-else>
            <div class="flex flex-col gap-2 rounded-lg border border-border/50 p-3">
              <Label for="regenerate-code" class="text-xs text-muted-foreground">
                {{ t("accountSecurity.regenerateDescription") }}
              </Label>
              <div class="flex gap-2">
                <Input
                  id="regenerate-code"
                  v-model="regenerateCode"
                  inputmode="numeric"
                  autocomplete="one-time-code"
                  maxlength="6"
                  :aria-label="t('accountSecurity.code')"
                  :placeholder="t('accountSecurity.code')"
                />
                <Button
                  variant="outline"
                  class="shrink-0"
                  :disabled="pendingAction"
                  @click="regenerateRecoveryCodes"
                >
                  {{ t("accountSecurity.regenerate") }}
                </Button>
              </div>
            </div>
            <div
              v-if="twoFactor.required_for_admins"
              class="rounded-lg border border-border/50 bg-muted/15 p-3 text-xs text-muted-foreground"
            >
              {{ t("accountSecurity.twoFactorRequiredManaged") }}
            </div>
            <div v-else class="flex flex-col gap-2 rounded-lg border border-destructive/30 p-3">
              <Label for="disable-code" class="text-xs text-muted-foreground">
                {{ t("accountSecurity.disableDescription") }}
              </Label>
              <div class="flex gap-2">
                <Input
                  id="disable-code"
                  v-model="disableCode"
                  inputmode="numeric"
                  autocomplete="one-time-code"
                  maxlength="6"
                  :aria-label="t('accountSecurity.code')"
                  :placeholder="t('accountSecurity.code')"
                />
                <Button
                  variant="destructive"
                  class="shrink-0"
                  :disabled="pendingAction"
                  @click="confirmDisableTwoFactor = true"
                >
                  {{ t("accountSecurity.disableTwoFactor") }}
                </Button>
              </div>
            </div>
          </template>

          <div
            v-if="recoveryCodes.length"
            class="rounded-lg border border-status-warning/40 bg-status-warning/5 p-3"
          >
            <div class="flex items-start justify-between gap-3">
              <div>
                <p class="text-xs font-medium">{{ t("accountSecurity.recoveryCodes") }}</p>
                <p class="mb-2 text-xs text-muted-foreground">
                  {{ t("accountSecurity.recoveryCodesDescription") }}
                </p>
              </div>
              <div class="flex shrink-0 items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  :aria-label="t('accountSecurity.copyRecoveryCodes')"
                  @click="copyText(recoveryCodes.join('\n'))"
                >
                  <Copy class="size-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  :aria-label="t('accountSecurity.downloadRecoveryCodes')"
                  @click="downloadRecoveryCodes"
                >
                  <Download class="size-4" />
                </Button>
              </div>
            </div>
            <div class="grid grid-cols-2 gap-1 font-mono text-xs">
              <span v-for="code in recoveryCodes" :key="code">{{ code }}</span>
            </div>
            <div class="mt-3 flex flex-wrap items-center justify-between gap-2">
              <p class="text-xs text-muted-foreground">
                {{ t("accountSecurity.recoveryCodesShowOnce") }}
              </p>
              <Button variant="outline" size="sm" @click="dismissRecoveryCodes">
                {{ t("accountSecurity.recoveryCodesSaved") }}
              </Button>
            </div>
          </div>
        </div>
      </PanelSection>

      <PanelSection
        id="linked-accounts"
        :title="t('accountSecurity.linkedAccounts')"
        :description="t('accountSecurity.linkedAccountsDescription')"
        :icon="Link2"
      >
        <div class="space-y-2">
          <div
            v-for="identity in identities"
            :key="identity.id"
            class="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/50 px-3 py-2.5 text-sm"
          >
            <span class="min-w-0 truncate"> {{ identity.provider }} · {{ identity.email }} </span>
            <Button
              variant="ghost"
              size="sm"
              class="gap-1 text-destructive hover:text-destructive"
              :disabled="pendingAction"
              @click="
                unlinkIdentityTarget = identity;
                showUnlinkIdentity = true;
              "
            >
              <Unlink class="size-3.5" />
              {{ t("accountSecurity.unlink") }}
            </Button>
          </div>
          <p
            v-if="!identities.length"
            class="rounded-lg border border-border/50 bg-muted/15 px-3 py-6 text-center text-sm text-muted-foreground"
          >
            {{ t("accountSecurity.noLinkedAccounts") }}
          </p>
        </div>

        <div v-if="providers.length" class="mt-4 flex flex-wrap gap-2">
          <Button
            v-for="provider in providers"
            :key="provider.id"
            variant="outline"
            size="sm"
            :disabled="pendingAction"
            @click="linkProvider(provider.id)"
          >
            {{ t("accountSecurity.link") }} {{ provider.name }}
          </Button>
        </div>
        <p v-else class="mt-2 text-xs text-muted-foreground">
          {{ t("accountSecurity.noProvidersAvailable") }}
        </p>
      </PanelSection>

      <PanelSection
        id="session"
        :title="t('accountSecurity.session')"
        :description="t('accountSecurity.sessionDescription')"
        :icon="ShieldCheck"
      >
        <Button variant="outline" @click="confirmSignOut = true">
          {{ t("accountSecurity.signOutCurrent") }}
        </Button>
      </PanelSection>

      <p v-if="authStore.isAdmin" class="text-xs text-muted-foreground">
        {{ t("accountSecurity.instanceSignInHint") }}
        <RouterLink
          to="/config/console-auth"
          class="text-foreground underline underline-offset-4 hover:no-underline"
        >
          {{ t("accountSecurity.manageInstanceSignIn") }}
        </RouterLink>
      </p>
    </div>

    <ConfirmDialog
      :open="showRemovePasskey"
      :title="t('accountSecurity.removePasskeyTitle')"
      :description="
        t('accountSecurity.removePasskeyDescription', { name: removePasskeyTarget?.name ?? '' })
      "
      :confirm-text="t('accountSecurity.deletePasskey')"
      :loading="pendingAction"
      @update:open="showRemovePasskey = $event"
      @confirm="removePasskey"
      @cancel="showRemovePasskey = false"
    />

    <ConfirmDialog
      :open="confirmDisableTwoFactor"
      :title="t('accountSecurity.disableTwoFactorTitle')"
      :description="t('accountSecurity.disableTwoFactorDescription')"
      :confirm-text="t('accountSecurity.disableTwoFactor')"
      :loading="pendingAction"
      @update:open="confirmDisableTwoFactor = $event"
      @confirm="disableTwoFactor"
      @cancel="confirmDisableTwoFactor = false"
    />

    <ConfirmDialog
      :open="showUnlinkIdentity"
      :title="t('accountSecurity.unlinkTitle')"
      :description="
        t('accountSecurity.unlinkDescription', { provider: unlinkIdentityTarget?.provider ?? '' })
      "
      :confirm-text="t('accountSecurity.unlink')"
      :loading="pendingAction"
      @update:open="showUnlinkIdentity = $event"
      @confirm="unlinkProvider"
      @cancel="showUnlinkIdentity = false"
    />

    <ConfirmDialog
      :open="confirmSignOut"
      :title="t('account.signOutTitle')"
      :description="t('account.signOutDescription')"
      :confirm-text="t('accountSecurity.signOutCurrent')"
      @update:open="confirmSignOut = $event"
      @confirm="signOutCurrent"
      @cancel="confirmSignOut = false"
    />
  </AccountShell>
</template>
