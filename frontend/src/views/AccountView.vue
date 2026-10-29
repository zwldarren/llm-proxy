<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import {
  CalendarDays,
  Clock,
  IdCard,
  KeyRound,
  Loader2,
  Mail,
  ShieldCheck,
  Wallet,
} from "@lucide/vue";
import { toast } from "vue-sonner";
import AccountShell from "@/components/layout/AccountShell.vue";
import LoadingState from "@/components/common/LoadingState.vue";
import PanelSection from "@/components/common/PanelSection.vue";
import StatusBadge from "@/components/common/StatusBadge.vue";
import AccountBudgetBanner from "@/components/common/AccountBudgetBanner.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { meApi, type MeBudget, type MeProfile } from "@/services/api/me";
import { HttpError } from "@/services/http";
import { useAuthStore } from "@/stores/auth";
import { getErrorMessage } from "@/utils/error";

const { t, locale } = useI18n();
const authStore = useAuthStore();

const profile = ref<MeProfile | null>(null);
const budget = ref<MeBudget | null>(null);
const loading = ref(true);
const loadError = ref("");

// Passwordless (passkey/OIDC) accounts prove a recent step-up instead of a
// current password, both for renaming and for the email verification call.
const hasPassword = computed(() => profile.value?.has_password ?? true);

const showUsernameDialog = ref(false);
const usernameNew = ref("");
const usernamePassword = ref("");
const usernameError = ref("");
const savingUsername = ref(false);

const showEmailDialog = ref(false);
const emailDraft = ref("");
const sendingEmail = ref(false);

/**
 * Fetch the profile without toggling the page-level loading state. Mutations
 * use this so the page refreshes in place instead of flashing a spinner over
 * content the operator is already reading.
 */
async function fetchProfile() {
  profile.value = await meApi.getProfile();
  try {
    budget.value = await meApi.getBudget();
  } catch {
    // Budget is supplementary: a failing spend query must not hide identity.
    budget.value = null;
  }
}

async function load() {
  loading.value = true;
  loadError.value = "";
  try {
    await fetchProfile();
  } catch (e) {
    loadError.value = getErrorMessage(e) || t("errors.fetchFailed");
  } finally {
    loading.value = false;
  }
}

async function refresh() {
  try {
    await fetchProfile();
  } catch (e) {
    console.error("Failed to refresh profile:", e);
    toast.error(t("errors.fetchFailedBackground"));
  }
}

onMounted(load);

function openUsernameDialog() {
  usernameNew.value = "";
  usernamePassword.value = "";
  usernameError.value = "";
  showUsernameDialog.value = true;
}

async function changeUsername() {
  usernameError.value = "";
  const newName = usernameNew.value.trim();

  if (!newName || newName.length > 64 || !/^[a-zA-Z0-9_-]+$/.test(newName)) {
    usernameError.value = t("profile.usernameInvalid");
    return;
  }
  if (newName.toLowerCase() === (profile.value?.username ?? "").toLowerCase()) {
    usernameError.value = t("profile.usernameUnchanged");
    return;
  }
  if (hasPassword.value && !usernamePassword.value) {
    usernameError.value = t("profile.passwordRequired");
    return;
  }

  savingUsername.value = true;
  try {
    const res = await meApi.changeUsername(
      hasPassword.value ? usernamePassword.value : null,
      newName
    );
    // The backend rotated the HttpOnly cookie; refresh the identity the UI
    // renders so the sidebar does not keep showing the old username.
    await authStore.refreshSession();
    showUsernameDialog.value = false;
    toast.success(t("profile.usernameChanged", { username: res.username }));
    await refresh();
  } catch (e) {
    if (e instanceof HttpError && e.status === 409) {
      usernameError.value = t("profile.usernameTaken");
    } else {
      usernameError.value = getErrorMessage(e) || t("errors.saveFailed");
    }
  } finally {
    savingUsername.value = false;
  }
}

function openEmailDialog() {
  emailDraft.value = profile.value?.email ?? "";
  showEmailDialog.value = true;
}

async function sendEmailVerification() {
  const address = emailDraft.value.trim();
  if (!address || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(address)) {
    toast.error(t("accountSecurity.invalidEmail"));
    return;
  }
  sendingEmail.value = true;
  try {
    await meApi.requestEmailVerification(address);
    showEmailDialog.value = false;
    emailDraft.value = "";
    toast.success(t("accountSecurity.emailSent"));
  } catch (e) {
    toast.error(getErrorMessage(e) || t("errors.saveFailed"));
  } finally {
    sendingEmail.value = false;
  }
}

const initials = () =>
  (profile.value?.username ?? authStore.username ?? "U").slice(0, 2).toUpperCase();

function emailActionLabel(): string {
  if (!profile.value?.email) return t("common.add");
  return profile.value.email_verified ? t("account.change") : t("account.verify");
}

function formatDate(value: string): string {
  return new Date(value).toLocaleDateString(locale.value, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatDateTime(value: string): string {
  return new Date(value).toLocaleString(locale.value, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function authMethodLabel(method: string): string {
  const key = `account.authMethods.${method}`;
  const translated = t(key);
  return translated === key ? method : translated;
}
</script>

<template>
  <AccountShell>
    <LoadingState v-if="loading" />

    <Alert v-else-if="loadError" variant="destructive">
      <AlertDescription class="flex items-center justify-between gap-4">
        <span>{{ loadError }}</span>
        <Button variant="outline" size="sm" @click="load">{{ t("common.retry") }}</Button>
      </AlertDescription>
    </Alert>

    <div v-else-if="profile">
      <PanelSection id="profile" :title="t('account.profile')" :icon="IdCard">
        <!-- Identity summary: the person, then the fields they can change. -->
        <div class="flex items-start gap-4">
          <div
            class="flex size-14 shrink-0 select-none items-center justify-center rounded-xl border border-border/50 bg-muted text-lg font-semibold text-muted-foreground"
            aria-hidden="true"
          >
            {{ initials() }}
          </div>
          <div class="min-w-0 flex-1">
            <div class="flex flex-wrap items-center gap-2">
              <h3 class="truncate text-lg font-semibold">{{ profile.username }}</h3>
              <Badge variant="secondary">
                {{ profile.role === "admin" ? t("team.admin") : t("team.viewer") }}
              </Badge>
            </div>
            <p class="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span class="min-w-0 truncate">
                {{ profile.email || t("account.noEmail") }}
              </span>
              <StatusBadge
                v-if="profile.email"
                variant="status"
                :status="profile.email_verified ? 'success' : 'warning'"
              >
                {{
                  profile.email_verified
                    ? t("accountSecurity.emailVerified")
                    : t("account.emailUnverified")
                }}
              </StatusBadge>
            </p>
          </div>
        </div>

        <dl class="mt-6 divide-y divide-border/50 text-sm">
          <div class="flex flex-wrap items-center justify-between gap-3 py-3">
            <dt class="flex items-center gap-2 text-muted-foreground">
              <IdCard class="size-4" aria-hidden="true" />
              {{ t("account.username") }}
            </dt>
            <dd class="flex items-center gap-2">
              <span class="font-mono text-xs">{{ profile.username }}</span>
              <Button variant="outline" size="sm" @click="openUsernameDialog">
                {{ t("account.change") }}
              </Button>
            </dd>
          </div>

          <div class="flex flex-wrap items-center justify-between gap-3 py-3">
            <dt class="flex items-center gap-2 text-muted-foreground">
              <Mail class="size-4" aria-hidden="true" />
              {{ t("account.email") }}
            </dt>
            <dd class="flex max-w-full flex-wrap items-center gap-2">
              <span v-if="profile.email" class="min-w-0 truncate text-xs">
                {{ profile.email }}
              </span>
              <span v-else class="text-xs text-muted-foreground">{{ t("account.noEmail") }}</span>
              <StatusBadge
                v-if="profile.email"
                variant="status"
                :status="profile.email_verified ? 'success' : 'warning'"
              >
                {{
                  profile.email_verified
                    ? t("accountSecurity.emailVerified")
                    : t("account.emailUnverified")
                }}
              </StatusBadge>
              <Button variant="outline" size="sm" @click="openEmailDialog">
                {{ emailActionLabel() }}
              </Button>
            </dd>
          </div>

          <div class="flex flex-wrap items-center justify-between gap-3 py-3">
            <dt class="flex items-center gap-2 text-muted-foreground">
              <ShieldCheck class="size-4" aria-hidden="true" />
              {{ t("account.role") }}
            </dt>
            <dd class="text-xs">
              {{ profile.role === "admin" ? t("team.admin") : t("team.viewer") }}
            </dd>
          </div>

          <div class="flex flex-wrap items-center justify-between gap-3 py-3">
            <dt class="flex items-center gap-2 text-muted-foreground">
              <CalendarDays class="size-4" aria-hidden="true" />
              {{ t("account.memberSince") }}
            </dt>
            <dd class="text-data text-xs">{{ formatDate(profile.created_at) }}</dd>
          </div>
        </dl>
      </PanelSection>

      <PanelSection
        id="access"
        :title="t('account.access')"
        :description="t('account.accessDescription')"
        :icon="Wallet"
        class="mt-6"
      >
        <AccountBudgetBanner v-if="budget && budget.budget_usd !== null" :budget="budget" />
        <p v-else class="text-xs text-muted-foreground">{{ t("account.noBudget") }}</p>

        <dl class="mt-6 divide-y divide-border/50 text-sm">
          <div class="flex flex-wrap items-start justify-between gap-3 py-3">
            <dt class="flex items-center gap-2 text-muted-foreground">
              <KeyRound class="size-4" aria-hidden="true" />
              {{ t("account.allowedModels") }}
            </dt>
            <dd
              v-if="profile.allowed_models?.length"
              class="flex max-h-24 max-w-full flex-wrap justify-end gap-1 overflow-y-auto"
            >
              <Badge
                v-for="model in profile.allowed_models"
                :key="model"
                variant="outline"
                class="font-mono text-[10px]"
              >
                {{ model }}
              </Badge>
            </dd>
            <dd v-else class="text-xs">{{ t("account.allModels") }}</dd>
          </div>

          <div class="flex flex-wrap items-center justify-between gap-3 py-3">
            <dt class="flex items-center gap-2 text-muted-foreground">
              <Clock class="size-4" aria-hidden="true" />
              {{ t("account.lastSignIn") }}
            </dt>
            <dd class="text-data text-xs">
              {{ profile.last_login_at ? formatDateTime(profile.last_login_at) : "—" }}
            </dd>
          </div>

          <div v-if="profile.session" class="flex flex-wrap items-start justify-between gap-3 py-3">
            <dt class="flex items-center gap-2 text-muted-foreground">
              <ShieldCheck class="size-4" aria-hidden="true" />
              {{ t("account.currentSession") }}
            </dt>
            <dd class="flex flex-col items-end gap-1 text-right text-xs">
              <span
                >{{ authMethodLabel(profile.session.auth_method) }} ·
                {{ profile.session.client_ip }}</span
              >
              <span class="text-muted-foreground">
                {{
                  t("account.sessionStarted", { date: formatDateTime(profile.session.created_at) })
                }}
              </span>
              <span class="text-muted-foreground">
                {{
                  t("account.sessionExpires", { date: formatDateTime(profile.session.expires_at) })
                }}
              </span>
            </dd>
          </div>
        </dl>
      </PanelSection>
    </div>

    <!-- Change username -->
    <Dialog v-model:open="showUsernameDialog">
      <DialogContent class="brand-panel">
        <DialogHeader>
          <DialogTitle>{{ t("profile.changeUsername") }}</DialogTitle>
          <DialogDescription>{{ t("profile.changeUsernameDescription") }}</DialogDescription>
        </DialogHeader>
        <form class="flex flex-col gap-4 py-2" @submit.prevent="changeUsername">
          <div class="flex flex-col gap-2">
            <Label for="account-new-username">{{ t("profile.newUsername") }}</Label>
            <Input
              id="account-new-username"
              v-model="usernameNew"
              :placeholder="profile?.username ?? ''"
              :disabled="savingUsername"
            />
            <p class="text-xs text-muted-foreground">{{ t("profile.usernameFormatHint") }}</p>
          </div>
          <div v-if="hasPassword" class="flex flex-col gap-2">
            <Label for="account-username-password">{{ t("profile.currentPassword") }}</Label>
            <Input
              id="account-username-password"
              v-model="usernamePassword"
              type="password"
              autocomplete="current-password"
              :disabled="savingUsername"
            />
          </div>
          <p v-else class="text-xs text-muted-foreground">
            {{ t("profile.reauthRequired") }}
          </p>
          <p v-if="usernameError" class="text-xs text-destructive">{{ usernameError }}</p>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              :disabled="savingUsername"
              @click="showUsernameDialog = false"
            >
              {{ t("common.cancel") }}
            </Button>
            <Button type="submit" :disabled="savingUsername">
              <Loader2 v-if="savingUsername" class="mr-2 size-4 animate-spin" />
              {{ t("profile.changeUsername") }}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>

    <!-- Login email: the address changes only after the verification link is opened. -->
    <Dialog v-model:open="showEmailDialog">
      <DialogContent class="brand-panel">
        <DialogHeader>
          <DialogTitle>{{ t("account.emailTitle") }}</DialogTitle>
          <DialogDescription>{{ t("accountSecurity.emailHelp") }}</DialogDescription>
        </DialogHeader>
        <form class="flex flex-col gap-4 py-2" @submit.prevent="sendEmailVerification">
          <div class="flex flex-col gap-2">
            <Label for="account-login-email">{{ t("account.email") }}</Label>
            <Input
              id="account-login-email"
              v-model="emailDraft"
              type="email"
              autocomplete="email"
              :placeholder="t('accountSecurity.emailPlaceholder')"
              :disabled="sendingEmail"
            />
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              :disabled="sendingEmail"
              @click="showEmailDialog = false"
            >
              {{ t("common.cancel") }}
            </Button>
            <Button type="submit" :disabled="sendingEmail || !emailDraft">
              <Loader2 v-if="sendingEmail" class="mr-2 size-4 animate-spin" />
              {{ t("accountSecurity.sendVerification") }}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  </AccountShell>
</template>
