<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { onBeforeRouteLeave } from "vue-router";
import { useClipboard } from "@vueuse/core";
import { ChevronDown, Loader2, Plus, ShieldCheck, Trash2 } from "@lucide/vue";
import { toast } from "vue-sonner";
import AppLayout from "@/components/layout/AppLayout.vue";
import LoadingState from "@/components/common/LoadingState.vue";
import PageHeader from "@/components/common/PageHeader.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  consoleAuthApi,
  type ConsoleAuthSettings,
  type OidcProviderSettings,
} from "@/services/api/consoleAuth";
import { getErrorMessage } from "@/utils/error";

const { t } = useI18n();
const { copy } = useClipboard({ legacy: true, copiedDuring: 1500 });
const settings = ref<ConsoleAuthSettings>({
  public_base_url: "",
  require_2fa_for_admins: false,
  smtp: {
    enabled: false,
    host: "",
    port: 587,
    username: "",
    password: "",
    from_address: "",
    starttls: true,
    use_ssl: false,
  },
  oidc_providers: [],
});
const settingsLoaded = ref(false);
const loading = ref(true);
const saving = ref(false);
const error = ref("");
const savedSnapshot = ref("");
// IDs that came from the server are immutable: the backend matches the stored
// encrypted client secret by ID, so a rename would drop the secret on save.
const savedProviderIds = ref<Set<string>>(new Set());
const expandedProviders = ref<Set<number>>(new Set());
const fieldErrors = ref<Record<string, string>>({});

const PROVIDER_ID_PATTERN = /^[a-z0-9][a-z0-9_-]*$/;
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

const hasFieldErrors = computed(() => Object.keys(fieldErrors.value).length > 0);
const fieldErrorList = computed(() =>
  Object.entries(fieldErrors.value).map(([id, message]) => ({ id, message }))
);

function isSecureHttpUrl(value: string): boolean {
  try {
    const url = new URL(value.trim());
    return (
      url.protocol === "https:" || (url.protocol === "http:" && LOOPBACK_HOSTS.has(url.hostname))
    );
  } catch {
    return false;
  }
}

function clearFieldError(fieldId: string) {
  error.value = "";
  if (!(fieldId in fieldErrors.value)) return;
  const next = { ...fieldErrors.value };
  delete next[fieldId];
  fieldErrors.value = next;
  error.value = "";
}

function describedBy(...ids: Array<string | false | undefined>): string | undefined {
  const value = ids.filter(Boolean).join(" ");
  return value || undefined;
}

function expandProvider(index: number) {
  if (expandedProviders.value.has(index)) return;
  const next = new Set(expandedProviders.value);
  next.add(index);
  expandedProviders.value = next;
}

function isProviderExpanded(index: number): boolean {
  return expandedProviders.value.has(index);
}

function toggleProvider(index: number) {
  const next = new Set(expandedProviders.value);
  if (next.has(index)) next.delete(index);
  else next.add(index);
  expandedProviders.value = next;
}

function clearSmtpErrors() {
  const next = { ...fieldErrors.value };
  delete next["smtp-host"];
  delete next["smtp-port"];
  delete next["smtp-from"];
  fieldErrors.value = next;
  error.value = "";
}

async function focusField(fieldId: string) {
  const providerMatch = /^provider-(\d+)-/.exec(fieldId);
  if (providerMatch?.[1]) expandProvider(Number(providerMatch[1]));
  await nextTick();
  const field = document.getElementById(fieldId);
  field?.scrollIntoView?.({ block: "center" });
  field?.focus();
}

function validate(): boolean {
  const errors: Record<string, string> = {};
  const required = (labelKey: string) => t("consoleAuth.fieldRequired", { field: t(labelKey) });
  const current = settings.value;

  if (current.public_base_url.trim() && !isSecureHttpUrl(current.public_base_url)) {
    errors["public-url"] = t("consoleAuth.publicUrlInvalid");
  }

  if (current.smtp.enabled) {
    if (!current.smtp.host.trim()) errors["smtp-host"] = required("consoleAuth.host");
    if (
      !Number.isInteger(current.smtp.port) ||
      current.smtp.port < 1 ||
      current.smtp.port > 65535
    ) {
      errors["smtp-port"] = t("consoleAuth.portInvalid");
    }
    if (!current.smtp.from_address.trim()) {
      errors["smtp-from"] = required("consoleAuth.fromAddress");
    } else if (!/^[^\s@]+@[^\s@]+$/.test(current.smtp.from_address.trim())) {
      errors["smtp-from"] = t("consoleAuth.emailInvalid");
    }
  }

  const seenProviderIds = new Set<string>();
  current.oidc_providers.forEach((provider, index) => {
    const prefix = `provider-${index}`;
    if (!provider.id) {
      errors[`${prefix}-id`] = required("consoleAuth.providerId");
    } else if (!PROVIDER_ID_PATTERN.test(provider.id)) {
      errors[`${prefix}-id`] = t("consoleAuth.providerIdInvalid");
    } else if (seenProviderIds.has(provider.id)) {
      errors[`${prefix}-id`] = t("consoleAuth.providerIdDuplicate");
    }
    seenProviderIds.add(provider.id);
    if (!provider.name.trim()) errors[`${prefix}-name`] = required("consoleAuth.providerName");
    if (!provider.issuer.trim()) {
      errors[`${prefix}-issuer`] = required("consoleAuth.issuer");
    } else if (!isSecureHttpUrl(provider.issuer)) {
      errors[`${prefix}-issuer`] = t("consoleAuth.issuerInvalid");
    }
    if (!provider.client_id.trim()) {
      errors[`${prefix}-client-id`] = required("consoleAuth.clientId");
    }
  });

  fieldErrors.value = errors;
  error.value = hasFieldErrors.value ? t("consoleAuth.validationSummary") : "";
  return !hasFieldErrors.value;
}

const isDirty = computed(
  () => settingsLoaded.value && JSON.stringify(settings.value) !== savedSnapshot.value
);

function newProvider(): OidcProviderSettings {
  return {
    id: "",
    name: "",
    issuer: "",
    client_id: "",
    client_secret: "",
    scopes: "openid email profile",
    enabled: true,
  };
}

function snapshotCurrentSettings() {
  savedSnapshot.value = JSON.stringify(settings.value);
}

async function load() {
  loading.value = true;
  error.value = "";
  try {
    settings.value = await consoleAuthApi.get();
    settingsLoaded.value = true;
    savedProviderIds.value = new Set(settings.value.oidc_providers.map((p) => p.id));
    expandedProviders.value = new Set();
    snapshotCurrentSettings();
  } catch (e) {
    error.value = getErrorMessage(e);
  } finally {
    loading.value = false;
  }
}

function isProviderIdLocked(provider: OidcProviderSettings): boolean {
  return savedProviderIds.value.has(provider.id);
}

/**
 * The callback URL the identity provider must be configured with.
 *
 * It is derived from the public base URL and the provider ID by the backend
 * (``/api/auth/oauth/{id}/callback``). Showing it here is the difference
 * between "fill in five fields and hope" and a working OIDC registration: the
 * ID is immutable once saved, so the value is stable after the first save.
 */
function redirectUri(provider: OidcProviderSettings): string {
  const base = (settings.value?.public_base_url ?? "").replace(/\/+$/, "");
  if (!base || !provider.id || !PROVIDER_ID_PATTERN.test(provider.id)) return "";
  return `${base}/api/auth/oauth/${provider.id}/callback`;
}

async function copyRedirectUri(provider: OidcProviderSettings): Promise<void> {
  const value = redirectUri(provider);
  if (!value) return;
  await copy(value);
  toast.success(t("common.copied"));
}

async function save() {
  if (!settingsLoaded.value || saving.value || !isDirty.value) return;
  if (!validate()) {
    const firstField = fieldErrorList.value[0]?.id;
    if (firstField) await focusField(firstField);
    return;
  }
  saving.value = true;
  error.value = "";
  try {
    settings.value = await consoleAuthApi.update(settings.value);
    savedProviderIds.value = new Set(settings.value.oidc_providers.map((p) => p.id));
    snapshotCurrentSettings();
    fieldErrors.value = {};
    toast.success(t("common.saved"));
  } catch (e) {
    error.value = getErrorMessage(e);
    await nextTick();
    const summary = document.getElementById("console-auth-error");
    summary?.scrollIntoView?.({ block: "center" });
    summary?.focus();
  } finally {
    saving.value = false;
  }
}

function discardChanges() {
  if (!isDirty.value || saving.value) return;
  settings.value = JSON.parse(savedSnapshot.value) as ConsoleAuthSettings;
  expandedProviders.value = new Set();
  fieldErrors.value = {};
  error.value = "";
}

onBeforeRouteLeave(() => {
  if (!isDirty.value) return true;
  return window.confirm(t("consoleAuth.leaveConfirmation"));
});

function warnBeforeUnload(event: BeforeUnloadEvent) {
  if (!isDirty.value) return;
  event.preventDefault();
  event.returnValue = "";
}

function addProvider() {
  const index = settings.value.oidc_providers.length;
  settings.value.oidc_providers.push(newProvider());
  expandProvider(index);
}

function removeProvider(index: number) {
  settings.value.oidc_providers.splice(index, 1);
  const nextExpanded = new Set<number>();
  for (const expandedIndex of expandedProviders.value) {
    if (expandedIndex < index) nextExpanded.add(expandedIndex);
    else if (expandedIndex > index) nextExpanded.add(expandedIndex - 1);
  }
  expandedProviders.value = nextExpanded;
  fieldErrors.value = Object.fromEntries(
    Object.entries(fieldErrors.value).filter(([id]) => !id.startsWith("provider-"))
  );
  error.value = "";
}

onMounted(() => {
  window.addEventListener("beforeunload", warnBeforeUnload);
  void load();
});

onUnmounted(() => {
  window.removeEventListener("beforeunload", warnBeforeUnload);
});
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar px-4 sm:px-6 py-4">
        <PageHeader
          :title="t('consoleAuth.title')"
          :description="t('consoleAuth.description')"
          :icon="ShieldCheck"
        />
      </header>
    </template>

    <div class="config-content overflow-y-auto px-4 sm:px-6 py-6">
      <div class="mx-auto w-full max-w-3xl">
        <LoadingState v-if="loading" />

        <Alert v-else-if="error && !settingsLoaded" variant="destructive">
          <AlertDescription class="flex flex-wrap items-center justify-between gap-3">
            <span>{{ error }}</span>
            <Button variant="outline" size="sm" @click="load">{{ t("common.retry") }}</Button>
          </AlertDescription>
        </Alert>

        <form
          v-else-if="settingsLoaded"
          class="flex flex-col gap-6"
          :aria-busy="saving"
          novalidate
          @submit.prevent="save"
        >
          <Alert
            v-if="error || hasFieldErrors"
            id="console-auth-error"
            variant="destructive"
            tabindex="-1"
          >
            <AlertDescription>
              <p>{{ error || t("consoleAuth.validationSummary") }}</p>
              <ul v-if="hasFieldErrors" class="list-disc space-y-1 pl-5">
                <li v-for="issue in fieldErrorList" :key="issue.id">
                  <button
                    type="button"
                    class="text-left underline underline-offset-4 hover:text-foreground"
                    @click="focusField(issue.id)"
                  >
                    {{ issue.message }}
                  </button>
                </li>
              </ul>
            </AlertDescription>
          </Alert>

          <section class="rounded-xl border border-border/60 bg-card p-5 sm:p-6">
            <h2 class="font-medium">{{ t("consoleAuth.consoleAccess") }}</h2>
            <p class="mt-1 max-w-[65ch] text-xs leading-relaxed text-muted-foreground">
              {{ t("consoleAuth.publicUrlHelp") }}
            </p>
            <div class="mt-5 space-y-2">
              <Label for="public-url">{{ t("consoleAuth.publicUrl") }}</Label>
              <Input
                id="public-url"
                v-model="settings.public_base_url"
                type="url"
                inputmode="url"
                autocomplete="url"
                spellcheck="false"
                placeholder="https://proxy.example.com"
                :aria-invalid="!!fieldErrors['public-url']"
                :aria-describedby="fieldErrors['public-url'] ? 'public-url-error' : undefined"
                @input="clearFieldError('public-url')"
              />
              <p
                v-if="fieldErrors['public-url']"
                id="public-url-error"
                class="text-xs text-destructive"
              >
                {{ fieldErrors["public-url"] }}
              </p>
            </div>
            <div
              class="mt-5 flex items-center justify-between gap-4 border-t border-border/50 pt-5"
            >
              <div>
                <p class="text-sm font-medium">{{ t("consoleAuth.requireAdmin2fa") }}</p>
                <p class="mt-0.5 text-xs text-muted-foreground">
                  {{ t("consoleAuth.requireAdmin2faHelp") }}
                </p>
              </div>
              <Switch
                v-model="settings.require_2fa_for_admins"
                :aria-label="t('consoleAuth.requireAdmin2fa')"
              />
            </div>
            <p v-if="settings.require_2fa_for_admins" class="mt-3 text-xs text-status-warning">
              {{ t("consoleAuth.requireAdmin2faWarning") }}
            </p>
          </section>

          <section class="rounded-xl border border-border/60 bg-card p-5 sm:p-6">
            <h2 class="font-medium">{{ t("consoleAuth.smtp") }}</h2>
            <p class="mt-1 text-xs text-muted-foreground">{{ t("consoleAuth.smtpHelp") }}</p>
            <div class="mt-5 flex items-center justify-between gap-4">
              <p class="text-sm font-medium">{{ t("consoleAuth.smtpEnabled") }}</p>
              <Switch
                v-model="settings.smtp.enabled"
                :aria-label="t('consoleAuth.smtpEnabled')"
                @update:model-value="clearSmtpErrors"
              />
            </div>
            <div v-if="settings.smtp.enabled" class="mt-5 grid gap-4 sm:grid-cols-2">
              <div class="space-y-2">
                <Label for="smtp-host">{{ t("consoleAuth.host") }}</Label>
                <Input
                  id="smtp-host"
                  v-model="settings.smtp.host"
                  autocomplete="off"
                  maxlength="255"
                  :aria-invalid="!!fieldErrors['smtp-host']"
                  :aria-describedby="fieldErrors['smtp-host'] ? 'smtp-host-error' : undefined"
                  @input="clearFieldError('smtp-host')"
                />
                <p
                  v-if="fieldErrors['smtp-host']"
                  id="smtp-host-error"
                  class="text-xs text-destructive"
                >
                  {{ fieldErrors["smtp-host"] }}
                </p>
              </div>
              <div class="space-y-2">
                <Label for="smtp-port">{{ t("consoleAuth.port") }}</Label>
                <Input
                  id="smtp-port"
                  v-model.number="settings.smtp.port"
                  type="number"
                  inputmode="numeric"
                  min="1"
                  max="65535"
                  class="text-data"
                  :aria-invalid="!!fieldErrors['smtp-port']"
                  :aria-describedby="fieldErrors['smtp-port'] ? 'smtp-port-error' : undefined"
                  @input="clearFieldError('smtp-port')"
                />
                <p
                  v-if="fieldErrors['smtp-port']"
                  id="smtp-port-error"
                  class="text-xs text-destructive"
                >
                  {{ fieldErrors["smtp-port"] }}
                </p>
              </div>
              <div class="space-y-2">
                <Label for="smtp-user">{{ t("consoleAuth.username") }}</Label>
                <Input
                  id="smtp-user"
                  v-model="settings.smtp.username"
                  autocomplete="off"
                  maxlength="255"
                />
              </div>
              <div class="space-y-2">
                <Label for="smtp-password">{{ t("consoleAuth.password") }}</Label>
                <Input
                  id="smtp-password"
                  v-model="settings.smtp.password"
                  type="password"
                  autocomplete="new-password"
                />
              </div>
              <div class="space-y-2 sm:col-span-2">
                <Label for="smtp-from">{{ t("consoleAuth.fromAddress") }}</Label>
                <Input
                  id="smtp-from"
                  v-model="settings.smtp.from_address"
                  type="email"
                  autocomplete="off"
                  maxlength="320"
                  :aria-invalid="!!fieldErrors['smtp-from']"
                  :aria-describedby="fieldErrors['smtp-from'] ? 'smtp-from-error' : undefined"
                  @input="clearFieldError('smtp-from')"
                />
                <p
                  v-if="fieldErrors['smtp-from']"
                  id="smtp-from-error"
                  class="text-xs text-destructive"
                >
                  {{ fieldErrors["smtp-from"] }}
                </p>
              </div>
            </div>
            <div v-if="settings.smtp.enabled" class="mt-4 flex flex-wrap gap-x-6">
              <label class="flex min-h-11 items-center gap-2 text-sm">
                <Switch v-model="settings.smtp.starttls" />
                {{ t("consoleAuth.starttls") }}
              </label>
              <label class="flex min-h-11 items-center gap-2 text-sm">
                <Switch v-model="settings.smtp.use_ssl" />
                {{ t("consoleAuth.smtps") }}
              </label>
            </div>
          </section>

          <section class="rounded-xl border border-border/60 bg-card p-5 sm:p-6">
            <div class="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 class="font-medium">{{ t("consoleAuth.oidc") }}</h2>
                <p class="mt-1 max-w-[65ch] text-xs text-muted-foreground">
                  {{ t("consoleAuth.oidcHelp") }}
                </p>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                class="gap-2 max-md:h-11"
                @click="addProvider"
              >
                <Plus class="size-4" />
                {{ t("consoleAuth.addProvider") }}
              </Button>
            </div>

            <div class="mt-5 divide-y divide-border/50">
              <article
                v-for="(provider, index) in settings.oidc_providers"
                :key="index"
                class="py-5 first:pt-0 last:pb-0"
              >
                <div
                  class="flex flex-wrap items-center justify-between gap-3"
                  :class="{ 'mb-4': isProviderExpanded(index) }"
                >
                  <div class="flex min-w-0 items-center gap-3">
                    <h3 class="truncate text-sm font-medium">
                      {{ provider.name || t("consoleAuth.provider") }}
                    </h3>
                    <Switch
                      v-model="provider.enabled"
                      :aria-label="
                        t('consoleAuth.providerEnabled', {
                          name: provider.name || t('consoleAuth.provider'),
                        })
                      "
                    />
                  </div>
                  <div class="flex shrink-0 items-center gap-1">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      class="gap-1.5 text-xs"
                      :aria-expanded="isProviderExpanded(index)"
                      :aria-controls="`provider-${index}-details`"
                      @click="toggleProvider(index)"
                    >
                      {{
                        isProviderExpanded(index)
                          ? t("consoleAuth.hideProviderDetails")
                          : t("consoleAuth.showProviderDetails")
                      }}
                      <ChevronDown
                        class="size-3.5 transition-transform duration-200"
                        :class="{ 'rotate-180': isProviderExpanded(index) }"
                      />
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      class="shrink-0 text-muted-foreground hover:text-destructive"
                      :aria-label="t('consoleAuth.removeProvider')"
                      :title="t('consoleAuth.removeProvider')"
                      @click="removeProvider(index)"
                    >
                      <Trash2 class="size-4" />
                    </Button>
                  </div>
                </div>

                <div
                  v-show="isProviderExpanded(index)"
                  :id="`provider-${index}-details`"
                  class="grid gap-4 sm:grid-cols-2"
                >
                  <div class="space-y-2">
                    <Label :for="`provider-${index}-id`">{{ t("consoleAuth.providerId") }}</Label>
                    <Input
                      :id="`provider-${index}-id`"
                      v-model="provider.id"
                      class="font-mono text-xs"
                      :disabled="isProviderIdLocked(provider)"
                      :pattern="'[a-z0-9][a-z0-9_-]*'"
                      maxlength="50"
                      :aria-invalid="!!fieldErrors[`provider-${index}-id`]"
                      :aria-describedby="
                        describedBy(
                          isProviderIdLocked(provider) && `provider-${index}-id-hint`,
                          fieldErrors[`provider-${index}-id`] && `provider-${index}-id-error`
                        )
                      "
                      @input="clearFieldError(`provider-${index}-id`)"
                    />
                    <p
                      v-if="fieldErrors[`provider-${index}-id`]"
                      :id="`provider-${index}-id-error`"
                      class="text-xs text-destructive"
                    >
                      {{ fieldErrors[`provider-${index}-id`] }}
                    </p>
                    <p
                      v-if="isProviderIdLocked(provider)"
                      :id="`provider-${index}-id-hint`"
                      class="text-xs text-muted-foreground"
                    >
                      {{ t("consoleAuth.providerIdLockedHint") }}
                    </p>
                  </div>
                  <div class="space-y-2">
                    <Label :for="`provider-${index}-name`">{{
                      t("consoleAuth.providerName")
                    }}</Label>
                    <Input
                      :id="`provider-${index}-name`"
                      v-model="provider.name"
                      maxlength="100"
                      :aria-invalid="!!fieldErrors[`provider-${index}-name`]"
                      :aria-describedby="
                        fieldErrors[`provider-${index}-name`]
                          ? `provider-${index}-name-error`
                          : undefined
                      "
                      @input="clearFieldError(`provider-${index}-name`)"
                    />
                    <p
                      v-if="fieldErrors[`provider-${index}-name`]"
                      :id="`provider-${index}-name-error`"
                      class="text-xs text-destructive"
                    >
                      {{ fieldErrors[`provider-${index}-name`] }}
                    </p>
                  </div>
                  <div class="space-y-2 sm:col-span-2">
                    <Label :for="`provider-${index}-issuer`">{{ t("consoleAuth.issuer") }}</Label>
                    <Input
                      :id="`provider-${index}-issuer`"
                      v-model="provider.issuer"
                      type="url"
                      inputmode="url"
                      autocomplete="off"
                      spellcheck="false"
                      maxlength="500"
                      :aria-invalid="!!fieldErrors[`provider-${index}-issuer`]"
                      :aria-describedby="
                        fieldErrors[`provider-${index}-issuer`]
                          ? `provider-${index}-issuer-error`
                          : undefined
                      "
                      @input="clearFieldError(`provider-${index}-issuer`)"
                    />
                    <p
                      v-if="fieldErrors[`provider-${index}-issuer`]"
                      :id="`provider-${index}-issuer-error`"
                      class="text-xs text-destructive"
                    >
                      {{ fieldErrors[`provider-${index}-issuer`] }}
                    </p>
                  </div>
                  <div class="space-y-2">
                    <Label :for="`provider-${index}-client-id`">{{
                      t("consoleAuth.clientId")
                    }}</Label>
                    <Input
                      :id="`provider-${index}-client-id`"
                      v-model="provider.client_id"
                      autocomplete="off"
                      spellcheck="false"
                      maxlength="255"
                      :aria-invalid="!!fieldErrors[`provider-${index}-client-id`]"
                      :aria-describedby="
                        fieldErrors[`provider-${index}-client-id`]
                          ? `provider-${index}-client-id-error`
                          : undefined
                      "
                      @input="clearFieldError(`provider-${index}-client-id`)"
                    />
                    <p
                      v-if="fieldErrors[`provider-${index}-client-id`]"
                      :id="`provider-${index}-client-id-error`"
                      class="text-xs text-destructive"
                    >
                      {{ fieldErrors[`provider-${index}-client-id`] }}
                    </p>
                  </div>
                  <div class="space-y-2">
                    <Label :for="`provider-${index}-client-secret`">{{
                      t("consoleAuth.clientSecret")
                    }}</Label>
                    <Input
                      :id="`provider-${index}-client-secret`"
                      v-model="provider.client_secret"
                      type="password"
                      autocomplete="new-password"
                    />
                  </div>
                  <div class="space-y-2 sm:col-span-2">
                    <Label :for="`provider-${index}-scopes`">{{ t("consoleAuth.scopes") }}</Label>
                    <Input
                      :id="`provider-${index}-scopes`"
                      v-model="provider.scopes"
                      maxlength="255"
                      class="font-mono text-xs"
                      autocomplete="off"
                      spellcheck="false"
                    />
                  </div>
                  <div class="space-y-2 sm:col-span-2">
                    <p class="text-sm font-medium">{{ t("consoleAuth.redirectUri") }}</p>
                    <div class="flex items-center gap-2">
                      <code
                        :id="`provider-${index}-redirect`"
                        class="min-w-0 flex-1 break-all rounded-md border border-border/50 bg-muted/30 px-3 py-2 font-mono text-[11px]"
                      >
                        {{ redirectUri(provider) || t("consoleAuth.redirectUriPending") }}
                      </code>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        class="shrink-0"
                        :disabled="!redirectUri(provider)"
                        @click="copyRedirectUri(provider)"
                      >
                        {{ t("common.copy") }}
                      </Button>
                    </div>
                    <p class="text-xs text-muted-foreground">
                      {{ t("consoleAuth.redirectUriHelp") }}
                    </p>
                  </div>
                </div>
              </article>

              <p
                v-if="!settings.oidc_providers.length"
                class="py-6 text-center text-sm text-muted-foreground"
              >
                {{ t("consoleAuth.noProviders") }}
              </p>
            </div>
          </section>

          <div
            class="sticky bottom-0 z-10 flex flex-wrap items-center justify-between gap-3 border-t border-border/60 bg-background py-4"
          >
            <span class="text-xs text-muted-foreground" role="status">
              {{ isDirty ? t("consoleAuth.unsavedChanges") : t("consoleAuth.noPendingChanges") }}
            </span>
            <div class="flex items-center gap-2">
              <Button
                type="button"
                variant="ghost"
                class="max-md:h-11"
                :disabled="saving || !isDirty"
                @click="discardChanges"
              >
                {{ t("consoleAuth.discardChanges") }}
              </Button>
              <Button type="submit" class="gap-2 max-md:h-11" :disabled="saving || !isDirty">
                <Loader2 v-if="saving" class="size-4 animate-spin" />
                {{ saving ? t("consoleAuth.saving") : t("common.save") }}
              </Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  </AppLayout>
</template>
