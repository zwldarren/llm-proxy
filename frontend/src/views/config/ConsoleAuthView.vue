<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useClipboard } from "@vueuse/core";
import { Loader2, Plus, ShieldCheck, Trash2 } from "@lucide/vue";
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

const PROVIDER_ID_PATTERN = /^[a-z0-9][a-z0-9_-]*$/;

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
    snapshotCurrentSettings();
  } catch (e) {
    error.value = e instanceof Error ? e.message : t("common.error");
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
  const invalid = settings.value.oidc_providers.find(
    (provider) => !PROVIDER_ID_PATTERN.test(provider.id)
  );
  if (invalid) {
    error.value = t("consoleAuth.providerIdInvalid");
    return;
  }
  saving.value = true;
  error.value = "";
  try {
    settings.value = await consoleAuthApi.update(settings.value);
    savedProviderIds.value = new Set(settings.value.oidc_providers.map((p) => p.id));
    snapshotCurrentSettings();
    toast.success(t("common.saved"));
  } catch (e) {
    error.value = e instanceof Error ? e.message : t("common.error");
  } finally {
    saving.value = false;
  }
}

function addProvider() {
  settings.value?.oidc_providers.push(newProvider());
}

function removeProvider(index: number) {
  settings.value?.oidc_providers.splice(index, 1);
}

onMounted(load);
</script>

<template>
  <AppLayout>
    <div class="mx-auto w-full max-w-3xl">
      <PageHeader
        :title="t('consoleAuth.title')"
        :description="t('consoleAuth.description')"
        :icon="ShieldCheck"
      />

      <LoadingState v-if="loading" />

      <Alert v-else-if="error && !settingsLoaded" variant="destructive" class="mt-6">
        <AlertDescription class="flex flex-wrap items-center justify-between gap-3">
          <span>{{ error }}</span>
          <Button variant="outline" size="sm" @click="load">{{ t("common.retry") }}</Button>
        </AlertDescription>
      </Alert>

      <form
        v-else-if="settingsLoaded"
        class="mt-6 flex flex-col gap-6"
        :aria-busy="saving"
        @submit.prevent="save"
      >
        <Alert v-if="error" variant="destructive">
          <AlertDescription>{{ error }}</AlertDescription>
        </Alert>

        <section class="rounded-xl border border-border/60 bg-card p-5 sm:p-6">
          <h2 class="font-medium">{{ t("consoleAuth.publicUrl") }}</h2>
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
            />
          </div>
          <div class="mt-5 flex items-center justify-between gap-4 border-t border-border/50 pt-5">
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
            <Switch v-model="settings.smtp.enabled" :aria-label="t('consoleAuth.smtpEnabled')" />
          </div>
          <div class="mt-5 grid gap-4 sm:grid-cols-2">
            <div class="space-y-2">
              <Label for="smtp-host">{{ t("consoleAuth.host") }}</Label>
              <Input
                id="smtp-host"
                v-model="settings.smtp.host"
                autocomplete="off"
                :disabled="!settings.smtp.enabled"
              />
            </div>
            <div class="space-y-2">
              <Label for="smtp-port">{{ t("consoleAuth.port") }}</Label>
              <Input
                id="smtp-port"
                v-model.number="settings.smtp.port"
                type="number"
                inputmode="numeric"
                class="text-data"
                :disabled="!settings.smtp.enabled"
              />
            </div>
            <div class="space-y-2">
              <Label for="smtp-user">{{ t("consoleAuth.username") }}</Label>
              <Input
                id="smtp-user"
                v-model="settings.smtp.username"
                autocomplete="off"
                :disabled="!settings.smtp.enabled"
              />
            </div>
            <div class="space-y-2">
              <Label for="smtp-password">{{ t("consoleAuth.password") }}</Label>
              <Input
                id="smtp-password"
                v-model="settings.smtp.password"
                type="password"
                autocomplete="new-password"
                :disabled="!settings.smtp.enabled"
              />
            </div>
            <div class="space-y-2 sm:col-span-2">
              <Label for="smtp-from">{{ t("consoleAuth.fromAddress") }}</Label>
              <Input
                id="smtp-from"
                v-model="settings.smtp.from_address"
                type="email"
                autocomplete="off"
                :disabled="!settings.smtp.enabled"
              />
            </div>
          </div>
          <div class="mt-4 flex flex-wrap gap-x-6">
            <label class="flex min-h-11 items-center gap-2 text-sm">
              <Switch v-model="settings.smtp.starttls" :disabled="!settings.smtp.enabled" />
              {{ t("consoleAuth.starttls") }}
            </label>
            <label class="flex min-h-11 items-center gap-2 text-sm">
              <Switch v-model="settings.smtp.use_ssl" :disabled="!settings.smtp.enabled" />
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
              <div class="mb-4 flex items-center justify-between gap-3">
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

              <div class="grid gap-4 sm:grid-cols-2">
                <div class="space-y-2">
                  <Label :for="`provider-${index}-id`">{{ t("consoleAuth.providerId") }}</Label>
                  <Input
                    :id="`provider-${index}-id`"
                    v-model="provider.id"
                    class="font-mono text-xs"
                    :disabled="isProviderIdLocked(provider)"
                    :pattern="'[a-z0-9][a-z0-9_-]*'"
                    :aria-describedby="
                      isProviderIdLocked(provider) ? `provider-${index}-id-hint` : undefined
                    "
                  />
                  <p
                    v-if="isProviderIdLocked(provider)"
                    :id="`provider-${index}-id-hint`"
                    class="text-xs text-muted-foreground"
                  >
                    {{ t("consoleAuth.providerIdLockedHint") }}
                  </p>
                </div>
                <div class="space-y-2">
                  <Label :for="`provider-${index}-name`">{{ t("consoleAuth.providerName") }}</Label>
                  <Input :id="`provider-${index}-name`" v-model="provider.name" />
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
                  />
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
                  />
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
          <Button type="submit" class="gap-2 max-md:h-11" :disabled="saving || !isDirty">
            <Loader2 v-if="saving" class="size-4 animate-spin" />
            {{ saving ? t("common.loading") : t("common.save") }}
          </Button>
        </div>
      </form>
    </div>
  </AppLayout>
</template>
