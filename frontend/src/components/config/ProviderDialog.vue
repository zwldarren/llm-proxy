<script setup lang="ts">
import { ExternalLink, KeyRound, Plus, RefreshCw, Server, Trash2, Unlink } from "@lucide/vue";
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useForm } from "vee-validate";
import { toast } from "vue-sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Sheet, SheetContent, SheetFooter, SheetHeader } from "@/components/ui/sheet";
import BrandMark from "@/components/common/BrandMark.vue";
import { useProviderTypes } from "@/composables/useProviderTypes";
import { configApi } from "@/services/api/config";
import { providerFormSchema, type ProviderFormValues } from "@/schemas/providerForm";
import type {
  AppAttribution,
  ProviderCreate,
  ProviderOAuthStatus,
  ProviderRead,
  ProviderUpdate,
} from "@/types/schemas";

const props = defineProps<{
  open: boolean;
  provider?: ProviderRead | null;
}>();

const emit = defineEmits<{
  (e: "update:open", value: boolean): void;
  (e: "saved"): void;
}>();

const { t } = useI18n();
const { typeOptions, providerMark, ensureLoaded } = useProviderTypes();

const isLoading = ref(false);
const savedProvider = ref<ProviderRead | null>(null);
const activeProvider = computed(() => props.provider ?? savedProvider.value);
const isEditing = computed(() => !!activeProvider.value);
const apiKeyEditing = ref(false);

const oauthStatus = ref<ProviderOAuthStatus | null>(null);
const oauthLoading = ref(false);
const showOAuthOptions = ref(false);
const oauthRedirectUri = ref("");
const oauthClientId = ref("");
const oauthClientSecret = ref("");
const pendingAuthUrl = ref("");
const callbackUrlInput = ref("");

function defaultOAuthRedirectUri(): string {
  if (typeof window === "undefined") return "http://127.0.0.1:1455/auth/callback";
  const host = window.location.hostname.toLowerCase();
  if (host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "[::1]") {
    return `${window.location.origin}/api/config/providers/oauth/callback`;
  }
  return "http://127.0.0.1:1455/auth/callback";
}

function resetOAuthState(provider: ProviderRead | null) {
  oauthStatus.value = provider?.oauth_status ?? null;
  oauthLoading.value = false;
  showOAuthOptions.value = false;
  oauthRedirectUri.value = defaultOAuthRedirectUri();
  oauthClientId.value = "";
  oauthClientSecret.value = "";
  pendingAuthUrl.value = "";
  callbackUrlInput.value = "";
}

async function loadOAuthStatus(providerName: string) {
  try {
    oauthStatus.value = await configApi.getProviderOAuthStatus(providerName);
  } catch {
    // Keep existing status on transient fetch failures.
  }
}

function onOAuthPopupMessage(event: MessageEvent) {
  if (typeof window === "undefined" || event.origin !== window.location.origin) return;
  const data = event.data as
    | { type?: string; status?: string; provider?: string; message?: string }
    | undefined;
  if (!data || data.type !== "llm-proxy-provider-oauth") return;
  const currentName = activeProvider.value?.name;
  if (data.provider && currentName && data.provider !== currentName) return;

  if (data.status === "success" && currentName) {
    pendingAuthUrl.value = "";
    callbackUrlInput.value = "";
    void loadOAuthStatus(currentName);
    toast.success(t("common.success"), { description: t("providers.oauthConnectSuccess") });
    emit("saved");
  } else if (data.status === "error") {
    toast.error(t("common.error"), {
      description: data.message || t("providers.oauthConnectFailed"),
    });
  }
}

if (typeof window !== "undefined") {
  window.addEventListener("message", onOAuthPopupMessage);
}

onBeforeUnmount(() => {
  if (typeof window !== "undefined") {
    window.removeEventListener("message", onOAuthPopupMessage);
  }
});

async function startOAuthFlow(forceNewRegistration = false) {
  const providerName = activeProvider.value?.name;
  if (!providerName) return;
  oauthLoading.value = true;
  try {
    const res = await configApi.startProviderOAuth(providerName, {
      redirect_uri: oauthRedirectUri.value.trim() || undefined,
      client_id: oauthClientId.value.trim() || undefined,
      client_secret: oauthClientSecret.value.trim() || undefined,
      force_new_registration: forceNewRegistration,
    });
    pendingAuthUrl.value = res.authorization_url;
    oauthRedirectUri.value = res.redirect_uri;
    if (typeof window !== "undefined") {
      window.open(res.authorization_url, "_blank", "width=560,height=720");
    }
  } catch (e: unknown) {
    const errorMessage = e instanceof Error ? e.message : t("providers.oauthConnectFailed");
    toast.error(t("common.error"), { description: errorMessage });
  } finally {
    oauthLoading.value = false;
  }
}

async function completeOAuthManually() {
  const providerName = activeProvider.value?.name;
  const callbackUrl = callbackUrlInput.value.trim();
  if (!providerName || !callbackUrl) return;
  oauthLoading.value = true;
  try {
    oauthStatus.value = await configApi.completeProviderOAuth(providerName, {
      callback_url: callbackUrl,
    });
    pendingAuthUrl.value = "";
    callbackUrlInput.value = "";
    toast.success(t("common.success"), { description: t("providers.oauthConnectSuccess") });
    emit("saved");
  } catch (e: unknown) {
    const errorMessage = e instanceof Error ? e.message : t("providers.oauthConnectFailed");
    toast.error(t("common.error"), { description: errorMessage });
  } finally {
    oauthLoading.value = false;
  }
}

async function refreshOAuthTokens() {
  const providerName = activeProvider.value?.name;
  if (!providerName) return;
  oauthLoading.value = true;
  try {
    oauthStatus.value = await configApi.refreshProviderOAuth(providerName);
    toast.success(t("common.success"), { description: t("providers.oauthRefreshSuccess") });
    emit("saved");
  } catch (e: unknown) {
    const errorMessage = e instanceof Error ? e.message : t("providers.oauthRefreshFailed");
    toast.error(t("common.error"), { description: errorMessage });
  } finally {
    oauthLoading.value = false;
  }
}

async function disconnectOAuth() {
  const providerName = activeProvider.value?.name;
  if (!providerName) return;
  oauthLoading.value = true;
  try {
    await configApi.disconnectProviderOAuth(providerName);
    oauthStatus.value = {
      connected: false,
      has_plan_usage_scope: false,
      has_refresh_token: false,
    };
    pendingAuthUrl.value = "";
    callbackUrlInput.value = "";
    toast.success(t("common.success"), { description: t("providers.oauthDisconnectSuccess") });
    emit("saved");
  } catch (e: unknown) {
    const errorMessage = e instanceof Error ? e.message : t("errors.unknown");
    toast.error(t("common.error"), { description: errorMessage });
  } finally {
    oauthLoading.value = false;
  }
}

/**
 * Form-level validator that wraps the Zod v4 schema.
 * Uses safeParse to avoid the Zod v3 _def.defaultValue() incompatibility
 * that @vee-validate/zod encounters with Zod v4.
 */
function validateProviderForm(values: Record<string, unknown>): true | Record<string, string> {
  const result = providerFormSchema.safeParse(values);
  if (result.success) return true;
  const errors: Record<string, string> = {};
  for (const issue of result.error.issues) {
    const path = issue.path.join(".");
    if (path && !errors[path]) {
      errors[path] = issue.message;
    }
  }
  return errors;
}

const { defineField, handleSubmit, resetForm, errors, setFieldValue } = useForm<ProviderFormValues>(
  {
    validationSchema: validateProviderForm,
    initialValues: {
      name: "",
      type: "openai",
      api_key: "",
      base_url: "",
      icon_url: null,
    },
  }
);

const [name, nameAttrs] = defineField("name");
const [type] = defineField("type");
const [apiKey, apiKeyAttrs] = defineField("api_key");
const [baseUrl, baseUrlAttrs] = defineField("base_url");
const [iconUrl, iconUrlAttrs] = defineField("icon_url");
const [nativeWebSearchEnabled] = defineField("native_web_search");

/**
 * Preview of the mark the provider row will show: the URL being typed, else
 * whatever the type/name resolve to in the icon catalog.
 */
const iconPreview = computed(() =>
  providerMark({ type: type.value ?? "", name: name.value ?? "", icon_url: iconUrl.value })
);

/**
 * Gemini upstream API dialect switch (provider_metadata.api_variant).
 * ON = Google's GA Interactions API, OFF = legacy generateContent (default).
 * See docs/adr/0010-gemini-interactions-variant.md.
 */
const geminiInteractions = ref(false);

/**
 * Type selector options from the backend catalog.
 *
 * When editing a provider whose type is no longer in the catalog (e.g. its
 * adapter was removed from the backend), the current type is appended so the
 * form still renders and can be saved unchanged.
 */
const providerTypes = computed(() => {
  const options = [...typeOptions.value];
  const current = activeProvider.value?.type;
  if (current && !options.some((o) => o.value === current)) {
    options.push({ label: current, value: current });
  }
  return options;
});

const customHeaders = ref<{ key: string; value: string }[]>([]);
type EndpointType = "embeddings" | "chat_completion";
const endpointTypes = computed<{ label: string; value: EndpointType }[]>(() => [
  { label: t("providers.endpointEmbeddings"), value: "embeddings" },
  { label: t("providers.endpointChatCompletion"), value: "chat_completion" },
]);
const endpointBaseUrls = ref<{ type: EndpointType; url: string }[]>([]);

/**
 * App-attribution identity (OpenRouter's HTTP-Referer / X-OpenRouter-Title).
 *
 * Held as plain refs rather than vee-validate fields, mirroring
 * ``customHeaders``: the values map onto a nested request object, and the
 * backend applies its own defaults when a field is left empty. Categories are
 * edited as a comma-separated string because OpenRouter accepts at most two
 * per request and the header form is itself comma-separated.
 */
type AppAttributionVisibility = NonNullable<AppAttribution["visibility"]>;
type AppAttributionField = "url" | "title" | "categories";

const appAttributionUrl = ref("");
const appAttributionTitle = ref("");
const appAttributionCategories = ref("");
const appAttributionVisibility = ref<AppAttributionVisibility | null>(null);
const appAttributionErrors = ref<Partial<Record<AppAttributionField, string>>>({});

/**
 * HTTP header values accept printable ASCII only; a non-ASCII or control
 * character would make httpx reject every request through the provider.
 */
const HEADER_VALUE_UNSAFE = /[^\x20-\x7e]/;

function parseAppAttributionCategories(): string[] {
  return appAttributionCategories.value
    .split(",")
    .map((category) => category.trim())
    .filter(Boolean);
}

function buildAppAttribution(): AppAttribution {
  return {
    url: appAttributionUrl.value.trim() || null,
    title: appAttributionTitle.value.trim() || null,
    // OpenRouter accepts at most two categories; extras are surfaced as a
    // validation error before submit and never sent.
    categories: parseAppAttributionCategories().slice(0, 2),
    visibility: appAttributionVisibility.value,
  };
}

/**
 * Validate the attribution fields before submit. The values are sent verbatim
 * as HTTP headers, so malformed URLs and non-ASCII/control characters must be
 * caught here rather than failing the upstream request at runtime.
 */
function validateAppAttribution(): boolean {
  const nextErrors: Partial<Record<AppAttributionField, string>> = {};
  const url = appAttributionUrl.value.trim();
  if (url) {
    try {
      const parsed = new URL(url);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        nextErrors.url = t("providers.appAttributionUrlInvalid");
      }
    } catch {
      nextErrors.url = t("providers.appAttributionUrlInvalid");
    }
  }
  if (HEADER_VALUE_UNSAFE.test(url)) {
    nextErrors.url = t("providers.appAttributionHeaderInvalid");
  }
  if (HEADER_VALUE_UNSAFE.test(appAttributionTitle.value.trim())) {
    nextErrors.title = t("providers.appAttributionHeaderInvalid");
  }
  const categories = parseAppAttributionCategories();
  if (categories.length > 2) {
    nextErrors.categories = t("providers.appAttributionCategoriesMax");
  } else if (categories.some((category) => HEADER_VALUE_UNSAFE.test(category))) {
    nextErrors.categories = t("providers.appAttributionHeaderInvalid");
  }
  appAttributionErrors.value = nextErrors;
  return Object.keys(nextErrors).length === 0;
}

function setAppAttributionVisibility(value: unknown): void {
  appAttributionVisibility.value = value === "public" || value === "hidden" ? value : null;
}

const canAddEndpointUrl = computed(
  () => endpointBaseUrls.value.length < endpointTypes.value.length
);

function populateForm(provider: ProviderRead | null) {
  if (provider) {
    setFieldValue("name", provider.name);
    setFieldValue("type", provider.type);
    setFieldValue("api_key", "");
    setFieldValue("base_url", provider.base_url || "");
    setFieldValue("icon_url", provider.icon_url);
    setFieldValue("native_web_search", provider.native_web_search ?? false);
    geminiInteractions.value = provider.provider_metadata?.api_variant === "interactions";

    apiKeyEditing.value = false;
    customHeaders.value = Object.entries(provider.custom_headers || {}).map(([key, value]) => ({
      key,
      value,
    }));
    endpointBaseUrls.value = Object.entries(provider.endpoint_base_urls || {}).map(
      ([type, url]) => ({ type: type as EndpointType, url })
    );
    appAttributionUrl.value = provider.app_attribution?.url ?? "";
    appAttributionTitle.value = provider.app_attribution?.title ?? "";
    appAttributionCategories.value = (provider.app_attribution?.categories ?? []).join(", ");
    appAttributionVisibility.value = provider.app_attribution?.visibility ?? null;
    appAttributionErrors.value = {};
  } else {
    setFieldValue("name", "");
    setFieldValue("type", "openai");
    setFieldValue("api_key", "");
    setFieldValue("base_url", "");
    setFieldValue("icon_url", null);
    setFieldValue("native_web_search", false);
    geminiInteractions.value = false;

    apiKeyEditing.value = false;
    customHeaders.value = [];
    endpointBaseUrls.value = [];
    appAttributionUrl.value = "";
    appAttributionTitle.value = "";
    appAttributionCategories.value = "";
    appAttributionVisibility.value = null;
    appAttributionErrors.value = {};
  }
}

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      savedProvider.value = null;
      resetForm(); // Reset validation state
      populateForm(props.provider ?? null);
      resetOAuthState(props.provider ?? null);
      if (props.provider && props.provider.type === "openai-chatgpt") {
        void loadOAuthStatus(props.provider.name);
      }
      // Warm the provider-type catalog so the type selector is populated
      // (cached after the first fetch; failure surfaces an empty selector
      // rather than stale data).
      void ensureLoaded();
    } else {
      savedProvider.value = null;
      resetOAuthState(null);
    }
  }
);

const close = () => {
  savedProvider.value = null;
  resetOAuthState(null);
  resetForm();
  emit("update:open", false);
};

/**
 * Build the endpoint_base_urls dict from the editable URL rows.
 */
function buildEndpointBaseUrls(): Record<string, string> {
  return endpointBaseUrls.value.reduce(
    (acc, curr) => {
      if (curr.type && curr.url) acc[curr.type] = curr.url;
      return acc;
    },
    {} as Record<string, string>
  );
}

/**
 * Build the full provider_metadata dict for the payload. Only called for
 * Gemini providers (the caller gates on the provider type).
 *
 * The backend stores parameter_overrides / endpoint_base_urls /
 * native_web_search INSIDE provider_metadata, and the update endpoint
 * replaces the whole dict when the field is present — so the payload must
 * carry every key, not just api_variant, or unrelated metadata would be
 * wiped. Unknown keys set via the API are preserved from the read model.
 */
function buildProviderMetadata(values: ProviderFormValues): Record<string, unknown> {
  const metadata: Record<string, unknown> = {
    ...(activeProvider.value?.provider_metadata || {}),
  };
  if (activeProvider.value?.parameter_overrides) {
    metadata.parameter_overrides = activeProvider.value.parameter_overrides;
  }
  const endpointUrls = buildEndpointBaseUrls();
  if (Object.keys(endpointUrls).length > 0) {
    metadata.endpoint_base_urls = endpointUrls;
  }
  metadata.native_web_search = values.native_web_search ?? false;
  if (geminiInteractions.value) {
    metadata.api_variant = "interactions";
  } else {
    delete metadata.api_variant;
  }
  return metadata;
}

const onSubmit = handleSubmit(async (values) => {
  if (!validateAppAttribution()) {
    toast.error(t("common.error"), {
      description: t("providers.appAttributionInvalid"),
    });
    return;
  }
  isLoading.value = true;
  try {
    const providerData: ProviderCreate = {
      name: values.name,
      type: values.type,
      api_key: values.api_key || "",
      base_url: values.base_url || "",
      icon_url: values.icon_url ?? null,
      native_web_search: values.native_web_search ?? false,
    };

    // Process custom headers
    providerData.custom_headers = customHeaders.value.reduce(
      (acc, curr) => {
        if (curr.key) acc[curr.key] = curr.value;
        return acc;
      },
      {} as Record<string, string>
    );

    // Process endpoint base URLs
    providerData.endpoint_base_urls = buildEndpointBaseUrls();

    // App-attribution identity; unset fields fall back to the adapter defaults.
    providerData.app_attribution = buildAppAttribution();

    // Gemini API dialect switch; other provider types leave metadata untouched.
    if (values.type === "gemini") {
      providerData.provider_metadata = buildProviderMetadata(values);
    }

    if (isEditing.value && activeProvider.value) {
      const updateData: ProviderUpdate = {
        type: providerData.type,
        base_url: providerData.base_url,
        custom_headers: providerData.custom_headers,
        endpoint_base_urls: providerData.endpoint_base_urls,
        app_attribution: providerData.app_attribution,
        icon_url: providerData.icon_url,
        native_web_search: providerData.native_web_search,
      };
      if (providerData.provider_metadata) {
        updateData.provider_metadata = providerData.provider_metadata;
      }
      if (providerData.api_key) {
        updateData.api_key = providerData.api_key;
      }
      await configApi.updateProvider(activeProvider.value.name, updateData);
      toast.success(t("common.success"), {
        description: t("providers.updateSuccess"),
      });
      emit("saved");
      close();
    } else {
      const created = await configApi.createProvider(providerData);
      toast.success(t("common.success"), {
        description: t("providers.createSuccess"),
      });
      emit("saved");
      if (values.type === "openai-chatgpt") {
        // Keep the sheet open after creation so the operator can complete ChatGPT OAuth immediately.
        savedProvider.value = created;
        resetOAuthState(created);
        void loadOAuthStatus(created.name);
      } else {
        close();
      }
    }
  } catch (e: unknown) {
    const errorMessage = e instanceof Error ? e.message : t("errors.unknown");
    toast.error(t("common.error"), { description: errorMessage });
  } finally {
    isLoading.value = false;
  }
});
</script>

<template>
  <Sheet :open="open" @update:open="(val) => emit('update:open', val)">
    <SheetContent side="right" size="lg">
      <SheetHeader :icon="Server">
        <template #title>
          {{ isEditing ? t("providers.editProvider") : t("providers.addProvider") }}
        </template>
        <template #description>
          {{
            isEditing
              ? t("providers.editProviderDescription")
              : t("providers.addProviderDescription")
          }}
        </template>
      </SheetHeader>

      <form @submit="onSubmit" class="flex-1 flex flex-col min-h-0">
        <Tabs default-value="general" class="flex-1 flex flex-col min-h-0 w-full">
          <div class="px-4 sm:px-6 pt-4 shrink-0">
            <TabsList class="grid w-full grid-cols-2 h-auto">
              <TabsTrigger value="general">{{ t("common.general") }}</TabsTrigger>
              <TabsTrigger value="advanced">{{ t("common.advanced") }}</TabsTrigger>
            </TabsList>
          </div>

          <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4">
            <TabsContent value="general" class="space-y-4 mt-0">
              <div class="grid gap-2">
                <Label for="name" :class="{ 'text-destructive': errors.name }"
                  >{{ t("providers.name") }} <span class="text-destructive">*</span></Label
                >
                <Input
                  id="name"
                  v-model="name"
                  v-bind="nameAttrs"
                  :placeholder="t('placeholders.providerName')"
                  :disabled="isEditing"
                  maxlength="255"
                  class="min-w-0"
                  :aria-invalid="!!errors.name"
                  :aria-describedby="errors.name ? 'provider-name-error' : undefined"
                />
                <p
                  v-if="errors.name"
                  id="provider-name-error"
                  role="alert"
                  class="text-sm text-destructive mt-1 wrap-break-word"
                >
                  {{ errors.name }}
                </p>
              </div>
              <div class="grid gap-2">
                <Label for="type"
                  >{{ t("providers.type") }} <span class="text-destructive">*</span></Label
                >
                <Select v-model="type">
                  <SelectTrigger
                    id="type"
                    :aria-invalid="!!errors.type"
                    :aria-describedby="errors.type ? 'provider-type-error' : undefined"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem
                      v-for="typeOption in providerTypes"
                      :key="typeOption.value"
                      :value="typeOption.value"
                    >
                      {{ typeOption.label }}
                    </SelectItem>
                  </SelectContent>
                </Select>
                <p
                  v-if="errors.type"
                  id="provider-type-error"
                  role="alert"
                  class="text-sm text-destructive mt-1 wrap-break-word"
                >
                  {{ errors.type }}
                </p>
              </div>
              <div v-if="type === 'openai-chatgpt'" class="grid gap-3 rounded-lg border border-border/60 p-3.5 bg-muted/20">
                <div class="flex items-start justify-between gap-2">
                  <div>
                    <Label class="flex items-center gap-1.5">
                      <KeyRound class="w-3.5 h-3.5" />
                      {{ t("providers.oauthTitle") }}
                    </Label>
                    <p class="text-[11px] text-muted-foreground mt-0.5">
                      {{ t("providers.oauthDescription") }}
                    </p>
                  </div>
                  <Badge
                    v-if="isEditing"
                    :variant="oauthStatus?.connected ? 'secondary' : 'outline'"
                    class="text-[11px] shrink-0"
                  >
                    {{ oauthStatus?.connected ? t("providers.oauthConnected") : t("providers.oauthNotConnected") }}
                  </Badge>
                </div>

                <p v-if="!isEditing" class="text-xs text-muted-foreground">
                  {{ t("providers.oauthSaveFirst") }}
                </p>

                <template v-else>
                  <div
                    v-if="oauthStatus?.connected"
                    class="space-y-1.5 rounded-md border border-border/50 bg-background/60 p-2.5 text-xs"
                  >
                    <div v-if="oauthStatus.account_email || oauthStatus.account_name" class="flex items-center justify-between gap-2">
                      <span class="text-muted-foreground">{{ t("providers.oauthAccount") }}</span>
                      <span class="font-mono truncate">{{ oauthStatus.account_email || oauthStatus.account_name }}</span>
                    </div>
                    <div class="flex items-center justify-between gap-2">
                      <span class="text-muted-foreground">{{ t("providers.oauthPlanScope") }}</span>
                      <Badge variant="outline" class="font-mono text-[10px]">
                        {{ oauthStatus.has_plan_usage_scope ? "chatgpt.tokens.use.direct" : t("providers.oauthScopeMissing") }}
                      </Badge>
                    </div>
                    <div v-if="oauthStatus.client_id" class="flex items-center justify-between gap-2">
                      <span class="text-muted-foreground">{{ t("providers.oauthClientId") }}</span>
                      <span class="font-mono text-[11px] truncate max-w-[220px]">{{ oauthStatus.client_id }}</span>
                    </div>
                    <div v-if="oauthStatus.expires_at" class="flex items-center justify-between gap-2">
                      <span class="text-muted-foreground">{{ t("providers.oauthExpiresAt") }}</span>
                      <span class="font-mono text-[11px]">{{ new Date(oauthStatus.expires_at).toLocaleString() }}</span>
                    </div>
                  </div>

                  <div class="flex flex-wrap items-center gap-2">
                    <Button
                      type="button"
                      size="sm"
                      :disabled="oauthLoading"
                      @click="startOAuthFlow(false)"
                    >
                      <ExternalLink class="w-3.5 h-3.5" />
                      {{ oauthStatus?.connected ? t("providers.oauthReauthorize") : t("providers.oauthSignIn") }}
                    </Button>
                    <Button
                      v-if="oauthStatus?.connected && oauthStatus?.has_refresh_token"
                      type="button"
                      variant="outline"
                      size="sm"
                      :disabled="oauthLoading"
                      @click="refreshOAuthTokens"
                    >
                      <RefreshCw class="w-3.5 h-3.5" />
                      {{ t("providers.oauthRefresh") }}
                    </Button>
                    <Button
                      v-if="oauthStatus?.connected"
                      type="button"
                      variant="ghost"
                      size="sm"
                      class="text-destructive hover:text-destructive hover:bg-destructive/10"
                      :disabled="oauthLoading"
                      @click="disconnectOAuth"
                    >
                      <Unlink class="w-3.5 h-3.5" />
                      {{ t("providers.oauthDisconnect") }}
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      class="ml-auto text-xs"
                      @click="showOAuthOptions = !showOAuthOptions"
                    >
                      {{ t("providers.oauthOptions") }}
                    </Button>
                  </div>

                  <div v-if="showOAuthOptions" class="space-y-2.5 pt-2 border-t border-border/50">
                    <div class="grid gap-1">
                      <Label for="oauth-redirect-uri" class="text-xs">{{ t("providers.oauthRedirectUri") }}</Label>
                      <Input
                        id="oauth-redirect-uri"
                        v-model="oauthRedirectUri"
                        type="url"
                        class="h-8 text-xs font-mono"
                      />
                      <p class="text-[11px] text-muted-foreground">
                        {{ t("providers.oauthRedirectUriHelp") }}
                      </p>
                    </div>
                    <div class="grid gap-1">
                      <Label for="oauth-client-id" class="text-xs">{{ t("providers.oauthStaticClientId") }}</Label>
                      <Input
                        id="oauth-client-id"
                        v-model="oauthClientId"
                        placeholder="dynamic_agent_client"
                        class="h-8 text-xs font-mono"
                      />
                    </div>
                    <div class="grid gap-1">
                      <Label for="oauth-client-secret" class="text-xs">{{ t("providers.oauthClientSecret") }}</Label>
                      <Input
                        id="oauth-client-secret"
                        v-model="oauthClientSecret"
                        type="password"
                        class="h-8 text-xs font-mono"
                      />
                    </div>
                  </div>

                  <div v-if="pendingAuthUrl" class="space-y-2 pt-2 border-t border-border/50">
                    <div class="flex items-center justify-between gap-2 text-xs">
                      <span class="text-muted-foreground">{{ t("providers.oauthPendingHint") }}</span>
                      <a
                        :href="pendingAuthUrl"
                        target="_blank"
                        rel="noreferrer"
                        class="inline-flex items-center gap-1 font-medium underline underline-offset-4"
                      >
                        {{ t("providers.oauthOpenLink") }}
                        <ExternalLink class="w-3 h-3" />
                      </a>
                    </div>
                    <div class="grid gap-1.5">
                      <Label for="oauth-callback-url" class="text-xs">{{ t("providers.oauthCallbackPaste") }}</Label>
                      <div class="flex items-center gap-2">
                        <Input
                          id="oauth-callback-url"
                          v-model="callbackUrlInput"
                          placeholder="http://127.0.0.1:1455/auth/callback?code=..."
                          class="flex-1 h-8 text-xs font-mono"
                        />
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          :disabled="oauthLoading || !callbackUrlInput.trim()"
                          @click="completeOAuthManually"
                        >
                          {{ t("providers.oauthCompleteManual") }}
                        </Button>
                      </div>
                    </div>
                  </div>
                </template>
              </div>
              <div v-else class="grid gap-2">
                <Label for="apikey">{{ t("providers.apiKey") }}</Label>
                <div v-if="isEditing && !apiKeyEditing" class="flex items-center gap-2">
                  <!-- The backend masks the key as `sk-...abcd` on purpose: the
                       visible head and tail are how an operator tells one key
                       from another. A password field hid exactly that. -->
                  <Input
                    id="apikey"
                    :value="activeProvider?.masked_api_key || t('providers.apiKeyUnknown')"
                    type="text"
                    readonly
                    class="flex-1 bg-muted/50 font-mono"
                  />
                  <Button variant="outline" @click="apiKeyEditing = true">
                    {{ t("common.edit") }}
                  </Button>
                </div>
                <Input
                  v-else
                  id="apikey"
                  v-model="apiKey"
                  v-bind="apiKeyAttrs"
                  type="password"
                  :placeholder="isEditing ? t('placeholders.apiKeyOptional') : ''"
                  :disabled="isEditing && !apiKeyEditing"
                  class="w-full"
                  :aria-invalid="!!errors.api_key"
                  :aria-describedby="errors.api_key ? 'provider-api-key-error' : undefined"
                />
                <p
                  v-if="errors.api_key"
                  id="provider-api-key-error"
                  role="alert"
                  class="text-sm text-destructive mt-1"
                >
                  {{ errors.api_key }}
                </p>
                <p v-else-if="isEditing" class="text-[11px] text-muted-foreground">
                  {{ t("providers.apiKeyEditHelp") }}
                </p>
              </div>
              <div class="grid gap-2">
                <Label for="baseurl">{{ t("providers.baseUrl") }}</Label>
                <Input
                  id="baseurl"
                  v-model="baseUrl"
                  v-bind="baseUrlAttrs"
                  :placeholder="t('placeholders.baseUrl')"
                  maxlength="2048"
                  class="min-w-0"
                  type="url"
                  :aria-invalid="!!errors.base_url"
                  :aria-describedby="errors.base_url ? 'provider-base-url-error' : undefined"
                />
                <p
                  v-if="errors.base_url"
                  id="provider-base-url-error"
                  role="alert"
                  class="text-sm text-destructive mt-1 wrap-break-word"
                >
                  {{ errors.base_url }}
                </p>
                <p v-else class="text-[11px] text-muted-foreground wrap-break-word">
                  {{ t("providers.baseUrlHelp") }}
                </p>
              </div>
              <div class="grid gap-2">
                <div class="flex items-center justify-between gap-3">
                  <Label for="iconurl">{{ t("models.iconUrl") }}</Label>
                  <BrandMark :icon="iconPreview" :label="name || type || 'Provider'" size="lg" />
                </div>
                <Input
                  id="iconurl"
                  v-model="iconUrl"
                  v-bind="iconUrlAttrs"
                  :placeholder="t('models.iconUrlPlaceholder')"
                  maxlength="2048"
                  class="min-w-0"
                  type="url"
                  :aria-invalid="!!errors.icon_url"
                  :aria-describedby="errors.icon_url ? 'provider-icon-url-error' : undefined"
                />
                <p
                  v-if="errors.icon_url"
                  id="provider-icon-url-error"
                  role="alert"
                  class="text-sm text-destructive mt-1 wrap-break-word"
                >
                  {{ errors.icon_url }}
                </p>
                <p v-else class="text-[11px] text-muted-foreground wrap-break-word">
                  {{ t("providers.iconUrlHelp") }}
                </p>
              </div>
            </TabsContent>

            <TabsContent value="advanced" class="space-y-4 mt-0">
              <div v-if="type === 'gemini'" class="grid gap-2">
                <div class="flex items-center justify-between">
                  <div>
                    <Label for="provider-gemini-interactions">{{
                      t("providers.geminiInteractions")
                    }}</Label>
                    <p class="text-[11px] text-muted-foreground">
                      {{ t("providers.geminiInteractionsHelp") }}
                    </p>
                  </div>
                  <Switch id="provider-gemini-interactions" v-model="geminiInteractions" />
                </div>
              </div>
              <div
                v-if="type === 'openrouter'"
                class="grid gap-3 rounded-lg border border-border/60 p-3"
              >
                <div>
                  <Label>{{ t("providers.appAttribution") }}</Label>
                  <p class="text-[11px] text-muted-foreground">
                    {{ t("providers.appAttributionHelp") }}
                  </p>
                </div>
                <div class="grid gap-2">
                  <Label for="app-attribution-url" class="text-xs">{{
                    t("providers.appAttributionUrl")
                  }}</Label>
                  <Input
                    id="app-attribution-url"
                    v-model="appAttributionUrl"
                    type="url"
                    maxlength="2048"
                    :placeholder="t('providers.appAttributionUrlPlaceholder')"
                    class="h-8 text-xs font-mono"
                    :aria-invalid="!!appAttributionErrors.url"
                    :aria-describedby="
                      appAttributionErrors.url ? 'app-attribution-url-error' : undefined
                    "
                  />
                  <p
                    v-if="appAttributionErrors.url"
                    id="app-attribution-url-error"
                    role="alert"
                    class="text-sm text-destructive"
                  >
                    {{ appAttributionErrors.url }}
                  </p>
                  <p v-else class="text-[11px] text-muted-foreground">
                    {{ t("providers.appAttributionUrlHelp") }}
                  </p>
                </div>
                <div class="grid gap-2">
                  <Label for="app-attribution-title" class="text-xs">{{
                    t("providers.appAttributionTitle")
                  }}</Label>
                  <Input
                    id="app-attribution-title"
                    v-model="appAttributionTitle"
                    maxlength="128"
                    :placeholder="t('providers.appAttributionTitlePlaceholder')"
                    class="h-8 text-xs"
                    :aria-invalid="!!appAttributionErrors.title"
                    :aria-describedby="
                      appAttributionErrors.title ? 'app-attribution-title-error' : undefined
                    "
                  />
                  <p
                    v-if="appAttributionErrors.title"
                    id="app-attribution-title-error"
                    role="alert"
                    class="text-sm text-destructive"
                  >
                    {{ appAttributionErrors.title }}
                  </p>
                </div>
                <div class="grid gap-2">
                  <Label for="app-attribution-categories" class="text-xs">{{
                    t("providers.appAttributionCategories")
                  }}</Label>
                  <Input
                    id="app-attribution-categories"
                    v-model="appAttributionCategories"
                    maxlength="256"
                    :placeholder="t('providers.appAttributionCategoriesPlaceholder')"
                    class="h-8 text-xs font-mono"
                    :aria-invalid="!!appAttributionErrors.categories"
                    :aria-describedby="
                      appAttributionErrors.categories
                        ? 'app-attribution-categories-error'
                        : undefined
                    "
                  />
                  <p
                    v-if="appAttributionErrors.categories"
                    id="app-attribution-categories-error"
                    role="alert"
                    class="text-sm text-destructive"
                  >
                    {{ appAttributionErrors.categories }}
                  </p>
                </div>
                <div class="grid gap-2">
                  <Label for="app-attribution-visibility" class="text-xs">{{
                    t("providers.appAttributionVisibility")
                  }}</Label>
                  <Select
                    :model-value="appAttributionVisibility ?? 'default'"
                    @update:model-value="setAppAttributionVisibility"
                  >
                    <SelectTrigger id="app-attribution-visibility" class="h-8 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">
                        {{ t("providers.appAttributionVisibilityDefault") }}
                      </SelectItem>
                      <SelectItem value="public">
                        {{ t("providers.appAttributionVisibilityPublic") }}
                      </SelectItem>
                      <SelectItem value="hidden">
                        {{ t("providers.appAttributionVisibilityHidden") }}
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div class="grid gap-2">
                <div class="flex items-center justify-between">
                  <div>
                    <Label for="provider-native-web-search">{{
                      t("providers.nativeWebSearch")
                    }}</Label>
                    <p class="text-[11px] text-muted-foreground">
                      {{ t("providers.nativeWebSearchHelp") }}
                    </p>
                  </div>
                  <Switch
                    id="provider-native-web-search"
                    :model-value="nativeWebSearchEnabled"
                    @update:model-value="(val: boolean) => setFieldValue('native_web_search', val)"
                  />
                </div>
              </div>
              <div class="grid gap-2">
                <Label>{{ t("providers.customHeaders") }}</Label>
                <div class="space-y-2">
                  <div
                    v-for="(header, index) in customHeaders"
                    :key="index"
                    class="flex items-center gap-2"
                  >
                    <Input
                      v-model="header.key"
                      :placeholder="t('labels.key')"
                      class="flex-1 h-8 text-xs font-mono"
                    />
                    <Input
                      v-model="header.value"
                      :placeholder="t('labels.value')"
                      class="flex-1 h-8 text-xs font-mono"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      class="h-10 w-10"
                      :aria-label="`${t('common.delete')} ${header.key || index + 1}`"
                      :title="t('common.delete')"
                      @click="customHeaders.splice(index, 1)"
                    >
                      <Trash2 class="w-4 h-4 icon-btn-muted hover:text-destructive" />
                    </Button>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    class="w-full border-dashed text-xs h-8"
                    @click="customHeaders.push({ key: '', value: '' })"
                  >
                    <Plus class="w-3 h-3" />
                    {{ t("common.add") }}
                  </Button>
                </div>
              </div>
              <div class="grid gap-2">
                <Label>{{ t("providers.endpointBaseUrls") }}</Label>
                <div class="space-y-2">
                  <div
                    v-for="(endpoint, index) in endpointBaseUrls"
                    :key="index"
                    class="flex items-center gap-2"
                  >
                    <Select v-model="endpoint.type">
                      <SelectTrigger class="h-8 text-xs w-32">
                        <SelectValue :placeholder="t('placeholders.selectEndpointType')" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem v-for="et in endpointTypes" :key="et.value" :value="et.value">
                          {{ et.label }}
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <Input
                      v-model="endpoint.url"
                      :placeholder="t('placeholders.endpointUrl')"
                      class="flex-1 h-8 text-xs font-mono"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      class="h-10 w-10"
                      :aria-label="`${t('common.delete')} ${endpoint.url || index + 1}`"
                      :title="t('common.delete')"
                      @click="endpointBaseUrls.splice(index, 1)"
                    >
                      <Trash2 class="w-4 h-4 icon-btn-muted hover:text-destructive" />
                    </Button>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    class="w-full border-dashed text-xs h-8"
                    :disabled="!canAddEndpointUrl"
                    @click="endpointBaseUrls.push({ type: 'embeddings', url: '' })"
                  >
                    <Plus class="w-3 h-3" />
                    {{ t("common.add") }}
                  </Button>
                </div>
                <p class="text-[11px] text-muted-foreground">
                  {{ t("providers.endpointBaseUrlsHelp") }}
                </p>
              </div>
            </TabsContent>
          </div>
        </Tabs>

        <SheetFooter>
          <Button variant="outline" type="button" @click="close">{{ t("common.cancel") }}</Button>
          <Button type="submit" :disabled="isLoading">{{ t("common.save") }}</Button>
        </SheetFooter>
      </form>
    </SheetContent>
  </Sheet>
</template>
