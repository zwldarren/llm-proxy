<script setup lang="ts">
import { Plus, Server, Trash2 } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useForm } from "vee-validate";
import { toast } from "vue-sonner";
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
import { Sheet, SheetContent } from "@/components/ui/sheet";
import SheetHeaderBand from "@/components/common/SheetHeaderBand.vue";
import { useProviderTypes } from "@/composables/useProviderTypes";
import { configApi } from "@/services/api/config";
import { providerFormSchema, type ProviderFormValues } from "@/schemas/providerForm";
import type { ProviderCreate, ProviderRead, ProviderUpdate } from "@/types/schemas";

const props = defineProps<{
  open: boolean;
  provider?: ProviderRead | null;
}>();

const emit = defineEmits<{
  (e: "update:open", value: boolean): void;
  (e: "saved"): void;
}>();

const { t } = useI18n();
const { typeOptions, ensureLoaded } = useProviderTypes();

const isLoading = ref(false);
const isEditing = computed(() => !!props.provider);
const apiKeyEditing = ref(false);

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
  const current = props.provider?.type;
  if (current && !options.some((o) => o.value === current)) {
    options.push({ label: current, value: current });
  }
  return options;
});

const customHeaders = ref<{ key: string; value: string }[]>([]);
type EndpointType = "embeddings" | "chat_completion";
const endpointTypes: { label: string; value: EndpointType }[] = [
  { label: "Embeddings", value: "embeddings" },
  { label: "Chat Completion", value: "chat_completion" },
];
const endpointBaseUrls = ref<{ type: EndpointType; url: string }[]>([]);

const canAddEndpointUrl = computed(() => endpointBaseUrls.value.length < endpointTypes.length);

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
  }
}

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      resetForm(); // Reset validation state
      populateForm(props.provider ?? null);
      // Warm the provider-type catalog so the type selector is populated
      // (cached after the first fetch; failure surfaces an empty selector
      // rather than stale data).
      void ensureLoaded();
    }
  }
);

const close = () => {
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
    ...(props.provider?.provider_metadata || {}),
  };
  if (props.provider?.parameter_overrides) {
    metadata.parameter_overrides = props.provider.parameter_overrides;
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
  // Validate api_key for create mode
  if (!isEditing.value && !values.api_key?.trim()) {
    toast.error(t("common.error"), {
      description: t("errors.validation.apiKeyRequired"),
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

    // Gemini API dialect switch; other provider types leave metadata untouched.
    if (values.type === "gemini") {
      providerData.provider_metadata = buildProviderMetadata(values);
    }

    if (isEditing.value && props.provider) {
      const updateData: ProviderUpdate = {
        type: providerData.type,
        base_url: providerData.base_url,
        custom_headers: providerData.custom_headers,
        endpoint_base_urls: providerData.endpoint_base_urls,
        icon_url: providerData.icon_url,
        native_web_search: providerData.native_web_search,
      };
      if (providerData.provider_metadata) {
        updateData.provider_metadata = providerData.provider_metadata;
      }
      if (providerData.api_key) {
        updateData.api_key = providerData.api_key;
      }
      await configApi.updateProvider(props.provider.name, updateData);
    } else {
      await configApi.createProvider(providerData);
    }

    toast.success(t("common.success"), {
      description: isEditing.value ? t("providers.updateSuccess") : t("providers.createSuccess"),
    });
    emit("saved");
    close();
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
    <SheetContent
      side="right"
      class="w-full sm:max-w-[520px] lg:max-w-[600px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card transition-colors duration-300 pb-[env(safe-area-inset-bottom\,0px)]"
    >
      <!-- Header band -->
      <SheetHeaderBand :icon="Server">
        <template #title>
          {{ isEditing ? t("providers.editProvider") : t("providers.addProvider") }}
        </template>
        <template #description>
          {{ isEditing ? provider?.name : t("providers.description") }}
        </template>
      </SheetHeaderBand>

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
                />
                <p v-if="errors.name" class="text-sm text-destructive mt-1 wrap-break-word">
                  {{ errors.name }}
                </p>
              </div>
              <div class="grid gap-2">
                <Label for="type"
                  >{{ t("providers.type") }} <span class="text-destructive">*</span></Label
                >
                <Select v-model="type">
                  <SelectTrigger>
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
                <p v-if="errors.type" class="text-sm text-destructive mt-1 wrap-break-word">
                  {{ errors.type }}
                </p>
              </div>
              <div class="grid gap-2">
                <Label for="apikey">{{ t("providers.apiKey") }}</Label>
                <div v-if="isEditing && !apiKeyEditing" class="flex items-center gap-2">
                  <Input value="********" type="password" disabled class="flex-1 bg-muted/50" />
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
                />
                <p v-if="errors.api_key" class="text-sm text-destructive mt-1">
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
                />
                <p v-if="errors.base_url" class="text-sm text-destructive mt-1 wrap-break-word">
                  {{ errors.base_url }}
                </p>
                <p v-else class="text-[11px] text-muted-foreground wrap-break-word">
                  {{ t("providers.baseUrlHelp") }}
                </p>
              </div>
              <div class="grid gap-2">
                <Label for="iconurl">{{ t("models.iconUrl") }}</Label>
                <Input
                  id="iconurl"
                  v-model="iconUrl"
                  v-bind="iconUrlAttrs"
                  :placeholder="t('models.iconUrlPlaceholder')"
                  maxlength="2048"
                  class="min-w-0"
                  type="url"
                />
                <p v-if="errors.icon_url" class="text-sm text-destructive mt-1 wrap-break-word">
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
                    <Label>{{ t("providers.geminiInteractions") }}</Label>
                    <p class="text-[11px] text-muted-foreground">
                      {{ t("providers.geminiInteractionsHelp") }}
                    </p>
                  </div>
                  <Switch v-model="geminiInteractions" />
                </div>
              </div>
              <div class="grid gap-2">
                <div class="flex items-center justify-between">
                  <div>
                    <Label>{{ t("providers.nativeWebSearch") }}</Label>
                    <p class="text-[11px] text-muted-foreground">
                      {{ t("providers.nativeWebSearchHelp") }}
                    </p>
                  </div>
                  <Switch
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
                    <Plus class="w-3 h-3 mr-2" />
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
                    <Plus class="w-3 h-3 mr-2" />
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

        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" type="button" @click="close">{{ t("common.cancel") }}</Button>
          <Button type="submit" :disabled="isLoading">{{ t("common.save") }}</Button>
        </div>
      </form>
    </SheetContent>
  </Sheet>
</template>
