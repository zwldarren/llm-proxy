<script setup lang="ts">
import { Plus, Trash2 } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import SheetHeader from "@/components/ui/sheet/SheetHeader.vue";
import SheetTitle from "@/components/ui/sheet/SheetTitle.vue";
import SheetDescription from "@/components/ui/sheet/SheetDescription.vue";
import { Switch } from "@/components/ui/switch";
import { type TracingProviderEditor } from "@/composables/useTracingProviders";
import type { TracingProvider, TracingProviderField } from "@/types/schemas";

/**
 * Focused side sheet for editing one tracing provider's connection details.
 * Kept out of the settings card so the page stays compact no matter how many
 * providers are configured.
 *
 * The form is rendered from the provider's field metadata (`/api/me/tracing/
 * providers`), not from a per-provider template: a new backend becomes
 * configurable here by declaring its fields, and a provider a user saved with
 * an older schema still shows whatever its metadata declares today.
 */
const props = defineProps<{
  provider: TracingProvider | null;
  editor: TracingProviderEditor;
}>();

const emit = defineEmits<{
  (e: "close"): void;
}>();

const { t, te } = useI18n();

const providerType = computed(() =>
  props.editor.providerTypes.value.find((type) => type.name === props.provider?.provider)
);

const fields = computed<TracingProviderField[]>(() => providerType.value?.fields ?? []);

/**
 * Field labels/help/choice labels live in i18n, with the provider-specific entry
 * winning over the shared one (`endpoint` means a collector for otel and the
 * LangSmith gateway for langsmith). The backend's English description is the
 * last fallback, so a provider added without translations still renders.
 */
function translated(field: TracingProviderField, kind: string): string | null {
  const provider = props.provider?.provider;
  const candidates: string[] = [];
  if (provider) candidates.push(`tracing.fields.${provider}.${field.name}.${kind}`);
  candidates.push(`tracing.fields.${field.name}.${kind}`);
  const key = candidates.find((candidate) => te(candidate));
  return key ? t(key) : null;
}

function fieldLabel(field: TracingProviderField): string {
  return translated(field, "label") ?? humanize(field.name);
}

function fieldHelp(field: TracingProviderField): string {
  return translated(field, "help") ?? field.description ?? "";
}

function choiceLabel(field: TracingProviderField, choice: string): string {
  return translated(field, `choices.${choice}`) ?? choice;
}

function humanize(name: string): string {
  const words = name.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * The value shown for a field: what the user saved, or the provider's declared
 * default when nothing is stored (e.g. `send_content` = `full`).
 */
function fieldValue(provider: TracingProvider, field: TracingProviderField): unknown {
  const value = props.editor.getProviderFieldValue(provider, field.name);
  if (value === undefined || value === null || value === "") return field.default;
  return value;
}

function setField(provider: TracingProvider, field: TracingProviderField, value: unknown): void {
  if (!provider.id) return;
  props.editor.updateProviderField(provider.id, field.name, value);
}

// ── headers field ───────────────────────────────────────────────────────
// A header map is edited as ordered key/value rows. Rows are projected back
// into the settings object on every edit, so no draft state can drift from
// what will be saved.

function headerEntries(provider: TracingProvider, field: TracingProviderField): [string, string][] {
  const value = props.editor.getProviderFieldValue(provider, field.name);
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  return Object.entries(value as Record<string, unknown>).map(([key, entry]) => [
    key,
    entry == null ? "" : String(entry),
  ]);
}

function setHeaders(
  provider: TracingProvider,
  field: TracingProviderField,
  entries: [string, string][]
): void {
  setField(provider, field, Object.fromEntries(entries));
}

function addHeader(provider: TracingProvider, field: TracingProviderField): void {
  const entries = headerEntries(provider, field);
  const keys = new Set(entries.map(([key]) => key));
  let index = entries.length + 1;
  while (keys.has(`header-${index}`)) index += 1;
  setHeaders(provider, field, [...entries, [`header-${index}`, ""]]);
}

function renameHeader(
  provider: TracingProvider,
  field: TracingProviderField,
  oldKey: string,
  newKey: string
): void {
  // A cleared key keeps its row (so the user can type a new one); the backend
  // drops keyless entries when it builds the request headers.
  setHeaders(
    provider,
    field,
    headerEntries(provider, field).map(([key, value]): [string, string] =>
      key === oldKey ? [newKey.trim(), value] : [key, value]
    )
  );
}

function updateHeaderValue(
  provider: TracingProvider,
  field: TracingProviderField,
  key: string,
  value: string
): void {
  setHeaders(
    provider,
    field,
    headerEntries(provider, field).map(([entryKey, entryValue]): [string, string] =>
      entryKey === key ? [entryKey, value] : [entryKey, entryValue]
    )
  );
}

function removeHeader(provider: TracingProvider, field: TracingProviderField, key: string): void {
  setHeaders(
    provider,
    field,
    headerEntries(provider, field).filter(([entryKey]) => entryKey !== key)
  );
}
</script>

<template>
  <Sheet
    :open="provider !== null"
    @update:open="
      (v) => {
        if (!v) emit('close');
      }
    "
  >
    <SheetContent
      class="w-full sm:max-w-[480px] flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
    >
      <SheetHeader class="px-5 py-4 border-b border-border/60 bg-muted/10 shrink-0 pr-12">
        <SheetTitle class="text-base font-semibold tracking-tight">
          {{ t("tracing.editProvider") }}
        </SheetTitle>
        <SheetDescription class="text-xs leading-normal">
          {{ t("tracing.editProviderDescription") }}
        </SheetDescription>
      </SheetHeader>

      <div v-if="provider" class="flex-1 overflow-y-auto p-5 space-y-4">
        <!-- Name -->
        <div class="space-y-1.5">
          <Label class="text-xs font-semibold text-muted-foreground">
            {{ t("tracing.providerName") }}
          </Label>
          <Input
            :model-value="provider.name"
            :placeholder="t('tracing.providerNamePlaceholder')"
            class="text-sm w-full bg-background"
            @update:model-value="
              provider.id && editor.updateProviderName(provider.id!, String($event))
            "
          />
        </div>

        <!-- Provider fields, in the order the backend declares them -->
        <template v-for="field in fields" :key="field.name">
          <!-- Key/value map (headers) -->
          <div v-if="field.type === 'headers'" class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">
              {{ fieldLabel(field) }}
            </Label>
            <div class="space-y-2">
              <div
                v-for="[key, value] in headerEntries(provider, field)"
                :key="key"
                class="flex items-center gap-2"
              >
                <Input
                  :model-value="key"
                  :aria-label="`${fieldLabel(field)} ${t('tracing.headerKey')}`"
                  class="font-mono text-xs w-2/5 bg-background"
                  @update:model-value="renameHeader(provider!, field, key, String($event))"
                />
                <Input
                  :model-value="value"
                  :aria-label="`${fieldLabel(field)} ${t('tracing.headerValue')}`"
                  type="password"
                  class="font-mono text-xs flex-1 bg-background"
                  @update:model-value="updateHeaderValue(provider!, field, key, String($event))"
                />
                <Button
                  variant="ghost"
                  size="icon"
                  class="size-8 shrink-0 text-muted-foreground hover:text-destructive"
                  :aria-label="t('tracing.removeHeader')"
                  @click="removeHeader(provider!, field, key)"
                >
                  <Trash2 class="size-3.5" />
                </Button>
              </div>
              <Button variant="outline" size="sm" @click="addHeader(provider!, field)">
                <Plus class="size-3.5 mr-1" />
                {{ t("tracing.addHeader") }}
              </Button>
            </div>
            <p v-if="fieldHelp(field)" class="text-xs text-muted-foreground">
              {{ fieldHelp(field) }}
            </p>
          </div>

          <!-- Boolean switch -->
          <div v-else-if="field.type === 'boolean'" class="flex items-center justify-between gap-4">
            <div class="space-y-0.5">
              <Label class="text-xs font-semibold text-muted-foreground">
                {{ fieldLabel(field) }}
              </Label>
              <p v-if="fieldHelp(field)" class="text-xs text-muted-foreground">
                {{ fieldHelp(field) }}
              </p>
            </div>
            <Switch
              :model-value="Boolean(fieldValue(provider, field))"
              :aria-label="fieldLabel(field)"
              @update:model-value="setField(provider!, field, Boolean($event))"
            />
          </div>

          <!-- Select -->
          <div v-else-if="field.type === 'select'" class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">
              {{ fieldLabel(field) }}
            </Label>
            <Select
              :model-value="String(fieldValue(provider, field) ?? '')"
              @update:model-value="setField(provider!, field, $event)"
            >
              <SelectTrigger class="w-full bg-background" :aria-label="fieldLabel(field)">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem v-for="choice in field.choices ?? []" :key="choice" :value="choice">
                  {{ choiceLabel(field, choice) }}
                </SelectItem>
              </SelectContent>
            </Select>
            <p v-if="fieldHelp(field)" class="text-xs text-muted-foreground">
              {{ fieldHelp(field) }}
            </p>
          </div>

          <!-- Number -->
          <div v-else-if="field.type === 'number'" class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">
              {{ fieldLabel(field) }}
            </Label>
            <!-- Width lives on the wrapper so NumberFieldRoot (which hosts the
                 absolutely-positioned stepper buttons) sizes with the input. -->
            <div class="w-32">
              <NumberInput
                :model-value="
                  fieldValue(provider, field) == null ? null : Number(fieldValue(provider, field))
                "
                :min="field.min ?? 0"
                :max="field.max"
                :aria-label="fieldLabel(field)"
                class="text-data w-full bg-background"
                @update:model-value="
                  setField(provider!, field, $event == null ? null : Number($event))
                "
              />
            </div>
            <p v-if="fieldHelp(field)" class="text-xs text-muted-foreground">
              {{ fieldHelp(field) }}
            </p>
          </div>

          <!-- Text / password -->
          <div v-else class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">
              {{ fieldLabel(field) }}
              <span v-if="field.required" class="text-destructive">*</span>
            </Label>
            <Input
              :model-value="String(fieldValue(provider, field) ?? '')"
              :type="field.type === 'password' ? 'password' : 'text'"
              :aria-label="fieldLabel(field)"
              class="font-mono text-sm w-full bg-background"
              @update:model-value="setField(provider!, field, String($event))"
            />
            <p v-if="fieldHelp(field)" class="text-xs text-muted-foreground">
              {{ fieldHelp(field) }}
            </p>
          </div>
        </template>
      </div>

      <div
        class="px-5 py-3.5 border-t border-border/60 shrink-0 flex items-center justify-end gap-3"
      >
        <Button variant="outline" size="sm" @click="emit('close')">
          {{ t("common.close") }}
        </Button>
      </div>
    </SheetContent>
  </Sheet>
</template>
