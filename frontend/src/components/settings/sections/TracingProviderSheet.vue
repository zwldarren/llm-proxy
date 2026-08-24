<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import SheetHeader from "@/components/ui/sheet/SheetHeader.vue";
import SheetTitle from "@/components/ui/sheet/SheetTitle.vue";
import SheetDescription from "@/components/ui/sheet/SheetDescription.vue";
import { type TracingProviderEditor } from "@/composables/useTracingProviders";
import type { TracingProvider } from "@/types/schemas";

/**
 * Focused side sheet for editing one tracing provider's connection details.
 * Kept out of the settings card so the page stays compact no matter how many
 * providers are configured.
 */
defineProps<{
  provider: TracingProvider | null;
  editor: TracingProviderEditor;
}>();

const emit = defineEmits<{
  (e: "close"): void;
}>();

const { t } = useI18n();
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

        <!-- Base URL -->
        <div class="space-y-1.5">
          <Label class="text-xs font-semibold text-muted-foreground">
            {{ t("tracing.baseUrl") }}
          </Label>
          <Input
            :model-value="(editor.getProviderFieldValue(provider, 'base_url') as string) || ''"
            :placeholder="t('tracing.baseUrlPlaceholder')"
            class="font-mono text-sm w-full bg-background"
            @update:model-value="
              provider.id && editor.updateProviderField(provider.id!, 'base_url', String($event))
            "
          />
          <p class="text-xs text-muted-foreground">{{ t("tracing.baseUrlHelp") }}</p>
        </div>

        <!-- Public + Secret keys -->
        <div class="grid grid-cols-1 gap-4">
          <div class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">
              {{ t("tracing.publicKey") }}
              <span class="text-destructive">*</span>
            </Label>
            <Input
              :model-value="(editor.getProviderFieldValue(provider, 'public_key') as string) || ''"
              :placeholder="t('tracing.publicKeyPlaceholder')"
              class="font-mono text-sm w-full bg-background"
              @update:model-value="
                provider.id &&
                editor.updateProviderField(provider.id!, 'public_key', String($event))
              "
            />
          </div>
          <div class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">
              {{ t("tracing.secretKey") }}
              <span class="text-destructive">*</span>
            </Label>
            <Input
              :model-value="(editor.getProviderFieldValue(provider, 'secret_key') as string) || ''"
              type="password"
              :placeholder="t('tracing.secretKeyPlaceholder')"
              class="font-mono text-sm w-full bg-background"
              @update:model-value="
                provider.id &&
                editor.updateProviderField(provider.id!, 'secret_key', String($event))
              "
            />
          </div>
        </div>

        <!-- Timeout -->
        <div class="space-y-1.5">
          <Label class="text-xs font-semibold text-muted-foreground">
            {{ t("tracing.timeout") }}
          </Label>
          <!-- Width lives on the wrapper so NumberFieldRoot (which hosts the
               absolutely-positioned stepper buttons) sizes with the input.
               Passing w-32 to NumberInput only shrinks the inner input while
               the root stays full-width, leaving the chevrons detached. -->
          <div class="w-32">
            <NumberInput
              :model-value="
                (editor.getProviderFieldValue(provider, 'timeout') as number | null) || null
              "
              :min="1"
              :placeholder="t('tracing.timeoutPlaceholder')"
              class="font-mono text-sm w-full bg-background"
              @update:model-value="
                provider.id &&
                editor.updateProviderField(
                  provider.id!,
                  'timeout',
                  $event == null ? null : Number($event)
                )
              "
            />
          </div>
        </div>
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
