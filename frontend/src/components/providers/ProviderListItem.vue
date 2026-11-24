<script setup lang="ts">
import { Edit, Globe, Trash2 } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import BrandMark from "@/components/common/BrandMark.vue";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useProviderTypes } from "@/composables/useProviderTypes";
import type { ProviderRead } from "@/types/schemas";
import { cn } from "@/lib/utils";

interface Props {
  provider: ProviderRead;
  isLoading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  isLoading: false,
});

const emit = defineEmits<{
  edit: [];
  delete: [];
}>();

const { t } = useI18n();
const { providerMark } = useProviderTypes();

const isEnabled = computed(() => props.provider.enabled !== false);
// Adapter-declared branding, else a catalog match on the type/name.
const icon = computed(() => providerMark(props.provider));
</script>

<template>
  <article
    :class="
      cn(
        'group px-4 sm:px-6 py-2.5 border-b border-border transition-colors duration-150 hover:bg-muted/50',
        !isEnabled && 'opacity-70'
      )
    "
  >
    <div class="flex items-center gap-3">
      <!-- Icon -->
      <BrandMark
        :icon="icon"
        :label="provider.name"
        size="lg"
        :decorative="false"
        :class="cn(!isEnabled && 'grayscale-[0.5]')"
      />

      <!-- Name + type + base url (stacked, matching the other list items) -->
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2 min-w-0">
          <h2 class="text-sm font-medium text-foreground truncate" :title="provider.name">
            {{ provider.name }}
          </h2>
          <Badge variant="secondary" class="text-[11px] uppercase font-medium px-1.5 py-0 shrink-0">
            {{ provider.type }}
          </Badge>
          <Badge
            v-if="
              provider.type === 'gemini' &&
              provider.provider_metadata?.api_variant === 'interactions'
            "
            variant="outline"
            class="text-[11px] font-medium px-1.5 py-0 text-muted-foreground shrink-0"
          >
            {{ t("providers.interactionsBadge") }}
          </Badge>
          <Badge
            v-if="provider.type === 'openai-chatgpt'"
            variant="outline"
            class="text-[11px] font-medium px-1.5 py-0 text-muted-foreground shrink-0"
          >
            {{
              provider.oauth_status?.connected
                ? t("providers.oauthConnected")
                : t("providers.oauthSignInRequired")
            }}
          </Badge>
          <Badge
            v-if="!isEnabled"
            variant="outline"
            class="text-[11px] font-medium px-1.5 py-0 text-muted-foreground shrink-0"
          >
            {{ t("providers.inactive") }}
          </Badge>
        </div>
        <div
          class="mt-1 flex items-center gap-1.5 min-w-0 text-muted-foreground"
          :title="provider.base_url || t('labels.default')"
        >
          <Globe class="w-3.5 h-3.5 shrink-0 text-muted-foreground/70" />
          <span class="truncate font-mono text-xs">
            {{ provider.base_url || t("labels.default") }}
          </span>
        </div>
      </div>

      <!-- Actions -->
      <div
        class="flex items-center justify-end gap-1 shrink-0 opacity-100 transition-opacity [@media(hover:hover)]:opacity-0 [@media(hover:hover)]:group-hover:opacity-100 [@media(hover:hover)]:group-focus-within:opacity-100"
      >
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9"
          :disabled="isLoading"
          :aria-label="t('common.editNamed', { name: provider.name })"
          @click.stop="emit('edit')"
        >
          <Edit class="w-4 h-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
          :disabled="isLoading"
          :aria-label="t('common.deleteNamed', { name: provider.name })"
          @click.stop="emit('delete')"
        >
          <Trash2 class="w-4 h-4" />
        </Button>
      </div>
    </div>
  </article>
</template>
