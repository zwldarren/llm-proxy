<script setup lang="ts">
import { Database, Edit, Globe, Trash2 } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
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
const { providerIconUrl: resolveIconUrl, providerIsMono: resolveIsMono } = useProviderTypes();

const iconFailed = ref(false);

const isEnabled = computed(() => props.provider.enabled !== false);
// Resolution order: custom icon_url -> backend type metadata -> static map.
const iconUrl = computed(() => resolveIconUrl(props.provider));
// Mono styling: backend-declared variant wins; fall back to the static map.
const isMono = computed(() => resolveIsMono(props.provider));
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
      <div
        :class="
          cn(
            'w-9 h-9 rounded-lg flex items-center justify-center overflow-hidden shrink-0',
            iconUrl ? 'bg-card border border-border' : 'bg-muted',
            !isEnabled && 'grayscale-[0.5]'
          )
        "
        role="img"
        :aria-label="provider.name"
      >
        <img
          v-if="iconUrl && !iconFailed"
          :src="iconUrl"
          :alt="provider.name"
          :class="[isMono && !provider.icon_url ? 'icon-mono' : null, 'w-5 h-5 object-contain']"
          loading="lazy"
          @error="iconFailed = true"
        />
        <Database v-else class="w-4 h-4 text-muted-foreground" />
      </div>

      <!-- Name + type + base url (stacked, matching the other list items) -->
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2 min-w-0">
          <h3 class="text-sm font-medium text-foreground truncate" :title="provider.name">
            {{ provider.name }}
          </h3>
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
        class="flex items-center justify-end gap-1 shrink-0 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity"
      >
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9"
          :disabled="isLoading"
          :aria-label="t('common.edit')"
          @click.stop="emit('edit')"
        >
          <Edit class="w-4 h-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
          :disabled="isLoading"
          :aria-label="t('common.delete')"
          @click.stop="emit('delete')"
        >
          <Trash2 class="w-4 h-4" />
        </Button>
      </div>
    </div>
  </article>
</template>
