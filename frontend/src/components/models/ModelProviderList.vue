<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { ModelProviderMapping } from "@/types/schemas";

/**
 * Clickable provider list for model rows: the first three names filter the
 * view on click, the rest collapse into a "+N" counter, and an en dash shows
 * when the model has no providers. Text size is inherited from the caller.
 */
const props = defineProps<{ providers: ModelProviderMapping[] }>();

const emit = defineEmits<{ filter: [provider: string] }>();

const { t } = useI18n();

const visibleProviders = computed(() => props.providers.slice(0, 3));
</script>

<template>
  <div class="flex items-center gap-1 flex-wrap font-mono text-muted-foreground">
    <template v-if="providers.length > 0">
      <template v-for="(p, i) in visibleProviders" :key="p.provider_name">
        <span v-if="i > 0" class="text-border" aria-hidden="true">·</span>
        <Tooltip>
          <TooltipTrigger as-child>
            <button
              type="button"
              class="rounded-sm transition-colors hover:text-foreground hover:underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
              :aria-label="t('models.filterByProvider') + ': ' + p.provider_name"
              @click.stop="emit('filter', p.provider_name)"
            >
              {{ p.provider_name }}
            </button>
          </TooltipTrigger>
          <TooltipContent>
            {{ t("models.filterByProvider") + ": " + p.provider_name }}
          </TooltipContent>
        </Tooltip>
      </template>
      <span v-if="providers.length > 3" class="text-muted-foreground/70">
        +{{ providers.length - 3 }}
      </span>
    </template>
    <span v-else>–</span>
  </div>
</template>
