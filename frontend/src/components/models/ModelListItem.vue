<script setup lang="ts">
import { Check, Edit, Trash2 } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { deriveModelCapabilities } from "@/components/plaza/capabilities";
import CapabilityIcons from "@/components/plaza/CapabilityIcons.vue";
import ModelIcon from "@/components/models/ModelIcon.vue";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import ModelPricingCell from "@/components/models/ModelPricingCell.vue";
import { formatContextLength } from "@/utils/format";
import type { ModelRead } from "@/types/schemas";

interface Props {
  model: ModelRead;
  isLoading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  isLoading: false,
});

const emit = defineEmits<{
  edit: [];
  delete: [];
  filterProvider: [provider: string];
}>();

const { t } = useI18n();

const providers = computed(() => props.model.providers ?? []);
const hasRoutingInfo = computed(() => Boolean(props.model.auto_eligible));
const capabilities = computed(() => deriveModelCapabilities(props.model));
</script>

<template>
  <article
    class="group px-4 sm:px-6 py-2.5 border-b border-border transition-colors duration-150 hover:bg-muted/50"
  >
    <div class="flex items-center gap-3">
      <!-- Icon -->
      <ModelIcon :name="model.name" :icon-url="model.icon_url" :decorative="false" />

      <!-- Name + description + meta line -->
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-1.5">
          <h3 class="text-sm font-medium text-foreground truncate" :title="model.name">
            {{ model.name }}
          </h3>
          <CapabilityIcons :capabilities="capabilities" />
        </div>
        <!-- Inline description: one scannable line, OpenRouter-list style -->
        <p
          v-if="model.description"
          class="mt-0.5 truncate text-xs text-muted-foreground"
          :title="model.description"
        >
          {{ model.description }}
        </p>
        <!-- Meta line: providers · context; routing is kept apart as its own chip -->
        <div
          class="mt-0.5 flex items-center gap-x-2 gap-y-0.5 flex-wrap font-mono text-[11px] text-muted-foreground"
        >
          <template v-if="providers.length > 0">
            <template v-for="(p, i) in providers.slice(0, 3)" :key="p.provider_name">
              <span v-if="i > 0" class="text-border" aria-hidden="true">·</span>
              <Tooltip>
                <TooltipTrigger as-child>
                  <button
                    type="button"
                    class="rounded-sm transition-colors hover:text-foreground hover:underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
                    :aria-label="t('models.filterByProvider') + ': ' + p.provider_name"
                    @click.stop="emit('filterProvider', p.provider_name)"
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
          <span v-else>-</span>
          <template v-if="model.context_length">
            <span class="text-border" aria-hidden="true">·</span>
            <span class="tabular-nums shrink-0">
              {{ formatContextLength(model.context_length) }}
              <span class="font-sans lowercase tracking-wide text-muted-foreground/70">{{
                t("models.contextShort")
              }}</span>
            </span>
          </template>
          <!-- Smart-routing status: tinted chip, deliberately distinct from the
               plain data to its left instead of glued to the context value -->
          <template v-if="hasRoutingInfo">
            <span
              class="ml-1 inline-flex items-center gap-1 rounded-full border border-status-success/30 bg-status-success/10 px-1.5 py-px font-sans text-[10px] font-medium text-status-success shrink-0"
              :title="t('models.autoEligible')"
            >
              <Check class="size-2.5" aria-hidden="true" />
              <span v-if="model.quality_tier" class="capitalize">
                {{ model.quality_tier.toLowerCase() }}
              </span>
              <span v-else>{{ t("common.routing") }}</span>
            </span>
            <span
              v-if="model.routing_assignments?.length"
              class="font-mono text-[11px] text-muted-foreground/80"
            >
              {{ model.routing_assignments.join(", ") }}
            </span>
          </template>
        </div>
      </div>

      <!-- Pricing: stacked IN/OUT/CACHED block, the primary data on the row -->
      <div class="hidden sm:block shrink-0 text-right">
        <ModelPricingCell :model="model" />
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
