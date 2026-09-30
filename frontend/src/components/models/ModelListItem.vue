<script setup lang="ts">
import { Check, Edit, Trash2 } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import CapabilityIcons from "@/components/plaza/CapabilityIcons.vue";
import { CAPABILITY_ORDER } from "@/components/plaza/capabilities";
import ModelContextCell from "@/components/models/ModelContextCell.vue";
import ModelIcon from "@/components/models/ModelIcon.vue";
import ModelProviderList from "@/components/models/ModelProviderList.vue";
import ModelStatusChip from "@/components/models/ModelStatusChip.vue";
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
// Fixed display order so icons don't shuffle between rows.
const capabilities = computed(() =>
  CAPABILITY_ORDER.filter((cap) => props.model.capabilities?.includes(cap))
);
</script>

<template>
  <article class="group border-b border-border transition-colors duration-150 hover:bg-muted/50">
    <div class="flex items-center gap-3 px-4 py-2.5 sm:px-6">
      <ModelIcon :name="model.name" :icon-url="model.icon_url" :decorative="false" />

      <!-- Name block: identifier line (name + chips) over the meta line -->
      <div class="min-w-0 flex-1">
        <div class="flex items-center gap-1.5">
          <Tooltip>
            <TooltipTrigger as-child>
              <h3 class="truncate font-mono text-[13px] font-medium text-foreground">
                {{ model.name }}
              </h3>
            </TooltipTrigger>
            <TooltipContent>{{ model.name }}</TooltipContent>
          </Tooltip>
          <ModelStatusChip v-if="model.status" :status="model.status" />
          <CapabilityIcons :capabilities="capabilities" />
          <!-- Smart-routing chip: kept apart from the plain data, on the
               identifier line where it qualifies the model itself -->
          <template v-if="hasRoutingInfo">
            <Tooltip>
              <TooltipTrigger as-child>
                <span
                  class="inline-flex shrink-0 items-center gap-1 rounded-full border border-status-success/30 bg-status-success/10 px-1.5 py-px text-[10px] font-medium text-status-success"
                >
                  <Check class="size-2.5" aria-hidden="true" />
                  <span v-if="model.quality_tier" class="capitalize">
                    {{ model.quality_tier.toLowerCase() }}
                  </span>
                  <span v-else>{{ t("common.routing") }}</span>
                </span>
              </TooltipTrigger>
              <TooltipContent>{{ t("models.autoEligible") }}</TooltipContent>
            </Tooltip>
            <span
              v-if="model.routing_assignments?.length"
              class="hidden font-mono text-[11px] text-muted-foreground sm:inline"
            >
              {{ model.routing_assignments.join(", ") }}
            </span>
          </template>
        </div>
        <!-- Meta line: providers · context (inline below lg; a column ≥lg) · description.
             Providers keep their natural width; the description absorbs the shrink. -->
        <div
          class="mt-0.5 flex min-w-0 items-center gap-x-2 font-mono text-[11px] text-muted-foreground"
        >
          <!-- Providers keep their natural width; the description absorbs the shrink. -->
          <div class="shrink-0 overflow-hidden">
            <ModelProviderList
              nowrap
              :providers="providers"
              @filter="emit('filterProvider', $event)"
            />
          </div>
          <template v-if="model.context_length">
            <span class="text-border lg:hidden" aria-hidden="true">·</span>
            <span class="shrink-0 tabular-nums lg:hidden">
              {{ formatContextLength(model.context_length) }}
              <span class="font-sans lowercase tracking-wide">{{ t("models.contextShort") }}</span>
            </span>
          </template>
          <template v-if="model.description">
            <span class="text-border" aria-hidden="true">·</span>
            <span class="min-w-0 truncate font-sans">{{ model.description }}</span>
          </template>
        </div>
      </div>

      <!-- Data columns: fixed widths shared with ModelListHead so every row
           aligns like the table. The full grammar shows at lg; between sm and
           lg only IN/OUT stay (CACHED is one tap away in either's popover);
           below sm the meta line above carries the context value. -->
      <div class="model-col-ctx hidden justify-end lg:flex">
        <ModelContextCell :context-length="model.context_length" />
      </div>
      <div class="model-col-price hidden justify-end sm:flex">
        <ModelPricingCell :model="model" field="input" />
      </div>
      <div class="model-col-price hidden justify-end sm:flex">
        <ModelPricingCell :model="model" field="output" />
      </div>
      <div class="model-col-price hidden justify-end lg:flex">
        <ModelPricingCell :model="model" field="cached" />
      </div>

      <!-- Actions -->
      <div
        class="model-col-actions flex items-center justify-end gap-1 opacity-100 transition-opacity sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100"
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
