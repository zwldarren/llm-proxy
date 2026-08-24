<script setup lang="ts">
import { Check, Edit, ImageOff, Trash2 } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { getIconUrl, isMonoIcon } from "@/utils/icons";
import { CAPABILITY_META, deriveModelCapabilities } from "@/components/plaza/capabilities";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import ModelPricingCell from "@/components/models/ModelPricingCell.vue";
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

const iconFailed = ref(false);

const iconUrl = computed(() => getIconUrl(props.model.icon_url, props.model.name));
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
      <div
        :class="[
          'w-8 h-8 rounded-lg flex items-center justify-center shrink-0 overflow-hidden',
          iconUrl ? 'bg-card border border-border' : 'bg-primary/10',
        ]"
        role="img"
        :aria-label="model.name"
      >
        <img
          v-if="iconUrl && !iconFailed"
          :src="iconUrl"
          :alt="model.name"
          :class="[isMonoIcon(model.name) ? 'icon-mono' : null, 'w-5 h-5 object-contain']"
          loading="lazy"
          @error="iconFailed = true"
        />
        <ImageOff v-else class="w-4 h-4 text-muted-foreground" />
      </div>

      <!-- Name + routing badges + providers -->
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-1.5 flex-wrap">
          <h3 class="text-sm font-medium text-foreground truncate" :title="model.name">
            {{ model.name }}
          </h3>
          <Badge
            v-for="cap in capabilities"
            :key="cap"
            variant="outline"
            :class="['text-[11px] px-1.5 py-0 shrink-0', CAPABILITY_META[cap].badgeClass]"
          >
            <component :is="CAPABILITY_META[cap].icon" class="size-3 mr-0.5" />
            {{ t(CAPABILITY_META[cap].labelKey) }}
          </Badge>
          <template v-if="hasRoutingInfo">
            <span
              v-if="model.auto_eligible"
              class="inline-flex items-center justify-center size-4 rounded-full bg-status-success/15 text-status-success shrink-0"
              :title="t('models.autoEligible')"
            >
              <Check class="size-2.5" />
            </span>
            <Badge
              v-if="model.quality_tier"
              :variant="
                model.quality_tier === 'PREMIUM'
                  ? 'default'
                  : model.quality_tier === 'BALANCED'
                    ? 'secondary'
                    : 'outline'
              "
              class="text-[11px] uppercase font-medium px-1.5 py-0"
            >
              {{ model.quality_tier }}
            </Badge>
            <Badge
              v-for="mode in model.routing_assignments || []"
              :key="mode"
              variant="outline"
              class="text-[11px] px-1.5 py-0 border-action-blue/30 bg-action-blue/5 text-action-blue uppercase"
            >
              {{ mode }}
            </Badge>
          </template>
        </div>
        <div class="mt-1 flex items-center gap-1.5 flex-wrap">
          <template v-if="providers.length > 0">
            <Badge
              v-for="p in providers.slice(0, 3)"
              :key="p.provider_name"
              variant="outline"
              class="cursor-pointer border-border/60 bg-background/55 px-1.5 py-0 font-mono text-[11px] transition-colors hover:bg-accent"
              @click.stop="emit('filterProvider', p.provider_name)"
            >
              {{ p.provider_name }}
            </Badge>
            <span v-if="providers.length > 3" class="text-[11px] text-muted-foreground font-medium">
              +{{ providers.length - 3 }}
            </span>
          </template>
          <span v-else class="text-xs text-muted-foreground italic">-</span>
        </div>
      </div>

      <!-- Pricing -->
      <div class="hidden sm:block w-24 shrink-0 text-right">
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
