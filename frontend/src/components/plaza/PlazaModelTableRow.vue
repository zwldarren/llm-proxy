<script setup lang="ts">
import { Check, Copy, ExternalLink } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import CapabilityIcons from "@/components/plaza/CapabilityIcons.vue";
import { usePlazaModelRow } from "@/components/plaza/usePlazaModelRow";
import ModelIcon from "@/components/models/ModelIcon.vue";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TableCell, TableRow } from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { ModelCatalogEntry } from "@/types/schemas";
import { formatContextLength } from "@/utils/format";

/**
 * Dense table row for the model catalog (table view).
 * Mirrors the list item's data in a scannable, OpenRouter-style layout.
 */

const props = defineProps<{ model: ModelCatalogEntry }>();

const { t } = useI18n();

const { capabilities, safeHomepageUrl, copied, copyName, tierBadgeVariant } = usePlazaModelRow(
  () => props.model
);
</script>

<template>
  <TableRow class="group">
    <!-- Model: icon + name + capability icons -->
    <TableCell>
      <div class="flex items-center gap-2.5 min-w-0">
        <ModelIcon :name="model.name" :icon-url="model.icon_url" size="sm" />
        <span
          class="font-mono text-[13px] font-medium text-foreground truncate"
          :title="model.name"
        >
          {{ model.name }}
        </span>
        <CapabilityIcons :capabilities="capabilities" />
      </div>
    </TableCell>
    <!-- Tier -->
    <TableCell>
      <Badge
        v-if="model.quality_tier"
        :variant="tierBadgeVariant(model.quality_tier)"
        class="text-[11px] uppercase font-medium px-1.5 py-0"
      >
        {{ model.quality_tier }}
      </Badge>
      <span v-else class="text-xs text-muted-foreground">–</span>
    </TableCell>
    <!-- Context -->
    <TableCell class="text-right">
      <span v-if="model.context_length != null" class="text-data text-xs text-muted-foreground">
        {{ formatContextLength(model.context_length) }}
      </span>
      <span v-else class="text-xs text-muted-foreground">–</span>
    </TableCell>
    <!-- Providers -->
    <TableCell>
      <span
        class="block max-w-56 truncate font-mono text-xs text-muted-foreground"
        :title="model.provider_names.join(', ')"
      >
        {{ model.provider_names.join(" · ") || "–" }}
      </span>
    </TableCell>
    <!-- Actions -->
    <TableCell class="text-right">
      <div
        class="flex items-center justify-end gap-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity"
      >
        <Tooltip>
          <TooltipTrigger as-child>
            <Button
              variant="ghost"
              size="icon"
              class="h-8 w-8"
              :aria-label="t('plaza.copyName')"
              @click="copyName"
            >
              <component :is="copied ? Check : Copy" class="size-3.5" aria-hidden="true" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{{ copied ? t("common.copied") : t("plaza.copyName") }}</TooltipContent>
        </Tooltip>
        <Tooltip v-if="safeHomepageUrl">
          <TooltipTrigger as-child>
            <Button variant="ghost" size="icon" class="h-8 w-8" as-child>
              <a
                :href="safeHomepageUrl"
                target="_blank"
                rel="noopener noreferrer"
                :aria-label="t('plaza.openHomepage')"
              >
                <ExternalLink class="size-3.5" />
              </a>
            </Button>
          </TooltipTrigger>
          <TooltipContent>{{ t("plaza.openHomepage") }}</TooltipContent>
        </Tooltip>
      </div>
    </TableCell>
  </TableRow>
</template>
