<script setup lang="ts">
import { Check, ChevronDown, Copy, ExternalLink, ImageOff, Layers } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { useClipboard } from "@vueuse/core";
import { CAPABILITY_META, CAPABILITY_ORDER } from "@/components/plaza/capabilities";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ModelCatalogEntry } from "@/types/schemas";
import { formatContextLength, formatTokens } from "@/utils/format";
import { getIconUrl, isMonoIcon } from "@/utils/icons";

interface Props {
  model: ModelCatalogEntry;
}

const props = defineProps<Props>();

const { t, locale } = useI18n();

const isOpen = ref(false);
const iconFailed = ref(false);
const iconUrl = computed(() => getIconUrl(props.model.icon_url, props.model.name));
const exactContext = computed(() => formatTokens(props.model.context_length, locale.value));

const safeHomepageUrl = computed(() => {
  const url = props.model.homepage_url;
  if (!url) return null;
  return /^(https?):\/\//i.test(url) ? url : null;
});

const capabilities = computed(() =>
  CAPABILITY_ORDER.filter((cap) => props.model.capabilities?.includes(cap))
);

const { copy, copied } = useClipboard({ legacy: true, copiedDuring: 1500 });

async function copyName() {
  try {
    await copy(props.model.name);
  } catch {
    toast.error(t("plaza.copyFailed"));
  }
}

function tierBadgeVariant(tier: string | null | undefined): "default" | "secondary" | "outline" {
  if (tier === "PREMIUM") return "default";
  if (tier === "BALANCED") return "secondary";
  return "outline";
}
</script>

<template>
  <article
    class="group border-b border-border transition-colors duration-150"
    :class="isOpen ? 'bg-muted/30' : 'hover:bg-muted/50'"
  >
    <!-- Row header: icon · name + badges · context · provider count · chevron -->
    <button
      type="button"
      class="flex w-full items-center gap-3 px-4 sm:px-6 py-3 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/60"
      :aria-expanded="isOpen"
      :aria-label="isOpen ? t('plaza.collapseDetails') : t('plaza.expandDetails')"
      @click="isOpen = !isOpen"
    >
      <span
        :class="[
          'w-9 h-9 rounded-lg flex items-center justify-center shrink-0 overflow-hidden',
          iconUrl ? 'bg-card border border-border' : 'bg-primary/10',
        ]"
      >
        <img
          v-if="iconUrl && !iconFailed"
          :src="iconUrl"
          alt=""
          :class="[isMonoIcon(model.name) ? 'icon-mono' : null, 'w-5 h-5 object-contain']"
          loading="lazy"
          @error="iconFailed = true"
        />
        <ImageOff v-else class="w-4 h-4 text-muted-foreground" />
      </span>

      <span class="flex-1 min-w-0">
        <span class="flex items-center gap-1.5 flex-wrap">
          <span
            class="font-mono text-[13px] font-medium text-foreground truncate"
            :title="model.name"
          >
            {{ model.name }}
          </span>
          <Badge
            v-if="model.quality_tier"
            :variant="tierBadgeVariant(model.quality_tier)"
            class="text-[11px] uppercase font-medium px-1.5 py-0 shrink-0"
          >
            {{ model.quality_tier }}
          </Badge>
          <Badge
            v-for="cap in capabilities"
            :key="cap"
            variant="outline"
            :class="['text-[11px] px-1.5 py-0 shrink-0', CAPABILITY_META[cap].badgeClass]"
          >
            <component :is="CAPABILITY_META[cap].icon" class="size-3 mr-0.5" />
            {{ t(CAPABILITY_META[cap].labelKey) }}
          </Badge>
        </span>
        <span class="mt-0.5 flex items-center gap-2 text-[11px] text-muted-foreground">
          <span class="inline-flex items-center gap-1">
            <Layers class="size-3 text-muted-foreground/70" />
            {{ t("plaza.providerCount", model.provider_names.length) }}
          </span>
          <span v-if="model.context_length != null" aria-hidden="true">·</span>
          <span v-if="model.context_length != null" class="font-mono tabular-nums">
            {{ formatContextLength(model.context_length) }} {{ t("plaza.context") }}
          </span>
        </span>
      </span>

      <ChevronDown
        class="size-4 shrink-0 text-muted-foreground/50 transition-transform duration-200"
        :class="isOpen && 'rotate-180 text-foreground'"
        aria-hidden="true"
      />
    </button>

    <!-- Expanded details -->
    <Transition name="plaza-expand">
      <div v-if="isOpen" class="px-4 sm:px-6 pb-4">
        <div class="sm:pl-12 space-y-3">
          <p
            :class="[
              'max-w-3xl text-xs leading-relaxed',
              model.description ? 'text-muted-foreground' : 'text-muted-foreground/50 italic',
            ]"
          >
            {{ model.description || t("plaza.noDescription") }}
          </p>

          <!-- Structured detail rows -->
          <dl class="grid gap-x-8 gap-y-2 sm:grid-cols-2 max-w-2xl">
            <div class="flex items-baseline gap-2 text-xs">
              <dt class="shrink-0 text-[11px] text-muted-foreground/80 w-20">
                {{ t("plaza.capabilitiesLabel") }}
              </dt>
              <dd class="flex flex-wrap items-center gap-1.5">
                <template v-if="capabilities.length">
                  <span
                    v-for="cap in capabilities"
                    :key="cap"
                    class="inline-flex items-center gap-1 text-muted-foreground"
                  >
                    <component
                      :is="CAPABILITY_META[cap].icon"
                      :class="['size-3.5', CAPABILITY_META[cap].iconClass]"
                    />
                    {{ t(CAPABILITY_META[cap].labelKey) }}
                  </span>
                </template>
                <span v-else class="text-muted-foreground">{{ t("plaza.capability.chat") }}</span>
              </dd>
            </div>
            <div v-if="model.context_length != null" class="flex items-baseline gap-2 text-xs">
              <dt class="shrink-0 text-[11px] text-muted-foreground/80 w-20">
                {{ t("plaza.context") }}
              </dt>
              <dd class="font-mono tabular-nums text-muted-foreground">{{ exactContext }}</dd>
            </div>
            <div
              v-if="model.provider_names.length"
              class="flex items-baseline gap-2 text-xs sm:col-span-2"
            >
              <dt class="shrink-0 text-[11px] text-muted-foreground/80 w-20">
                {{ t("plaza.providers") }}
              </dt>
              <dd class="flex flex-wrap items-center gap-1.5">
                <Badge
                  v-for="p in model.provider_names"
                  :key="p"
                  variant="outline"
                  class="border-border/60 bg-background/55 px-1.5 py-0 font-mono text-[11px]"
                >
                  {{ p }}
                </Badge>
              </dd>
            </div>
          </dl>

          <!-- Actions -->
          <div class="flex flex-wrap items-center gap-2 pt-1">
            <Button
              variant="outline"
              size="sm"
              class="h-8"
              :aria-label="t('plaza.copyName')"
              @click="copyName"
            >
              <component :is="copied ? Check : Copy" class="size-3.5 mr-1.5" aria-hidden="true" />
              {{ copied ? t("common.copied") : t("plaza.copyName") }}
            </Button>
            <Button v-if="safeHomepageUrl" variant="outline" size="sm" class="h-8" as-child>
              <a :href="safeHomepageUrl" target="_blank" rel="noopener noreferrer">
                <ExternalLink class="size-3.5 mr-1.5" />
                {{ t("plaza.viewHomepage") }}
              </a>
            </Button>
          </div>
        </div>
      </div>
    </Transition>
  </article>
</template>

<style scoped>
/* Opacity/transform-only entrance (layout properties are never animated). */
.plaza-expand-enter-active {
  transition:
    opacity 200ms cubic-bezier(0.16, 1, 0.3, 1),
    transform 200ms cubic-bezier(0.16, 1, 0.3, 1);
}

.plaza-expand-enter-from {
  opacity: 0;
  transform: translateY(-4px);
}

.plaza-expand-leave-active {
  transition:
    opacity 150ms cubic-bezier(0.16, 1, 0.3, 1),
    transform 150ms cubic-bezier(0.16, 1, 0.3, 1);
}

.plaza-expand-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
