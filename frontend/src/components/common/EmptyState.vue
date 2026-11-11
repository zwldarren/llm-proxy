<script setup lang="ts">
import { Inbox, Plus, RefreshCw } from "@lucide/vue";
import type { Component } from "vue";
import { Button } from "@/components/ui/button";
import { useI18n } from "vue-i18n";

interface Props {
  /** Text to display in the empty state (optional if using slots) */
  text?: string;
  /** Optional icon component to display */
  icon?: Component;
  /** Size of the icon */
  iconSize?: "sm" | "md" | "lg";
  /** Optional CTA button configuration */
  ctaText?: string;
  /** Icon component to show in the CTA button (defaults to Plus) */
  ctaIcon?: Component;
  /** Whether to show the CTA button */
  showCta?: boolean;
  /** Whether to show a retry button */
  showRetry?: boolean;
  /**
   * Accessible name for the region. Defaults to "No results found", which is
   * wrong for the error and never-configured cases: pass the matching text.
   */
  ariaLabel?: string;
}

withDefaults(defineProps<Props>(), {
  text: undefined,
  icon: undefined,
  iconSize: "md",
  ctaIcon: undefined,
  showCta: false,
  showRetry: false,
  ariaLabel: undefined,
});

const emit = defineEmits<{
  click: [];
  retry: [];
}>();

const { t } = useI18n();

const sizeClasses = {
  sm: "w-5 h-5",
  md: "w-6 h-6",
  lg: "w-8 h-8",
};
</script>

<template>
  <div
    class="flex flex-col items-center justify-center py-12 px-4 text-center animate-slide-in-up"
    role="region"
    :aria-label="ariaLabel ?? t('common.noResults')"
  >
    <div
      :class="[
        'icon-container mb-4 rounded-lg bg-muted border border-border',
        iconSize === 'lg' ? 'p-3' : iconSize === 'md' ? 'p-2.5' : 'p-2',
      ]"
      aria-hidden="true"
    >
      <component v-if="icon" :is="icon" :class="[sizeClasses[iconSize], 'text-primary']" />
      <!-- A magnifier used to stand in for every empty state, which read as
           "your search found nothing" even for a collection that never had
           anything in it. -->
      <Inbox v-else :class="[sizeClasses[iconSize], 'text-primary']" />
    </div>
    <p v-if="text" class="text-muted-foreground mb-4 max-w-md wrap-break-word">{{ text }}</p>
    <slot name="title" />
    <slot name="description" />
    <div v-if="showRetry || showCta" class="flex items-center gap-3 mt-4 flex-wrap justify-center">
      <Button v-if="showRetry" variant="outline" @click="emit('retry')">
        <RefreshCw class="w-4 h-4 mr-2" />
        {{ t("common.refresh") }}
      </Button>
      <Button v-if="showCta && ctaText" variant="secondary" @click="emit('click')">
        <component :is="ctaIcon ?? Plus" class="w-4 h-4 mr-2" />
        {{ ctaText }}
      </Button>
    </div>
    <slot />
  </div>
</template>
