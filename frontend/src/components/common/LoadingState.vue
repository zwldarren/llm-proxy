<script setup lang="ts">
import { Loader2 } from "@lucide/vue";
import { useI18n } from "vue-i18n";

interface Props {
  /** Custom loading text, defaults to i18n 'common.loading' */
  text?: string;
  /** Whether to show the loading text */
  showText?: boolean;
  /** Size of the spinner icon */
  size?: "sm" | "md" | "lg";
}

const props = withDefaults(defineProps<Props>(), {
  text: undefined,
  showText: true,
  size: "md",
});

const { t } = useI18n();

const sizeClasses = {
  sm: "w-5 h-5",
  md: "w-8 h-8",
  lg: "w-12 h-12",
};
</script>

<template>
  <!-- Flat-by-default: a single quiet spinner, no glows or gradient overlays.
       The global prefers-reduced-motion rule neutralizes the spin. -->
  <div
    class="loading-state"
    role="status"
    aria-busy="true"
    :aria-label="text ?? t('common.loading')"
  >
    <Loader2
      :class="[sizeClasses[props.size], 'animate-spin text-muted-foreground']"
      aria-hidden="true"
    />
    <span v-if="showText" class="text-sm">{{ text ?? t("common.loading") }}</span>
  </div>
</template>
