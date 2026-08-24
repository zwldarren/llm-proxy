<script setup lang="ts">
import { Loader2 } from "@lucide/vue";
import { useI18n } from "vue-i18n";

interface Props {
  /** Loading display mode - unified to spinner */
  mode?: "spinner";
  /** Custom loading text, defaults to i18n 'common.loading' */
  text?: string;
  /** Whether to show the loading text */
  showText?: boolean;
  /** Size of the spinner icon */
  size?: "sm" | "md" | "lg";
}

const props = withDefaults(defineProps<Props>(), {
  mode: "spinner",
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
  <div
    class="loading-state"
    role="status"
    aria-busy="true"
    :aria-label="text ?? t('common.loading')"
  >
    <div class="relative">
      <Loader2 :class="[sizeClasses[props.size], 'animate-spin text-primary']" />
      <div
        class="absolute inset-0 rounded-full animate-spin"
        style="
          background: linear-gradient(
            to right,
            hsl(var(--primary) / var(--opacity-light)),
            transparent,
            transparent
          );
        "
      />
    </div>
    <span v-if="showText" class="text-sm">{{ text ?? t("common.loading") }}</span>
    <span class="sr-only">{{ text ?? t("common.loading") }}</span>
  </div>
</template>
