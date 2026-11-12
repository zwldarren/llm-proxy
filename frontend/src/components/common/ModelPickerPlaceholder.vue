<script setup lang="ts">
/**
 * Model picker fallback, shared by the Chat and Images headers.
 *
 * Shown while the catalog is empty: loading, failed, or genuinely empty. It
 * stays interactive — the list loads once per session, so a failed call must
 * be retryable in place instead of leaving an inert placeholder.
 *
 * It wears the same chip surface as the picker it stands in for, so the header
 * does not change shape between "list loaded" and "list missing", and the
 * `enabled:` hover is what separates a retry from a status label.
 */
import { AlertCircle, Loader2, RefreshCw } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { HEADER_CHIP_CLASS } from "@/lib/controlMetric";

interface Props {
  /** A `/v1/models` call is in flight. */
  loading: boolean;
  /** Why the list is missing; distinguishes a failed load from an empty one. */
  error?: string | null;
  /** Placeholder copy, e.g. "Models unavailable — retry". */
  label: string;
}

withDefaults(defineProps<Props>(), { error: null });

const emit = defineEmits<{ retry: [] }>();

const { t } = useI18n();
</script>

<template>
  <button
    type="button"
    :disabled="loading"
    :aria-busy="loading"
    :aria-label="loading ? t('common.loading') : label"
    :title="loading ? t('common.loading') : label"
    :class="[
      HEADER_CHIP_CLASS,
      'touch-target min-w-0 font-mono text-[11px] font-medium enabled:cursor-pointer disabled:cursor-progress',
      error ? 'border-action-amber/50 text-action-amber' : 'text-muted-foreground',
    ]"
    @click="emit('retry')"
  >
    <span class="flex min-w-0 items-center gap-2">
      <Loader2 v-if="loading" class="w-3 h-3 shrink-0 animate-spin opacity-70" />
      <AlertCircle v-else-if="error" class="w-3 h-3 shrink-0" />
      <RefreshCw v-else class="w-3 h-3 shrink-0 opacity-70" />
      <span class="min-w-0 max-w-[200px] truncate">
        {{ loading ? `${t("common.loading")}…` : label }}
      </span>
    </span>
  </button>
</template>
