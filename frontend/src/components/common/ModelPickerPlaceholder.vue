<script setup lang="ts">
/**
 * Model picker fallback, shared by the Chat and Images headers.
 *
 * Shown while the catalog is empty: loading, failed, or genuinely empty. It
 * stays interactive — the list loads once per session, so a failed call must
 * be retryable in place instead of leaving an inert placeholder.
 */
import { AlertCircle, Loader2, RefreshCw } from "@lucide/vue";
import { useI18n } from "vue-i18n";

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
    @click="emit('retry')"
    class="border rounded-md h-8 px-2.5 gap-3 flex items-center min-w-0 font-mono text-[11px] font-medium transition-colors enabled:cursor-pointer enabled:hover:bg-muted/10 focus-visible:ring-1 focus-visible:ring-foreground focus-visible:ring-offset-0 focus-visible:outline-none disabled:cursor-progress"
    :class="
      error ? 'border-action-amber/50 text-action-amber' : 'border-border/60 text-muted-foreground'
    "
  >
    <div class="flex items-center gap-2 min-w-0">
      <Loader2 v-if="loading" class="w-3 h-3 animate-spin shrink-0 opacity-70" />
      <AlertCircle v-else-if="error" class="w-3 h-3 shrink-0" />
      <RefreshCw v-else class="w-3 h-3 shrink-0 opacity-70" />
      <span class="truncate min-w-0 max-w-[200px]">
        {{ loading ? `${t("common.loading")}…` : label }}
      </span>
    </div>
  </button>
</template>
