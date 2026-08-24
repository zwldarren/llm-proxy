<script setup lang="ts">
import { computed, ref, type Component } from "vue";
import { useI18n } from "vue-i18n";
import type { FallbackAttempt, RetryAttempt } from "@/types/schemas";

type AttemptEntry = FallbackAttempt | RetryAttempt;

const props = defineProps<{
  attempts: AttemptEntry[];
  icon: Component;
  iconAnimated?: boolean;
  title: string;
  successText: string;
  failureText: string;
  showMoreText: string;
  statusCode?: number | null;
  displayLimit?: number;
}>();

const { t } = useI18n();

const DISPLAY_LIMIT = props.displayLimit ?? 20;
const showAll = ref(false);

const visible = computed(() => {
  if (!Array.isArray(props.attempts)) return [];
  if (showAll.value || props.attempts.length <= DISPLAY_LIMIT) {
    return props.attempts;
  }
  return props.attempts.slice(0, DISPLAY_LIMIT);
});

const hasMore = computed(() => {
  return Array.isArray(props.attempts) && props.attempts.length > DISPLAY_LIMIT;
});

const isSuccess = computed(() => {
  const code = props.statusCode;
  return typeof code === "number" && code >= 200 && code < 300;
});
</script>

<template>
  <div
    v-if="attempts.length > 0"
    class="rounded-md border bg-muted/10 border-border/40 overflow-hidden"
  >
    <div
      class="flex items-center gap-2 px-3 sm:px-4 py-2.5 sm:py-3 bg-muted/20 border-b border-border/40"
    >
      <component
        :is="icon"
        class="w-3.5 h-3.5 sm:w-4 sm:h-4 text-muted-foreground shrink-0"
        :class="{ 'animate-spin-slow': iconAnimated }"
      />
      <h3 class="text-xs sm:text-sm font-semibold text-foreground">
        {{ title }}
      </h3>
    </div>
    <div class="px-3 sm:px-4 py-3 sm:py-4">
      <p class="text-[11px] sm:text-xs font-medium mb-3">
        <span v-if="isSuccess" class="text-status-success">
          {{ t(successText, { count: attempts.length }) }}
        </span>
        <span v-else class="text-status-error">
          {{ t(failureText, { count: attempts.length }) }}
        </span>
      </p>

      <!-- Vertical Timeline -->
      <div class="relative pl-6 border-l-2 border-border/40 space-y-4 ml-2.5">
        <div
          v-for="(attempt, index) in visible"
          :key="index"
          class="relative p-3 rounded-lg bg-background/50 border border-border/20 hover:border-border/40 transition-colors"
        >
          <!-- Timeline dot node -->
          <div
            class="absolute -left-[25px] top-4 w-2.5 h-2.5 rounded-full border-2 border-border/40 bg-background ring-1 ring-border/20"
            aria-hidden="true"
          />
          <div class="flex items-start justify-between gap-2">
            <div class="flex items-center gap-1.5 min-w-0">
              <span class="font-mono text-xs font-semibold text-foreground truncate">
                {{ attempt.provider }}
              </span>
              <!-- Slot for per-attempt extra info (e.g. provider_type badge, attempt label) -->
              <slot name="attempt-meta" :attempt="attempt" />
            </div>
            <span
              v-if="attempt.status_code"
              class="font-mono text-xs font-semibold px-1.5 py-0.5 bg-muted/60 border rounded text-muted-foreground shrink-0"
            >
              {{ attempt.status_code }}
            </span>
          </div>
          <div
            v-if="attempt.error_type"
            class="text-xs text-muted-foreground/90 font-medium mt-1.5"
          >
            Type:
            <span class="font-mono text-[11px] text-foreground/80">{{ attempt.error_type }}</span>
          </div>
          <div
            v-if="attempt.error_message"
            class="text-xs text-muted-foreground mt-1 bg-destructive/5 dark:bg-destructive/10 border border-destructive/10 p-2 rounded break-all"
          >
            {{ attempt.error_message }}
          </div>
        </div>

        <!-- Show more button -->
        <button
          v-if="hasMore && !showAll"
          @click="showAll = true"
          class="w-full text-xs font-semibold text-muted-foreground hover:text-foreground py-2 rounded-lg border border-dashed border-border/30 hover:border-border/60 transition-colors cursor-pointer"
        >
          {{ t(showMoreText, { count: attempts.length - DISPLAY_LIMIT }) }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
@media (prefers-reduced-motion: reduce) {
  .animate-spin-slow {
    animation: none;
  }
}
</style>
