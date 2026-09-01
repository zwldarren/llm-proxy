<script setup lang="ts">
/**
 * Expand-all / collapse-all control shared by the Logs I/O lists
 * (conversation messages, ordered output items, offered tools).
 */
import { useI18n } from "vue-i18n";

defineEmits<{ (e: "expand"): void; (e: "collapse"): void }>();

const { t } = useI18n();
</script>

<template>
  <div class="ml-auto flex items-center gap-1">
    <button
      type="button"
      class="text-[11px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted/60 rounded-md transition-colors px-1.5 py-1 cursor-pointer"
      @click="$emit('expand')"
    >
      {{ t("logs.expandAll") }}
    </button>
    <span class="text-border">·</span>
    <button
      type="button"
      class="text-[11px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted/60 rounded-md transition-colors px-1.5 py-1 cursor-pointer"
      @click="$emit('collapse')"
    >
      {{ t("logs.collapseAll") }}
    </button>
  </div>
</template>
