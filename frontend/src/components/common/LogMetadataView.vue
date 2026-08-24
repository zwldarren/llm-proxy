<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { FileJson, Database } from "@lucide/vue";
import JsonViewer from "@/components/common/JsonViewer.vue";
import type { LogRead } from "@/types/schemas";

const props = defineProps<{
  log: LogRead;
}>();

const { t } = useI18n();

// Determine if metadata exists and is non-empty
const hasMetadata = computed((): boolean => {
  const meta = props.log?.log_metadata;
  if (meta === null || meta === undefined) return false;
  // Object values: arrays count if non-empty, objects count if they have keys
  if (Array.isArray(meta)) return meta.length > 0;
  return Object.keys(meta).length > 0;
});
</script>

<template>
  <div class="space-y-5">
    <!-- Section header with icon -->
    <div class="flex items-center gap-2.5 pb-2.5 border-b border-border/60">
      <div
        class="p-2 rounded-lg min-h-8 min-w-8 bg-muted/60 border border-border/40 text-muted-foreground flex items-center justify-center"
        aria-hidden="true"
      >
        <FileJson class="size-4" />
      </div>
      <h3 class="font-semibold text-base text-foreground" id="metadata-heading">
        {{ t("logs.metadata") }}
      </h3>
    </div>

    <!-- Empty state when metadata is null/undefined/empty -->
    <div
      v-if="!hasMetadata"
      role="status"
      aria-labelledby="metadata-heading"
      class="flex flex-col items-center justify-center gap-3 py-12 px-4 text-center border border-dashed border-border/40 rounded-md bg-muted/5"
    >
      <Database class="size-7 text-muted-foreground/30" aria-hidden="true" />
      <div class="space-y-1">
        <p class="text-sm text-muted-foreground font-medium">
          {{ t("logs.noMetadata") }}
        </p>
        <p class="text-xs text-muted-foreground/60 max-w-sm">
          {{ t("logs.noMetadataDescription") }}
        </p>
      </div>
    </div>

    <!-- JsonViewer for non-empty metadata -->
    <div v-else class="metadata-viewer" role="region" aria-label="Log metadata JSON">
      <JsonViewer :data="log.log_metadata" maxHeight="max-h-[600px]" :deep="4" />
    </div>
  </div>
</template>

<style scoped>
.metadata-viewer :deep(.vue-json-pretty) {
  font-variant-numeric: tabular-nums;
}

@media (prefers-reduced-motion: reduce) {
  .metadata-viewer,
  .metadata-viewer :deep(*) {
    transition: none !important;
    animation: none !important;
  }
}
</style>
