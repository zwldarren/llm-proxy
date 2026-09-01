<script setup lang="ts">
import { copyToClipboard } from "@/utils/clipboard";
import { Check, Copy, AlertTriangle, Loader2 } from "@lucide/vue";
import { computed, ref, defineAsyncComponent, watch } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";

// Async load vue-json-pretty with error and loading fallbacks
const VueJsonPretty = defineAsyncComponent({
  loader: async () => {
    await import("vue-json-pretty/lib/styles.css");
    return import("vue-json-pretty");
  },
  loadingComponent: {
    template:
      '<div class="flex items-center gap-2 py-4 text-muted-foreground text-xs"><Loader2 class="w-4 h-4 animate-spin" /><span>Loading JSON viewer…</span></div>',
    components: { Loader2 },
  },
  errorComponent: {
    props: ["error"],
    template:
      '<div class="flex items-center gap-2 py-4 text-destructive text-xs"><AlertTriangle class="w-4 h-4 shrink-0" /><span>Failed to load JSON viewer</span></div>',
    components: { AlertTriangle },
  },
  timeout: 10000,
});

// Define JSONDataType locally since it's not exported from vue-json-pretty main module
type JSONDataType = string | number | boolean | unknown[] | Record<string, unknown> | null;

const props = withDefaults(
  defineProps<{
    data: unknown;
    label?: string;
    maxHeight?: string;
    collapsedOnClickBrackets?: boolean;
    deep?: number;
    /**
     * Borderless/transparent variant for nesting inside an already-contained
     * region (collapsible rows, banners): the parent supplies the surface, so
     * the viewer must not draw a second box (no card-in-card). Default (boxed)
     * is for viewers sitting directly on a page/sheet surface.
     */
    flat?: boolean;
  }>(),
  {
    deep: 3,
    collapsedOnClickBrackets: true,
    flat: false,
  }
);

const emit = defineEmits<{
  copy: [value: string];
  error: [message: string];
}>();

const { t } = useI18n();
const copyError = ref(false);
const copyErrorMessage = ref("");
const copied = ref(false);

// Reset copy error when copied state changes
watch(copied, (val) => {
  if (val) {
    copyError.value = false;
    copyErrorMessage.value = "";
  }
});

// Safe JSON stringify that handles circular references and other edge cases
const safeStringify = (data: unknown): string => {
  const seen = new WeakSet<object>();
  try {
    return JSON.stringify(
      data,
      (_key: string, value: unknown) => {
        if (typeof value === "object" && value !== null) {
          if (seen.has(value)) {
            return "[Circular]";
          }
          seen.add(value);
        }
        return value;
      },
      2
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown serialization error";
    emit("error", message);
    return `[Serialization Error: ${message}]`;
  }
};

const handleCopy = async () => {
  try {
    copyError.value = false;
    copyErrorMessage.value = "";
    const text = safeStringify(props.data);

    await copyToClipboard(text);

    emit("copy", text);
    copied.value = true;
    setTimeout(() => {
      copied.value = false;
    }, 2000);
  } catch (err) {
    copyError.value = true;
    copyErrorMessage.value = err instanceof Error ? err.message : "Copy failed";
    emit("error", copyErrorMessage.value);
  }
};

// Cast unknown data to JSONDataType for vue-json-pretty
const jsonData = computed((): JSONDataType | undefined => {
  if (props.data === null || props.data === undefined) {
    return undefined;
  }
  return props.data as JSONDataType;
});

// Determine if data is empty (null, undefined, empty object, empty array)
const isEmpty = computed((): boolean => {
  const d = props.data;
  if (d === null || d === undefined) return true;
  if (typeof d === "object") {
    if (Array.isArray(d)) return d.length === 0;
    return Object.keys(d).length === 0;
  }
  return false;
});

// Default collapsed depth - show first level for large objects
const defaultCollapsedNodes = (data: Record<string, unknown> | unknown[]): number => {
  if (!data || typeof data !== "object") return 0;
  if (Array.isArray(data)) {
    return data.length > 10 ? 1 : 0;
  }
  const keys = Object.keys(data);
  return keys.length > 5 ? 1 : 0;
};

// Copy button label for screen readers
const copyLabel = computed(() => {
  if (copyError.value) return t("common.copyFailed") || "Copy failed";
  if (copied.value) return t("common.copied") || "Copied";
  return t("common.copy") || "Copy to clipboard";
});
</script>

<template>
  <div class="space-y-2">
    <!-- Header with label and copy button -->
    <div v-if="label" class="flex items-center justify-between">
      <h4
        class="text-xs uppercase font-medium text-muted-foreground truncate min-w-0"
        :title="label"
      >
        {{ label }}
      </h4>
      <Button
        variant="ghost"
        size="icon"
        class="h-10 w-10 shrink-0"
        :disabled="copied && !copyError"
        :aria-label="copyLabel"
        :title="copyLabel"
        @click="handleCopy"
      >
        <Check
          v-if="copied && !copyError"
          class="w-4 h-4 text-status-success motion-safe:animate-in motion-safe:fade-in motion-safe:zoom-in-50"
        />
        <AlertTriangle v-else-if="copyError" class="w-4 h-4 text-destructive" />
        <Copy v-else class="w-4 h-4 text-muted-foreground" />
      </Button>
    </div>

    <!-- JSON content container -->
    <div
      class="overflow-x-auto relative group"
      :class="
        flat
          ? maxHeight || 'max-h-96'
          : `${maxHeight || 'max-h-96'} bg-muted/50 rounded-lg border border-border`
      "
    >
      <!-- Floating copy button (when no label) -->
      <div
        v-if="!label"
        class="absolute top-2 right-2 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity z-10"
      >
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9 bg-background/50 hover:bg-background"
          :aria-label="copyLabel"
          :title="copyLabel"
          @click="handleCopy"
        >
          <Check
            v-if="copied && !copyError"
            class="w-4 h-4 text-status-success motion-safe:animate-in motion-safe:fade-in motion-safe:zoom-in-50"
          />
          <AlertTriangle v-else-if="copyError" class="w-4 h-4 text-destructive" />
          <Copy v-else class="w-4 h-4 text-muted-foreground" />
        </Button>
      </div>

      <!-- Empty state -->
      <div
        v-if="isEmpty"
        class="flex items-center justify-center py-8 text-muted-foreground text-xs"
      >
        <span v-if="data === null">{{ t("common.null") || "null" }}</span>
        <span v-else-if="data === undefined">{{ t("common.undefined") || "undefined" }}</span>
        <span v-else class="opacity-50">{{ t("common.empty") || "Empty" }}</span>
      </div>

      <!-- JSON tree -->
      <div v-else class="break-all" :class="flat ? '' : 'p-3'">
        <VueJsonPretty
          :data="jsonData"
          :deep="deep"
          :collapsedOnClickBrackets="collapsedOnClickBrackets"
          :showLine="true"
          :showLength="true"
          :showIcon="true"
          :showLineNumber="false"
          :collapsedOnClickBracketsVal="
            defaultCollapsedNodes(data as Record<string, unknown> | unknown[])
          "
          class="text-xs font-mono"
        />
      </div>
    </div>

    <!-- Screen reader live region for copy status -->
    <div role="status" aria-live="polite" class="sr-only">
      <template v-if="copied && !copyError">{{
        t("common.copied") !== "common.copied" ? t("common.copied") : "Copied to clipboard"
      }}</template>
      <template v-else-if="copyError">{{ copyErrorMessage }}</template>
    </div>
  </div>
</template>

<style>
/* Override vue-json-pretty styles to match theme */
.vue-json-pretty {
  --vue-json-pretty-theme-background-color: transparent;
  --vue-json-pretty-theme-font-family:
    ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  --vue-json-pretty-theme-font-size: 12px;
  --vue-json-pretty-theme-line-height: 1.6;
  --vue-json-pretty-theme-token-property: hsl(var(--foreground));
  --vue-json-pretty-theme-token-string: hsl(var(--json-string));
  --vue-json-pretty-theme-token-number: hsl(var(--json-number));
  --vue-json-pretty-theme-token-boolean: hsl(var(--json-boolean));
  --vue-json-pretty-theme-token-null: hsl(var(--json-null));
  --vue-json-pretty-theme-token-key: hsl(var(--json-key));
}

.vue-json-pretty__icon {
  color: var(--vue-json-pretty-theme-token-null);
}

.vue-json-pretty__string {
  color: var(--vue-json-pretty-theme-token-string);
}

.vue-json-pretty__number {
  color: var(--vue-json-pretty-theme-token-number);
}

.vue-json-pretty__boolean {
  color: var(--vue-json-pretty-theme-token-boolean);
}

.vue-json-pretty__null {
  color: var(--vue-json-pretty-theme-token-null);
}

.vue-json-pretty__key {
  color: var(--vue-json-pretty-theme-token-key);
}

/* Override vue-json-pretty hover styles for both light and dark mode */
/* Light mode hover - subtle gray background */
.vjs-tree-node:hover,
.vjs-tree-node.is-highlight {
  background-color: hsl(var(--muted) / 0.4) !important;
}

/* Dark mode hover - subtle light background that maintains text readability */
.dark .vjs-tree-node:hover,
.dark .vjs-tree-node.is-highlight {
  background-color: hsl(var(--muted) / 0.3) !important;
}

/* Also override the actions background in dark mode */
.dark .vjs-tree-node:hover .vjs-tree-node-actions,
.dark .vjs-tree-node.is-highlight .vjs-tree-node-actions {
  background-color: hsl(var(--muted) / 0.4) !important;
}

/* Ensure value colors are maintained on hover in dark mode */
.dark .vjs-tree-node:hover .vjs-value-string,
.dark .vjs-tree-node.is-highlight .vjs-value-string {
  color: hsl(var(--json-string)) !important;
}

.dark .vjs-tree-node:hover .vjs-value-number,
.dark .vjs-tree-node.is-highlight .vjs-value-number,
.dark .vjs-tree-node:hover .vjs-value-boolean,
.dark .vjs-tree-node.is-highlight .vjs-value-boolean {
  color: hsl(var(--json-number)) !important;
}

.dark .vjs-tree-node:hover .vjs-value-null,
.dark .vjs-tree-node.is-highlight .vjs-value-null,
.dark .vjs-tree-node:hover .vjs-value-undefined,
.dark .vjs-tree-node.is-highlight .vjs-value-undefined {
  color: hsl(var(--json-null)) !important;
}

.dark .vjs-tree-node:hover .vjs-key,
.dark .vjs-tree-node.is-highlight .vjs-key {
  color: hsl(var(--json-key)) !important;
}

/* Override the brackets hover color in dark mode */
.dark .vjs-tree-brackets:hover {
  color: hsl(var(--json-key)) !important;
}

/* Override carets hover color in dark mode */
.dark .vjs-carets:hover {
  color: hsl(var(--json-key)) !important;
}

/* Ensure long strings wrap properly inside the JSON viewer */
.vue-json-pretty .vjs-value-string {
  word-break: break-all;
  overflow-wrap: break-word;
  white-space: pre-wrap;
}

/* Respect reduced motion for copy feedback animation */
@media (prefers-reduced-motion: reduce) {
  .motion-safe\:animate-in {
    animation: none !important;
  }
}
</style>
