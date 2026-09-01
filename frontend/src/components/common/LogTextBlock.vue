/** * Long-text block for the Logs I/O views. * * Text-node layout is O(chars), and log bodies can
carry multi-MB strings — * inserting one whole into the DOM freezes scrolling for hundreds of ms. *
This block slices the text at `maxChars` before it reaches the DOM and * offers copy-all /
expand-on-demand for the tail, so the page stays * responsive by default and stays honest on demand.
* * Typography/container classes are passed through via `class`; the shared *
pre-wrap/break-all/leading behaviour is baked in. */
<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";
import { useI18n } from "vue-i18n";
import { copyToClipboard } from "@/utils/clipboard";
import { formatCharCount } from "@/utils/logFormat";

const props = withDefaults(
  defineProps<{
    text: string;
    /** Slice length before truncation kicks in. */
    maxChars?: number;
  }>(),
  { maxChars: 100_000 }
);

const { t } = useI18n();

defineOptions({ inheritAttrs: false });

const expanded = ref(false);
const copied = ref(false);
let copyTimer: ReturnType<typeof setTimeout> | undefined;

const isTruncated = computed(() => props.text.length > props.maxChars);
const shown = computed(() =>
  isTruncated.value && !expanded.value ? props.text.slice(0, props.maxChars) : props.text
);

async function copyAll() {
  try {
    await copyToClipboard(props.text);
    copied.value = true;
    if (copyTimer) clearTimeout(copyTimer);
    copyTimer = setTimeout(() => (copied.value = false), 1500);
  } catch {
    // Clipboard unavailable (insecure context) — nothing to signal inline.
  }
}

onBeforeUnmount(() => {
  if (copyTimer) clearTimeout(copyTimer);
});
</script>

<template>
  <div class="min-w-0">
    <div v-bind="$attrs" class="whitespace-pre-wrap break-all leading-relaxed">{{ shown }}</div>
    <div v-if="isTruncated" class="mt-1.5 flex items-center gap-2 flex-wrap">
      <span class="text-[11px] font-mono text-muted-foreground/60">
        {{
          t("logs.textTruncated", {
            shown: formatCharCount(props.maxChars),
            total: formatCharCount(props.text.length),
          })
        }}
      </span>
      <button
        v-if="!expanded"
        type="button"
        class="text-[11px] font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
        @click="expanded = true"
      >
        {{ t("logs.showAllText") }}
      </button>
      <button
        type="button"
        class="text-[11px] font-medium transition-colors cursor-pointer"
        :class="copied ? 'text-status-success' : 'text-muted-foreground hover:text-foreground'"
        @click="copyAll"
      >
        {{ copied ? t("common.copied") : t("common.copy") }}
      </button>
    </div>
  </div>
</template>
