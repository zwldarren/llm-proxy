<script setup lang="ts">
import { useI18n } from "vue-i18n";

const { t } = useI18n();

defineProps<{
  hasModel: boolean;
}>();

const emit = defineEmits<{
  prompt: [text: string];
}>();

const templates = [
  {
    key: "templateStreaming",
    descKey: "templateStreamingDesc",
    text: "Stream a short paragraph about how proxies handle backpressure, token by token.",
  },
  {
    key: "templateToolCall",
    descKey: "templateToolCallDesc",
    text: "Call a function `get_weather(city)` and return the result as JSON.",
  },
  {
    key: "templateVision",
    descKey: "templateVisionDesc",
    text: "Describe the attached image and list any visible text.",
  },
  {
    key: "templateMultiTurn",
    descKey: "templateMultiTurnDesc",
    text: "Remember my API key prefix across turns and recall it when I ask.",
  },
];
</script>

<template>
  <div class="flex-1 flex flex-col justify-center max-w-2xl mx-auto w-full py-16 px-4">
    <!-- Header — operational, no exclamation headline, no tracked mono eyebrow -->
    <div class="mb-8">
      <h3
        class="text-2xl sm:text-3xl font-medium tracking-tight text-foreground brand-heading mb-2"
      >
        {{ t("chat.apiConsoleTitle") }}
      </h3>
      <p class="text-sm text-muted-foreground leading-relaxed max-w-md">
        {{ t("chat.apiConsoleSubtitle") }}
      </p>
    </div>

    <!-- Request templates -->
    <div v-if="hasModel" class="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2">
      <button
        v-for="tpl in templates"
        :key="tpl.key"
        @click="emit('prompt', tpl.text)"
        class="group/tpl flex flex-col text-left p-4 bg-card/30 border border-border/40 hover:border-primary/30 hover:bg-muted/5 rounded-md transition-colors duration-150 cursor-pointer min-h-[96px] outline-none focus-visible:ring-1 focus-visible:ring-primary"
      >
        <span
          class="text-sm font-medium text-foreground/80 group-hover/tpl:text-foreground transition-colors"
        >
          {{ t(`chat.${tpl.key}`) }}
        </span>
        <span class="text-[11px] text-muted-foreground mt-1 leading-relaxed">
          {{ t(`chat.${tpl.descKey}`) }}
        </span>
        <span
          class="mt-3 self-end text-[11px] font-medium text-muted-foreground/50 group-hover/tpl:text-foreground transition-colors"
        >
          {{ t("chat.runTemplate") }} &rarr;
        </span>
      </button>
    </div>

    <!-- No model selected state -->
    <div v-else class="border-l border-action-amber/70 pl-4 py-2 mt-2">
      <p class="text-[11px] font-medium text-action-amber mb-1 uppercase tracking-wide">
        {{ t("chat.modelRequired") }}
      </p>
      <p class="text-xs text-muted-foreground leading-relaxed">
        {{ t("chat.selectModelHelp") }}
      </p>
    </div>
  </div>
</template>
