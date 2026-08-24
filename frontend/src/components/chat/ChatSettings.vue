<script setup lang="ts">
import {
  Brain,
  ChevronDown,
  ChevronUp,
  Globe,
  Paintbrush,
  Plus,
  Sliders,
  Sparkles,
  Thermometer,
  Trash2,
  Type,
  Volume2,
  Wand2,
  Wrench,
  X,
} from "@lucide/vue";
import { computed, onUnmounted, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useModelStore } from "@/stores/models";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { WebSearchConfig, WebSearchContextSize } from "@/composables/useChat";
import type { CustomVariable, ToolDefinition } from "@/adapters/types";

const props = defineProps<{
  open: boolean;
  temperature: number;
  temperatureEnabled: boolean;
  maxTokens: number | null;
  maxTokensEnabled: boolean;
  topP: number;
  topPEnabled: boolean;
  frequencyPenalty: number;
  frequencyPenaltyEnabled: boolean;
  presencePenalty: number;
  presencePenaltyEnabled: boolean;
  systemPrompt: string;
  systemPromptEnabled: boolean;
  reasoningEffort: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "";
  reasoningEffortEnabled: boolean;
  customVariables: CustomVariable[];
  tools: ToolDefinition[];
  webSearch: WebSearchConfig;
  speechVoice: string;
  speechSpeed: number;
  speechModel: string;
}>();

const emit = defineEmits<{
  (e: "update:open", value: boolean): void;
  (e: "update:temperature", value: number): void;
  (e: "update:temperatureEnabled", value: boolean): void;
  (e: "update:maxTokens", value: number | null): void;
  (e: "update:maxTokensEnabled", value: boolean): void;
  (e: "update:topP", value: number): void;
  (e: "update:topPEnabled", value: boolean): void;
  (e: "update:frequencyPenalty", value: number): void;
  (e: "update:frequencyPenaltyEnabled", value: boolean): void;
  (e: "update:presencePenalty", value: number): void;
  (e: "update:presencePenaltyEnabled", value: boolean): void;
  (e: "update:systemPrompt", value: string): void;
  (e: "update:systemPromptEnabled", value: boolean): void;
  (
    e: "update:reasoningEffort",
    value: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | ""
  ): void;
  (e: "update:reasoningEffortEnabled", value: boolean): void;
  (e: "update:customVariables", value: CustomVariable[]): void;
  (e: "update:tools", value: ToolDefinition[]): void;
  (e: "update:webSearch", value: WebSearchConfig): void;
  (e: "update:speechVoice", value: string): void;
  (e: "update:speechSpeed", value: number): void;
  (e: "update:speechModel", value: string): void;
  (e: "reset"): void;
}>();

const { t } = useI18n();

type ReasoningEffort = typeof props.reasoningEffort;

const emitUpdate = (name: keyof typeof props, value: unknown) => {
  (emit as (e: string, v: unknown) => void)(`update:${name}`, value);
};

function syncProp<T>(name: keyof typeof props) {
  return computed<T>({
    get: () => props[name] as T,
    set: (value: T) => emitUpdate(name, value),
  });
}

const localTemperature = syncProp<number>("temperature");
const localTemperatureEnabled = syncProp<boolean>("temperatureEnabled");
const localMaxTokens = syncProp<number | null>("maxTokens");
const localMaxTokensEnabled = syncProp<boolean>("maxTokensEnabled");
const localTopP = syncProp<number>("topP");
const localTopPEnabled = syncProp<boolean>("topPEnabled");
const localFrequencyPenalty = syncProp<number>("frequencyPenalty");
const localFrequencyPenaltyEnabled = syncProp<boolean>("frequencyPenaltyEnabled");
const localPresencePenalty = syncProp<number>("presencePenalty");
const localPresencePenaltyEnabled = syncProp<boolean>("presencePenaltyEnabled");
const localSystemPrompt = syncProp<string>("systemPrompt");
const localSystemPromptEnabled = syncProp<boolean>("systemPromptEnabled");
const localReasoningEffort = syncProp<ReasoningEffort>("reasoningEffort");
const localReasoningEffortEnabled = syncProp<boolean>("reasoningEffortEnabled");
const localCustomVariables = syncProp<CustomVariable[]>("customVariables");
const localTools = syncProp<ToolDefinition[]>("tools");
const localWebSearch = syncProp<WebSearchConfig>("webSearch");
const localSpeechVoice = syncProp<string>("speechVoice");
const localSpeechSpeed = syncProp<number>("speechSpeed");
const localSpeechModel = syncProp<string>("speechModel");

const modelStore = useModelStore();
const speechModels = computed(() => modelStore.models);
const hasConfiguredTtsModel = computed(() => speechModels.value.length > 0);

const browserVoices = ref<SpeechSynthesisVoice[]>([]);

let updateVoices: (() => void) | null = null;
if (typeof speechSynthesis !== "undefined") {
  updateVoices = () => {
    browserVoices.value = speechSynthesis.getVoices();
  };
  updateVoices();
  speechSynthesis.addEventListener("voiceschanged", updateVoices);
}

onUnmounted(() => {
  if (typeof speechSynthesis !== "undefined") {
    speechSynthesis.cancel();
    if (updateVoices) {
      speechSynthesis.removeEventListener("voiceschanged", updateVoices);
    }
  }
});

const availableVoices = computed(() => browserVoices.value.map((v) => v.name));
const isBrowserTts = computed(() => localSpeechModel.value === "browser");
const isGptTts = computed(() => ["tts-1", "tts-1-hd"].includes(localSpeechModel.value));

const GPT_TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"];

watch(localSpeechModel, (newModel, oldModel) => {
  if (newModel === "browser") {
    const voices = availableVoices.value;
    if (voices.length && !voices.includes(localSpeechVoice.value)) {
      localSpeechVoice.value = voices[0] ?? "";
    }
  } else if (newModel === "tts-1" || newModel === "tts-1-hd") {
    if (!GPT_TTS_VOICES.includes(localSpeechVoice.value)) {
      localSpeechVoice.value = "alloy";
    }
  } else if (oldModel === "browser") {
    localSpeechVoice.value = "";
  }
});

watch(availableVoices, (voices) => {
  if (isBrowserTts.value && voices.length && !voices.includes(localSpeechVoice.value)) {
    localSpeechVoice.value = voices[0] ?? "";
  }
});

const PREDEFINED_TOOLS = [
  {
    label: "Weather Tool",
    name: "get_current_weather",
    description: "Get the current weather for a location",
    parameters: JSON.stringify(
      {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state, e.g. San Francisco, CA",
          },
          unit: {
            type: "string",
            enum: ["celsius", "fahrenheit"],
          },
        },
        required: ["location"],
      },
      null,
      2
    ),
  },
  {
    label: "Stock Tool",
    name: "get_stock_price",
    description: "Get the current stock price for a ticker symbol",
    parameters: JSON.stringify(
      {
        type: "object",
        properties: {
          ticker: {
            type: "string",
            description: "The stock ticker symbol, e.g. GOOG",
          },
        },
        required: ["ticker"],
      },
      null,
      2
    ),
  },
];

const addTool = () => {
  const newList = [
    ...localTools.value,
    {
      name: "new_tool",
      description: "Description of the tool",
      parameters: JSON.stringify(
        {
          type: "object",
          properties: {
            param1: { type: "string", description: "Parameter description" },
          },
          required: ["param1"],
        },
        null,
        2
      ),
      enabled: true,
    },
  ];
  emit("update:tools", newList);
};

const addPredefinedTool = (tmpl: (typeof PREDEFINED_TOOLS)[number]) => {
  const newList = [
    ...localTools.value,
    {
      name: tmpl.name,
      description: tmpl.description,
      parameters: tmpl.parameters,
      enabled: true,
    },
  ];
  emit("update:tools", newList);
};

const removeTool = (index: number) => {
  emit(
    "update:tools",
    localTools.value.filter((_, i) => i !== index)
  );
};

const updateTool = (
  index: number,
  field: keyof ToolDefinition,
  value: ToolDefinition[keyof ToolDefinition]
) => {
  emit(
    "update:tools",
    localTools.value.map((tool, i) => (i === index ? { ...tool, [field]: value } : tool))
  );
};

const addCustomVariable = () => {
  const newList = [
    ...localCustomVariables.value,
    { key: "", value: "", type: "string" as const, enabled: true },
  ];
  emit("update:customVariables", newList);
};

const removeCustomVariable = (index: number) => {
  emit(
    "update:customVariables",
    localCustomVariables.value.filter((_, i) => i !== index)
  );
};

const updateCustomVariable = (
  index: number,
  field: keyof CustomVariable,
  value: CustomVariable[keyof CustomVariable]
) => {
  emit(
    "update:customVariables",
    localCustomVariables.value.map((v, i) => (i === index ? { ...v, [field]: value } : v))
  );
};

const updateCustomVariableType = (index: number, type: CustomVariable["type"]) => {
  emit(
    "update:customVariables",
    localCustomVariables.value.map((v, i) =>
      i === index ? { ...v, type, value: type === "boolean" ? "true" : "" } : v
    )
  );
};

const updateWebSearch = (
  field: keyof WebSearchConfig,
  value: WebSearchConfig[keyof WebSearchConfig]
) => {
  emit("update:webSearch", { ...localWebSearch.value, [field]: value });
};

const close = () => emit("update:open", false);
</script>

<template>
  <!-- Settings Panel Overlay (mobile) -->
  <transition
    enter-active-class="settings-overlay-enter-active"
    leave-active-class="settings-overlay-leave-active"
  >
    <div
      v-if="open"
      class="fixed inset-0 overlay-light backdrop-blur-[2px] z-40 lg:hidden"
      @click="close"
    />
  </transition>

  <!-- Settings Panel -->
  <transition
    enter-active-class="settings-panel-enter-active"
    leave-active-class="settings-panel-leave-active"
  >
    <aside
      v-if="open"
      class="fixed right-0 top-0 bottom-0 w-80 z-40 lg:relative lg:z-auto shrink-0 border-l border-border/50 bg-card lg:shadow-none lg:bg-card overflow-hidden will-change-transform"
    >
      <div class="h-full flex flex-col">
        <!-- Panel Header -->
        <div class="p-4 border-b border-border/50 lg:pt-4 pt-[calc(env(safe-area-inset-top)+1rem)]">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <div class="icon-container p-1.5">
                <Sliders class="w-4 h-4 text-primary" />
              </div>
              <h3 class="font-semibold text-sm">{{ t("chat.advancedSettings") }}</h3>
            </div>
            <div class="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                @click="emit('reset')"
                class="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
              >
                <Paintbrush class="w-3 h-3 mr-1" />
                {{ t("chat.resetSettings") }}
              </Button>
              <Button variant="ghost" size="icon" class="h-10 w-10" @click="close">
                <X class="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>

        <!-- Settings Content -->
        <div class="flex-1 overflow-y-auto p-4 space-y-5">
          <!-- System Prompt -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-medium flex items-center gap-2">
                <Wand2 class="w-3.5 h-3.5 text-action-violet" />
                {{ t("chat.systemPrompt") }}
              </Label>
              <Switch v-model="localSystemPromptEnabled" class="scale-75" />
            </div>
            <div :class="{ 'opacity-50 pointer-events-none': !localSystemPromptEnabled }">
              <Textarea
                v-model="localSystemPrompt"
                placeholder="You are a helpful assistant..."
                :disabled="!localSystemPromptEnabled"
                class="min-h-20 text-sm bg-muted/30 border-border/50 resize-none"
                rows="3"
              />
              <p class="text-[11px] text-muted-foreground mt-1">{{ t("chat.systemPromptHelp") }}</p>
            </div>
          </div>

          <!-- Temperature -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-medium flex items-center gap-2">
                <Thermometer class="w-3.5 h-3.5 text-action-amber" />
                {{ t("chat.temperature") }}
              </Label>
              <div class="flex items-center gap-2">
                <Badge
                  variant="outline"
                  class="font-mono text-[11px] h-5"
                  :class="{ 'opacity-50': !localTemperatureEnabled }"
                >
                  {{ localTemperature.toFixed(2) }}
                </Badge>
                <Switch v-model="localTemperatureEnabled" class="scale-75" />
              </div>
            </div>
            <div
              :class="{ 'opacity-50 pointer-events-none': !localTemperatureEnabled }"
              class="space-y-1.5"
            >
              <Slider
                :model-value="[localTemperature]"
                @update:model-value="localTemperature = ($event as number[])[0] ?? 0"
                :min="0"
                :max="2"
                :step="0.01"
                :disabled="!localTemperatureEnabled"
                class="w-full"
              />
              <div class="flex justify-between text-[11px] text-muted-foreground mt-1">
                <span>{{ t("chat.precise") }}</span>
                <span>{{ t("chat.creative") }}</span>
              </div>
            </div>
          </div>

          <!-- Max Tokens -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-medium flex items-center gap-2">
                <Type class="w-3.5 h-3.5 text-action-blue" />
                {{ t("chat.maxTokens") }}
              </Label>
              <div class="flex items-center gap-2">
                <Badge
                  v-if="localMaxTokens && localMaxTokensEnabled"
                  variant="outline"
                  class="font-mono text-[11px] h-5"
                >
                  {{ localMaxTokens }}
                </Badge>
                <Badge
                  v-else-if="localMaxTokensEnabled"
                  variant="secondary"
                  class="text-[11px] h-5"
                >
                  {{ t("chat.unlimited") }}
                </Badge>
                <Switch v-model="localMaxTokensEnabled" class="scale-75" />
              </div>
            </div>
            <div :class="{ 'opacity-50 pointer-events-none': !localMaxTokensEnabled }">
              <NumberInput
                v-model.number="localMaxTokens"
                :placeholder="t('chat.maxTokensPlaceholder')"
                :disabled="!localMaxTokensEnabled"
                class="h-9 text-sm bg-muted/30 border-border/50 font-mono"
                min="1"
              />
              <p class="text-[11px] text-muted-foreground mt-1">{{ t("chat.maxTokensHelp") }}</p>
            </div>
          </div>

          <!-- Top P -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-medium flex items-center gap-2">
                <Sparkles class="w-3.5 h-3.5 text-action-amber" />
                {{ t("chat.topP") }}
              </Label>
              <div class="flex items-center gap-2">
                <Badge
                  variant="outline"
                  class="font-mono text-[11px] h-5"
                  :class="{ 'opacity-50': !localTopPEnabled }"
                >
                  {{ localTopP.toFixed(2) }}
                </Badge>
                <Switch v-model="localTopPEnabled" class="scale-75" />
              </div>
            </div>
            <div
              :class="{ 'opacity-50 pointer-events-none': !localTopPEnabled }"
              class="space-y-1.5"
            >
              <Slider
                :model-value="[localTopP]"
                @update:model-value="localTopP = ($event as number[])[0] ?? 0"
                :min="0"
                :max="1"
                :step="0.01"
                :disabled="!localTopPEnabled"
                class="w-full"
              />
              <p class="text-[11px] text-muted-foreground mt-1">{{ t("chat.topPHelp") }}</p>
            </div>
          </div>

          <!-- Frequency Penalty -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-medium flex items-center gap-2">
                <ChevronDown class="w-3.5 h-3.5 text-action-blue" />
                {{ t("chat.frequencyPenalty") }}
              </Label>
              <div class="flex items-center gap-2">
                <Badge
                  variant="outline"
                  class="font-mono text-[11px] h-5"
                  :class="{ 'opacity-50': !localFrequencyPenaltyEnabled }"
                >
                  {{ localFrequencyPenalty.toFixed(2) }}
                </Badge>
                <Switch v-model="localFrequencyPenaltyEnabled" class="scale-75" />
              </div>
            </div>
            <div
              :class="{ 'opacity-50 pointer-events-none': !localFrequencyPenaltyEnabled }"
              class="space-y-1.5"
            >
              <Slider
                :model-value="[localFrequencyPenalty]"
                @update:model-value="localFrequencyPenalty = ($event as number[])[0] ?? 0"
                :min="-2"
                :max="2"
                :step="0.01"
                :disabled="!localFrequencyPenaltyEnabled"
                class="w-full"
              />
              <p class="text-[11px] text-muted-foreground mt-1">
                {{ t("chat.frequencyPenaltyHelp") }}
              </p>
            </div>
          </div>

          <!-- Presence Penalty -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-medium flex items-center gap-2">
                <ChevronUp class="w-3.5 h-3.5 text-muted-foreground" />
                {{ t("chat.presencePenalty") }}
              </Label>
              <div class="flex items-center gap-2">
                <Badge
                  variant="outline"
                  class="font-mono text-[11px] h-5"
                  :class="{ 'opacity-50': !localPresencePenaltyEnabled }"
                >
                  {{ localPresencePenalty.toFixed(2) }}
                </Badge>
                <Switch v-model="localPresencePenaltyEnabled" class="scale-75" />
              </div>
            </div>
            <div
              :class="{ 'opacity-50 pointer-events-none': !localPresencePenaltyEnabled }"
              class="space-y-1.5"
            >
              <Slider
                :model-value="[localPresencePenalty]"
                @update:model-value="localPresencePenalty = ($event as number[])[0] ?? 0"
                :min="-2"
                :max="2"
                :step="0.01"
                :disabled="!localPresencePenaltyEnabled"
                class="w-full"
              />
              <p class="text-[11px] text-muted-foreground mt-1">
                {{ t("chat.presencePenaltyHelp") }}
              </p>
            </div>
          </div>

          <!-- Reasoning Effort -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-medium flex items-center gap-2">
                <Brain class="w-3.5 h-3.5 text-action-rose" />
                {{ t("chat.reasoningEffort") }}
              </Label>
              <Switch v-model="localReasoningEffortEnabled" class="scale-75" />
            </div>
            <div :class="{ 'opacity-50 pointer-events-none': !localReasoningEffortEnabled }">
              <Select
                :model-value="localReasoningEffort"
                @update:model-value="localReasoningEffort = $event as ReasoningEffort"
                :disabled="!localReasoningEffortEnabled"
              >
                <SelectTrigger class="h-9 text-sm">
                  <SelectValue :placeholder="t('chat.reasoningEffortDefault')" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">none</SelectItem>
                  <SelectItem value="minimal">minimal</SelectItem>
                  <SelectItem value="low">low</SelectItem>
                  <SelectItem value="medium">medium</SelectItem>
                  <SelectItem value="high">high</SelectItem>
                  <SelectItem value="xhigh">xhigh</SelectItem>
                </SelectContent>
              </Select>
              <p class="text-[11px] text-muted-foreground mt-1">
                {{
                  t("chat.reasoningEffortHelp") ||
                  "Controls the reasoning effort for models supporting reasoning capabilities."
                }}
              </p>
            </div>
          </div>

          <!-- Text-to-Speech (TTS) Settings Section -->
          <Separator class="my-4 border-border/50" />
          <div class="space-y-3">
            <Label class="text-xs font-semibold text-muted-foreground flex items-center gap-2">
              <Volume2 class="w-3.5 h-3.5 text-muted-foreground" />
              {{ t("chat.ttsSettings") }}
            </Label>

            <!-- TTS Model Selection -->
            <div class="space-y-1">
              <Label class="text-[11px] text-muted-foreground">{{ t("chat.ttsModel") }}</Label>
              <Select v-model="localSpeechModel">
                <SelectTrigger class="h-9 text-sm">
                  <SelectValue :placeholder="t('chat.selectModel')" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem v-if="!hasConfiguredTtsModel" value="browser">
                    {{ t("chat.browserTts") }}
                  </SelectItem>
                  <SelectItem v-for="m in speechModels" :key="m.id" :value="m.name">
                    {{ m.name }}
                  </SelectItem>
                  <SelectItem v-if="hasConfiguredTtsModel" value="browser">
                    {{ t("chat.browserTts") }}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <!-- TTS Voice Selection -->
            <div class="space-y-1">
              <Label class="text-[11px] text-muted-foreground">{{ t("chat.speechVoice") }}</Label>
              <Select v-if="isBrowserTts" v-model="localSpeechVoice">
                <SelectTrigger class="h-9 text-sm">
                  <SelectValue :placeholder="t('chat.selectVoice')" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem v-for="v in availableVoices" :key="v" :value="v">
                    {{ v }}
                  </SelectItem>
                </SelectContent>
              </Select>
              <Select v-else-if="isGptTts" v-model="localSpeechVoice">
                <SelectTrigger class="h-9 text-sm">
                  <SelectValue :placeholder="t('chat.selectVoice')" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem v-for="v in GPT_TTS_VOICES" :key="v" :value="v">
                    {{ v }}
                  </SelectItem>
                </SelectContent>
              </Select>
              <Input
                v-else
                type="text"
                v-model="localSpeechVoice"
                placeholder="e.g. alloy"
                class="h-9 bg-background text-sm px-3 shadow-none"
              />
            </div>

            <!-- TTS Speed Selection -->
            <div class="space-y-1">
              <Label class="text-[11px] text-muted-foreground">{{ t("chat.speechSpeed") }}</Label>
              <div class="flex items-center gap-2">
                <Input
                  type="number"
                  v-model.number="localSpeechSpeed"
                  min="0.25"
                  max="4.0"
                  step="0.05"
                  class="h-9 bg-background text-sm px-3 shadow-none text-center flex-1"
                />
                <span class="text-muted-foreground text-[11px] whitespace-nowrap font-sans"
                  >x (0.25 - 4.0)</span
                >
              </div>
            </div>
          </div>

          <!-- Custom Variables Section -->
          <Separator class="my-4 border-border/50" />
          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-semibold text-muted-foreground flex items-center gap-2">
                <Sparkles class="w-3.5 h-3.5 text-action-amber" />
                {{ t("chat.customVariables") }}
              </Label>
              <Button
                variant="outline"
                size="sm"
                class="h-7 px-2 text-xs"
                @click="addCustomVariable"
              >
                <Plus class="w-3 h-3 mr-1" />
                {{ t("chat.add") }}
              </Button>
            </div>

            <div
              v-if="localCustomVariables.length === 0"
              class="text-center py-4 border border-dashed border-border/60 rounded-lg bg-muted/10"
            >
              <p class="text-[11px] text-muted-foreground">
                {{ t("chat.noCustomVariables") }}
              </p>
            </div>

            <div v-else class="space-y-3 max-h-60 overflow-y-auto pr-1">
              <div
                v-for="(v, index) in localCustomVariables"
                :key="index"
                class="p-2 border border-border/50 rounded-lg bg-muted/10 space-y-2 animate-in fade-in slide-in-from-top-1 duration-200"
              >
                <div class="flex items-center justify-between gap-2">
                  <div class="flex items-center gap-1.5">
                    <Switch
                      :model-value="v.enabled"
                      @update:model-value="updateCustomVariable(index, 'enabled', $event)"
                      class="scale-75"
                    />
                    <span class="text-[11px] font-medium text-muted-foreground">
                      {{ v.enabled ? t("chat.enabled") : t("chat.disabled") }}
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-6 w-6 text-muted-foreground hover:text-destructive"
                    @click="removeCustomVariable(index)"
                  >
                    <Trash2 class="w-3.5 h-3.5" />
                  </Button>
                </div>

                <div class="grid grid-cols-2 gap-2">
                  <div class="space-y-1">
                    <Label class="text-[11px] text-muted-foreground">{{
                      t("chat.variableKey")
                    }}</Label>
                    <Input
                      :model-value="v.key"
                      @update:model-value="updateCustomVariable(index, 'key', $event as string)"
                      placeholder="e.g. seed"
                      class="h-7 text-xs bg-background"
                    />
                  </div>
                  <div class="space-y-1">
                    <Label class="text-[11px] text-muted-foreground">{{
                      t("chat.variableType")
                    }}</Label>
                    <Select
                      :model-value="v.type"
                      @update:model-value="
                        updateCustomVariableType(index, $event as CustomVariable['type'])
                      "
                    >
                      <SelectTrigger class="h-7 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="string">String</SelectItem>
                        <SelectItem value="number">Number</SelectItem>
                        <SelectItem value="boolean">Boolean</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div class="space-y-1">
                  <Label class="text-[11px] text-muted-foreground">{{
                    t("chat.variableValue")
                  }}</Label>
                  <Input
                    v-if="v.type !== 'boolean'"
                    :model-value="v.value"
                    @update:model-value="updateCustomVariable(index, 'value', $event as string)"
                    :placeholder="v.type === 'number' ? 'e.g. 42' : 'e.g. high'"
                    class="h-7 text-xs bg-background"
                  />
                  <Select
                    v-else
                    :model-value="v.value"
                    @update:model-value="updateCustomVariable(index, 'value', $event as string)"
                  >
                    <SelectTrigger class="h-7 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="true">True</SelectItem>
                      <SelectItem value="false">False</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </div>

          <!-- Web Search Section -->
          <Separator class="my-4 border-border/50" />
          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-semibold text-muted-foreground flex items-center gap-2">
                <Globe class="w-3.5 h-3.5 text-action-blue" />
                {{ t("chat.webSearch") }}
              </Label>
              <Switch
                :model-value="localWebSearch.enabled"
                @update:model-value="updateWebSearch('enabled', $event)"
                class="scale-75"
              />
            </div>

            <div
              class="space-y-3"
              :class="{ 'opacity-50 pointer-events-none': !localWebSearch.enabled }"
            >
              <p class="text-[11px] text-muted-foreground leading-relaxed">
                {{ t("chat.webSearchHelp") }}
              </p>

              <!-- Anthropic: max_uses -->
              <div class="space-y-1">
                <div class="flex items-center justify-between">
                  <Label class="text-[11px] text-muted-foreground">{{
                    t("chat.webSearchMaxUses")
                  }}</Label>
                  <Switch
                    :model-value="localWebSearch.maxUses !== null"
                    @update:model-value="updateWebSearch('maxUses', $event ? 5 : null)"
                    class="scale-75"
                  />
                </div>
                <NumberInput
                  v-if="localWebSearch.maxUses !== null"
                  :model-value="localWebSearch.maxUses"
                  @update:model-value="updateWebSearch('maxUses', $event)"
                  min="1"
                  max="100"
                  step="1"
                  class="h-8 text-xs bg-background"
                />
                <p class="text-[11px] text-muted-foreground">
                  {{ t("chat.webSearchMaxUsesHelp") }}
                </p>
              </div>

              <!-- OpenAI: search_context_size -->
              <div class="space-y-1">
                <Label class="text-[11px] text-muted-foreground">{{
                  t("chat.webSearchContextSize")
                }}</Label>
                <Select
                  :model-value="localWebSearch.searchContextSize"
                  @update:model-value="
                    updateWebSearch('searchContextSize', $event as WebSearchContextSize)
                  "
                >
                  <SelectTrigger class="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">low</SelectItem>
                    <SelectItem value="medium">medium</SelectItem>
                    <SelectItem value="high">high</SelectItem>
                  </SelectContent>
                </Select>
                <p class="text-[11px] text-muted-foreground">
                  {{ t("chat.webSearchContextSizeHelp") }}
                </p>
              </div>

              <!-- OpenAI: include sources -->
              <div class="flex items-center justify-between">
                <Label class="text-[11px] text-muted-foreground">{{
                  t("chat.webSearchIncludeSources")
                }}</Label>
                <Switch
                  :model-value="localWebSearch.includeSources"
                  @update:model-value="updateWebSearch('includeSources', $event)"
                  class="scale-75"
                />
              </div>
            </div>
          </div>

          <!-- Tools Section -->
          <Separator class="my-4 border-border/50" />
          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <Label class="text-xs font-semibold text-muted-foreground flex items-center gap-2">
                <Wrench class="w-3.5 h-3.5 text-action-amber" />
                {{ t("chat.toolsDefinition") }}
              </Label>
              <Button variant="outline" size="sm" class="h-7 px-2 text-xs" @click="addTool">
                <Plus class="w-3 h-3 mr-1" />
                {{ t("chat.addTool") }}
              </Button>
            </div>

            <!-- Quick Template Buttons -->
            <div class="flex flex-wrap gap-1.5">
              <span class="text-[11px] text-muted-foreground self-center mr-1">{{
                t("chat.toolTemplates")
              }}</span>
              <Button
                v-for="tmpl in PREDEFINED_TOOLS"
                :key="tmpl.name"
                variant="secondary"
                size="sm"
                class="h-6 px-1.5 text-[11px] bg-action-amber/10 hover:bg-action-amber/20 text-action-amber border border-action-amber/20"
                @click="addPredefinedTool(tmpl)"
              >
                +
                {{
                  t("chat." + (tmpl.name === "get_current_weather" ? "weatherTool" : "stockTool"))
                }}
              </Button>
            </div>

            <div
              v-if="localTools.length === 0"
              class="text-center py-4 border border-dashed border-border/60 rounded-lg bg-muted/10"
            >
              <p class="text-[11px] text-muted-foreground">
                {{ t("chat.noTools") }}
              </p>
            </div>

            <div v-else class="space-y-3 max-h-80 overflow-y-auto pr-1">
              <div
                v-for="(tool, index) in localTools"
                :key="index"
                class="p-2.5 border border-border/50 rounded-lg bg-muted/10 space-y-2 animate-in fade-in slide-in-from-top-1 duration-200"
              >
                <div class="flex items-center justify-between gap-2">
                  <div class="flex items-center gap-1.5">
                    <Switch
                      :model-value="tool.enabled"
                      @update:model-value="updateTool(index, 'enabled', $event)"
                      class="scale-75"
                    />
                    <span class="text-[11px] font-medium text-muted-foreground">
                      {{ tool.enabled ? t("chat.enabled") : t("chat.disabled") }}
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-6 w-6 text-muted-foreground hover:text-destructive"
                    @click="removeTool(index)"
                  >
                    <Trash2 class="w-3.5 h-3.5" />
                  </Button>
                </div>

                <div class="space-y-1">
                  <Label class="text-[11px] text-muted-foreground font-semibold">{{
                    t("chat.toolName")
                  }}</Label>
                  <Input
                    :model-value="tool.name"
                    @update:model-value="updateTool(index, 'name', $event as string)"
                    :placeholder="t('chat.toolNamePlaceholder')"
                    class="h-7 text-xs bg-background font-mono"
                  />
                </div>

                <div class="space-y-1">
                  <Label class="text-[11px] text-muted-foreground font-semibold">{{
                    t("chat.toolDescription")
                  }}</Label>
                  <Input
                    :model-value="tool.description"
                    @update:model-value="updateTool(index, 'description', $event as string)"
                    :placeholder="t('chat.toolDescriptionPlaceholder')"
                    class="h-7 text-xs bg-background"
                  />
                </div>

                <div class="space-y-1">
                  <Label class="text-[11px] text-muted-foreground font-semibold">{{
                    t("chat.toolParameters")
                  }}</Label>
                  <Textarea
                    :model-value="tool.parameters"
                    @update:model-value="updateTool(index, 'parameters', $event as string)"
                    :placeholder="t('chat.toolParametersPlaceholder')"
                    rows="4"
                    class="text-xs bg-background font-mono resize-y"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Panel Footer -->
        <Separator />
        <div class="p-4 bg-muted/20">
          <div class="flex items-center gap-2 text-[11px] text-muted-foreground">
            <Sparkles class="w-3 h-3" />
            <span>{{ t("chat.settingsApplyNextMessage") }}</span>
          </div>
        </div>
      </div>
    </aside>
  </transition>
</template>
