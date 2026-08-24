<script setup lang="ts">
import {
  ChevronDown,
  Eraser,
  Loader2,
  ArrowUp,
  Settings2,
  Wrench,
  Paperclip,
  X,
  FileText,
  Code,
  File,
  Globe,
  Cpu,
} from "@lucide/vue";
import { useThrottleFn, useWindowSize } from "@vueuse/core";
import { storeToRefs } from "pinia";
import { computed, nextTick, onActivated, onMounted, onUnmounted, ref, watch } from "vue";

defineOptions({ name: "ChatView" });
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import ChatEmptyState from "@/components/chat/ChatEmptyState.vue";
import ChatMessage from "@/components/chat/ChatMessage.vue";
import ChatSettings from "@/components/chat/ChatSettings.vue";
import RunInspector from "@/components/playground/RunInspector.vue";
import RunSpecimen from "@/components/playground/RunSpecimen.vue";
import SpecimenTray from "@/components/playground/SpecimenTray.vue";
import type { ToolDefinition } from "@/adapters/types";
import type { CustomVariable } from "@/adapters/types";
import { Button } from "@/components/ui/button";
import { getErrorMessage } from "@/utils/error";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import type { ChatMessage as ChatMessageType, ContentPart } from "@/types/schemas";
import { useChat, type ChatOptions } from "@/composables/useChat";
import type { WebSearchConfig } from "@/composables/useChat";
import { getDefaultWebSearchConfig, loadWebSearchConfig } from "@/composables/useChat";
import { chatApi } from "@/services/api/chat";
import { useChatStore } from "@/stores/chat";
import { useAuthStore } from "@/stores/auth";
import { STORAGE_KEYS } from "@/constants/storageKeys";

/** Tool name used by the backend for server-side web search (Responses API). */
const WEB_SEARCH_TOOL_NAME = "web_search";

import ConfirmDialog from "@/components/common/ConfirmDialog.vue";
import { Input } from "@/components/ui/input";
import { useModelStore } from "@/stores/models";
import { getModelIconUrl, isMonoIcon } from "@/utils/icons";
import { endpointName, formatLatency } from "@/utils/runs";

interface ModelOption {
  id: string;
  provider: string;
}

const loadSelectedModel = (): string | null => {
  return localStorage.getItem(STORAGE_KEYS.CHAT_MODEL);
};

const loadSelectedEndpoint = (): string => {
  return localStorage.getItem(STORAGE_KEYS.CHAT_ENDPOINT) || "/v1/chat/completions";
};

const models = ref<ModelOption[]>([]);
const selectedModel = ref<string | null>(loadSelectedModel());
const input = ref("");

const getModelIcon = (modelId: string | null | undefined) => {
  if (!modelId) return null;
  const apiModel = models.value.find((m) => m.id === modelId);
  const configModel = modelStore.models.find((cm) => cm.name === modelId);
  return getModelIconUrl(
    modelId,
    apiModel?.provider || configModel?.providers?.[0]?.provider_name,
    configModel?.icon_url
  );
};
const isLoadingModels = ref(false);

interface AttachedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  url: string;
  base64?: string;
  text?: string;
}

const attachedFiles = ref<AttachedFile[]>([]);
const fileInputRef = ref<HTMLInputElement | null>(null);
const isDragging = ref(false);

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB limit

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
};

const TEXT_FILE_EXTENSIONS = new Set([
  "txt",
  "md",
  "markdown",
  "py",
  "js",
  "ts",
  "tsx",
  "jsx",
  "html",
  "css",
  "json",
  "yaml",
  "yml",
  "xml",
  "csv",
  "sh",
  "bash",
  "sql",
  "rs",
  "go",
  "c",
  "cpp",
  "h",
  "java",
  "kt",
  "rb",
  "pl",
  "pm",
  "php",
  "ini",
  "conf",
  "env",
  "toml",
  "gradle",
  "bat",
  "ps1",
]);

const isTextFile = (filename: string): boolean => {
  const ext = filename.split(".").pop()?.toLowerCase();
  return ext ? TEXT_FILE_EXTENSIONS.has(ext) : false;
};

const triggerFileInput = () => {
  fileInputRef.value?.click();
};

const removeAttachedFile = (id: string) => {
  const file = attachedFiles.value.find((f) => f.id === id);
  if (file?.url.startsWith("blob:")) {
    URL.revokeObjectURL(file.url);
  }
  attachedFiles.value = attachedFiles.value.filter((f) => f.id !== id);
};

const processFiles = (files: FileList) => {
  Array.from(files).forEach((file) => {
    if (file.size > MAX_FILE_SIZE) {
      toast.warning(t("chat.fileSizeLimit", { size: formatFileSize(MAX_FILE_SIZE) }));
      return;
    }

    const id = `file_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    const url = URL.createObjectURL(file);

    const attached: AttachedFile = {
      id,
      name: file.name,
      size: file.size,
      type: file.type,
      url,
    };

    if (isTextFile(file.name)) {
      const reader = new FileReader();
      reader.onload = (e) => {
        attached.text = e.target?.result as string;
      };
      reader.readAsText(file);
    } else {
      const reader = new FileReader();
      reader.onload = (e) => {
        attached.base64 = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
    attachedFiles.value.push(attached);
  });
};

const handleFileChange = (e: Event) => {
  const target = e.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    processFiles(target.files);
    target.value = "";
  }
};

const handleDrop = (e: DragEvent) => {
  isDragging.value = false;
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    processFiles(files);
  }
};

const handlePaste = (e: ClipboardEvent) => {
  const files = e.clipboardData?.files;
  if (files && files.length > 0) {
    e.preventDefault();
    processFiles(files);
  }
};

const { t } = useI18n();

// UI state
const messagesContainer = ref<HTMLElement | null>(null);
const textareaRef = ref<HTMLTextAreaElement | null>(null);
const isAutoScroll = ref(true);
const showSettings = ref(false);
const showClearConfirmDialog = ref(false);

// Mobile keyboard handling - use visualViewport for reliable keyboard detection
const { width: windowWidth } = useWindowSize();
const isMobile = computed(() => windowWidth.value < 768);
const isKeyboardVisible = ref(false);
const viewportHeight = ref(typeof window !== "undefined" ? window.innerHeight : 0);

let visualViewportCleanup: (() => void) | null = null;

onMounted(() => {
  if (typeof visualViewport !== "undefined" && visualViewport) {
    const viewport = visualViewport;
    const updateViewport = () => {
      viewportHeight.value = viewport.height;
      isKeyboardVisible.value = viewport.height < window.innerHeight - 50;
    };
    viewport.addEventListener("resize", updateViewport);
    updateViewport();
    visualViewportCleanup = () => viewport.removeEventListener("resize", updateViewport);
  }
});

onUnmounted(() => {
  // Abort any in-flight streaming request
  stopGeneration();
  visualViewportCleanup?.();
  // Clean up blob URLs from attached files to prevent memory leaks
  for (const file of attachedFiles.value) {
    if (file.url.startsWith("blob:")) {
      URL.revokeObjectURL(file.url);
    }
  }
});

const handleFocus = () => {
  if (isMobile.value) {
    isKeyboardVisible.value = true;
    // Use requestAnimationFrame for more reliable timing after keyboard appears
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        textareaRef.value?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    });
  }
};

const handleBlur = () => {
  isKeyboardVisible.value = false;
};

// Settings
const defaultSettings = {
  temperature: 0.7,
  temperatureEnabled: true,
  maxTokens: null,
  maxTokensEnabled: false,
  topP: 1.0,
  topPEnabled: false,
  frequencyPenalty: 0,
  frequencyPenaltyEnabled: false,
  presencePenalty: 0,
  presencePenaltyEnabled: false,
  systemPrompt: "",
  systemPromptEnabled: true,
  reasoningEffort: "" as "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "",
  reasoningEffortEnabled: false,
};

const modelStore = useModelStore();

const parseSpeed = (raw: string | null): number => {
  const v = parseFloat(raw || "1.0");
  if (isNaN(v)) return 1.0;
  return Math.min(4.0, Math.max(0.25, v));
};

const speechVoice = ref(localStorage.getItem(STORAGE_KEYS.SPEECH_VOICE) || "alloy");
const speechSpeed = ref(parseSpeed(localStorage.getItem(STORAGE_KEYS.SPEECH_SPEED)));
const speechModel = ref(localStorage.getItem(STORAGE_KEYS.SPEECH_MODEL) || "tts-1");

const speechVoices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"];

watch(speechVoice, (val) => {
  localStorage.setItem(STORAGE_KEYS.SPEECH_VOICE, val);
});

watch(speechModel, (val) => {
  localStorage.setItem(STORAGE_KEYS.SPEECH_MODEL, val);
});

watch(speechSpeed, (val) => {
  if (isNaN(val)) {
    speechSpeed.value = 1.0;
    return;
  }
  const clamped = Math.min(4.0, Math.max(0.25, val));
  if (clamped !== val) speechSpeed.value = clamped;
  localStorage.setItem(STORAGE_KEYS.SPEECH_SPEED, speechSpeed.value.toString());
});

const selectedEndpoint = ref(loadSelectedEndpoint());

watch(selectedEndpoint, (val) => {
  localStorage.setItem(STORAGE_KEYS.CHAT_ENDPOINT, val);
});

watch(selectedModel, (val) => {
  if (val) {
    localStorage.setItem(STORAGE_KEYS.CHAT_MODEL, val);
  } else {
    localStorage.removeItem(STORAGE_KEYS.CHAT_MODEL);
  }
});

const loadTools = (): ToolDefinition[] => {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.CHAT_TOOLS);
    if (raw) return JSON.parse(raw);
  } catch (e) {
    console.error("Failed to parse tools from localStorage:", e);
  }
  return [];
};
const tools = ref<ToolDefinition[]>(loadTools());
watch(
  tools,
  (val) => {
    try {
      localStorage.setItem(STORAGE_KEYS.CHAT_TOOLS, JSON.stringify(val));
    } catch (e) {
      console.error("Failed to save tools to localStorage:", e);
    }
  },
  { deep: true }
);

const webSearch = ref<WebSearchConfig>(loadWebSearchConfig());
watch(
  webSearch,
  (val) => {
    try {
      localStorage.setItem(STORAGE_KEYS.CHAT_WEB_SEARCH, JSON.stringify(val));
    } catch (e) {
      console.error("Failed to save web search config to localStorage:", e);
    }
  },
  { deep: true }
);

const temperature = ref(defaultSettings.temperature);
const temperatureEnabled = ref(defaultSettings.temperatureEnabled);
const maxTokens = ref<number | null>(defaultSettings.maxTokens);
const maxTokensEnabled = ref(defaultSettings.maxTokensEnabled);
const topP = ref(defaultSettings.topP);
const topPEnabled = ref(defaultSettings.topPEnabled);
const frequencyPenalty = ref(defaultSettings.frequencyPenalty);
const frequencyPenaltyEnabled = ref(defaultSettings.frequencyPenaltyEnabled);
const presencePenalty = ref(defaultSettings.presencePenalty);
const presencePenaltyEnabled = ref(defaultSettings.presencePenaltyEnabled);
const systemPrompt = ref(defaultSettings.systemPrompt);
const systemPromptEnabled = ref(defaultSettings.systemPromptEnabled);
const reasoningEffort = ref(defaultSettings.reasoningEffort);
const reasoningEffortEnabled = ref(defaultSettings.reasoningEffortEnabled);
const customVariables = ref<CustomVariable[]>([]);

const resetSettings = () => {
  temperature.value = defaultSettings.temperature;
  temperatureEnabled.value = defaultSettings.temperatureEnabled;
  maxTokens.value = defaultSettings.maxTokens;
  maxTokensEnabled.value = defaultSettings.maxTokensEnabled;
  topP.value = defaultSettings.topP;
  topPEnabled.value = defaultSettings.topPEnabled;
  frequencyPenalty.value = defaultSettings.frequencyPenalty;
  frequencyPenaltyEnabled.value = defaultSettings.frequencyPenaltyEnabled;
  presencePenalty.value = defaultSettings.presencePenalty;
  presencePenaltyEnabled.value = defaultSettings.presencePenaltyEnabled;
  systemPrompt.value = defaultSettings.systemPrompt;
  systemPromptEnabled.value = defaultSettings.systemPromptEnabled;
  reasoningEffort.value = defaultSettings.reasoningEffort;
  reasoningEffortEnabled.value = defaultSettings.reasoningEffortEnabled;
  speechVoice.value = "alloy";
  speechSpeed.value = 1.0;
  speechModel.value = "tts-1";
  customVariables.value = [];
  tools.value = [];
  webSearch.value = getDefaultWebSearchConfig();
};

// Computed
const hasActiveSettings = computed(() => {
  return (
    temperatureEnabled.value !== defaultSettings.temperatureEnabled ||
    temperature.value !== defaultSettings.temperature ||
    maxTokensEnabled.value !== defaultSettings.maxTokensEnabled ||
    maxTokens.value !== defaultSettings.maxTokens ||
    topPEnabled.value !== defaultSettings.topPEnabled ||
    topP.value !== defaultSettings.topP ||
    frequencyPenaltyEnabled.value !== defaultSettings.frequencyPenaltyEnabled ||
    frequencyPenalty.value !== defaultSettings.frequencyPenalty ||
    presencePenaltyEnabled.value !== defaultSettings.presencePenaltyEnabled ||
    presencePenalty.value !== defaultSettings.presencePenalty ||
    systemPromptEnabled.value !== defaultSettings.systemPromptEnabled ||
    systemPrompt.value !== defaultSettings.systemPrompt ||
    reasoningEffortEnabled.value !== defaultSettings.reasoningEffortEnabled ||
    reasoningEffort.value !== defaultSettings.reasoningEffort ||
    customVariables.value.some((v) => v.enabled) ||
    tools.value.some((t) => t.enabled && t.name.trim())
  );
});

const canSend = computed(
  () =>
    (input.value.trim().length > 0 || attachedFiles.value.length > 0) &&
    !isLoading.value &&
    !isSubmitting.value &&
    !!selectedModel.value &&
    apiKey.value.trim().length > 0
);

const loadModels = async () => {
  try {
    isLoadingModels.value = true;
    const res = await chatApi.getModels(apiKey.value);

    // Map each model to resolve the provider name from the modelStore
    models.value = res.data.map((m) => {
      let provider = m.provider;
      if (!provider) {
        const configModel = modelStore.models.find((cm) => cm.name === m.id);
        provider = configModel?.providers?.[0]?.provider_name || "";
      }
      return {
        id: m.id,
        provider: provider,
      };
    });

    // Restore saved model if it exists in the models list, otherwise default to first model
    const savedModel = localStorage.getItem(STORAGE_KEYS.CHAT_MODEL);
    if (savedModel && models.value.some((m) => m.id === savedModel)) {
      selectedModel.value = savedModel;
    } else if (models.value.length > 0) {
      selectedModel.value = models.value[0]?.id ?? null;
    }
  } catch (error) {
    console.error("Failed to load models:", error);
    models.value = [];
    selectedModel.value = null;
  } finally {
    isLoadingModels.value = false;
  }
};

const authStore = useAuthStore();

const apiKey = computed(() => {
  return authStore.sessionApiKey ?? "";
});

const isFirstLoad = ref(true);

onMounted(async () => {
  if (isFirstLoad.value) {
    if (authStore.isAdmin) {
      try {
        await modelStore.fetchModels();
      } catch (e) {
        console.error("Failed to fetch model configs:", e);
      }
    }
    await loadModels();
    isFirstLoad.value = false;
  }
});

onActivated(async () => {
  if (!isFirstLoad.value) {
    if (authStore.isAdmin) {
      try {
        // Force reload model configurations from the server config
        await modelStore.fetchModels(true);
      } catch (e) {
        console.error("Failed to fetch model configs on activation:", e);
      }
    }
  }
});

// Automatically sync model options if the global model configuration store updates
watch(
  () => modelStore.models,
  async () => {
    await loadModels();
  }
);

const isNearBottom = (threshold = 100): boolean => {
  if (!messagesContainer.value) return true;
  const { scrollTop, scrollHeight, clientHeight } = messagesContainer.value;
  return scrollHeight - scrollTop - clientHeight < threshold;
};

const scrollToBottom = async (smooth = true) => {
  await nextTick();
  if (messagesContainer.value) {
    messagesContainer.value.scrollTo({
      top: messagesContainer.value.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  }
};

const smartScrollToBottom = async (smooth = true) => {
  if (isNearBottom()) {
    await scrollToBottom(smooth);
  }
};

const handleScroll = useThrottleFn(() => {
  isAutoScroll.value = isNearBottom();
}, 100);

const { messages, isLoading, sendMessage, clearChat, stopGeneration } = useChat();

const isSubmitting = ref(false);

// Run specimens live in the chat store so the tray survives route navigation.
const chatStore = useChatStore();
const { runs } = storeToRefs(chatStore);
const selectedRunId = ref<string | null>(null);

const selectedRun = computed(() => runs.value.find((r) => r.id === selectedRunId.value) ?? null);
const selectedRunNumber = computed(
  () => runs.value.findIndex((r) => r.id === selectedRunId.value) + 1
);

// Inspector and settings share the right lane — opening one closes the other.
const toggleRunSelection = (id: string) => {
  if (selectedRunId.value === id) {
    selectedRunId.value = null;
  } else {
    selectedRunId.value = id;
    showSettings.value = false;
  }
};

const toggleSettings = () => {
  showSettings.value = !showSettings.value;
  if (showSettings.value) selectedRunId.value = null;
};

const getBaseChatOptions = () => {
  const opts: ChatOptions = {};
  if (systemPromptEnabled.value && systemPrompt.value.trim()) {
    opts.system_prompt = systemPrompt.value.trim();
  }
  if (temperatureEnabled.value) {
    opts.temperature = temperature.value;
  }
  if (maxTokensEnabled.value && maxTokens.value !== null) {
    opts.max_tokens = maxTokens.value;
  }
  if (topPEnabled.value) {
    opts.top_p = topP.value;
  }
  if (frequencyPenaltyEnabled.value) {
    opts.frequency_penalty = frequencyPenalty.value;
  }
  if (presencePenaltyEnabled.value) {
    opts.presence_penalty = presencePenalty.value;
  }
  if (reasoningEffortEnabled.value && reasoningEffort.value) {
    opts.reasoning_effort = reasoningEffort.value;
  }
  customVariables.value.forEach((v) => {
    if (v.enabled && v.key.trim()) {
      let val: string | number | boolean = v.value;
      if (v.type === "number") {
        const num = Number(v.value);
        if (!isNaN(num)) val = num;
      } else if (v.type === "boolean") {
        val = v.value.toLowerCase() === "true" || v.value === "1";
      }
      opts[v.key.trim()] = val;
    }
  });

  const activeTools = tools.value.filter((t) => t.enabled && t.name.trim());
  if (activeTools.length > 0) {
    opts.tools = activeTools;
  }

  opts.webSearch = webSearch.value;

  if (selectedEndpoint.value === "/v1/audio/speech") {
    opts.voice = speechVoice.value;
    opts.speed = speechSpeed.value;
  } else {
    delete opts.voice;
    delete opts.speed;
  }

  return opts;
};

const handleSendMessage = async () => {
  if (
    (!input.value.trim() && attachedFiles.value.length === 0) ||
    !selectedModel.value ||
    isLoading.value ||
    isSubmitting.value
  )
    return;

  const key = apiKey.value.trim();
  if (!key) {
    toast.warning(t("chat.apiKeyRequired"), {
      description: t("common.error"),
    });
    return;
  }

  const rawInput = input.value;
  const filesToSend = [...attachedFiles.value];
  input.value = "";
  attachedFiles.value = [];
  isAutoScroll.value = true;
  isSubmitting.value = true;

  try {
    await smartScrollToBottom();

    let finalContent: string | ContentPart[] = rawInput;

    if (filesToSend.length > 0) {
      const parts: ContentPart[] = [];
      let textFileContent = "";

      for (const file of filesToSend) {
        if (file.text !== undefined) {
          textFileContent += `\n\n---\n*Attached file: \`${file.name}\`*\n\`\`\`${file.name.split(".").pop() || ""}\n${file.text}\n\`\`\``;
        } else if (file.type.startsWith("image/")) {
          parts.push({
            type: "image_url",
            image_url: {
              url: file.base64 || file.url,
            },
          });
        } else {
          parts.push({
            type: "file",
            file: {
              file_data: file.base64 || file.url,
              filename: file.name,
            },
          } as unknown as ContentPart);
        }
      }

      const combinedText = (rawInput + textFileContent).trim();
      if (combinedText) {
        parts.unshift({
          type: "text",
          text: combinedText,
        });
      }

      finalContent = parts;
    } else {
      finalContent = rawInput.trim();
    }

    await sendMessage(
      finalContent,
      selectedModel.value,
      key,
      selectedEndpoint.value,
      () => {
        if (isNearBottom()) {
          scrollToBottom(false);
        }
      },
      getBaseChatOptions()
    );

    await smartScrollToBottom();
  } catch (error) {
    input.value = rawInput;
    attachedFiles.value = filesToSend;
    toast.error(t("chat.messageFailed"), {
      description: getErrorMessage(error),
    });
  } finally {
    isSubmitting.value = false;
  }
};

const handleRetryMessage = async (msg: ChatMessageType) => {
  if (isLoading.value || isSubmitting.value) return;

  const model = selectedModel.value;
  if (!model) {
    toast.warning(t("chat.selectModel"));
    return;
  }

  const key = apiKey.value.trim();
  if (!key) {
    toast.warning(t("chat.apiKeyRequired"), {
      description: t("common.error"),
    });
    return;
  }

  // Find the index of the message to retry
  const idx = messages.value.findIndex((m) => m.id === msg.id);
  if (idx === -1) return;

  // Find the preceding user message index
  let userMsgIdx = -1;
  for (let i = idx - 1; i >= 0; i--) {
    if (messages.value[i].role === "user") {
      userMsgIdx = i;
      break;
    }
  }

  if (userMsgIdx === -1) {
    toast.error(t("chat.retryFailedUserMsgNotFound"));
    return;
  }

  const userMessageContent = messages.value[userMsgIdx].content;

  // Remove the assistant message and all subsequent messages
  messages.value.splice(idx);

  // Remove the user message as well because sendMessage will push a new one
  messages.value.splice(userMsgIdx, 1);

  isSubmitting.value = true;
  isAutoScroll.value = true;

  try {
    await sendMessage(
      userMessageContent,
      model,
      key,
      selectedEndpoint.value,
      () => {
        if (isNearBottom()) {
          scrollToBottom(false);
        }
      },
      getBaseChatOptions()
    );
    await smartScrollToBottom();
  } catch (error) {
    toast.error(t("chat.messageFailed"), {
      description: getErrorMessage(error),
    });
  } finally {
    isSubmitting.value = false;
  }
};

const toolOutputs = ref<Record<string, string>>({});

const pendingToolCalls = computed(() => {
  if (messages.value.length === 0) return [];
  const lastMsg = messages.value[messages.value.length - 1];
  if (lastMsg?.role !== "assistant" || !lastMsg.tool_calls || lastMsg.tool_calls.length === 0) {
    return [];
  }

  const responseIds = new Set(
    messages.value.filter((m) => m.role === "tool").map((m) => m.tool_call_id)
  );

  return lastMsg.tool_calls.filter(
    (tc) => tc && tc.id && !responseIds.has(tc.id) && tc.function?.name !== WEB_SEARCH_TOOL_NAME
  );
});

const isToolCallFormCancelled = ref(false);

watch(
  pendingToolCalls,
  (newCalls) => {
    if (newCalls.length > 0) {
      isToolCallFormCancelled.value = false;
    }
    for (const tc of newCalls) {
      if (!(tc.id in toolOutputs.value)) {
        toolOutputs.value[tc.id] = JSON.stringify({ status: "pending", data: "" }, null, 2);
      }
    }
  },
  { immediate: true }
);

const cancelPendingToolCalls = () => {
  isToolCallFormCancelled.value = true;
  const calls = [...pendingToolCalls.value];
  for (const tc of calls) {
    const toolMsg: ChatMessageType = {
      role: "tool",
      content: JSON.stringify({ error: "Tool execution cancelled by user" }, null, 2),
      tool_call_id: tc.id,
      name: tc.function.name,
    };
    useChatStore().pushMessage(toolMsg);
  }
  toolOutputs.value = {};
};

const submitToolOutputs = async () => {
  if (isLoading.value || isSubmitting.value || !selectedModel.value) return;

  const key = apiKey.value.trim();
  if (!key) {
    toast.warning(t("chat.apiKeyRequired"), {
      description: t("common.error"),
    });
    return;
  }

  isAutoScroll.value = true;
  isSubmitting.value = true;

  try {
    const calls = [...pendingToolCalls.value];
    for (const tc of calls) {
      const output = toolOutputs.value[tc.id] || "{}";
      const toolMsg: ChatMessageType = {
        role: "tool",
        content: output,
        tool_call_id: tc.id,
        name: tc.function.name,
      };
      useChatStore().pushMessage(toolMsg);
    }

    await smartScrollToBottom();

    const opts = getBaseChatOptions();
    opts.isToolResponse = true;
    await sendMessage(
      "",
      selectedModel.value,
      key,
      selectedEndpoint.value,
      () => {
        if (isNearBottom()) {
          scrollToBottom(false);
        }
      },
      opts
    );

    await smartScrollToBottom();
  } catch (error) {
    toast.error(t("chat.messageFailed"), {
      description: getErrorMessage(error),
    });
  } finally {
    isSubmitting.value = false;
  }
};

const handleClearChat = () => {
  showClearConfirmDialog.value = true;
};

const confirmClearChat = () => {
  clearChat();
  selectedRunId.value = null;
  showClearConfirmDialog.value = false;
};

const setQuickPrompt = (text: string) => {
  input.value = text;
};

watch(
  () => messages.value.length,
  () => {
    if (isAutoScroll.value && isNearBottom()) {
      // During active streaming, use auto scroll to avoid jank from
      // multiple overlapping smooth-scroll animations.
      scrollToBottom(!(isLoading.value || isSubmitting.value));
    }
  }
);
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header
        class="config-header-bar flex items-center justify-between px-4 sm:px-6 h-14 z-20 shrink-0"
      >
        <div class="flex items-center gap-2">
          <!-- Model Selector -->
          <Select v-if="models.length > 0" v-model="selectedModel">
            <SelectTrigger
              class="border border-border/60 bg-transparent hover:bg-muted/10 shadow-none focus-visible:ring-1 focus-visible:ring-foreground focus-visible:ring-offset-0 rounded-md h-8 px-2.5 gap-3 transition-colors text-foreground flex items-center min-w-0 font-mono text-[11px]"
            >
              <div class="flex items-center gap-2 min-w-0">
                <div
                  v-if="selectedModel"
                  class="w-4 h-4 rounded flex items-center justify-center shrink-0 overflow-hidden bg-background/50 border border-border/40"
                >
                  <img
                    v-if="getModelIcon(selectedModel)"
                    :src="getModelIcon(selectedModel)!"
                    :alt="selectedModel"
                    :class="[
                      isMonoIcon(selectedModel) ? 'icon-mono' : null,
                      'w-3.5 h-3.5 object-contain',
                    ]"
                    loading="lazy"
                  />
                  <Cpu v-else class="w-2.5 h-2.5 text-muted-foreground" />
                </div>
                <span class="font-medium truncate min-w-0 max-w-[200px]" v-if="selectedModel">
                  {{ selectedModel }}
                </span>
                <span v-else class="text-muted-foreground">{{ t("chat.selectModel") }}</span>
              </div>
            </SelectTrigger>
            <SelectContent class="min-w-64 rounded-md border border-border/80 shadow-md">
              <SelectItem
                v-for="modelOption in models"
                :key="modelOption.id"
                :value="modelOption.id"
                class="rounded-sm"
              >
                <div class="flex items-center gap-2.5 min-w-0 w-full py-0.5">
                  <div
                    class="w-4 h-4 rounded flex items-center justify-center shrink-0 overflow-hidden bg-background/50 border border-border/40"
                  >
                    <img
                      v-if="getModelIcon(modelOption.id)"
                      :src="getModelIcon(modelOption.id)!"
                      :alt="modelOption.id"
                      :class="[
                        isMonoIcon(modelOption.id) ? 'icon-mono' : null,
                        'w-3.5 h-3.5 object-contain',
                      ]"
                      loading="lazy"
                    />
                    <Cpu v-else class="w-2.5 h-2.5 text-muted-foreground" />
                  </div>
                  <span
                    class="min-w-0 truncate max-w-[40vw] sm:max-w-[250px] text-[11px] font-mono text-foreground leading-none"
                    :title="modelOption.id"
                    >{{ modelOption.id }}</span
                  >
                </div>
              </SelectItem>
            </SelectContent>
          </Select>

          <!-- Model Selector Placeholder/Loading -->
          <div
            v-else
            class="border border-border/60 bg-transparent opacity-85 rounded-md h-8 px-2.5 gap-3 text-muted-foreground flex items-center min-w-0 font-mono text-[11px] select-none cursor-not-allowed"
          >
            <div class="flex items-center gap-2 min-w-0">
              <Loader2
                v-if="isLoadingModels"
                class="w-3 h-3 animate-spin shrink-0 text-muted-foreground/70"
              />
              <span class="font-medium text-muted-foreground/70 truncate min-w-0 max-w-[200px]">
                {{ isLoadingModels ? `${t("common.loading")}…` : t("chat.selectModel") }}
              </span>
            </div>
          </div>

          <!-- Hairline divider between the primary model picker and the secondary endpoint -->
          <div class="hidden sm:block h-4 w-px bg-border/60 shrink-0" aria-hidden="true" />

          <!-- Endpoint Selector (secondary, mono) -->
          <Select v-model="selectedEndpoint">
            <SelectTrigger
              class="border border-border/60 bg-transparent hover:bg-muted/10 shadow-none focus-visible:ring-1 focus-visible:ring-foreground focus-visible:ring-offset-0 rounded-md h-8 px-2.5 gap-2 transition-colors text-foreground flex items-center font-mono text-[11px]"
            >
              <div class="flex items-center gap-2">
                <span class="text-muted-foreground font-mono text-[11px] select-none">API</span>
                <span class="font-medium text-muted-foreground">{{ selectedEndpoint }}</span>
              </div>
            </SelectTrigger>
            <SelectContent class="rounded-md border border-border/80 shadow-md">
              <SelectItem value="/v1/chat/completions" class="font-mono text-[11px] rounded-sm">
                /v1/chat/completions
              </SelectItem>
              <SelectItem value="/v1/messages" class="font-mono text-[11px] rounded-sm">
                /v1/messages
              </SelectItem>
              <SelectItem value="/v1/responses" class="font-mono text-[11px] rounded-sm">
                /v1/responses
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div class="flex items-center gap-1">
          <!-- Settings Toggle -->
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  @click="toggleSettings"
                  class="relative h-9 w-9 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  :class="{ 'bg-muted text-foreground': showSettings }"
                >
                  <Settings2 class="w-4 h-4" />
                  <span
                    v-if="hasActiveSettings && !showSettings"
                    class="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-primary rounded-full ring-2 ring-background"
                  />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{{ t("chat.advancedSettings") }}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <!-- Clear Chat -->
          <TooltipProvider v-if="messages.length > 0">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  @click="handleClearChat"
                  class="h-9 w-9 rounded-lg text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                >
                  <Eraser class="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{{ t("chat.clearChatTitle") }}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </header>
    </template>

    <!-- Main Content Area -->
    <div class="brand-main-shell flex-1 flex relative overflow-hidden">
      <!-- Chat Area -->
      <div class="flex-1 flex flex-col overflow-hidden">
        <!-- Messages -->
        <div
          ref="messagesContainer"
          class="flex-1 overflow-y-auto px-4 scroll-smooth"
          @scroll="handleScroll"
          aria-live="polite"
          aria-atomic="false"
          aria-label="Chat messages"
          role="log"
          tabindex="0"
        >
          <div class="max-w-3xl mx-auto flex flex-col gap-6 py-8 min-h-full">
            <ChatEmptyState
              v-if="messages.length === 0"
              :has-model="!!selectedModel"
              @prompt="setQuickPrompt"
            />

            <!-- Message List -->
            <transition-group
              tag="div"
              class="flex flex-col gap-6"
              enter-active-class="transition-all duration-300 ease-out"
              leave-active-class="transition-all duration-300 ease-in"
              enter-from-class="opacity-0 translate-y-2"
              leave-to-class="opacity-0 translate-y-2"
            >
              <ChatMessage
                v-for="msg in messages"
                :key="msg.id"
                :message="msg"
                @retry="handleRetryMessage"
              />
            </transition-group>

            <!-- Pending Tool Calls Form -->
            <div
              v-if="pendingToolCalls.length > 0 && !isLoading && !isToolCallFormCancelled"
              class="brand-panel p-4 rounded-xl border border-action-amber/30 bg-action-amber/5 space-y-4 max-w-3xl w-full mx-auto animate-in fade-in slide-in-from-bottom-2 duration-300"
            >
              <div class="flex items-center gap-2 text-action-amber">
                <Wrench class="w-5 h-5" />
                <h4 class="font-semibold text-sm">{{ t("chat.requiredToolOutputs") }}</h4>
              </div>
              <p class="text-xs text-muted-foreground">
                {{ t("chat.toolOutputsDescription") }}
              </p>

              <div class="space-y-4">
                <div v-for="tc in pendingToolCalls" :key="tc.id" class="space-y-1.5">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-semibold text-foreground font-mono">
                      {{ tc.function.name }}()
                    </span>
                    <span class="text-[11px] font-mono text-muted-foreground">
                      ID: {{ tc.id }}
                    </span>
                  </div>
                  <div
                    class="bg-muted/30 p-2 rounded text-[11px] font-mono text-muted-foreground mb-1.5 border border-border/50 max-h-32 overflow-y-auto"
                  >
                    {{ t("chat.toolArguments") }} {{ tc.function.arguments }}
                  </div>
                  <Textarea
                    v-model="toolOutputs[tc.id]"
                    :placeholder="t('chat.toolOutputPlaceholder')"
                    rows="3"
                    class="font-mono text-xs bg-background"
                  />
                </div>
              </div>

              <div class="flex justify-end gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  class="h-8 text-xs"
                  @click="cancelPendingToolCalls"
                >
                  {{ t("common.cancel") }}
                </Button>
                <Button size="sm" class="h-8 text-xs" @click="submitToolOutputs">
                  {{ t("chat.submitToolOutputs") }}
                </Button>
              </div>
            </div>
          </div>
        </div>

        <!-- Input Area -->
        <div class="flex-none p-4 pb-4 bg-background z-10">
          <div class="max-w-3xl mx-auto relative">
            <!-- Scroll to Bottom Button (positioned above input) -->
            <Transition
              enter-active-class="transition duration-300 ease-out"
              enter-from-class="translate-y-4 opacity-0"
              enter-to-class="translate-y-0 opacity-100"
              leave-active-class="transition duration-300 ease-in"
              leave-from-class="translate-y-0 opacity-100"
              leave-to-class="translate-y-4 opacity-0"
            >
              <button
                type="button"
                v-if="!isAutoScroll && messages.length > 0"
                @click="scrollToBottom(true)"
                class="absolute -top-12 right-0 z-50 bg-background/90 backdrop-blur-sm border border-border text-muted-foreground p-1.5 rounded-full shadow-sm hover:bg-muted hover:text-foreground transition-all focus-visible:outline-none"
                :aria-label="t('chat.scrollToBottom')"
              >
                <ChevronDown class="w-4.5 h-4.5" />
              </button>
            </Transition>

            <!-- Speech Settings Bar -->
            <div
              v-if="selectedEndpoint === '/v1/audio/speech'"
              class="flex items-center gap-4 mb-2.5 p-2 px-3 rounded-lg border border-border/40 bg-card/80 animate-in fade-in slide-in-from-bottom-1 duration-200"
            >
              <div class="flex items-center gap-2 text-xs">
                <span class="text-muted-foreground font-medium">{{ t("chat.speechVoice") }}:</span>
                <Select v-model="speechVoice">
                  <SelectTrigger
                    class="h-7 w-28 border border-border/50 bg-background/50 text-[11px] shadow-none"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem v-for="v in speechVoices" :key="v" :value="v" class="text-[11px]">
                      {{ v }}
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div class="flex items-center gap-2 text-xs">
                <span class="text-muted-foreground font-medium">{{ t("chat.speechSpeed") }}:</span>
                <Input
                  type="number"
                  v-model.number="speechSpeed"
                  min="0.25"
                  max="4.0"
                  step="0.05"
                  class="h-7 w-16 border border-border/50 bg-background/50 text-[11px] px-2 py-0 shadow-none text-center"
                />
                <span class="text-muted-foreground text-[11px]">x (0.25 - 4.0)</span>
              </div>
            </div>
            <form
              @submit.prevent="handleSendMessage"
              @dragover.prevent="isDragging = true"
              @dragleave.prevent="isDragging = false"
              @drop.prevent="handleDrop"
              class="relative flex flex-col brand-panel transition-all duration-200 p-3 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary"
              :class="{ 'border-primary bg-muted/10 ring-1 ring-primary': isDragging }"
            >
              <!-- Attached Files Preview Grid -->
              <div
                v-if="attachedFiles.length > 0"
                class="flex flex-wrap gap-2 px-1 pb-3 border-b border-border/20 mb-2.5"
              >
                <div
                  v-for="file in attachedFiles"
                  :key="file.id"
                  class="relative flex items-center gap-2 p-1.5 pr-2.5 rounded-md border border-border bg-background text-foreground text-xs font-medium animate-in fade-in zoom-in-95 duration-150 group/item"
                >
                  <img
                    v-if="file.type.startsWith('image/')"
                    :src="file.url"
                    class="w-7 h-7 object-cover rounded-sm border border-border/50"
                  />
                  <FileText
                    v-else-if="file.type === 'application/pdf'"
                    class="w-5 h-5 text-muted-foreground shrink-0 ml-1"
                  />
                  <Code
                    v-else-if="isTextFile(file.name)"
                    class="w-5 h-5 text-muted-foreground shrink-0 ml-1"
                  />
                  <File v-else class="w-5 h-5 text-muted-foreground shrink-0 ml-1" />

                  <div class="min-w-0 max-w-32">
                    <p class="truncate text-[11px] font-semibold font-sans" :title="file.name">
                      {{ file.name }}
                    </p>
                    <p class="text-[11px] text-muted-foreground font-mono leading-none">
                      {{ formatFileSize(file.size) }}
                    </p>
                  </div>

                  <button
                    type="button"
                    @click="removeAttachedFile(file.id)"
                    class="p-0.5 rounded-full hover:bg-muted text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                  >
                    <X class="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <!-- Textarea Row -->
              <textarea
                ref="textareaRef"
                v-model="input"
                rows="1"
                :placeholder="t('chat.typeMessage')"
                :aria-label="t('chat.typeMessage')"
                class="w-full bg-transparent border-0 outline-none ring-0 focus:outline-none focus:ring-0 py-2 px-2.5 min-h-[38px] max-h-48 resize-none text-[14.5px] leading-relaxed placeholder:text-muted-foreground/50 field-sizing-content text-foreground font-sans"
                @focus="handleFocus"
                @blur="handleBlur"
                @paste="handlePaste"
                @keydown.enter="
                  (e) => {
                    if (!e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }
                "
              />

              <!-- Actions Footer Row -->
              <div
                class="flex items-center justify-between border-t border-border/20 pt-2.5 mt-1.5 px-1 pb-0.5"
              >
                <div class="flex items-center gap-1.5">
                  <!-- Attach file button -->
                  <button
                    type="button"
                    @click="triggerFileInput"
                    :disabled="selectedEndpoint === '/v1/audio/speech'"
                    class="h-8.5 w-8.5 flex items-center justify-center rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-all duration-150 shrink-0 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                    :title="
                      selectedEndpoint === '/v1/audio/speech'
                        ? 'Attachments not supported for speech generation'
                        : t('chat.uploadFile')
                    "
                  >
                    <Paperclip class="w-4 h-4" />
                  </button>

                  <!-- Web Search Toggle -->
                  <button
                    type="button"
                    @click="webSearch.enabled = !webSearch.enabled"
                    class="h-8.5 w-8.5 flex items-center justify-center rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-all duration-150 shrink-0 cursor-pointer"
                    :class="{
                      'text-foreground bg-muted/40 ring-1 ring-border': webSearch.enabled,
                    }"
                    :title="t('chat.webSearch')"
                  >
                    <Globe class="w-4 h-4" />
                  </button>

                  <input
                    type="file"
                    ref="fileInputRef"
                    class="hidden"
                    multiple
                    @change="handleFileChange"
                  />
                </div>

                <div class="flex items-center gap-3">
                  <span
                    class="hidden md:inline text-[11px] text-muted-foreground font-mono select-none uppercase tracking-wider"
                  >
                    ↵ send / ⇧↵ new line
                  </span>

                  <Button
                    type="submit"
                    size="icon"
                    :disabled="!canSend"
                    class="h-11 w-11 shrink-0 flex items-center justify-center transition-[transform,colors] duration-200 active:scale-95 rounded-md"
                    :class="
                      isLoading || isSubmitting
                        ? 'bg-muted text-foreground'
                        : canSend
                          ? 'bg-foreground text-background hover:bg-foreground/90'
                          : 'bg-muted/30 text-muted-foreground/40 cursor-not-allowed border border-border/20'
                    "
                    :aria-label="isSubmitting ? t('chat.sending') : t('chat.sendMessage')"
                  >
                    <Loader2 v-if="isLoading || isSubmitting" class="w-4 h-4 animate-spin" />
                    <ArrowUp v-else class="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </form>
          </div>
        </div>

        <!-- Run tray: every API request deposits an inspectable specimen -->
        <SpecimenTray :count="runs.length">
          <RunSpecimen
            v-for="(run, i) in runs"
            :key="run.id"
            :status="run.status"
            :selected="selectedRunId === run.id"
            @click="toggleRunSelection(run.id)"
          >
            <span class="text-foreground/80">#{{ String(i + 1).padStart(2, "0") }}</span>
            <span>{{ endpointName(run.endpoint) }}</span>
            <span class="text-muted-foreground/70">{{ formatLatency(run.latencyMs) }}</span>
          </RunSpecimen>
        </SpecimenTray>
      </div>

      <RunInspector
        v-if="selectedRun"
        :open="!!selectedRun"
        :run-number="selectedRunNumber"
        :status="selectedRun.status"
        :endpoint="selectedRun.endpoint"
        :model="selectedRun.model"
        :started-at="selectedRun.startedAt"
        :latency-ms="selectedRun.latencyMs"
        :response-chars="selectedRun.responseChars"
        :error-message="selectedRun.errorMessage"
        :payload="selectedRun.payload"
        @close="selectedRunId = null"
      />

      <ChatSettings
        v-model:open="showSettings"
        v-model:temperature="temperature"
        v-model:temperatureEnabled="temperatureEnabled"
        v-model:maxTokens="maxTokens"
        v-model:maxTokensEnabled="maxTokensEnabled"
        v-model:topP="topP"
        v-model:topPEnabled="topPEnabled"
        v-model:frequencyPenalty="frequencyPenalty"
        v-model:frequencyPenaltyEnabled="frequencyPenaltyEnabled"
        v-model:presencePenalty="presencePenalty"
        v-model:presencePenaltyEnabled="presencePenaltyEnabled"
        v-model:systemPrompt="systemPrompt"
        v-model:systemPromptEnabled="systemPromptEnabled"
        v-model:reasoningEffort="reasoningEffort"
        v-model:reasoningEffortEnabled="reasoningEffortEnabled"
        v-model:customVariables="customVariables"
        v-model:tools="tools"
        v-model:webSearch="webSearch"
        v-model:speechVoice="speechVoice"
        v-model:speechSpeed="speechSpeed"
        v-model:speechModel="speechModel"
        @reset="resetSettings"
      />
    </div>

    <!-- Clear Chat Confirmation Dialog -->
    <ConfirmDialog
      v-model:open="showClearConfirmDialog"
      :title="t('chat.clearChatTitle')"
      :description="t('chat.clearChatDescription')"
      :confirm-text="t('common.delete')"
      :cancel-text="t('common.cancel')"
      :danger="true"
      @confirm="confirmClearChat"
    />
  </AppLayout>
</template>
