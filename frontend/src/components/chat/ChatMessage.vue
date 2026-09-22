<script setup lang="ts">
import {
  Brain,
  Wrench,
  Copy,
  Volume2,
  VolumeX,
  Check,
  Loader2,
  X,
  FileText,
  RotateCcw,
} from "@lucide/vue";
import MarkdownIt from "markdown-it";
import { computed, ref, onUnmounted, watch } from "vue";
import { highlightCode } from "@/utils/highlighter";
import { useI18n } from "vue-i18n";

import type { ChatMessage } from "@/types/schemas";
import { sanitizeMarkdownHtml } from "@/utils/sanitize";
import { useAuthStore } from "@/stores/auth";
import { useChatStore } from "@/stores/chat";
import { useModelStore } from "@/stores/models";
import { toast } from "vue-sonner";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import { copyToClipboard } from "@/utils/clipboard";

const previewImageUrl = ref<string | null>(null);
const openImageModal = (url: string) => {
  previewImageUrl.value = url;
};
const closeImageModal = () => {
  previewImageUrl.value = null;
};

const props = defineProps<{
  message: ChatMessage;
}>();

const emit = defineEmits<{
  (e: "retry", message: ChatMessage): void;
}>();

const { t } = useI18n();

const md = new MarkdownIt({
  html: true,
  linkify: true,
  breaks: true,
});

const renderCodeBlock = (code: string, lang: string) => {
  // Escape HTML to prevent injection issues in code content
  const escaped = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");

  const highlighted = highlightCode(escaped, lang);
  const langLabel = lang || "code";

  return `<div class="code-block-wrapper my-4 overflow-hidden rounded-xl border border-code-border bg-code-bg font-mono">
  <div class="flex items-center justify-between px-4 py-2 border-b border-code-border bg-code-header-bg text-xs text-muted-foreground select-none">
    <span class="font-medium lowercase">${langLabel}</span>
    <button type="button" class="code-copy-btn flex items-center gap-1 hover:text-foreground transition-colors cursor-pointer max-sm:min-h-11 max-sm:px-3" aria-label="Copy code">
      <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="copy-icon"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
      <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="check-icon hidden text-status-success"><polyline points="20 6 9 17 4 12"/></svg>
      <span class="copy-text text-[11px] font-sans">Copy</span>
    </button>
  </div>
  <pre class="p-4 overflow-x-auto text-[13px] leading-relaxed text-foreground/90"><code class="language-${langLabel}">${highlighted}</code></pre>
</div>
`;
};

md.renderer.rules.fence = (tokens, idx) => {
  const token = tokens[idx];
  return renderCodeBlock(token.content, token.info.trim());
};

md.renderer.rules.code_block = (tokens, idx) => {
  const token = tokens[idx];
  return renderCodeBlock(token.content, "");
};

const escapeAttr = (value: string) =>
  value.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// Markdown images are zoomable. Wrap the rendered image element in a real
// button so the lightbox stays reachable by keyboard (the same pattern
// LogImagePreview uses). The name comes from the image's alt text; a decorative
// image falls back to the localized "image" label rather than shipping a
// nameless button.
md.renderer.rules.image = (tokens, idx, options, env, self) => {
  const token = tokens[idx];
  const alt = self.renderInlineAsText(token.children ?? [], options, env);
  const altIndex = token.attrIndex("alt");
  if (altIndex >= 0) {
    token.attrs![altIndex][1] = alt;
  }
  const label = alt.trim() || t("images.image");
  return `<button type="button" class="zoomable-image focus-visible-ring" aria-label="${escapeAttr(label)}">${self.renderToken(tokens, idx, options)}</button>`;
};

const handleCodeCopy = async (btn: Element) => {
  const wrapper = btn.closest(".code-block-wrapper");
  const codeEl = wrapper?.querySelector("code");
  if (!codeEl) return;

  try {
    await copyToClipboard(codeEl.textContent || "");
    const copyText = btn.querySelector(".copy-text");
    const copyIcon = btn.querySelector(".copy-icon");
    const checkIcon = btn.querySelector(".check-icon");
    if (copyText) copyText.textContent = t("common.copied");
    if (copyIcon && checkIcon) {
      copyIcon.classList.add("hidden");
      checkIcon.classList.remove("hidden");
      setTimeout(() => {
        copyIcon.classList.remove("hidden");
        checkIcon.classList.add("hidden");
        if (copyText) copyText.textContent = t("common.copy");
      }, 2000);
    }
  } catch (err) {
    console.error("Failed to copy code snippet:", err);
  }
};

const handleContentClick = (e: MouseEvent) => {
  const target = e.target as HTMLElement;

  // Keyboard activation of a zoomable image reports the wrapper button, so
  // resolve the image through it instead of matching the image tag alone.
  const zoomWrapper = target.closest(".zoomable-image");
  const zoomedImg = zoomWrapper
    ? zoomWrapper.querySelector("img")
    : target instanceof HTMLImageElement
      ? target
      : null;
  if (zoomedImg?.src) {
    openImageModal(zoomedImg.src);
    return;
  }

  const btn = target.closest(".code-copy-btn");
  if (btn) {
    handleCodeCopy(btn);
  }
};

const isThinkingExpanded = ref(false);
const isToolCallsExpanded = ref(false);

const hasReasoningContent = computed(() => {
  return (props.message.reasoning_content?.trim().length ?? 0) > 0;
});

const activeToolCalls = computed(() => {
  return (props.message.tool_calls || []).filter((tc) => tc != null);
});

const hasToolCalls = computed(() => {
  return activeToolCalls.value.length > 0;
});

const toolUseCalls = computed(() => {
  const content = props.message.content;
  if (!Array.isArray(content)) return [];
  return content.filter(
    (
      part
    ): part is { type: "tool_use"; id: string; name: string; input: Record<string, unknown> } =>
      part.type === "tool_use"
  );
});

// Image parts from message content (for assistant messages with image generation)
const imageParts = computed(() => {
  const content = props.message.content;
  if (!Array.isArray(content)) return [];
  return content.filter(
    (part): part is { type: "image_url"; image_url: { url: string } } => part.type === "image_url"
  );
});

const hasImageParts = computed(() => imageParts.value.length > 0);

const extractText = (content: ChatMessage["content"]): string => {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((part): part is { type: "text"; text: string } => part.type === "text")
      .map((part) => part.text)
      .join("\n");
  }
  return "";
};

const markdownToHtml = (text: string) => sanitizeMarkdownHtml(md.render(text));

// A streaming response appends to `content` on every flush. Re-parsing the whole
// message each time (markdown + DOMPurify + syntax highlighting) is O(n²), so the
// settled prefix is rendered once, frozen at a blank-line boundary, and only the
// growing tail is re-rendered. `renderFull` runs when the stream ends: a full
// pass corrects anything a boundary split could not see (reference-style links,
// HTML blocks spanning blank lines).
const renderedContent = ref("");
let frozenHtml = "";
let frozenLength = 0;
let renderedSource = "";

const hasBalancedFences = (text: string): boolean => {
  const fences = text.match(/^[ \t]*```/gm);
  return !fences || fences.length % 2 === 0;
};

/** First blank-line boundary after `from` that is safe to freeze at (-1 if none). */
const findFreezePoint = (source: string, from: number): number => {
  let index = source.indexOf("\n\n", from);
  while (index !== -1) {
    const candidate = index + 2;
    if (hasBalancedFences(source.slice(0, candidate))) return candidate;
    index = source.indexOf("\n\n", candidate);
  }
  return -1;
};

const renderIncremental = () => {
  const source = extractText(props.message.content);
  if (!source.startsWith(renderedSource)) {
    // Content was replaced or edited rather than appended to: start over.
    frozenHtml = "";
    frozenLength = 0;
  }
  renderedSource = source;

  const freezePoint = findFreezePoint(source, frozenLength);
  if (freezePoint > frozenLength) {
    frozenHtml += markdownToHtml(source.slice(frozenLength, freezePoint));
    frozenLength = freezePoint;
  }
  renderedContent.value = frozenHtml + markdownToHtml(source.slice(frozenLength));
};

const renderFull = () => {
  renderedSource = extractText(props.message.content);
  frozenHtml = "";
  frozenLength = 0;
  renderedContent.value = markdownToHtml(renderedSource);
};

watch(
  () => props.message.content,
  () => renderIncremental(),
  { immediate: true, deep: true }
);

const formatToolArguments = (args: string): string => {
  try {
    const parsed = JSON.parse(args);
    return JSON.stringify(parsed, null, 2);
  } catch {
    return args;
  }
};

const chatStore = useChatStore();
const modelStore = useModelStore();

// The stream settled: re-render this message in one full pass so an incremental
// freeze boundary can never leave a stale interpretation behind.
watch(
  () => chatStore.isLoading,
  (loading) => {
    if (!loading) renderFull();
  }
);

const isTtsLoading = ref(false);
const isCopied = ref(false);
const ttsAbortController = ref<AbortController | null>(null);

// Track available browser voices (populated asynchronously)
const browserVoices = ref<SpeechSynthesisVoice[]>([]);

const updateBrowserVoices = () => {
  browserVoices.value = speechSynthesis.getVoices();
};

if (typeof speechSynthesis !== "undefined") {
  updateBrowserVoices();
  speechSynthesis.addEventListener("voiceschanged", updateBrowserVoices);
}

onUnmounted(() => {
  ttsAbortController.value?.abort();
  if (typeof speechSynthesis !== "undefined") {
    speechSynthesis.cancel();
    speechSynthesis.removeEventListener("voiceschanged", updateBrowserVoices);
  }
});

const getMessageText = (msg: ChatMessage): string => {
  const content = msg.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p): p is { type: "text"; text: string } => p.type === "text")
      .map((p) => p.text)
      .join("\n");
  }
  return "";
};

const handleCopy = async () => {
  try {
    const textToCopy = getMessageText(props.message);
    await copyToClipboard(textToCopy);
    isCopied.value = true;
    setTimeout(() => {
      isCopied.value = false;
    }, 2000);
  } catch (err) {
    console.error("Failed to copy text:", err);
  }
};

const getTtsModel = () => {
  const configuredTtsModel = localStorage.getItem(STORAGE_KEYS.SPEECH_MODEL);
  if (configuredTtsModel) return configuredTtsModel;

  if (modelStore.models.length === 0) return "browser";

  const activeModel = localStorage.getItem(STORAGE_KEYS.CHAT_MODEL);
  if (activeModel?.toLowerCase().includes("tts")) return activeModel;

  return modelStore.models.find((m) => m.name.toLowerCase().includes("tts"))?.name || "browser";
};

const handleBrowserTts = (text: string, voice: string): SpeechSynthesisUtterance => {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = parseFloat(localStorage.getItem(STORAGE_KEYS.SPEECH_SPEED) || "1.0");

  const voices = browserVoices.value.length > 0 ? browserVoices.value : speechSynthesis.getVoices();
  const exactMatch = voices.find((v) => v.name === voice);
  if (exactMatch) {
    utterance.voice = exactMatch;
    utterance.lang = exactMatch.lang;
  }

  return utterance;
};

const handleReadAloud = async () => {
  const msgId = props.message.id;
  if (!msgId) return;

  // If already playing this message, stop it
  if (chatStore.currentlyPlayingId === msgId) {
    speechSynthesis.cancel();
    chatStore.stopAudio();
    return;
  }

  // Ignore clicks while a TTS request is already in flight for this message.
  // Without this guard a second click fires a duplicate request; when both
  // resolve, the second handler revokes the first blob URL and playAudio
  // drops the second URL, leaving the message silently unplayed.
  if (isTtsLoading.value) return;

  // Cancel any in-flight TTS request
  ttsAbortController.value?.abort();
  ttsAbortController.value = new AbortController();

  const storeMsg = chatStore.messages.find((m) => m.id === msgId);
  const existingAudioUrl = storeMsg?.audioUrl || props.message.audioUrl;
  if (existingAudioUrl) {
    chatStore.playAudio(msgId, existingAudioUrl);
    return;
  }

  isTtsLoading.value = true;
  try {
    const textToSpeech = getMessageText(props.message);

    if (!textToSpeech.trim()) return;

    const ttsModel = getTtsModel();

    // Use browser TTS if no model is configured or user chose browser
    if (ttsModel === "browser") {
      const voice = localStorage.getItem(STORAGE_KEYS.SPEECH_VOICE) || "alloy";
      const utterance = handleBrowserTts(textToSpeech.trim(), voice);

      // Play using browser's speech synthesis
      speechSynthesis.cancel(); // Cancel any ongoing speech
      speechSynthesis.speak(utterance);

      // For browser TTS, we don't have a blob URL, so just track playing state
      chatStore.currentlyPlayingId = msgId;

      utterance.onend = () => {
        chatStore.currentlyPlayingId = null;
      };
      utterance.onerror = (event) => {
        chatStore.currentlyPlayingId = null;
        if ((event as SpeechSynthesisErrorEvent).error !== "canceled") {
          toast.error(t("chat.messageFailed"), {
            description: t("dialogs.errorGenerating"),
          });
        }
      };

      return;
    }

    // API-based TTS
    const apiKey = useAuthStore().sessionApiKey ?? "";
    const voice = localStorage.getItem(STORAGE_KEYS.SPEECH_VOICE) || "alloy";
    const speed = parseFloat(localStorage.getItem(STORAGE_KEYS.SPEECH_SPEED) || "1.0");

    const response = await fetch("/v1/audio/speech", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: ttsModel,
        input: textToSpeech.trim(),
        voice,
        speed,
        response_format: "mp3",
      }),
      signal: ttsAbortController.value?.signal,
    });

    if (!response.ok) {
      let errData;
      try {
        errData = await response.json();
      } catch {
        errData = await response.text();
      }
      const errMsg = errData?.error?.message || errData?.message || `HTTP ${response.status}`;
      throw new Error(errMsg);
    }

    const blob = await response.blob();

    // Revoke the previous blob URL to prevent a memory leak — but never while
    // audio playback is still using it, or the sound dies silently.
    const prevAudioUrl = storeMsg?.audioUrl || props.message.audioUrl;
    if (
      prevAudioUrl &&
      prevAudioUrl.startsWith("blob:") &&
      chatStore.currentlyPlayingId !== msgId
    ) {
      URL.revokeObjectURL(prevAudioUrl);
    }

    const audioUrl = URL.createObjectURL(blob);

    if (storeMsg) {
      storeMsg.audioUrl = audioUrl;
      // The store persists on an explicit dirty flag, not a deep watcher.
      chatStore.touchChat();
    }

    chatStore.playAudio(msgId, audioUrl);
  } catch (err: unknown) {
    // A newer read-aloud (or unmount) aborted this request — not an error.
    if ((err as Error).name === "AbortError") return;
    console.error("Read aloud failed:", err);
    const errorMessage = (err as Error).message || t("dialogs.errorGenerating");
    toast.error(t("chat.messageFailed"), {
      description: errorMessage,
    });
  } finally {
    isTtsLoading.value = false;
  }
};
</script>

<template>
  <div
    class="flex flex-col gap-3 group py-6 border-b border-border/20 last:border-0 relative outline-none focus-visible:ring-1 focus-visible:ring-border/50"
    tabindex="0"
  >
    <!-- Header: Meta & Actions -->
    <div class="flex items-center justify-between">
      <!-- Role Label -->
      <span
        class="text-[11px] uppercase tracking-widest font-mono font-semibold"
        :class="{
          'text-foreground/80': message.role === 'user',
          'text-foreground': message.role === 'assistant',
          'text-action-amber': message.role === 'tool',
        }"
      >
        {{ message.role === "user" ? "Human" : message.role === "tool" ? "Tool" : "Model" }}
      </span>

      <!-- Actions (Visible on hover) -->
      <div
        v-if="message.role === 'assistant' && getMessageText(message)"
        class="flex items-center gap-1 opacity-0 group-focus-within:opacity-100 group-hover:opacity-100 transition-opacity duration-200"
      >
        <button
          type="button"
          @click="handleReadAloud"
          class="p-1.5 rounded-md hover:bg-muted/40 text-muted-foreground hover:text-foreground transition-colors relative group/btn flex items-center justify-center"
          :aria-label="chatStore.currentlyPlayingId === message.id ? 'Stop' : 'Read Aloud'"
        >
          <Loader2 v-if="isTtsLoading" class="w-3.5 h-3.5 animate-spin text-primary" />
          <VolumeX
            v-else-if="chatStore.currentlyPlayingId === message.id"
            class="w-3.5 h-3.5 text-primary animate-pulse"
          />
          <Volume2 v-else class="w-3.5 h-3.5" />
          <span
            class="absolute top-full mt-1.5 left-1/2 -translate-x-1/2 bg-popover text-popover-foreground text-[11px] px-2 py-0.5 rounded-md shadow-sm opacity-0 group-hover/btn:opacity-100 transition-opacity pointer-events-none whitespace-nowrap border border-border z-10 hidden sm:block font-mono"
          >
            {{
              chatStore.currentlyPlayingId === message.id
                ? "Stop"
                : isTtsLoading
                  ? "Generating..."
                  : "Read Aloud"
            }}
          </span>
        </button>

        <button
          type="button"
          @click="handleCopy"
          class="p-1.5 rounded-md hover:bg-muted/40 text-muted-foreground hover:text-foreground transition-colors relative group/btn flex items-center justify-center"
          aria-label="Copy message"
        >
          <Check v-if="isCopied" class="w-3.5 h-3.5 text-status-success" />
          <Copy v-else class="w-3.5 h-3.5" />
          <span
            class="absolute top-full mt-1.5 left-1/2 -translate-x-1/2 bg-popover text-popover-foreground text-[11px] px-2 py-0.5 rounded-md shadow-sm opacity-0 group-hover/btn:opacity-100 transition-opacity pointer-events-none whitespace-nowrap border border-border z-10 hidden sm:block font-mono"
          >
            {{ isCopied ? "Copied!" : "Copy" }}
          </span>
        </button>

        <button
          type="button"
          @click="emit('retry', message)"
          class="p-1.5 rounded-md hover:bg-muted/40 text-muted-foreground hover:text-foreground transition-colors relative group/btn flex items-center justify-center"
          :aria-label="t('chat.retry')"
        >
          <RotateCcw class="w-3.5 h-3.5" />
          <span
            class="absolute top-full mt-1.5 left-1/2 -translate-x-1/2 bg-popover text-popover-foreground text-[11px] px-2 py-0.5 rounded-md shadow-sm opacity-0 group-hover/btn:opacity-100 transition-opacity pointer-events-none whitespace-nowrap border border-border z-10 hidden sm:block font-mono"
          >
            {{ t("chat.retry") }}
          </span>
        </button>
      </div>
    </div>

    <!-- Content -->
    <div class="min-w-0 flex flex-col gap-4">
      <!-- Tool Output Block -->
      <div v-if="message.role === 'tool'" class="w-full">
        <div
          class="text-[11px] font-mono font-medium text-action-amber mb-2 flex items-center gap-1.5"
        >
          <Wrench class="w-3.5 h-3.5" />
          <span>{{ message.name || message.tool_call_id }}()</span>
        </div>
        <pre
          class="text-xs font-mono text-muted-foreground/80 whitespace-pre-wrap overflow-x-auto bg-muted/10 rounded-md p-3 border border-border/40"
          >{{
            typeof message.content === "string"
              ? formatToolArguments(message.content)
              : JSON.stringify(message.content, null, 2)
          }}</pre>
      </div>

      <!-- Assistant / User Content Block -->
      <div v-else class="w-full flex flex-col gap-4">
        <!-- System Traces (Reasoning) -->
        <div
          v-if="hasReasoningContent"
          class="w-full rounded-md border border-border/40 bg-muted/5 p-3"
        >
          <button
            @click="isThinkingExpanded = !isThinkingExpanded"
            class="flex items-center gap-2 text-[11px] font-mono tracking-widest transition-colors w-full text-left uppercase text-muted-foreground hover:text-foreground"
          >
            <Brain class="w-3 h-3 text-action-violet animate-pulse" />
            <span>Reasoning Trace</span>
            <span class="opacity-50 lowercase ml-2"
              >({{ message.reasoning_content?.length || 0 }} chars)</span
            >
            <span class="ml-auto">{{ isThinkingExpanded ? "[-]" : "[+]" }}</span>
          </button>
          <div
            v-if="isThinkingExpanded"
            class="mt-3 pt-3 border-t border-border/40 text-[12px] font-mono text-muted-foreground/80 whitespace-pre-wrap max-h-96 overflow-y-auto"
          >
            {{ message.reasoning_content }}
          </div>
        </div>

        <!-- Tool Calls Traces -->
        <div v-if="hasToolCalls" class="w-full rounded-md border border-border/40 bg-muted/5 p-3">
          <button
            @click="isToolCallsExpanded = !isToolCallsExpanded"
            class="flex items-center gap-2 text-[11px] font-mono tracking-widest transition-colors w-full text-left uppercase text-muted-foreground hover:text-foreground"
          >
            <Wrench class="w-3 h-3 text-action-blue" />
            <span>Tool Calls</span>
            <span class="opacity-50 lowercase ml-2">({{ activeToolCalls.length }})</span>
            <span class="ml-auto">{{ isToolCallsExpanded ? "[-]" : "[+]" }}</span>
          </button>
          <div v-if="isToolCallsExpanded" class="mt-3 space-y-3 pt-3 border-t border-border/40">
            <div v-for="(toolCall, index) in activeToolCalls" :key="toolCall.id ?? `tool-${index}`">
              <div class="text-[11px] font-mono text-action-blue mb-1">
                › {{ toolCall.function.name }}()
              </div>
              <pre
                class="text-xs font-mono rounded-md text-muted-foreground/80 whitespace-pre-wrap p-3 bg-background border border-border/40"
                >{{ formatToolArguments(toolCall.function.arguments) }}</pre>
            </div>
          </div>
        </div>

        <!-- Web Search Traces -->
        <div
          v-if="message.web_search_calls?.some((c) => c != null)"
          class="w-full rounded-md border border-border/40 bg-muted/5 p-3"
        >
          <button
            @click="isToolCallsExpanded = !isToolCallsExpanded"
            class="flex items-center gap-2 text-[11px] font-mono tracking-widest transition-colors w-full text-left uppercase text-muted-foreground hover:text-foreground"
          >
            <span class="w-3 h-3 flex items-center justify-center text-action-blue">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                class="w-3 h-3"
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" />
                <path d="M2 12h20" />
              </svg>
            </span>
            <span>Web Search</span>
            <span class="ml-auto">{{ isToolCallsExpanded ? "[-]" : "[+]" }}</span>
          </button>
          <div v-if="isToolCallsExpanded" class="mt-3 space-y-3 pt-3 border-t border-border/40">
            <div
              v-for="(call, index) in (props.message.web_search_calls || []).filter(
                (c) => c != null
              )"
              :key="call.id ?? `search-${index}`"
            >
              <div class="flex items-center gap-2 mb-1">
                <span class="text-[11px] font-mono text-action-blue">› search</span>
                <span
                  v-if="call.status === 'in_progress'"
                  class="text-[11px] font-mono text-action-amber animate-pulse"
                  >SEARCHING</span
                >
                <span v-else class="text-[11px] font-mono text-status-success">COMPLETED</span>
              </div>
              <div class="text-xs font-mono text-muted-foreground/80 pl-2">"{{ call.query }}"</div>
            </div>
          </div>
        </div>

        <!-- Anthropic Tool Use -->
        <div
          v-if="toolUseCalls.length > 0"
          class="w-full rounded-md border border-border/40 bg-muted/5 p-3"
        >
          <button
            @click="isToolCallsExpanded = !isToolCallsExpanded"
            class="flex items-center gap-2 text-[11px] font-mono tracking-widest transition-colors w-full text-left uppercase text-muted-foreground hover:text-foreground"
          >
            <Wrench class="w-3 h-3 text-action-blue" />
            <span>Tool Use</span>
            <span class="opacity-50 lowercase ml-2">({{ toolUseCalls.length }})</span>
            <span class="ml-auto">{{ isToolCallsExpanded ? "[-]" : "[+]" }}</span>
          </button>
          <div v-if="isToolCallsExpanded" class="mt-3 space-y-3 pt-3 border-t border-border/40">
            <div v-for="(toolUse, index) in toolUseCalls" :key="toolUse.id ?? `tooluse-${index}`">
              <div class="text-[11px] font-mono text-action-blue mb-1">› {{ toolUse.name }}()</div>
              <pre
                class="text-xs font-mono rounded-md text-muted-foreground/80 whitespace-pre-wrap p-3 bg-background border border-border/40"
                >{{ JSON.stringify(toolUse.input, null, 2) }}</pre>
            </div>
          </div>
        </div>

        <!-- Message Body -->
        <div
          v-if="message.role === 'user'"
          class="w-full font-sans text-[15px] leading-relaxed text-foreground/90 whitespace-pre-wrap break-words"
        >
          <template v-if="typeof message.content === 'string'">
            {{ message.content }}
          </template>
          <template v-else-if="Array.isArray(message.content)">
            <div
              v-for="(part, idx) in message.content.filter((p) => p.type === 'text')"
              :key="'text-' + idx"
            >
              {{ part.text }}
            </div>

            <div
              v-if="message.content.some((p) => p.type === 'image_url')"
              class="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-4"
            >
              <button
                v-for="(part, idx) in message.content.filter((p) => p.type === 'image_url')"
                :key="'img-' + idx"
                type="button"
                class="focus-visible-ring relative overflow-hidden border border-border/40 bg-muted/20 aspect-video flex items-center justify-center cursor-zoom-in"
                :aria-label="t('chat.openImage', { n: idx + 1 })"
                @click="openImageModal(part.image_url.url)"
              >
                <img
                  :src="part.image_url.url"
                  :alt="t('chat.uploadedImageAlt', { n: idx + 1 })"
                  class="w-full h-full object-cover hover:opacity-90 transition-opacity"
                  loading="lazy"
                />
              </button>
            </div>

            <div
              v-if="message.content.some((p) => p.type === 'file')"
              class="flex flex-col gap-2 mt-4"
            >
              <div
                v-for="(part, idx) in message.content.filter((p) => p.type === 'file')"
                :key="'file-' + idx"
                class="flex items-center gap-3 p-2 border border-border/40 bg-muted/10 max-w-sm"
              >
                <FileText class="w-5 h-5 text-muted-foreground shrink-0" />
                <div class="min-w-0">
                  <p class="text-[13px] font-medium truncate font-sans">{{ part.file.filename }}</p>
                </div>
              </div>
            </div>
          </template>
        </div>

        <!-- eslint-disable vue/no-v-html -->
        <div
          v-else
          class="prose prose-sm dark:prose-invert max-w-none break-words font-sans prose-p:leading-[1.7] prose-p:text-[15px] prose-p:text-foreground/90 prose-headings:font-display prose-headings:font-medium prose-headings:tracking-tight prose-a:text-foreground prose-a:underline prose-a:decoration-border/60 hover:prose-a:decoration-foreground prose-code:before:content-none prose-code:after:content-none prose-code:bg-muted/50 prose-code:px-1.5 prose-code:py-0.5 prose-code:font-mono prose-code:text-[13px] prose-code:text-foreground prose-pre:bg-muted/10 prose-pre:border prose-pre:border-border/40 prose-pre:rounded-none prose-pre:p-4"
          v-html="renderedContent"
          @click="handleContentClick"
        ></div>

        <!-- Generated Images (for assistant messages with image generation) -->
        <div
          v-if="message.role === 'assistant' && hasImageParts"
          class="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-4"
        >
          <button
            v-for="(part, idx) in imageParts"
            :key="'img-' + idx"
            type="button"
            class="focus-visible-ring relative overflow-hidden border border-border/40 bg-muted/20 aspect-video flex items-center justify-center cursor-zoom-in"
            :aria-label="t('chat.openImage', { n: idx + 1 })"
            @click="openImageModal(part.image_url.url)"
          >
            <img
              :src="part.image_url.url"
              :alt="t('chat.generatedImageAlt', { n: idx + 1 })"
              class="w-full h-full object-cover hover:opacity-90 transition-opacity"
              loading="lazy"
            />
          </button>
        </div>

        <!-- Generated Audio Player -->
        <div
          v-if="message.audioUrl && message.explicitAudio"
          class="mt-4 pt-4 border-t border-border/40"
        >
          <div class="text-[11px] uppercase font-mono tracking-widest text-muted-foreground mb-2">
            {{ t("chat.generatedAudio") }}
          </div>
          <audio
            controls
            :src="message.audioUrl"
            class="w-full h-8 max-w-md grayscale"
            preload="metadata"
          ></audio>
        </div>
      </div>
    </div>
    <!-- Image Lightbox Modal (teleported to body to keep a single root for <TransitionGroup>) -->
    <Teleport to="body">
      <div
        v-if="previewImageUrl"
        class="fixed inset-0 z-50 flex items-center justify-center bg-background/95 backdrop-blur-sm p-4 transition-all duration-300 animate-in fade-in"
        @click.stop="closeImageModal"
      >
        <button
          type="button"
          class="absolute top-6 right-6 text-muted-foreground hover:text-foreground p-2 transition-colors cursor-pointer"
          @click.stop="closeImageModal"
        >
          <X class="w-6 h-6" />
        </button>
        <img
          :src="previewImageUrl"
          :alt="t('chat.imagePreviewAlt')"
          class="max-w-full max-h-[90vh] object-contain shadow-lg animate-in zoom-in-95 duration-200"
          @click.stop
        />
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
/* Use system default text selection highlight - no custom background */

:deep(.code-block-wrapper pre) {
  margin: 0 !important;
  background: transparent !important;
  padding: 1rem !important;
  border-radius: 0 !important;
  color: var(--color-foreground) !important;
}

:deep(.code-block-wrapper code) {
  background: transparent !important;
  padding: 0 !important;
  border-radius: 0 !important;
  color: inherit !important;
}

/* On touch devices, keep action bar always visible since hover doesn't work */
@media (hover: none) {
  .action-bar {
    opacity: 1 !important;
    pointer-events: auto !important;
  }
}

/* Markdown-rendered images: style and make them clickable for lightbox */
:deep(img) {
  max-width: 100%;
  max-height: 512px;
  object-fit: contain;
  border-radius: 0.5rem;
  border: 1px solid hsl(var(--border));
  cursor: zoom-in;
  transition: opacity 0.2s ease;
}

:deep(img:hover) {
  opacity: 0.9;
}

/* Markdown images are wrapped in a button for keyboard access, so the wrapper
   must not paint button chrome. */
:deep(.zoomable-image) {
  display: inline-block;
  padding: 0;
  border: 0;
  background: none;
  cursor: zoom-in;
}
</style>
