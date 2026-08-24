import { useDebounceFn } from "@vueuse/core";
import { defineStore } from "pinia";
import { ref, watch } from "vue";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import type { ChatMessage } from "@/types/schemas";
import type { ChatRun } from "@/types/runs";
import { appendRun } from "@/utils/runs";

const MAX_STORED_MESSAGES = 100;
/** Tray specimens are session-scoped; cap them since payloads can be large. */
const MAX_RUNS = 50;

/** Persisted runs are telemetry stubs only — payloads never hit localStorage. */
function loadRuns(): ChatRun[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.CHAT_RUNS);
    if (raw) return JSON.parse(raw) as ChatRun[];
  } catch {
    // ignore parse errors
  }
  return [];
}

function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function loadMessages(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.CHAT_MESSAGES);
    if (raw) {
      const parsed = JSON.parse(raw) as ChatMessage[];
      return parsed.map((m) => {
        // Clear blob URLs on load since they are invalid after page reload
        const audioUrl = m.audioUrl && m.audioUrl.startsWith("blob:") ? undefined : m.audioUrl;
        return {
          ...m,
          audioUrl,
          id: m.id || generateMessageId(),
        };
      });
    }
  } catch {
    // ignore parse errors
  }
  return [];
}

export const useChatStore = defineStore("chat", () => {
  const messages = ref<ChatMessage[]>(loadMessages());
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  const saveMessages = useDebounceFn(() => {
    try {
      const toSave = messages.value.slice(-MAX_STORED_MESSAGES);
      localStorage.setItem(STORAGE_KEYS.CHAT_MESSAGES, JSON.stringify(toSave));
    } catch {
      // ignore localStorage errors (e.g. quota exceeded)
    }
  }, 800);

  watch(messages, saveMessages, { deep: true });

  const currentlyPlayingId = ref<string | null>(null);
  let activeAudio: HTMLAudioElement | null = null;

  function playAudio(id: string, url: string, onEnded?: () => void) {
    // Pause any current playback. The caller owns the stop toggle (it stops
    // before ever calling us), so a same-id request here means "replace the
    // audio", never "toggle off" — dropping it leaves the user with silence
    // and no error.
    if (activeAudio) {
      activeAudio.pause();
      activeAudio.onended = null;
    }

    currentlyPlayingId.value = id;
    const audio = new Audio(url);
    audio.preload = "auto";
    activeAudio = audio;

    // Only reset state if this element is still the active one — a newer
    // playAudio call may have already replaced it.
    const finish = () => {
      if (activeAudio === audio) activeAudio = null;
      if (currentlyPlayingId.value === id) currentlyPlayingId.value = null;
    };

    audio.onended = () => {
      finish();
      if (onEnded) onEnded();
    };
    audio.onerror = () => {
      // Decode/load failure (not an autoplay-policy issue).
      console.error("Audio element failed to load:", url);
      finish();
      if (onEnded) onEnded();
    };

    const startPlayback = () => {
      const result = audio.play();
      if (result !== undefined) {
        result.catch((err) => {
          console.error("Audio playback failed:", err);
          finish();
          if (onEnded) onEnded();
        });
      }
    };

    // Calling play() before the element has its metadata is a classic source
    // of silent failures (WebKit especially). Try immediately — blob URLs are
    // usually ready instantly — and retry once the element is playable if it
    // wasn't loaded yet.
    if (audio.readyState >= HTMLMediaElement.HAVE_METADATA) {
      startPlayback();
    } else {
      audio.addEventListener("canplay", startPlayback, { once: true });
      startPlayback();
    }
  }

  function stopAudio() {
    if (activeAudio) {
      activeAudio.pause();
      activeAudio = null;
    }
    currentlyPlayingId.value = null;
  }

  function pushMessage(msg: ChatMessage) {
    if (!msg.id) {
      msg.id = generateMessageId();
    }
    messages.value.push(msg);
  }

  // Run specimens for the playground tray — one record per API request.
  // Payloads stay in memory only (they can hold base64 attachments, too heavy
  // for localStorage); lightweight telemetry stubs persist so a restored
  // transcript keeps a matching tray.
  const runs = ref<ChatRun[]>(loadRuns());

  const saveRuns = useDebounceFn(() => {
    try {
      const stubs = runs.value.slice(-MAX_RUNS).map((r) => ({ ...r, payload: null }));
      localStorage.setItem(STORAGE_KEYS.CHAT_RUNS, JSON.stringify(stubs));
    } catch {
      // ignore localStorage errors (e.g. quota exceeded)
    }
  }, 800);

  watch(runs, saveRuns, { deep: true });

  function upsertRun(run: ChatRun) {
    const idx = runs.value.findIndex((r) => r.id === run.id);
    if (idx >= 0) {
      runs.value[idx] = run;
    } else {
      runs.value = appendRun(runs.value, run, MAX_RUNS);
    }
  }

  function clearMessages() {
    messages.value = [];
    runs.value = [];
    error.value = null;
    stopAudio();
  }

  function setLoading(val: boolean) {
    isLoading.value = val;
  }

  function setError(val: string | null) {
    error.value = val;
  }

  return {
    messages,
    isLoading,
    error,
    currentlyPlayingId,
    runs,
    upsertRun,
    playAudio,
    stopAudio,
    pushMessage,
    clearMessages,
    setLoading,
    setError,
  };
});
