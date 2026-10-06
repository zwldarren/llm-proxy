import { useDebounceFn } from "@vueuse/core";
import { defineStore } from "pinia";
import { ref, watch } from "vue";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import type { ChatMessage } from "@/types/schemas";

const MAX_STORED_MESSAGES = 100;

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

  /**
   * Persistence is driven by an explicit dirty counter rather than a deep
   * watcher: streaming mutates the last message in place on every chunk, and a
   * deep watcher re-traverses the whole transcript for each one (the debounced
   * write never fixed the traversal). Every mutation of the message list must
   * call touchChat().
   */
  const revision = ref(0);

  function touchChat() {
    revision.value += 1;
  }

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
    touchChat();
  }

  watch(revision, () => {
    saveMessages();
  });

  function clearMessages() {
    messages.value = [];
    error.value = null;
    stopAudio();
    touchChat();
  }

  /** Session teardown: drop the transcript from memory AND localStorage so
   *  the next user on this browser never sees the previous user's chat.
   *  (clearMessages alone leaves the persisted keys in place if the tab
   *  closes before the debounced writer flushes.) */
  function reset() {
    clearMessages();
    try {
      localStorage.removeItem(STORAGE_KEYS.CHAT_MESSAGES);
      // Legacy key from the removed run tray — clean up stale data.
      localStorage.removeItem(STORAGE_KEYS.CHAT_RUNS);
    } catch {
      // ignore localStorage errors
    }
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
    playAudio,
    stopAudio,
    pushMessage,
    touchChat,
    clearMessages,
    reset,
    setLoading,
    setError,
  };
});
