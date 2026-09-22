// frontend/src/stores/chat.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import type { ChatMessage } from "@/types/schemas";
import { useChatStore } from "./chat";

beforeEach(() => {
  setActivePinia(createPinia());
  localStorage.clear();
  vi.useRealTimers();
});

const assistantMessage = (content: string): ChatMessage =>
  ({ id: "m1", role: "assistant", content }) as ChatMessage;

const persisted = (): ChatMessage[] =>
  JSON.parse(localStorage.getItem(STORAGE_KEYS.CHAT_MESSAGES) ?? "null") ?? [];

describe("chat store persistence", () => {
  it("persists a pushed message after the debounce window", async () => {
    vi.useFakeTimers();
    const store = useChatStore();

    store.pushMessage(assistantMessage("hello"));
    await vi.advanceTimersByTimeAsync(900);

    expect(persisted()).toHaveLength(1);
    expect(persisted()[0]!.content).toBe("hello");
  });

  it("persists in-place mutations only once the caller marks the store dirty", async () => {
    vi.useFakeTimers();
    const store = useChatStore();

    store.pushMessage(assistantMessage("hello"));
    await vi.advanceTimersByTimeAsync(900);

    // Streaming appends to the last message in place...
    store.messages[0]!.content = "hello world";
    await vi.advanceTimersByTimeAsync(900);
    // ...and without an explicit dirty flag nothing is re-persisted.
    expect(persisted()[0]!.content).toBe("hello");

    store.touchChat();
    await vi.advanceTimersByTimeAsync(900);
    expect(persisted()[0]!.content).toBe("hello world");
  });

  it("drops the transcript from localStorage on reset", async () => {
    vi.useFakeTimers();
    const store = useChatStore();

    store.pushMessage(assistantMessage("secret"));
    await vi.advanceTimersByTimeAsync(900);
    expect(localStorage.getItem(STORAGE_KEYS.CHAT_MESSAGES)).not.toBeNull();

    store.reset();

    expect(localStorage.getItem(STORAGE_KEYS.CHAT_MESSAGES)).toBeNull();
    expect(localStorage.getItem(STORAGE_KEYS.CHAT_RUNS)).toBeNull();
  });
});
