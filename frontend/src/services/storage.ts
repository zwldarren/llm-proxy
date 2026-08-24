import { STORAGE_KEYS } from "@/constants/storageKeys";

/**
 * Thin localStorage wrappers for use outside Vue's setup context (e.g., http.ts).
 * For reactive usage inside components/stores, prefer @vueuse/core useStorage.
 */

const TOKEN_KEY = STORAGE_KEYS.AUTH_TOKEN;

/** Cached result of the localStorage probe — computed once per session. */
let _cachedStorage: Storage | null | undefined;

function getStorage(): Storage | null {
  if (_cachedStorage !== undefined) return _cachedStorage;
  // globalThis covers both browser (window) and Node.js (global) environments
  const gs = globalThis as Record<string, unknown>;
  const ls = gs.localStorage as Storage | undefined;
  if (!ls) {
    _cachedStorage = null;
    return null;
  }
  try {
    // Touch a key to verify localStorage is accessible (handles
    // browsers that throw in private/incognito mode)
    ls.getItem("");
    _cachedStorage = ls;
    return ls;
  } catch {
    _cachedStorage = null;
    return null;
  }
}

function safeGetItem(key: string): string | null {
  const storage = getStorage();
  if (!storage) return null;
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function safeSetItem(key: string, value: string): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    storage.setItem(key, value);
  } catch {
    // Storage full or unavailable — silently ignore
  }
}

function safeRemoveItem(key: string): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    storage.removeItem(key);
  } catch {
    // Silently ignore
  }
}

export const tokenStorage = {
  get(): string | null {
    return safeGetItem(TOKEN_KEY);
  },
  set(token: string): void {
    safeSetItem(TOKEN_KEY, token);
  },
  remove(): void {
    safeRemoveItem(TOKEN_KEY);
  },
};
