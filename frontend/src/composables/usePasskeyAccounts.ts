import { useStorage } from "@vueuse/core";
import { STORAGE_KEYS } from "@/constants/storageKeys";

/**
 * Device-local usernames that have proven a passkey in this browser.
 *
 * This is only a UX hint for choosing the login screen's primary action. It
 * must never grant access, and an unknown username falls back to the password
 * form. With an empty identifier, any known account means a discoverable
 * passkey is available.
 */
export function usePasskeyAccounts() {
  const accounts = useStorage<string[]>(STORAGE_KEYS.PASSKEY_ACCOUNTS, [], localStorage);

  function remember(username: string | null | undefined) {
    const normalized = username?.trim().toLowerCase();
    if (!normalized || accounts.value.includes(normalized)) return;
    accounts.value = [...accounts.value, normalized];
  }

  function forget(username: string | null | undefined) {
    const normalized = username?.trim().toLowerCase();
    if (!normalized) return;
    accounts.value = accounts.value.filter((account) => account !== normalized);
  }

  function isKnown(username = "") {
    const normalized = username.trim().toLowerCase();
    return normalized ? accounts.value.includes(normalized) : accounts.value.length > 0;
  }

  return { remember, forget, isKnown };
}
