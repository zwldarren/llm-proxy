import { computed, ref } from "vue";
import type { ApiKeyResponse } from "@/services/api/apiKeys";

/**
 * One-shot plaintext reveal for a newly created API key.
 *
 * Escape and overlay dismissal update the same open state as the explicit close
 * button, so the decrypted key is always discarded when the dialog closes.
 */
export function useApiKeyReveal() {
  const createdKey = ref<ApiKeyResponse | null>(null);
  const showReveal = ref(false);

  const open = computed({
    get: () => showReveal.value,
    set: (value: boolean) => {
      if (value) {
        showReveal.value = true;
      } else {
        showReveal.value = false;
        createdKey.value = null;
      }
    },
  });

  function reveal(key: ApiKeyResponse) {
    createdKey.value = key;
    showReveal.value = true;
  }

  function close() {
    open.value = false;
  }

  return { createdKey, open, reveal, close };
}
