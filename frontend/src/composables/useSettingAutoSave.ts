import { useDebounceFn } from "@vueuse/core";
import { ref, toRaw, watch, onUnmounted, type Ref } from "vue";
import { toast } from "vue-sonner";
import { useI18n } from "vue-i18n";
import { getErrorMessage } from "@/utils/error";

export interface AutoSaveState<T> {
  state: Ref<T>;
  pending: Ref<boolean>;
  error: Ref<string | null>;
  initialize: (value: T) => void;
  save: () => Promise<boolean>;
}

export function useSettingAutoSave<T extends object>(
  initialValue: T,
  saveFn: (value: T) => Promise<T | void>,
  options: { debounceMs?: number; toastThrottleMs?: number; showSuccessToast?: boolean } = {}
): AutoSaveState<T> {
  const { t } = useI18n();
  const state = ref<T>(structuredClone(initialValue)) as Ref<T>;
  const lastSaved = ref<T>(structuredClone(initialValue));
  const pending = ref(false);
  const error = ref<string | null>(null);

  const scheduleSave = useDebounceFn(doSave, options.debounceMs ?? 300);
  let lastToastAt = 0;
  let isSaving = false;

  async function doSave(): Promise<boolean> {
    if (isSaving) return false;
    isSaving = true;
    pending.value = true;
    error.value = null;
    const snapshot = structuredClone(toRaw(state.value) as T);
    try {
      await saveFn(snapshot);
      lastSaved.value = snapshot;
      // Auto-save has no explicit submit button, so a throttled toast is the
      // durable confirmation that the change reached the server.
      if (options.showSuccessToast ?? true) {
        const now = Date.now();
        if (now - lastToastAt > (options.toastThrottleMs ?? 3000)) {
          toast.success(t("settings.saved"));
          lastToastAt = now;
        }
      }
      return true;
    } catch (e) {
      error.value = getErrorMessage(e) || t("settings.saveFailed");
      state.value = structuredClone(toRaw(lastSaved.value) as T) as Ref<T>["value"];
      return false;
    } finally {
      pending.value = false;
      isSaving = false;
      scheduleSaveIfDirty();
    }
  }

  function scheduleSaveIfDirty() {
    if (snapshotChecksum(state.value) !== snapshotChecksum(lastSaved.value)) {
      void scheduleSave();
    }
  }

  watch(
    () => state.value,
    () => {
      if (isSaving) return;
      scheduleSaveIfDirty();
    },
    { deep: true }
  );

  function initialize(value: T) {
    state.value = structuredClone(toRaw(value)) as Ref<T>["value"];
    lastSaved.value = structuredClone(toRaw(value));
  }

  onUnmounted(() => {
    if (scheduleSave.isPending.value) {
      scheduleSave.cancel();
      // Flush any pending save before the component is destroyed so changes
      // made just before navigation are not silently dropped.
      doSave().catch(() => {});
    }
  });

  return {
    state,
    pending,
    error,
    initialize,
    save: doSave,
  };
}

function snapshotChecksum(value: unknown): string {
  try {
    return JSON.stringify(toRaw(value));
  } catch {
    return "";
  }
}
