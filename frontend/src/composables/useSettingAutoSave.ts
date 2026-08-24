import { ref, toRaw, watch, onUnmounted, type Ref } from "vue";
import { toast } from "vue-sonner";
import { useI18n } from "vue-i18n";

export interface AutoSaveState<T> {
  state: Ref<T>;
  pending: Ref<boolean>;
  error: Ref<string | null>;
  initialize: (value: T) => void;
  save: () => Promise<void>;
}

export function useSettingAutoSave<T extends object>(
  initialValue: T,
  saveFn: (value: T) => Promise<T | void>,
  options: { debounceMs?: number; toastThrottleMs?: number; showSuccessToast?: boolean } = {}
): AutoSaveState<T> {
  const { t } = useI18n();
  const state = ref<T>(deepClone(initialValue)) as Ref<T>;
  const lastSaved = ref<T>(deepClone(initialValue));
  const pending = ref(false);
  const error = ref<string | null>(null);

  let saveTimer: ReturnType<typeof setTimeout> | null = null;
  let lastToastAt = 0;
  let isSaving = false;

  async function doSave() {
    if (isSaving) return;
    isSaving = true;
    pending.value = true;
    error.value = null;
    const snapshot = deepClone(toRaw(state.value) as T);
    try {
      await saveFn(snapshot);
      lastSaved.value = snapshot;
      if (options.showSuccessToast) {
        const now = Date.now();
        if (now - lastToastAt > (options.toastThrottleMs ?? 3000)) {
          toast.success(t("settings.saved"));
          lastToastAt = now;
        }
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : t("settings.saveFailed");
      state.value = deepClone(lastSaved.value) as Ref<T>["value"];
    } finally {
      pending.value = false;
      isSaving = false;
      if (snapshotChecksum(state.value) !== snapshotChecksum(lastSaved.value)) {
        scheduleSave();
      }
    }
  }

  function scheduleSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(doSave, options.debounceMs ?? 300);
  }

  watch(
    () => state.value,
    () => {
      if (isSaving) return;
      if (snapshotChecksum(state.value) === snapshotChecksum(lastSaved.value)) return;
      scheduleSave();
    },
    { deep: true }
  );

  function initialize(value: T) {
    state.value = deepClone(value) as Ref<T>["value"];
    lastSaved.value = deepClone(value);
  }

  onUnmounted(() => {
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveTimer = null;
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

function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== "object") return obj;
  if (obj instanceof Date) return new Date(obj.getTime()) as unknown as T;
  if (obj instanceof Set) return new Set([...obj].map(deepClone)) as unknown as T;
  if (obj instanceof Map) {
    return new Map([...obj].map(([k, v]) => [deepClone(k), deepClone(v)])) as unknown as T;
  }
  if (Array.isArray(obj)) {
    const arr = obj.map(deepClone);
    Object.keys(obj)
      .filter((k) => !/^\d+$/.test(k))
      .forEach((k) => {
        (arr as unknown as Record<string, unknown>)[k] = deepClone(
          (obj as Record<string, unknown>)[k]
        );
      });
    return arr as unknown as T;
  }
  const cloned = {} as Record<string, unknown>;
  for (const key of Object.keys(obj)) {
    cloned[key] = deepClone((obj as Record<string, unknown>)[key]);
  }
  return cloned as T;
}
