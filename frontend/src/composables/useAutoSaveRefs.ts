import type { AutoSaveState } from "@/composables/useSettingAutoSave";

/**
 * Alias the auto-save refs out of an `AutoSaveState` prop.
 *
 * Bindings mutate the shared ref value owned by the parent view, never the
 * prop object itself (vue/no-mutating-props). Destructuring the refs once
 * keeps templates terse (`state.x` instead of `autoSave.state.value.x`) while
 * preserving reactivity — the destructured `Ref`s stay linked to the parent.
 */
export function useAutoSaveRefs<T extends object>(autoSave: AutoSaveState<T>) {
  const { state, pending, error } = autoSave;
  return { state, pending, error };
}
