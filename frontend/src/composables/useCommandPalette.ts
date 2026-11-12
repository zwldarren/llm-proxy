import { ref } from "vue";

/**
 * One shared open flag for the global command palette.
 *
 * The palette mounts once, in App.vue, and owns its own keyboard activation.
 * The sidebar's search trigger has to open that same instance, so the flag
 * cannot live inside the palette component — it is module state that both
 * sides read. Keeping it here (rather than reaching into one component from
 * another) is what lets the palette stay a self-contained component.
 *
 * Mounting it once is not cosmetic: the console keeps several views alive at a
 * time, each of which used to render its own copy, and a per-view copy means
 * N dialogs and N copies of the Cmd/Ctrl+K handler for one piece of state.
 */
const open = ref(false);

/** Hint for the platform's palette accelerator, shown next to the trigger. */
export const COMMAND_PALETTE_KEY_HINT =
  typeof navigator !== "undefined" && /mac|iphone|ipad/i.test(navigator.platform ?? "")
    ? "⌘K"
    : "Ctrl K";

export function useCommandPalette() {
  return {
    paletteOpen: open,
    openCommandPalette: () => {
      open.value = true;
    },
    closeCommandPalette: () => {
      open.value = false;
    },
    toggleCommandPalette: () => {
      open.value = !open.value;
    },
  };
}
