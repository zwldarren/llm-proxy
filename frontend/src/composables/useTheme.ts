import { createGlobalState, useColorMode } from "@vueuse/core";
import { computed } from "vue";

export type Theme = "light" | "dark" | "system";

const useGlobalColorMode = createGlobalState(() =>
  useColorMode({
    modes: {
      light: "light",
      dark: "dark",
    },

    initialValue: "dark",
    disableTransition: false,
  })
);

export const initializeTheme = () => {
  const mode = useGlobalColorMode();
  void mode.store.value;
};

export function useTheme() {
  const mode = useGlobalColorMode();

  const theme = computed<Theme>({
    get: () => {
      const raw: string = mode.store.value;
      if (raw === "auto") return "system";
      return raw as Theme;
    },
    set: (v: Theme) => {
      mode.store.value = v === "system" ? "auto" : v;
    },
  });

  const isDark = computed(() => {
    const raw: string = mode.store.value;
    if (raw === "auto") return mode.system.value === "dark";
    return raw === "dark";
  });

  const setTheme = (newTheme: Theme) => {
    theme.value = newTheme;
  };

  return {
    theme,
    setTheme,
    isDark,
  };
}
