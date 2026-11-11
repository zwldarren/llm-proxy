import { useIntervalFn } from "@vueuse/core";
import { onUnmounted, ref, type Ref } from "vue";

/**
 * Options for configuring auto-refresh behavior.
 */
interface AutoRefreshOptions {
  /** Interval in milliseconds (default: 5000) */
  interval?: number;
  /** Whether refresh should be paused when a specific condition is met (e.g., dialog is open) */
  shouldPauseRefresh?: () => boolean;
  /** Callback when refresh is paused/resumed */
  onRefreshStateChange?: (isRefreshing: boolean) => void;
  /** Callback to execute the actual refresh operation */
  onRefresh?: () => void | Promise<void>;
}

/** Controls and state exposed by the auto-refresh composable. */
export interface AutoRefreshControls {
  autoRefresh: Ref<boolean>;
  startAutoRefresh: () => void;
  stopAutoRefresh: () => void;
  toggleAutoRefresh: () => void;
}

/**
 * Composable for managing auto-refresh functionality.
 * Provides start, stop, and toggle controls for automated data fetching.
 * The polling loop itself is VueUse's `useIntervalFn`: pausing and resuming
 * are library operations rather than a hand-managed timer.
 *
 * @example
 * const { autoRefresh, startAutoRefresh, stopAutoRefresh, toggleAutoRefresh } = useAutoRefresh({
 *   interval: 5000,
 *   shouldPauseRefresh: () => showDetailDialog.value,
 *   onRefresh: () => checkAndRefresh(),
 * });
 *
 * // In template
 * <button @click="toggleAutoRefresh">
 *   {{ autoRefresh ? 'Pause' : 'Resume' }}
 * </button>
 */
export function useAutoRefresh(options: AutoRefreshOptions = {}): AutoRefreshControls {
  const autoRefresh = ref(true);

  const { pause, resume } = useIntervalFn(
    () => {
      if (!options.shouldPauseRefresh?.()) void options.onRefresh?.();
    },
    options.interval ?? 5000,
    // The interval only starts when the consumer calls startAutoRefresh().
    { immediate: false }
  );

  /**
   * Start the auto-refresh timer.
   * Only starts if auto-refresh is enabled and the pause condition is not met.
   */
  const startAutoRefresh = () => {
    if (!autoRefresh.value || options.shouldPauseRefresh?.()) return;
    options.onRefreshStateChange?.(true);
    resume();
  };

  /**
   * Stop the auto-refresh timer.
   */
  const stopAutoRefresh = () => {
    pause();
    options.onRefreshStateChange?.(false);
  };

  /**
   * Toggle auto-refresh on/off.
   * Handles pausing/resuming the refresh timer.
   */
  const toggleAutoRefresh = () => {
    autoRefresh.value = !autoRefresh.value;
    if (autoRefresh.value) startAutoRefresh();
    else stopAutoRefresh();
  };

  // Clean up timer on component unmount
  onUnmounted(stopAutoRefresh);

  return {
    autoRefresh,
    startAutoRefresh,
    stopAutoRefresh,
    toggleAutoRefresh,
  };
}
