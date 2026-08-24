import { onUnmounted, ref } from "vue";

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

/**
 * Composable for managing auto-refresh functionality.
 * Provides start, stop, and toggle controls for automated data fetching.
 *
 * @example
 * const { autoRefresh, startAutoRefresh, stopAutoRefresh, toggleAutoRefresh } = useAutoRefresh({
 *   interval: 5000,
 *   shouldPauseRefresh: () => showDetailDialog.value,
 *   onRefreshStateChange: (isRefreshing) => console.log('Refresh state:', isRefreshing)
 * });
 *
 * // In template
 * <button @click="toggleAutoRefresh">
 *   {{ autoRefresh ? 'Pause' : 'Resume' }}
 * </button>
 */
export function useAutoRefresh(options?: AutoRefreshOptions) {
  const autoRefresh = ref(true);
  const refreshInterval = options?.interval ?? 5000;
  let refreshTimer: ReturnType<typeof setInterval> | null = null;

  /**
   * Start the auto-refresh timer.
   * Only starts if autoRefresh is enabled and the pause condition is not met.
   */
  const startAutoRefresh = () => {
    if (refreshTimer) {
      clearInterval(refreshTimer);
    }
    if (autoRefresh.value && !options?.shouldPauseRefresh?.()) {
      options?.onRefreshStateChange?.(true);
      refreshTimer = setInterval(() => {
        if (!options?.shouldPauseRefresh?.()) {
          options?.onRefresh?.();
        }
      }, refreshInterval);
    }
  };

  /**
   * Stop the auto-refresh timer.
   */
  const stopAutoRefresh = () => {
    if (refreshTimer) {
      clearInterval(refreshTimer);
      refreshTimer = null;
    }
    options?.onRefreshStateChange?.(false);
  };

  /**
   * Toggle auto-refresh on/off.
   * Handles pausing/resuming the refresh timer.
   */
  const toggleAutoRefresh = () => {
    autoRefresh.value = !autoRefresh.value;
    if (autoRefresh.value) {
      startAutoRefresh();
    } else {
      stopAutoRefresh();
    }
    options?.onRefreshStateChange?.(autoRefresh.value);
  };

  // Clean up timer on component unmount
  onUnmounted(() => {
    stopAutoRefresh();
  });

  return {
    autoRefresh,
    startAutoRefresh,
    stopAutoRefresh,
    toggleAutoRefresh,
  };
}
