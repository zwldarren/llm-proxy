import { defineStore } from "pinia";
import { ref } from "vue";
import type { SystemInfo } from "@/services/api/system";
import { systemApi } from "@/services/api/system";

export const useSystemStore = defineStore("system", () => {
  const info = ref<SystemInfo | null>(null);
  const loading = ref(false);

  /** In-flight request deduplication (AppLayout mounts once per page view). */
  let inFlight: Promise<SystemInfo> | null = null;

  async function fetchSystemInfo(force = false): Promise<SystemInfo> {
    // Non-forced calls resolve once per session: subsequent callers reuse the
    // cached info instead of hitting the API again. A null info (never fetched
    // or previous failure) triggers a real fetch.
    if (!force && info.value) return info.value;
    if (inFlight) return inFlight;

    loading.value = true;
    const promise = systemApi
      .getSystemInfo(force)
      .then((res) => {
        info.value = res;
        return res;
      })
      .finally(() => {
        if (inFlight === promise) inFlight = null;
        loading.value = false;
      });
    inFlight = promise;
    return promise;
  }

  return { info, loading, fetchSystemInfo };
});
