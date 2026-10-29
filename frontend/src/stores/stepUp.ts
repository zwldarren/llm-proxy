import { defineStore } from "pinia";
import { ref } from "vue";

type Waiter = {
  resolve: () => void;
  reject: (error: Error) => void;
};

/**
 * Single queue for step-up re-authentication.
 *
 * The HTTP layer calls {@link requireStepUp} when a sensitive request is
 * rejected with ``step_up_required``. Every request waiting on that rejection
 * shares one dialog and retries after the password or passkey proves the user
 * is still present.
 */
export const useStepUpStore = defineStore("stepUp", () => {
  const required = ref(false);
  const waiters: Waiter[] = [];

  function requireStepUp(): Promise<void> {
    required.value = true;
    return new Promise<void>((resolve, reject) => {
      waiters.push({ resolve, reject });
    });
  }

  function complete(): void {
    required.value = false;
    for (const waiter of waiters.splice(0)) waiter.resolve();
  }

  function cancel(): void {
    required.value = false;
    const error = new Error("Step-up cancelled");
    for (const waiter of waiters.splice(0)) waiter.reject(error);
  }

  return { required, requireStepUp, complete, cancel };
});
