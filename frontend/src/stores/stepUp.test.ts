import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it } from "vitest";
import { useStepUpStore } from "./stepUp";

describe("step-up queue", () => {
  beforeEach(() => setActivePinia(createPinia()));

  it("resolves every queued request after one successful verification", async () => {
    const store = useStepUpStore();
    const first = store.requireStepUp();
    const second = store.requireStepUp();
    expect(store.required).toBe(true);

    store.complete();

    await expect(first).resolves.toBeUndefined();
    await expect(second).resolves.toBeUndefined();
    expect(store.required).toBe(false);
  });

  it("rejects all queued requests when the dialog is dismissed", async () => {
    const store = useStepUpStore();
    const first = store.requireStepUp();
    const second = store.requireStepUp();

    store.cancel();

    await expect(first).rejects.toThrow("Step-up cancelled");
    await expect(second).rejects.toThrow("Step-up cancelled");
    expect(store.required).toBe(false);
  });
});
