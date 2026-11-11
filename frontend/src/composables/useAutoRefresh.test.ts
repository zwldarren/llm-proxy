import { mount } from "@vue/test-utils";
import { defineComponent, h } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useAutoRefresh, type AutoRefreshControls } from "@/composables/useAutoRefresh";

describe("useAutoRefresh", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("polls while started, pauses and resumes, and stops on unmount", () => {
    vi.useFakeTimers();
    const onRefresh = vi.fn();
    let controls!: AutoRefreshControls;

    const wrapper = mount(
      defineComponent({
        setup() {
          controls = useAutoRefresh({ interval: 1000, onRefresh });
          return () => h("div");
        },
      })
    );

    controls.startAutoRefresh();
    vi.advanceTimersByTime(3000);
    expect(onRefresh).toHaveBeenCalledTimes(3);

    controls.toggleAutoRefresh();
    vi.advanceTimersByTime(2000);
    expect(onRefresh).toHaveBeenCalledTimes(3);

    controls.toggleAutoRefresh();
    vi.advanceTimersByTime(1000);
    expect(onRefresh).toHaveBeenCalledTimes(4);

    wrapper.unmount();
    vi.advanceTimersByTime(2000);
    expect(onRefresh).toHaveBeenCalledTimes(4);
  });
});
