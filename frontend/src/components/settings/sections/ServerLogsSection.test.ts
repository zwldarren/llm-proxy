import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import { SettingsItem } from "@/components/settings";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { DEFAULT_LOGGING } from "@/constants/defaults";
import type { LoggingConfig } from "@/types/schemas";
import ServerLogsSection from "./ServerLogsSection.vue";

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));

function mountSection() {
  const autoSave = {
    state: ref({ ...DEFAULT_LOGGING }),
    pending: ref(false),
    error: ref(null),
  } as unknown as AutoSaveState<LoggingConfig>;
  return mount(ServerLogsSection, { props: { autoSave, isAdmin: true } });
}

describe("ServerLogsSection", () => {
  it("groups settings into Capture, Retention, and Audit without changing their order within a group", () => {
    const wrapper = mountSection();
    const groups = wrapper.findAll("section[aria-labelledby]");
    expect(groups).toHaveLength(3);
    expect(groups.map((group) => group.find("h3").text())).toEqual([
      "settings.logCaptureGroup",
      "settings.logRetentionGroup",
      "settings.logAuditGroup",
    ]);

    expect(groups[0].text()).toContain("settings.logInputOutput");
    expect(groups[0].text()).toContain("settings.logRawStream");
    expect(groups[0].text()).toContain("settings.maskSensitiveData");
    expect(groups[0].text()).toContain("settings.sensitiveKeys");

    expect(groups[1].text()).toContain("settings.logRetentionDays");
    expect(groups[1].text()).toContain("settings.cleanOldLogs");

    expect(groups[2].text()).toContain("settings.samplingRate");
    expect(groups[2].text()).toContain("settings.maxLoggedBodySize");
    expect(groups[2].text()).toContain("settings.auditSamplingRate");
    expect(groups[2].text()).toContain("settings.auditRetentionDays");
    expect(wrapper.findAllComponents(SettingsItem)).toHaveLength(10);
  });
});
