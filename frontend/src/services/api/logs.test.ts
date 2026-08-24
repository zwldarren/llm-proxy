import { describe, expect, it, type Mock, vi } from "vitest";

vi.mock("../http", () => {
  return {
    http: {
      get: vi.fn(() => Promise.resolve({ items: [], total: 0, page: 1, page_size: 50 })),
      delete: vi.fn(() => Promise.resolve({ deleted: 0 })),
    },
  };
});

import { http } from "../http";
import { logsApi } from "./logs";

function parseUrl(value: string) {
  return new URL(value, "http://test");
}

function getCallUrl(mockFn: Mock, callIndex: number): string {
  const calls = mockFn.mock.calls;
  const call = calls[callIndex];
  if (!call || call.length === 0) {
    throw new Error(`No call at index ${callIndex}`);
  }
  return call[0] as string;
}

describe("logsApi", () => {
  it("getLogs builds the list endpoint URL", async () => {
    await logsApi.getLogs({
      page: 2,
      page_size: 20,
      status_code: 500,
      model: "gpt-4",
      provider: "openai",
      user: "admin",
    });

    expect(http.get).toHaveBeenCalledTimes(1);
    const url = getCallUrl(http.get as Mock, 0);
    const parsed = parseUrl(url);
    expect(parsed.pathname).toBe("/api/logs");
    expect(parsed.searchParams.get("page")).toBe("2");
    expect(parsed.searchParams.get("page_size")).toBe("20");
    expect(parsed.searchParams.get("status_code")).toBe("500");
    expect(parsed.searchParams.get("model")).toBe("gpt-4");
    expect(parsed.searchParams.get("provider")).toBe("openai");
    expect(parsed.searchParams.get("user")).toBe("admin");
  });

  it("getLogs sends status_code_from/status_code_to for range filters", async () => {
    await logsApi.getLogs({
      status_code_from: 400,
      status_code_to: 499,
    });

    const url = getCallUrl(http.get as Mock, 1);
    const parsed = parseUrl(url);
    expect(parsed.searchParams.get("status_code_from")).toBe("400");
    expect(parsed.searchParams.get("status_code_to")).toBe("499");
    expect(parsed.searchParams.has("status_code")).toBe(false);
  });

  it("deleteOldLogs calls the purge endpoint with older_than_days", async () => {
    await logsApi.deleteOldLogs(30);

    expect(http.delete).toHaveBeenCalledTimes(1);
    const url = getCallUrl(http.delete as Mock, 0);
    const parsed = parseUrl(url);
    expect(parsed.pathname).toBe("/api/logs/cleanup");
    expect(parsed.searchParams.get("older_than_days")).toBe("30");
  });
});
