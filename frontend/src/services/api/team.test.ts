import { describe, expect, it, type Mock, vi } from "vitest";

vi.mock("../http", () => {
  return {
    http: {
      get: vi.fn(() => Promise.resolve({ items: [], total: 0 })),
      post: vi.fn(() => Promise.resolve({})),
      put: vi.fn(() => Promise.resolve({})),
    },
  };
});

import { http } from "../http";
import { teamApi } from "./team";

function parseUrl(value: string) {
  return new URL(value, "http://test");
}

describe("teamApi", () => {
  it("listMembers sends pagination, search, and sort parameters", async () => {
    await teamApi.listMembers({
      limit: 50,
      offset: 100,
      search: "adm",
      sort_by: "created",
      sort_dir: "desc",
    });

    const parsed = parseUrl((http.get as Mock).mock.calls[0]![0] as string);
    expect(parsed.pathname).toBe("/api/team/members");
    expect(parsed.searchParams.get("limit")).toBe("50");
    expect(parsed.searchParams.get("offset")).toBe("100");
    expect(parsed.searchParams.get("search")).toBe("adm");
    expect(parsed.searchParams.get("sort_by")).toBe("created");
    expect(parsed.searchParams.get("sort_dir")).toBe("desc");
  });

  it("listMembers omits empty optional parameters", async () => {
    await teamApi.listMembers({ limit: 50, offset: 0 });

    const parsed = parseUrl((http.get as Mock).mock.calls.at(-1)![0] as string);
    expect(parsed.searchParams.has("search")).toBe(false);
    expect(parsed.searchParams.has("sort_by")).toBe(false);
    expect(parsed.searchParams.get("offset")).toBe("0");
  });

  it("createMember forwards role and model allowlist", async () => {
    await teamApi.createMember({
      username: "member",
      password: "Temporary-pass-123!",
      role: "admin",
      allowed_models: ["gpt-4o"],
    });

    expect(http.post).toHaveBeenCalledWith("/api/team/members", {
      username: "member",
      password: "Temporary-pass-123!",
      role: "admin",
      allowed_models: ["gpt-4o"],
    });
  });

  it("updateMemberRateLimit sends an explicit null to clear the cap", async () => {
    await teamApi.updateMemberRateLimit(7, null);

    expect(http.put).toHaveBeenCalledWith("/api/team/members/7/rate-limit", {
      rate_limit_rpm: null,
    });
  });
});
