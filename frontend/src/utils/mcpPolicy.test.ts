import { describe, expect, it } from "vitest";
import type { McpSecurityPolicyConfig } from "@/types/schemas";
import {
  disallowedEnvKeys,
  isBlockedCommand,
  isCommandAllowed,
  isEnvKeyAllowed,
} from "@/utils/mcpPolicy";

function makePolicy(overrides: Partial<McpSecurityPolicyConfig> = {}): McpSecurityPolicyConfig {
  return {
    require_key_mcp_permissions: true,
    allowed_commands: [],
    blocked_commands: ["bash", "sh", "python", "node"],
    allowed_env_keys: [],
    blocked_env_keys: ["PATH", "HOME"],
    blocked_url_hosts: [],
    blocked_url_ips: [],
    ...overrides,
  };
}

describe("mcpPolicy", () => {
  describe("isBlockedCommand", () => {
    it("detects blocked commands case-insensitively and by path", () => {
      const policy = makePolicy();
      expect(isBlockedCommand("bash", policy)).toBe(true);
      expect(isBlockedCommand("/bin/bash", policy)).toBe(true);
      expect(isBlockedCommand("BASH", policy)).toBe(true);
      expect(isBlockedCommand("npx", policy)).toBe(false);
    });
  });

  describe("isCommandAllowed", () => {
    it("denies everything when the allowlist is empty", () => {
      const policy = makePolicy();
      expect(isCommandAllowed("npx", ["-y", "pkg"], policy)).toBe(false);
    });

    it("allows broad entries regardless of args", () => {
      const policy = makePolicy({ allowed_commands: ["npx"] });
      expect(isCommandAllowed("npx", ["-y", "anything"], policy)).toBe(true);
    });

    it("allows exact entries only for the matching positional arg", () => {
      const policy = makePolicy({ allowed_commands: ["npx mcp-searxng"] });
      expect(isCommandAllowed("npx", ["-y", "mcp-searxng"], policy)).toBe(true);
      expect(isCommandAllowed("npx", ["other-pkg"], policy)).toBe(false);
      expect(isCommandAllowed("npx", [], policy)).toBe(false);
    });

    it("never allows a blocked command even if listed", () => {
      const policy = makePolicy({ allowed_commands: ["bash"] });
      expect(isCommandAllowed("bash", [], policy)).toBe(false);
    });
  });

  describe("env key checks", () => {
    it("denies every key when the allowlist is empty", () => {
      const policy = makePolicy();
      expect(isEnvKeyAllowed("GITHUB_TOKEN", policy)).toBe(false);
    });

    it("allows listed keys case-insensitively unless blocked", () => {
      const policy = makePolicy({ allowed_env_keys: ["github_token", "PATH"] });
      expect(isEnvKeyAllowed("GITHUB_TOKEN", policy)).toBe(true);
      expect(isEnvKeyAllowed("PATH", policy)).toBe(false);
    });

    it("reports keys that would be dropped", () => {
      const policy = makePolicy({ allowed_env_keys: ["GITHUB_TOKEN"] });
      expect(disallowedEnvKeys(["GITHUB_TOKEN", "API_KEY", "PATH", ""], policy)).toEqual([
        "API_KEY",
        "PATH",
      ]);
    });
  });
});
