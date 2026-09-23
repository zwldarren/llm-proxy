import type { McpSecurityPolicyConfig } from "@/types/schemas";

/**
 * Client-side mirror of the backend MCP security policy checks.
 *
 * These helpers are only used to give the operator inline, proactive feedback
 * while editing an MCP server. The backend remains the source of truth — the
 * policy is re-validated server-side on every create/update.
 */

function normalizeName(command: string): string {
  return command.trim().toLowerCase().split("/").pop() ?? "";
}

/** Return True when the command is explicitly blocked by the policy. */
export function isBlockedCommand(command: string, policy: McpSecurityPolicyConfig): boolean {
  const name = normalizeName(command);
  return policy.blocked_commands.some((c) => c.trim().toLowerCase() === name);
}

/**
 * Return True when the command (with its args) is permitted by the allowlist.
 *
 * Mirrors `McpSecurityPolicy.is_allowed_command`: broad entries (e.g. `npx`)
 * permit any invocation, exact entries (e.g. `npx mcp-searxng`) permit only
 * the matching first positional argument.
 */
export function isCommandAllowed(
  command: string,
  args: string[],
  policy: McpSecurityPolicyConfig
): boolean {
  const name = normalizeName(command);
  if (!name || isBlockedCommand(command, policy)) {
    return false;
  }

  const allowed = policy.allowed_commands.map((c) => c.trim().toLowerCase()).filter(Boolean);
  if (allowed.length === 0) {
    return false;
  }

  const broad = allowed.filter((c) => !c.includes(" "));
  if (broad.includes(name)) {
    return true;
  }

  const exact = allowed.filter((c) => c.includes(" "));
  const positional = args.map((a) => a.trim().toLowerCase()).filter((a) => !a.startsWith("-"));
  if (!positional.length || exact.length === 0) {
    return false;
  }
  return exact.includes(`${name} ${positional[0]}`);
}

/** Return True when the env key may be set by a server config. */
export function isEnvKeyAllowed(key: string, policy: McpSecurityPolicyConfig): boolean {
  const upper = key.trim().toUpperCase();
  if (!upper) {
    return false;
  }
  if (policy.blocked_env_keys.some((k) => k.trim().toUpperCase() === upper)) {
    return false;
  }
  return policy.allowed_env_keys.some((k) => k.trim().toUpperCase() === upper);
}

/** Return the subset of the provided env keys that would be dropped. */
export function disallowedEnvKeys(keys: string[], policy: McpSecurityPolicyConfig): string[] {
  return keys.filter((k) => k.trim() && !isEnvKeyAllowed(k, policy));
}
