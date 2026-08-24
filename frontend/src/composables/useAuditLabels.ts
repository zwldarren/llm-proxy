import { useI18n } from "vue-i18n";

/**
 * Helpers for rendering backend audit enum values as localized, human-readable
 * labels. Falls back to a title-cased version of the raw snake_case value when
 * an explicit i18n entry is missing (e.g. for unknown future enum members).
 */
export function useAuditLabels() {
  const { t, te } = useI18n();

  const titleCase = (value: string): string =>
    value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  /** Resolve an i18n label for an enum group + value, with a title-case fallback. */
  const enumLabel = (
    group: "eventType" | "actionCategory" | "resourceType" | "outcome" | "authMethod",
    value: string | null | undefined
  ): string => {
    if (!value) return "-";
    const key = `logs.audit.${group}_${value}`;
    return te(key) ? t(key) : titleCase(value);
  };

  const formatEventType = (value: string | null | undefined) => enumLabel("eventType", value);
  const formatActionCategory = (value: string | null | undefined) =>
    enumLabel("actionCategory", value);
  const formatResourceType = (value: string | null | undefined) => enumLabel("resourceType", value);
  const formatAuthMethod = (value: string | null | undefined) => enumLabel("authMethod", value);

  const OUTCOME_STATUS_MAP: Record<string, "success" | "warning" | "error" | "unknown"> = {
    success: "success",
    failure: "warning",
    error: "error",
  };

  /**
   * Map a backend `outcome` value to a StatusBadge-compatible status.
   * - success -> "success"
   * - failure -> "warning"
   * - error   -> "error"
   */
  const outcomeStatus = (
    value: string | null | undefined
  ): "success" | "warning" | "error" | "unknown" =>
    value ? (OUTCOME_STATUS_MAP[value] ?? "unknown") : "unknown";

  const formatOutcome = (value: string | null | undefined) => enumLabel("outcome", value);

  return {
    formatEventType,
    formatActionCategory,
    formatResourceType,
    formatAuthMethod,
    formatOutcome,
    outcomeStatus,
  };
}
