import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { safeParseJson, validateJsonObject } from "@/utils/validation";

/**
 * Composable for handling JSON field input and parsing.
 * Common pattern used in forms where JSON objects (like parameter overrides)
 * are entered as text and need to be validated and parsed.
 *
 * @example
 * const paramOverridesJson = ref("{}");
 * const { parseJsonField } = useJsonField();
 *
 * const saveData = {
 *   ...otherData,
 *   parameter_overrides: parseJsonField(paramOverridesJson.value, {
 *     errorTitle: t("providers.invalidJsonError"),
 *     errorDescription: t("providers.invalidJsonDescription")
 *   })
 * };
 *
 * if (!saveData.parameter_overrides) {
 *   // Parse failed, error toast was shown
 *   return;
 * }
 */
export function useJsonField() {
  const { t } = useI18n();

  /**
   * Parse a JSON string value for form submission.
   * Shows error toast if parsing fails.
   *
   * @param jsonString - The JSON string to parse
   * @param options - Error handling options
   * @returns The parsed object, or null if parsing failed
   */
  const parseJsonField = (
    jsonString: string,
    options?: {
      /** Toast error title template */
      errorTitle?: string;
      /** Toast error description template */
      errorDescription?: string;
    }
  ): Record<string, unknown> | null => {
    // Validate JSON object format
    const validation = validateJsonObject(jsonString);
    if (!validation.valid) {
      toast.error(options?.errorTitle || t("common.error") || "Invalid JSON", {
        description: options?.errorDescription || validation.error,
      });
      return null;
    }

    // Parse safely using the utility function
    return safeParseJson(jsonString, null);
  };

  /**
   * Format an object to a JSON string for display in a textarea.
   * Handles null by returning "{}" empty object representation.
   *
   * @param obj - The object to format
   * @returns JSON string representation
   */
  const formatObjectToJson = (obj: Record<string, unknown> | null | undefined): string => {
    if (!obj) return "{}";
    return JSON.stringify(obj, null, 2);
  };

  /**
   * Initialize a JSON field from an object value.
   *
   * @param obj - Initial object value
   * @returns JSON string representation
   */
  const initJsonField = (obj: Record<string, unknown> | null | undefined): string => {
    return formatObjectToJson(obj);
  };

  return {
    parseJsonField,
    formatObjectToJson,
    initJsonField,
  };
}
