/**
 * Parameter overrides types for advanced configuration.
 *
 * Reference: https://www.newapi.ai/zh/docs/guide/console/channel-management
 */

/**
 * Operation modes for parameter overrides.
 */
type OperationMode =
  | "set"
  | "delete"
  | "move"
  | "append"
  | "prepend"
  | "copy"
  | "trim_prefix"
  | "trim_suffix"
  | "ensure_prefix"
  | "ensure_suffix"
  | "trim_space"
  | "to_lower"
  | "to_upper"
  | "replace"
  | "regex_replace";

/**
 * Condition matching modes.
 */
type ConditionMode = "full" | "prefix" | "suffix" | "contains" | "gt" | "gte" | "lt" | "lte";

/**
 * Logic mode for combining conditions.
 */
type LogicMode = "and" | "or";

/**
 * Condition for conditional operation execution.
 */
interface Condition {
  /** Condition matching mode */
  mode: ConditionMode;
  /** Value to compare against */
  value: unknown;
  /** Path to check (if different from operation path) */
  path?: string;
  /** Invert the condition result */
  invert: boolean;
  /** Pass if path doesn't exist */
  pass_missing_key: boolean;
}

/**
 * Single operation in parameter overrides.
 */
interface Operation {
  /** Operation mode */
  mode: OperationMode;
  /** Target path (e.g., 'temperature', 'messages.0.content') */
  path: string;
  /** Value for set/append/prepend/ensure operations */
  value?: unknown;
  /** Source path for move/copy operations */
  from_path?: string;
  /** Pattern for replace/regex_replace operations */
  pattern?: string;
  /** Replacement string for replace/regex_replace */
  replacement?: string;
  /** For set: skip if key exists; for append/prepend on objects: merge */
  keep_origin: boolean;
  /** Conditions that must be met to execute this operation */
  conditions: Condition[];
  /** Logic for combining conditions */
  logic: LogicMode;
}

/**
 * Configuration for parameter overrides.
 *
 * Supports two modes:
 * - Simple mode: Direct key-value pairs (e.g., { "temperature": 0.7 })
 * - Operations mode: { "operations": [...] }
 */
export interface ParameterOverridesConfig {
  /** List of operations (operations mode) */
  operations?: Operation[];
  /** Direct key-value pairs (simple mode) - any other keys */
  [key: string]: unknown;
}
