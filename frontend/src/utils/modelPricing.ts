/**
 * The single definition of "effective model pricing".
 *
 * Both consumers of the rule — the admin table's cost-column sorting and the
 * pricing cell's displayed value — must agree on which numbers count as a
 * model's price: every configured provider override plus the model-level
 * default. Keeping the collection logic here once means the sort order can
 * never drift from the price the cell shows for the same model.
 */

/** Every cost dimension carried by a model default or a provider override. */
export type ModelCostKey =
  | "input_cost_per_1m"
  | "output_cost_per_1m"
  | "cached_read_cost_per_1m"
  | "cached_write_cost_per_1m"
  | "audio_input_cost_per_1m"
  | "audio_output_cost_per_1m"
  | "image_input_cost_per_1m"
  | "cost_per_image"
  | "audio_cost_per_minute"
  | "tts_cost_per_1m_chars"
  | "web_search_cost_per_1k";

export const MODEL_COST_KEYS: readonly ModelCostKey[] = [
  "input_cost_per_1m",
  "output_cost_per_1m",
  "cached_read_cost_per_1m",
  "cached_write_cost_per_1m",
  "audio_input_cost_per_1m",
  "audio_output_cost_per_1m",
  "image_input_cost_per_1m",
  "cost_per_image",
  "audio_cost_per_minute",
  "tts_cost_per_1m_chars",
  "web_search_cost_per_1k",
];

/** Structural minimum: anything carrying the cost dimensions (model default or provider mapping). */
type PricedSource = Partial<Record<ModelCostKey, number | null>>;

/** A priced model: cost dimensions of its own plus per-provider overrides. */
export interface PricedModel extends PricedSource {
  providers?: readonly PricedSource[] | null;
}

/** Every configured value for a cost dimension: provider overrides plus the model default. */
export function costValues(model: PricedModel, key: ModelCostKey): number[] {
  const values: number[] = [];
  for (const p of model.providers ?? []) {
    const v = p[key];
    if (v != null) values.push(v);
  }
  const d = model[key];
  if (d != null) values.push(d);
  return values;
}

export interface CostRange {
  min: number;
  max: number;
}

/** The min–max range across every configured value, or null when nothing is priced. */
export function costRange(model: PricedModel, key: ModelCostKey): CostRange | null {
  const values = costValues(model, key);
  if (values.length === 0) return null;
  return { min: Math.min(...values), max: Math.max(...values) };
}

/** Effective price as a scalar: the lowest configured value for the dimension. */
export function effectiveCost(model: PricedModel, key: ModelCostKey): number | null {
  return costRange(model, key)?.min ?? null;
}

/**
 * Comparator over one cost dimension, flipped by direction. Range semantics
 * match what the pricing cell displays: order by range minimum, ties broken
 * by range maximum, then by name. Models without any pricing always sort
 * last, regardless of direction.
 */
export function compareModelsByCost(key: ModelCostKey, dir: 1 | -1) {
  return (a: PricedModel & { name: string }, b: PricedModel & { name: string }): number => {
    const ra = costRange(a, key);
    const rb = costRange(b, key);
    if (ra == null && rb == null) return a.name.localeCompare(b.name);
    if (ra == null) return 1;
    if (rb == null) return -1;
    return (ra.min - rb.min) * dir || (ra.max - rb.max) * dir || a.name.localeCompare(b.name);
  };
}
