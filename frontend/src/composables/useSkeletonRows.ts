import { computed } from "vue";

interface SkeletonRow {
  key: number;
  name: string;
  detail: string;
  delay: string;
}

/**
 * Builds skeleton rows with deterministic width classes and a per-row pulse
 * delay, so rows read as real records instead of one stamped mold.
 */
export function useSkeletonRows(rows: number, nameWidths: string[], detailWidths: string[]) {
  return computed<SkeletonRow[]>(() =>
    Array.from({ length: rows }, (_, i) => ({
      key: i,
      name: nameWidths[i % nameWidths.length],
      detail: detailWidths[i % detailWidths.length],
      delay: `${i * 70}ms`,
    }))
  );
}
