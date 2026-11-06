<script setup lang="ts">
/**
 * The non-visual half of a usage chart: one row per bucket so the numbers reach
 * a screen reader instead of the canvas's single accessible name. Shared by the
 * charts so the caption/header/scope markup cannot drift between them.
 */
export interface ChartDataTableRow {
  /** Stable bucket key; the label can repeat across days at hourly zoom. */
  key: string;
  label: string;
  values: string[];
}

interface Props {
  /** Mirrors the chart's aria-label. */
  caption: string;
  /** Header of the row-label column (the bucket). */
  rowHeader: string;
  /** One header per value column, in `values` order. */
  columns: string[];
  rows: ChartDataTableRow[];
}

defineProps<Props>();
</script>

<template>
  <!-- `sr-only` sits on a wrapper, not on the table: Firefox ignores
       `width`/`height`/`clip-path` on a table box, so a table carrying the class
       still paints its caption and rows at full size. -->
  <div class="sr-only">
    <table>
      <caption>
        {{
          caption
        }}
      </caption>
      <thead>
        <tr>
          <th scope="col">{{ rowHeader }}</th>
          <th v-for="column in columns" :key="column" scope="col">{{ column }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row.key">
          <th scope="row">{{ row.label }}</th>
          <td v-for="(value, index) in row.values" :key="index">{{ value }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
