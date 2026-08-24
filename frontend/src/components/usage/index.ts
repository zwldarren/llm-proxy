import { defineAsyncComponent } from "vue";

export const UsageByModel = defineAsyncComponent(() => import("./UsageByModel.vue"));
export const UsageByProvider = defineAsyncComponent(() => import("./UsageByProvider.vue"));
export const UsageTrendsChart = defineAsyncComponent(() => import("./UsageTrendsChart.vue"));
export const UsageMetricStrip = defineAsyncComponent(() => import("./UsageMetricStrip.vue"));
