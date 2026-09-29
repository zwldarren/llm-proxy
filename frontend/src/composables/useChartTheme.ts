import type { TooltipItem } from "chart.js";
import { computed } from "vue";
import { useTheme } from "@/composables/useTheme";

/**
 * Read a `main.css` custom property. Tokens are stored as bare HSL channels,
 * so `--border` resolves to e.g. `220 10% 22%`; the font stacks resolve to a
 * full family list. Canvas can't read CSS variables at all, but resolving them
 * here keeps every chart on the same single source of truth as the rest of the
 * UI instead of re-encoding the palette as literals.
 */
const cssValue = (token: string): string =>
  getComputedStyle(document.documentElement).getPropertyValue(token).trim();

/** Resolve a color token into a chart-ready value: `hsl(220 10% 22% / 0.3)`. */
export const cssColor = (token: string, alpha?: number): string => {
  const channels = cssValue(token);
  return alpha === undefined ? `hsl(${channels})` : `hsl(${channels} / ${alpha})`;
};

let monoFamily: string | undefined;

/**
 * The mono stack, for Chart.js `font.family` — numbers are always mono.
 *
 * Resolved once: the stack cannot change at runtime, and `getComputedStyle`
 * forces a style flush on every call.
 */
export const chartMonoFamily = (): string => {
  monoFamily ??= cssValue("--font-mono").replace(/\s+/g, " ") || "ui-monospace, monospace";
  return monoFamily;
};

export interface ChartTheme {
  tooltipBg: string;
  tooltipTitle: string;
  tooltipBody: string;
  tooltipBorder: string;
  gridColor: string;
  tickColor: string;
  actionBlue: string;
  actionViolet: string;
  actionAmber: string;
}

/**
 * Theme-reactive colors shared by every Chart.js surface on the dashboard.
 *
 * The palette hangs off `useTheme`'s reactive dark flag, so a theme toggle
 * repaints the charts without a reload.
 */
export function useChartTheme() {
  const { isDark } = useTheme();

  const themeColors = computed<ChartTheme>(() => {
    // `isDark` is the reactive dependency: the color-mode class is already on
    // <html> by the time it flips, so the tokens resolve to the new palette.
    const isDarkMode = isDark.value;
    return {
      tooltipBg: cssColor("--popover", 0.95),
      tooltipTitle: cssColor("--popover-foreground"),
      tooltipBody: cssColor("--muted-foreground"),
      tooltipBorder: cssColor("--border"),
      // The grid is a whisper on dark surfaces and more present on light ones.
      gridColor: cssColor("--border", isDarkMode ? 0.3 : 0.8),
      tickColor: cssColor("--muted-foreground"),
      actionBlue: cssColor("--action-blue"),
      actionViolet: cssColor("--action-violet"),
      actionAmber: cssColor("--action-amber"),
    };
  });

  return { isDark, themeColors };
}

/** The per-chart decisions the shared chrome cannot guess. */
interface BarChartOverrides {
  /** Formatter for y-axis tick values. */
  yTickLabel: (value: number) => string;
  /** Decimal places on the y axis; omit for the Chart.js default. */
  yPrecision?: number;
  /** Cap on how many x-axis ticks are drawn. */
  xMaxTicksLimit: number;
  /** Hover mode: `nearest` for dense multi-series, `index` for stacked bars. */
  interactionMode: "index" | "nearest";
  tooltip?: {
    title?: (items: TooltipItem<"bar">[]) => string;
    label?: (context: TooltipItem<"bar">) => string;
  };
}

/**
 * The option chrome every stacked bar chart on the dashboard shares: legend,
 * tooltip surface, and mono-styled axes. Callers supply only their own tick
 * formatting and tooltip callbacks, so the chrome stays consistent by
 * construction instead of by two files being edited in step.
 */
export function baseBarChartOptions(themeColors: ChartTheme, overrides: BarChartOverrides) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: "bottom" as const,
        labels: {
          boxWidth: 12,
          padding: 15,
          // Chart.js defaults to a sans stack; numbers and labels stay mono.
          font: { size: 11, family: chartMonoFamily() },
          color: themeColors.tickColor,
        },
      },
      tooltip: {
        mode: "index" as const,
        intersect: false,
        backgroundColor: themeColors.tooltipBg,
        titleColor: themeColors.tooltipTitle,
        bodyColor: themeColors.tooltipBody,
        borderColor: themeColors.tooltipBorder,
        borderWidth: 1,
        padding: 12,
        cornerRadius: 8,
        displayColors: true,
        boxPadding: 4,
        callbacks: {
          title: overrides.tooltip?.title,
          label: overrides.tooltip?.label,
        },
      },
    },
    scales: {
      x: {
        stacked: true,
        // Sparse ranges (a single day at day granularity) would otherwise
        // render one chart-width pillar; the cap keeps bars readable at any
        // bucket density without touching the tight multi-bucket spacing.
        maxBarThickness: 48,
        grid: { display: false },
        ticks: {
          color: themeColors.tickColor,
          font: { size: 10, family: chartMonoFamily() },
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: overrides.xMaxTicksLimit,
        },
      },
      y: {
        stacked: true,
        grid: {
          color: themeColors.gridColor,
          drawBorder: false,
        },
        ticks: {
          color: themeColors.tickColor,
          font: { size: 10, family: chartMonoFamily() },
          precision: overrides.yPrecision,
          callback: (value: string | number) => {
            const numValue = typeof value === "number" ? value : Number.parseFloat(value);
            return Number.isNaN(numValue) ? "" : overrides.yTickLabel(numValue);
          },
        },
      },
    },
    interaction: {
      mode: overrides.interactionMode,
      axis: "x" as const,
      intersect: false,
    },
  };
}
