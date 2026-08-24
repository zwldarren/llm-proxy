// frontend/src/utils/colorPalette.test.ts
import { describe, expect, it } from "vitest";
import { createCategoricalColorScale } from "@/utils/colorPalette";

// Usage-ranked, as the chart passes them (allModels order).
const MODELS = [
  "claude-sonnet-4-5",
  "gpt-4o",
  "gemini-2.5-pro",
  "claude-opus-4-5",
  "deepseek-v3",
  "gpt-4o-mini",
  "llama-3.3-70b",
  "qwen-2.5-coder",
  "mistral-large",
  "grok-3",
];

const HSL_RE = /^hsl\((\d+) (\d+)% (\d+)%\)$/;

const parseHsl = (color: string): [number, number, number] => {
  const match = HSL_RE.exec(color);
  if (!match) throw new Error(`Not an hsl() color: ${color}`);
  return [Number(match[1]), Number(match[2]), Number(match[3])];
};

const minPairwiseHueGap = (hues: number[]): number => {
  const sorted = [...hues].sort((a, b) => a - b);
  let min = Infinity;
  for (let i = 0; i < sorted.length; i++) {
    const next = sorted[(i + 1) % sorted.length]!;
    const gap = (next - sorted[i]! + 360) % 360;
    if (gap > 0) min = Math.min(min, gap);
  }
  return min;
};

describe("createCategoricalColorScale", () => {
  it("assigns every category a valid muted hsl color inside the allowed hue arc", () => {
    const scale = createCategoricalColorScale(MODELS);
    for (const name of MODELS) {
      const [h, s, l] = parseHsl(scale(name, true));
      // Hue arc excludes status red/orange (0–55°).
      expect(h).toBeGreaterThanOrEqual(55);
      expect(h).toBeLessThan(350);
      // Muted Monochrome Editorial chroma.
      expect(s).toBeGreaterThanOrEqual(20);
      expect(s).toBeLessThanOrEqual(45);
      expect(l).toBeGreaterThan(0);
      expect(l).toBeLessThan(100);
    }
  });

  it("gives the top 10 ranks pairwise-distinct colors (the reported bug)", () => {
    const scale = createCategoricalColorScale(MODELS);
    const dark = MODELS.map((m) => scale(m, true));
    const light = MODELS.map((m) => scale(m, false));
    expect(new Set(dark).size).toBe(MODELS.length);
    expect(new Set(light).size).toBe(MODELS.length);
  });

  it("maximizes hue separation for the top 10 (even 29.5° slot spacing)", () => {
    const scale = createCategoricalColorScale(MODELS);
    const hues = MODELS.map((m) => parseHsl(scale(m, true))[0]);
    // Ten evenly spaced slots on the 295° arc — allow 2° for rounding.
    expect(minPairwiseHueGap(hues)).toBeGreaterThanOrEqual(27);
  });

  it("spreads any prefix (Top 5) across the whole arc instead of clustering", () => {
    const scale = createCategoricalColorScale(MODELS);
    const top5 = MODELS.slice(0, 5).map((m) => parseHsl(scale(m, true))[0]);
    const span = Math.max(...top5) - Math.min(...top5);
    expect(span).toBeGreaterThanOrEqual(200);
  });

  it("alternates tone so hue-adjacent slots also differ in lightness/saturation", () => {
    const scale = createCategoricalColorScale(MODELS);
    const parsed = MODELS.map((m) => parseHsl(scale(m, true)));
    // Adjacent slots are 29.5° apart; any pair closer than 35° must differ
    // in BOTH saturation and lightness.
    for (let i = 0; i < parsed.length; i++) {
      for (let j = i + 1; j < parsed.length; j++) {
        const [hi, si, li] = parsed[i]!;
        const [hj, sj, lj] = parsed[j]!;
        const hueGap = Math.min(Math.abs(hi - hj), 360 - Math.abs(hi - hj));
        if (hueGap < 35) {
          expect(si).not.toBe(sj);
          expect(li).not.toBe(lj);
        }
      }
    }
  });

  it("is stable across repeated builds with the same ranking", () => {
    const a = createCategoricalColorScale(MODELS);
    const b = createCategoricalColorScale([...MODELS]);
    for (const name of MODELS) {
      expect(a(name, true)).toBe(b(name, true));
    }
  });

  it("keeps ranks beyond 10 distinct from their prime-slot neighbors", () => {
    const many = [...MODELS, ...MODELS.map((m) => `${m}-alt`)];
    const scale = createCategoricalColorScale(many);
    const colors = many.map((m) => scale(m, true));
    expect(new Set(colors).size).toBe(many.length);
  });

  it("returns a different theme variant for dark vs light mode", () => {
    const scale = createCategoricalColorScale(MODELS);
    for (const name of MODELS) {
      expect(scale(name, true)).not.toBe(scale(name, false));
    }
  });

  it("returns the neutral-slate fallback for names outside the set", () => {
    const scale = createCategoricalColorScale(MODELS);
    expect(scale("unknown-model", true)).toBe("hsl(220 5% 50%)");
  });
});
