// frontend/src/utils/icons.test.ts
import { describe, expect, it } from "vitest";
import {
  LOBE_COLOR_BRANDS,
  LOBE_ICONS_VERSION,
  LOBE_MONO_BRANDS,
} from "@/utils/lobeBrands.generated";
import { brandIcon, modelIcon, providerIcon, monogramInitial, resolveBrand } from "@/utils/icons";

const COLOR_BRAND = "deepseek";
const MONO_ONLY_BRAND = [...LOBE_MONO_BRANDS].find(
  (brand) => !LOBE_COLOR_BRANDS.includes(brand)
) as string;

describe("resolveBrand", () => {
  it.each([
    ["gpt-4o", "openai"],
    ["gpt4o", "openai"],
    ["gpt4o-mini", "openai"],
    ["o3", "openai"],
    ["claude-sonnet-4-5", "claude"],
    ["gemini-2.5-pro", "gemini"],
    ["qwen3-235b-a22b", "qwen"],
    ["glm-4.6", "zai"],
    ["deepseek-v3.2", "deepseek"],
    ["kimi-k2-thinking", "kimi"],
    ["MiniMax-M2", "minimax"],
    ["llama-3.3-70b-versatile", "meta"],
    ["phi-4-mini", "microsoft"],
    ["ernie-4.5-21b", "baidu"],
    ["nemotron-3-super-120b", "nvidia"],
    ["command-a", "commanda"],
    ["sonar-pro", "perplexity"],
    ["hermes-4-405b", "nousresearch"],
    ["mimo-v2-pro", "xiaomimimo"],
    ["devstral-2512", "mistral"],
    ["step-3", "stepfun"],
    ["hy-mt2-30b", "hunyuan"],
    ["amazon/nova-pro-v1", "nova"],
  ])("maps %s to %s", (modelId, brand) => {
    expect(resolveBrand(modelId)).toBe(brand);
  });

  it.each([
    // Rightmost segment first: the model family's own mark beats the
    // platform or company that publishes it.
    ["x-ai/grok-4", "grok"],
    ["meta-llama/Llama-4-Maverick", "meta"],
    ["mistralai/codestral-2508", "mistral"],
    ["moonshotai/Kimi-K2.6", "kimi"],
    ["together_ai/thinkingmachines/Inkling", "together"],
    ["@cf/cloudflare/clairic-flash", "cloudflare"],
    ["dots-studio/dots-3-note-preview:free", "dotsstudio"],
  ])("resolves the brand of %s to %s", (modelId, brand) => {
    expect(resolveBrand(modelId)).toBe(brand);
  });

  it("prefers the model's own brand over the platform reselling it", () => {
    expect(resolveBrand("deepinfra/bytedance/seed-2.0-code")).toBe("bytedance");
    expect(resolveBrand("openrouter/anthropic/claude-3.5-sonnet")).toBe("claude");
    expect(resolveBrand("aws-bedrock/claude-fable-5")).toBe("claude");
  });

  it.each([
    // The provider types the backend currently registers that the catalog
    // covers on its own (adapters that declare `LOBE_ICON_ID` are branded
    // either way; these are the ones the type name has to answer for).
    ["anthropic", "anthropic"],
    ["chutes", "chutes"],
    ["cohere", "cohere"],
    ["deepseek", "deepseek"],
    ["gemini", "gemini"],
    ["kimi-code", "kimi"],
    ["minimax", "minimax"],
    ["mistral", "mistral"],
    ["moonshot", "moonshot"],
    ["ollama", "ollama"],
    ["openai", "openai"],
    ["openai-compatible", "openai"],
    ["openrouter", "openrouter"],
    ["qwen-intl", "qwen"],
    ["vllm", "vllm"],
    ["xai", "xai"],
    ["zai-coding", "zai"],
    ["zhipu-coding", "zhipu"],
  ])("resolves provider type %s to %s", (providerType, brand) => {
    expect(resolveBrand(providerType)).toBe(brand);
  });

  it("leaves the provider types lobe has no mark for unmatched", () => {
    // These render as a monogram tile — a deliberate mark, not a broken one.
    expect(resolveBrand("sglang")).toBeNull();
    expect(resolveBrand("nanogpt")).toBeNull();
  });

  it("completes a brand from a truncated type name", () => {
    // `typesafe` is api.typesafe.ai; lobe ships the mark as `typesafeai`.
    expect(resolveBrand("typesafe")).toBe("typesafeai");
  });

  it.each([
    // Words that only look like brands.
    ["auto", null],
    ["agent-max", null],
    ["chat-model", null],
    ["mixtral-8x7b-instruct-v0.1", null],
    ["fast-model-1", null],
    ["Steelskull/L3.3-Electra-R1-70b", null],
    ["", null],
    ["   ", null],
    [null, null],
    [undefined, null],
  ])("resolves %s to %s", (identifier, brand) => {
    expect(resolveBrand(identifier as string)).toBe(brand);
  });

  it("is case-insensitive", () => {
    expect(resolveBrand("Claude-Opus-4")).toBe("claude");
    expect(resolveBrand("GEMMA-4-31B-IT")).toBe("gemma");
  });
});

describe("brandIcon", () => {
  it("prefers the fixed-palette mark when the brand ships one", () => {
    expect(brandIcon(COLOR_BRAND)).toEqual({
      url: `https://unpkg.com/@lobehub/icons-static-svg@${LOBE_ICONS_VERSION}/icons/${COLOR_BRAND}-color.svg`,
      mono: false,
    });
  });

  it("falls back to the mono mark for brands without a colour variant", () => {
    const icon = brandIcon(MONO_ONLY_BRAND);
    expect(icon?.mono).toBe(true);
    expect(icon?.url).toContain(`/${MONO_ONLY_BRAND}.svg`);
  });

  it("honours a forced variant and falls back when it is missing", () => {
    expect(brandIcon(COLOR_BRAND, "mono")).toEqual({
      url: expect.stringContaining(`/${COLOR_BRAND}.svg`),
      mono: true,
    });
    expect(brandIcon(MONO_ONLY_BRAND, "color")?.mono).toBe(true);
  });

  it("rejects brands outside the catalog", () => {
    expect(brandIcon("not-a-real-brand")).toBeNull();
    expect(brandIcon("")).toBeNull();
    expect(brandIcon(null)).toBeNull();
  });
});

describe("modelIcon", () => {
  it("uses the model's brand", () => {
    expect(modelIcon({ name: "qwen3-max" })?.mono).toBe(false);
  });

  it("falls back to the provider when the name carries no brand", () => {
    expect(modelIcon({ name: "fast-model-1", provider: "openrouter" })?.url).toContain(
      "/openrouter"
    );
  });

  it("prefers the model brand over the provider", () => {
    const icon = modelIcon({ name: "claude-sonnet-4-5", provider: "openrouter" });
    expect(icon?.url).toContain("/claude");
  });

  it("uses a configured icon_url verbatim", () => {
    expect(modelIcon({ name: "claude-opus-4", iconUrl: "https://example.com/x.svg" })).toEqual({
      url: "https://example.com/x.svg",
      mono: false,
    });
  });
});

describe("providerIcon", () => {
  it("uses the branding the provider type declares", () => {
    const icon = providerIcon({
      type: "openai",
      brand: { icon_id: "openai", icon_variant: "mono" },
    });
    expect(icon?.url).toContain("/openai.svg");
    expect(icon?.mono).toBe(true);
  });

  it("matches the type against the catalog when nothing is declared", () => {
    expect(providerIcon({ type: "sglang" })).toBeNull();
    expect(providerIcon({ type: "chutes" })?.url).toContain("/chutes");
  });

  it("matches the configured name when the type is unknown", () => {
    expect(providerIcon({ type: "openai-compatible", name: "Together" })?.url).toContain("/openai");
    expect(providerIcon({ type: "custom", name: "chutes" })?.url).toContain("/chutes");
  });

  it("uses a configured icon_url verbatim, never tinted", () => {
    expect(
      providerIcon({
        type: "openai",
        iconUrl: "https://example.com/logo.svg",
        brand: { icon_id: "openai", icon_variant: "mono" },
      })
    ).toEqual({ url: "https://example.com/logo.svg", mono: false });
  });
});

describe("monogramInitial", () => {
  it.each([
    ["deepseek", "D"],
    ["qwen3-max", "Q"],
    ["通义千问", "通"],
    ["  spaced", "S"],
    ["!@# weird", "W"],
    ["42", "4"],
    ["", "?"],
    [null, "?"],
  ])("takes the first letter of %s", (label, initial) => {
    expect(monogramInitial(label)).toBe(initial);
  });
});

describe("generated catalog", () => {
  it("only lists colour brands that also have a mono mark or a colour file", () => {
    expect(LOBE_MONO_BRANDS.length).toBeGreaterThan(100);
    expect(new Set(LOBE_MONO_BRANDS).size).toBe(LOBE_MONO_BRANDS.length);
    expect(new Set(LOBE_COLOR_BRANDS).size).toBe(LOBE_COLOR_BRANDS.length);
    expect(LOBE_ICONS_VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });

  it("resolves every catalog brand to a mark", () => {
    for (const brand of [...LOBE_MONO_BRANDS, ...LOBE_COLOR_BRANDS]) {
      expect(brandIcon(brand), brand).not.toBeNull();
    }
  });
});
