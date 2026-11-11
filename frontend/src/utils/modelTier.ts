const TIER_KEYS = {
  ECONOMY: "models.qualityTierEconomy",
  BALANCED: "models.qualityTierBalanced",
  PREMIUM: "models.qualityTierPremium",
} as const;

/**
 * Localized label for a smart-routing quality tier. Unknown values pass through
 * unchanged so a newer backend tier never renders as an empty badge.
 */
export function modelTierLabel(t: (key: string) => string, tier: string): string {
  const key = TIER_KEYS[tier as keyof typeof TIER_KEYS];
  return key ? t(key) : tier;
}
