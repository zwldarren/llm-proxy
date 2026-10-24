/**
 * URL scheme guard for image URLs that reach navigation sinks.
 *
 * Provider responses and stored log rows are untrusted: an image `url` field
 * can be `javascript:...`, which `window.open()` / an anchor click would
 * execute in the dashboard origin. Only image-bearing or navigable schemes
 * are returned; everything else is dropped.
 */
export function safeImageUrl(url: string | null | undefined): string | undefined {
  if (!url) return undefined;
  try {
    const parsed = new URL(url, window.location.origin);
    if (
      parsed.protocol === "http:" ||
      parsed.protocol === "https:" ||
      parsed.protocol === "blob:"
    ) {
      return url;
    }
    if (parsed.protocol === "data:" && /^data:image\/(png|jpe?g|webp|gif|avif);/i.test(url)) {
      return url;
    }
  } catch {
    // Relative strings that cannot be parsed are not usable as navigation.
  }
  return undefined;
}
