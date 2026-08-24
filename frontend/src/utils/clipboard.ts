/**
 * Copy text to clipboard with fallback support.
 *
 * Uses the modern Clipboard API when available (secure contexts).
 * Falls back to the legacy document.execCommand("copy") method
 * for insecure contexts (HTTP) or when the Clipboard API fails.
 */
export async function copyToClipboard(text: string): Promise<void> {
  // Try modern Clipboard API first
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Clipboard API rejected — fall through to legacy method
    }
  }

  // Fallback: use execCommand with a temporary textarea
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";
  document.body.appendChild(textarea);
  try {
    textarea.select();
    const success = document.execCommand("copy");
    if (!success) {
      throw new Error("execCommand copy returned false");
    }
  } finally {
    document.body.removeChild(textarea);
  }
}
