/**
 * DOMPurify-based sanitization utilities for safe v-html rendering.
 * Prevents XSS attacks when rendering user-controlled content.
 */
import DOMPurify from "dompurify";

/**
 * Escape HTML special characters to prevent XSS.
 * Use this for plain text that will be inserted into HTML context.
 */
const escapeHtml = (text: string): string => {
  const htmlEscapes: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };
  return text.replace(/[&<>"']/g, (char) => htmlEscapes[char] || char);
};

// LRU cache for highlight results to avoid repeated DOMPurify work
const _highlightCache = new Map<string, string>();
const _HIGHLIGHT_CACHE_MAX = 500;

/**
 * Sanitize text for use in search highlighting.
 * Escapes HTML in both the text and search term, then wraps matches in <mark> tags.
 * This prevents XSS attacks through malicious text or search input.
 *
 * Uses an LRU-style cache to avoid repeated DOMPurify work when the same
 * text+searchTerm pair is used multiple times in a single render cycle.
 *
 * @param text - The text to search within
 * @param searchTerm - The search term to highlight
 * @returns Safe HTML string with highlighted matches
 */
export const sanitizeHighlightText = (
  text: string | number | null | undefined,
  searchTerm: string | undefined
): string => {
  if (!text || !searchTerm || !searchTerm.trim()) {
    return escapeHtml(String(text ?? "-"));
  }

  const textStr = String(text);
  const term = searchTerm.trim();
  // Key on the full (term, text) pair. A delimiter-based key (e.g.
  // term|length|prefix) collides when different texts share a length and
  // prefix, or when a search term contains the delimiter — both serve the
  // wrong cached highlight. JSON.stringify is unambiguous for string pairs.
  const cacheKey = JSON.stringify([term, textStr]);

  const cached = _highlightCache.get(cacheKey);
  if (cached !== undefined) return cached;

  // Escape HTML in both text and search term first
  const escapedText = escapeHtml(textStr);
  const escapedTerm = escapeHtml(term);

  // Escape regex special characters for safe regex construction
  const escapedRegex = escapedTerm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  let result: string;
  try {
    const regex = new RegExp(`(${escapedRegex})`, "gi");
    // Wrap matches in <mark> tags - the content is already escaped
    result = escapedText.replace(
      regex,
      '<mark class="bg-yellow-200 dark:bg-yellow-900/60 px-0.5 rounded">$1</mark>'
    );
  } catch {
    // If regex fails (shouldn't happen after escaping), return escaped text
    result = escapedText;
  }

  // LRU eviction before insert
  if (_highlightCache.size >= _HIGHLIGHT_CACHE_MAX) {
    const firstKey = _highlightCache.keys().next();
    if (!firstKey.done) _highlightCache.delete(firstKey.value);
  }
  _highlightCache.set(cacheKey, result);
  return result;
};

/**
 * Configuration for markdown content sanitization.
 * Allows a broader set of tags for rich markdown rendering.
 */
const MARKDOWN_ALLOWED_TAGS = [
  "p",
  "br",
  "strong",
  "em",
  "code",
  "pre",
  "ul",
  "ol",
  "li",
  "a",
  "blockquote",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "table",
  "thead",
  "tbody",
  "tr",
  "th",
  "td",
  "span",
  "hr",
  "div",
  "button",
  "svg",
  "rect",
  "path",
  "polyline",
  "circle",
  "img",
];

const MARKDOWN_ALLOWED_ATTR = [
  "href",
  "title",
  "class",
  "target",
  "rel",
  "type",
  "aria-label",
  "width",
  "height",
  "viewBox",
  "fill",
  "stroke",
  "stroke-width",
  "stroke-linecap",
  "stroke-linejoin",
  "points",
  "x",
  "y",
  "rx",
  "ry",
  "cx",
  "cy",
  "r",
  "d",
  "src",
  "alt",
  "loading",
];

/**
 * Sanitize markdown-rendered HTML content.
 * Use this for markdown content that has been converted to HTML.
 */
export const sanitizeMarkdownHtml = (html: string): string => {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: MARKDOWN_ALLOWED_TAGS,
    ALLOWED_ATTR: MARKDOWN_ALLOWED_ATTR,
    ALLOW_DATA_ATTR: false,
    // Allow data: URIs for base64 images and standard https:/http: URIs
    ALLOWED_URI_REGEXP:
      /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|sms|cid|xmpp|xxx|data):|[^a-z]|[a-z+.-]+(?:[^a-z+.\-:]|$))/i,
    // Add rel="noopener noreferrer" to all links for security
    ADD_ATTR: ["target", "rel"],
  });
};

// Configure DOMPurify to add security attributes to links
DOMPurify.addHook("uponSanitizeAttribute", (node, data) => {
  if (data.attrName === "target" && node.tagName === "A") {
    // Ensure links opening in new tab have security attributes
    if (data.attrValue === "_blank") {
      node.setAttribute("rel", "noopener noreferrer");
    }
  }
});
