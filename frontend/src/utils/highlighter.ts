/**
 * Syntax highlighting for chat code blocks.
 *
 * highlight.js owns tokenization and HTML escaping. Only the languages the
 * console actually renders are registered, and the output is styled with the
 * existing `--json-*` design tokens (see `.hljs-*` in `main.css`).
 *
 * Unknown fence languages fall back to escaped plain text instead of
 * auto-detection: an explicit language is cheap and predictable, while
 * auto-detection would re-run every candidate grammar on each streamed chunk.
 */
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import css from "highlight.js/lib/languages/css";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

const LANGUAGES = {
  bash,
  css,
  javascript,
  json,
  markdown,
  python,
  sql,
  typescript,
  xml,
  yaml,
};

for (const [name, definition] of Object.entries(LANGUAGES)) {
  hljs.registerLanguage(name, definition);
}

const HTML_ESCAPE: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

const escapeHtml = (text: string): string =>
  text.replace(/[&<>"']/g, (character) => HTML_ESCAPE[character] ?? character);

export function highlightCode(code: string, lang: string): string {
  const language = (lang || "").toLowerCase().trim();
  if (language && hljs.getLanguage(language)) {
    return hljs.highlight(code, { language, ignoreIllegals: true }).value;
  }
  return escapeHtml(code);
}
