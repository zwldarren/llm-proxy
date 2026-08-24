/**
 * Lightweight client-side syntax highlighter for code snippets.
 * Matches keywords, strings, comments, functions, and numeric/boolean values.
 * Uses HSL colors aligned with the LLM Proxy design system.
 *
 * **Security**: All matched text is HTML-escaped before being wrapped in <span> tags.
 * The output is rendered via v-html, so unescaped user content is an XSS vector.
 */

const HTML_ESCAPE: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};
const HTML_ESCAPE_RE = /[&<>"']/g;

function escapeHtml(text: string): string {
  return text.replace(HTML_ESCAPE_RE, (ch) => HTML_ESCAPE[ch] || ch);
}

export function highlightCode(code: string, lang: string): string {
  const l = (lang || "").toLowerCase().trim();
  if (!l) return code;

  // JSON highlight
  if (l === "json") {
    return code.replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
      (match) => {
        let cls: string;
        if (/^"/.test(match)) {
          if (/:$/.test(match)) {
            // Key
            cls = "text-json-key font-medium";
          } else {
            // String
            cls = "text-json-string";
          }
        } else if (/true|false/.test(match)) {
          // Boolean
          cls = "text-json-boolean font-medium";
        } else if (/null/.test(match)) {
          // Null
          cls = "text-json-null";
        } else if (/-?\d/.test(match)) {
          // Number
          cls = "text-json-number";
        } else {
          cls = "text-muted-foreground";
        }
        return `<span class="${cls}">${escapeHtml(match)}</span>`;
      }
    );
  }

  // Save comments and strings to placeholders so they aren't parsed as keywords
  const placeholders: string[] = [];
  let tempCode = code;

  // 1. Comments
  const commentRegex =
    l === "sql"
      ? /(--.*)/g
      : ["python", "py", "yaml", "yml", "toml", "sh", "bash", "dockerfile", "docker"].includes(l)
        ? /(#.*)/g
        : /(\/\/.*|\/\*[\s\S]*?\*\/)/g;

  tempCode = tempCode.replace(commentRegex, (match) => {
    const id = `___COMMENT_${placeholders.length}___`;
    placeholders.push(
      `<span class="text-muted-foreground italic select-none">${escapeHtml(match)}</span>`
    );
    return id;
  });

  // 2. Strings
  const stringRegex = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)/g;
  tempCode = tempCode.replace(stringRegex, (match) => {
    const id = `___STRING_${placeholders.length}___`;
    placeholders.push(`<span class="text-json-string">${escapeHtml(match)}</span>`);
    return id;
  });

  // 3. Keywords
  let keywordRegex =
    /\b(const|let|var|function|def|class|return|if|else|for|while|import|from|export|as|try|except|catch|finally|async|await|with|yield|lambda|in|is|not|and|or|true|false|null|None|self|this|public|private|protected|interface|implements|extends|new|throw|typeof|instanceof)\b/g;

  if (l === "sql") {
    keywordRegex =
      /\b(select|insert|update|delete|from|where|join|left|right|inner|outer|on|group|by|order|having|limit|offset|and|or|not|null|is|in|into|values|create|table|drop|alter|index|primary|key|foreign|references|varchar|int|boolean|date|timestamp|text)\b/gi;
  } else if (l === "css") {
    keywordRegex = /\b(media|import|charset|keyframes|font-face|important)\b/g;
  } else if (["html", "xml"].includes(l)) {
    keywordRegex =
      /\b(div|span|button|a|p|h[1-6]|ul|ol|li|img|svg|path|rect|polyline|circle|meta|link|script|html|body|head|title)\b/g;
  }

  tempCode = tempCode.replace(keywordRegex, (match) => {
    return `<span class="text-json-keyword font-semibold">${escapeHtml(match)}</span>`;
  });

  // 4. Function calls
  const funcRegex = /\b([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*\()/g;
  tempCode = tempCode.replace(funcRegex, (match) => {
    // Skip if matched word is actually a keyword
    if (match.match(keywordRegex)) return escapeHtml(match);
    return `<span class="text-json-function">${escapeHtml(match)}</span>`;
  });

  // 5. Restore placeholders in reverse order
  for (let i = placeholders.length - 1; i >= 0; i--) {
    tempCode = tempCode.replace(`___STRING_${i}___`, placeholders[i]);
    tempCode = tempCode.replace(`___COMMENT_${i}___`, placeholders[i]);
  }

  return tempCode;
}
