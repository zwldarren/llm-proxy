/**
 * Image output-format vocabulary shared by the images playground, the log
 * response parser and the request builders.
 *
 * The backend is the authority: `ImageGenerationsSerializer` accepts
 * png/jpeg/webp/gif/svg, `/v1/images/edits` stays on the still-image formats,
 * and gif/svg use a different response envelope (no streaming, no partial
 * images, no compression).
 */

/** True for animated GIF output, the only format with frame controls. */
export function isGifOutputFormat(format?: string | null): boolean {
  return format === "gif";
}

/** True for the formats whose envelope cannot stream, preview, or compress. */
export function isEnvelopeOutputFormat(format?: string | null): boolean {
  return format === "gif" || format === "svg";
}

/** MIME type for a requested (or provider-echoed) output format. */
export function imageMimeType(format?: string | null): string {
  switch (format) {
    case "jpeg":
    case "jpg":
      return "image/jpeg";
    case "webp":
      return "image/webp";
    case "gif":
      return "image/gif";
    // Anthropic image blocks report a media type, which splits as "svg+xml".
    case "svg":
    case "svg+xml":
      return "image/svg+xml";
    default:
      return "image/png";
  }
}

/** Download file extension for an output format; `jpeg` writes `.jpg`. */
export function imageFileExtension(format?: string | null): string {
  switch (format) {
    case "jpeg":
    case "jpg":
      return "jpg";
    case "webp":
      return "webp";
    case "gif":
      return "gif";
    case "svg":
    case "svg+xml":
      return "svg";
    default:
      return "png";
  }
}
