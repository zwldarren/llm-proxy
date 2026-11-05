/**
 * Request bodies for the images playground.
 *
 * The cross-field rules of GPT Image 2.6 live in the backend serializer
 * (`ImageGenerationsSerializer._validate_output_format_constraints`); these
 * builders apply the same rules client-side so the console never sends a body
 * the proxy would reject with a 400.
 */
import type { ImageEditRequest, ImageGenerationRequest } from "@/types/schemas";
import { isEnvelopeOutputFormat, isGifOutputFormat } from "@/utils/imageOutputFormat";

/** The settings sheet's current values, as the view holds them. */
export interface ImageRequestSettings {
  model: string;
  prompt: string;
  n: number;
  /** "auto" or a WIDTHxHEIGHT resolution. */
  size: string;
  quality: string;
  background: string | null;
  moderation: string | null;
  outputFormat: string | null;
  outputCompression: number | null;
  partialImages: number | null;
  frameCount: number | null;
  frameDurationMs: number | null;
}

/**
 * Body for `POST /v1/images/generations`.
 *
 * GIF/SVG output drops compression and partial images, SVG pins `n` to 1, and
 * the frame controls only travel with GIF (the proxy rejects them otherwise).
 */
export function buildImageGenerationBody(settings: ImageRequestSettings): ImageGenerationRequest {
  const envelope = isEnvelopeOutputFormat(settings.outputFormat);
  const gif = isGifOutputFormat(settings.outputFormat);
  return {
    prompt: settings.prompt,
    model: settings.model,
    // SVG is one vector document, so the count is not the user's to choose.
    n: settings.outputFormat === "svg" ? 1 : settings.n,
    size: settings.size !== "auto" ? settings.size : undefined,
    quality: settings.quality,
    background: settings.background || undefined,
    moderation: settings.moderation || undefined,
    output_compression: envelope ? undefined : (settings.outputCompression ?? undefined),
    output_format: settings.outputFormat || undefined,
    frame_count: gif ? (settings.frameCount ?? undefined) : undefined,
    frame_duration_ms: gif ? (settings.frameDurationMs ?? undefined) : undefined,
    partial_images: envelope ? undefined : (settings.partialImages ?? undefined),
    response_format: "url",
  };
}

/**
 * Body for `POST /v1/images/edits`.
 *
 * The edits endpoint is generation-format only (png/jpeg/webp), so an envelope
 * format is never forwarded even if the caller still holds one.
 */
export function buildImageEditBody(
  settings: ImageRequestSettings,
  images: string[],
  mask: string | null
): ImageEditRequest {
  const body: ImageEditRequest = {
    prompt: settings.prompt,
    model: settings.model,
    images: images.map((imageUrl) => ({ image_url: imageUrl })),
    n: settings.n,
    size: settings.size !== "auto" ? settings.size : undefined,
    quality: settings.quality,
    background: settings.background || undefined,
    moderation: settings.moderation || undefined,
    output_compression: settings.outputCompression ?? undefined,
    output_format: isEnvelopeOutputFormat(settings.outputFormat)
      ? undefined
      : settings.outputFormat || undefined,
    partial_images: settings.partialImages ?? undefined,
  };
  // The mask key is omitted rather than nulled so the wire shape matches a
  // request that never set one.
  if (mask) body.mask = { image_url: mask };
  return body;
}
