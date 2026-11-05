import { describe, expect, it } from "vitest";
import { buildImageEditBody, buildImageGenerationBody } from "@/utils/imageRequest";
import type { ImageRequestSettings } from "@/utils/imageRequest";

/** Every setting at its default, so a test only states what it changes. */
function settings(overrides: Partial<ImageRequestSettings> = {}): ImageRequestSettings {
  return {
    model: "gpt-image-2.6-flare",
    prompt: "a looping mascot",
    n: 3,
    size: "1024x1024",
    quality: "high",
    background: "transparent",
    moderation: "auto",
    outputFormat: null,
    outputCompression: 80,
    partialImages: 2,
    frameCount: null,
    frameDurationMs: null,
    ...overrides,
  };
}

/**
 * The body as the proxy receives it. The builders leave absent fields as
 * `undefined`, which JSON drops — assert on the wire shape, not the object.
 */
function wire(body: object): Record<string, unknown> {
  return JSON.parse(JSON.stringify(body));
}

describe("buildImageGenerationBody", () => {
  it("keeps the still-image envelope for png/jpeg/webp", () => {
    const body = wire(buildImageGenerationBody(settings({ outputFormat: "webp" })));

    expect(body.n).toBe(3);
    expect(body.output_format).toBe("webp");
    expect(body.output_compression).toBe(80);
    expect(body.partial_images).toBe(2);
    expect(body).not.toHaveProperty("frame_count");
    expect(body).not.toHaveProperty("frame_duration_ms");
  });

  it("sends the frame controls with animated GIF and drops the envelope fields", () => {
    const body = wire(
      buildImageGenerationBody(
        settings({ outputFormat: "gif", frameCount: 16, frameDurationMs: 120 })
      )
    );

    expect(body.output_format).toBe("gif");
    expect(body.frame_count).toBe(16);
    expect(body.frame_duration_ms).toBe(120);
    // GIF cannot stream, so compression and partial images never travel.
    expect(body).not.toHaveProperty("output_compression");
    expect(body).not.toHaveProperty("partial_images");
    expect(body.n).toBe(3);
  });

  it("pins the count to 1 for SVG and never sends frame controls", () => {
    const body = wire(
      buildImageGenerationBody(
        settings({ outputFormat: "svg", n: 4, frameCount: 9, frameDurationMs: 200 })
      )
    );

    expect(body.output_format).toBe("svg");
    expect(body.n).toBe(1);
    expect(body).not.toHaveProperty("output_compression");
    expect(body).not.toHaveProperty("partial_images");
    expect(body).not.toHaveProperty("frame_count");
    expect(body).not.toHaveProperty("frame_duration_ms");
  });

  it("omits the format when none is chosen and keeps the still-image envelope", () => {
    const body = wire(buildImageGenerationBody(settings()));

    expect(body).not.toHaveProperty("output_format");
    expect(body.output_compression).toBe(80);
    expect(body.partial_images).toBe(2);
  });

  it("maps size auto and unset optional fields to undefined", () => {
    const body = wire(
      buildImageGenerationBody(
        settings({ size: "auto", background: null, moderation: null, outputCompression: null })
      )
    );

    expect(body).not.toHaveProperty("size");
    expect(body).not.toHaveProperty("background");
    expect(body).not.toHaveProperty("moderation");
    expect(body).not.toHaveProperty("output_compression");
    expect(body.response_format).toBe("url");
  });
});

describe("buildImageEditBody", () => {
  it("maps the uploaded images and the mask", () => {
    const body = buildImageEditBody(
      settings({ outputFormat: "png" }),
      ["data:a", "data:b"],
      "data:m"
    );

    expect(body.images).toEqual([{ image_url: "data:a" }, { image_url: "data:b" }]);
    expect(body.mask).toEqual({ image_url: "data:m" });
    expect(body.output_format).toBe("png");
    expect(body.n).toBe(3);
  });

  it("omits the mask key when no mask was uploaded", () => {
    expect(wire(buildImageEditBody(settings(), ["data:a"], null))).not.toHaveProperty("mask");
  });

  it("never forwards a gif/svg format the endpoint rejects", () => {
    for (const outputFormat of ["gif", "svg"]) {
      const body = wire(buildImageEditBody(settings({ outputFormat }), ["data:a"], null));
      expect(body).not.toHaveProperty("output_format");
      expect(body).not.toHaveProperty("frame_count");
      expect(body).not.toHaveProperty("frame_duration_ms");
    }
  });
});
