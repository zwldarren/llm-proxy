import { describe, expect, it } from "vitest";
import {
  imageFileExtension,
  imageMimeType,
  isEnvelopeOutputFormat,
  isGifOutputFormat,
} from "@/utils/imageOutputFormat";

describe("isGifOutputFormat", () => {
  it("is true only for animated GIF", () => {
    expect(isGifOutputFormat("gif")).toBe(true);
    expect(isGifOutputFormat("svg")).toBe(false);
    expect(isGifOutputFormat("png")).toBe(false);
    expect(isGifOutputFormat(null)).toBe(false);
    expect(isGifOutputFormat(undefined)).toBe(false);
  });
});

describe("isEnvelopeOutputFormat", () => {
  it("covers the formats that cannot stream, preview, or compress", () => {
    expect(isEnvelopeOutputFormat("gif")).toBe(true);
    expect(isEnvelopeOutputFormat("svg")).toBe(true);
    expect(isEnvelopeOutputFormat("png")).toBe(false);
    expect(isEnvelopeOutputFormat("jpeg")).toBe(false);
    expect(isEnvelopeOutputFormat("webp")).toBe(false);
    expect(isEnvelopeOutputFormat(null)).toBe(false);
  });
});

describe("imageMimeType", () => {
  it("maps every output format the proxy accepts", () => {
    expect(imageMimeType("png")).toBe("image/png");
    expect(imageMimeType("jpeg")).toBe("image/jpeg");
    expect(imageMimeType("jpg")).toBe("image/jpeg");
    expect(imageMimeType("webp")).toBe("image/webp");
    expect(imageMimeType("gif")).toBe("image/gif");
    expect(imageMimeType("svg")).toBe("image/svg+xml");
  });

  it("maps a media type's subtype as well as the request value", () => {
    // Anthropic image blocks report media_type, which splits as "svg+xml".
    expect(imageMimeType("svg+xml")).toBe("image/svg+xml");
  });

  it("falls back to PNG for an unknown or missing format", () => {
    expect(imageMimeType(undefined)).toBe("image/png");
    expect(imageMimeType(null)).toBe("image/png");
    expect(imageMimeType("avif")).toBe("image/png");
  });
});

describe("imageFileExtension", () => {
  it("writes .jpg for jpeg and .svg for both SVG spellings", () => {
    expect(imageFileExtension("jpeg")).toBe("jpg");
    expect(imageFileExtension("jpg")).toBe("jpg");
    expect(imageFileExtension("svg")).toBe("svg");
    expect(imageFileExtension("svg+xml")).toBe("svg");
    expect(imageFileExtension("gif")).toBe("gif");
    expect(imageFileExtension("webp")).toBe("webp");
    expect(imageFileExtension(undefined)).toBe("png");
  });
});
