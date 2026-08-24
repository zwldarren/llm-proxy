import { http } from "@/services/http";
import type {
  ImageEditRequest,
  ImageGenerationRequest,
  ImageGenerationResponse,
} from "@/types/schemas";

const BASE_URL = "/v1";

/**
 * Image generation is slow (dall-e-3 / gpt-image-1 / Gemini image models
 * routinely take 1-3 minutes). The default 30s HTTP timeout would abort the
 * request while the backend is still generating, so image calls get a
 * generous 5-minute budget.
 */
const IMAGE_TIMEOUT_MS = 5 * 60_000;

export const imagesApi = {
  generateImage: (data: ImageGenerationRequest, apiKey: string) =>
    http.post<ImageGenerationResponse>(`${BASE_URL}/images/generations`, data, {
      headers: { Authorization: `Bearer ${apiKey}` },
      timeoutMs: IMAGE_TIMEOUT_MS,
    }),

  editImage: (data: ImageEditRequest, apiKey: string) =>
    http.post<ImageGenerationResponse>(`${BASE_URL}/images/edits`, data, {
      headers: { Authorization: `Bearer ${apiKey}` },
      timeoutMs: IMAGE_TIMEOUT_MS,
    }),
};
