import type { ModelCatalogEntry } from "@/types/schemas";
import { http } from "../http";

const BASE_URL = "/api/catalog";

export const catalogApi = {
  /** Fetch the public model catalog (display-oriented). Available to any authenticated user. */
  getModels: () => http.get<ModelCatalogEntry[]>(`${BASE_URL}/models`),
};
