import { z } from "zod";

const urlRefine = (v: string | null | undefined) => !v || /^https?:\/\/.*/.test(v);
const urlError = "Must be a valid URL starting with http:// or https://";

export const providerFormSchema = z.object({
  name: z.string().min(1, "Name is required").max(255),
  type: z.string().min(1, "Type is required"),
  api_key: z.string().optional().default(""),
  base_url: z.string().optional().default("").refine(urlRefine, urlError),
  icon_url: z.string().optional().nullable().refine(urlRefine, urlError),
  native_web_search: z.boolean().optional().default(false),
});

export type ProviderFormValues = z.infer<typeof providerFormSchema>;
