import { computed, ref, watch } from "vue";
import type { ParameterOverridesConfig } from "@/types/parameterOverrides";
import type {
  ModelCapability,
  ModelCreate,
  ModelProviderMapping,
  ModelRead,
  ModelStatus,
} from "@/types/schemas";

/**
 * Form state for the model create/edit sheet: the draft model, capability
 * mapping, null-proxied optional fields, the nested provider-edit sheet, and
 * validation/payload building. Pure reactivity — no store, toast, or i18n —
 * so the whole form is unit-testable without mounting. The view owns dialog
 * visibility and the save/delete orchestration on top of this state.
 */

/** Fresh draft for the model sheet; openForCreate/openForEdit reseed it. */
export function emptyModelForm(): ModelCreate {
  return {
    name: "",
    providers: [],
    input_cost_per_1m: null,
    output_cost_per_1m: null,
    cached_read_cost_per_1m: null,
    cached_write_cost_per_1m: null,
    audio_input_cost_per_1m: null,
    audio_output_cost_per_1m: null,
    image_input_cost_per_1m: null,
    cost_per_image: null,
    audio_cost_per_minute: null,
    tts_cost_per_1m_chars: null,
    web_search_cost_per_1k: null,
    icon_url: null,
    parameter_overrides: null,
    auto_eligible: false,
    quality_tier: null,
    routing_assignments: null,
    supports_images: false,
    supports_image_generation: false,
    supports_tts: false,
    supports_stt: false,
    supports_embedding: false,
    supports_realtime: false,
    attachment: false,
    reasoning: false,
    tool_call: false,
    structured_output: false,
    temperature: false,
    experimental: false,
    open_weights: false,
    status: null,
    family: null,
    knowledge: null,
    release_date: null,
    max_output_tokens: null,
    description: null,
    homepage_url: null,
    context_length: null,
  };
}

/** Blank provider mapping row for the draft's providers list. */
function emptyProviderMapping(): ModelProviderMapping {
  return {
    provider_name: "",
    priority: 0,
    provider_model_name: "",
    input_cost_per_1m: null,
    output_cost_per_1m: null,
    cached_read_cost_per_1m: null,
    cached_write_cost_per_1m: null,
    audio_input_cost_per_1m: null,
    audio_output_cost_per_1m: null,
    image_input_cost_per_1m: null,
    cost_per_image: null,
    audio_cost_per_minute: null,
    tts_cost_per_1m_chars: null,
    web_search_cost_per_1k: null,
    parameter_overrides: {},
  };
}

/**
 * Backing boolean field per capability. Proxy-bound capabilities map to the
 * supports_* columns; informational capabilities share the models.dev name.
 */
const CAPABILITY_FIELD: Record<ModelCapability, string> = {
  vision: "supports_images",
  image_generation: "supports_image_generation",
  tts: "supports_tts",
  stt: "supports_stt",
  embedding: "supports_embedding",
  realtime: "supports_realtime",
  attachment: "attachment",
  reasoning: "reasoning",
  tool_call: "tool_call",
  structured_output: "structured_output",
  temperature: "temperature",
  open_weights: "open_weights",
  experimental: "experimental",
};

export function useModelForm() {
  const form = ref<ModelCreate>(emptyModelForm());
  const parameterOverrides = ref<ParameterOverridesConfig>({});
  const iconPreviewFailed = ref(false);

  function getCapability(cap: ModelCapability): boolean {
    return Boolean(form.value[CAPABILITY_FIELD[cap] as keyof ModelCreate]);
  }

  function setCapability(cap: ModelCapability, value: boolean) {
    (form.value as Record<string, unknown>)[CAPABILITY_FIELD[cap]] = value;
  }

  // ---- Nested provider-edit sheet ----

  const showProviderEditDialog = ref(false);
  const editingProviderIndex = ref<number | null>(null);
  const editingProviderData = ref<ModelProviderMapping>({
    provider_name: "",
    priority: 0,
    provider_model_name: "",
  });

  const addProvider = () => {
    form.value.providers.push(emptyProviderMapping());
  };

  const removeProvider = (index: number) => {
    form.value.providers.splice(index, 1);
  };

  const openProviderEditDialog = (index: number) => {
    editingProviderIndex.value = index;
    editingProviderData.value = { ...form.value.providers[index]! };
    showProviderEditDialog.value = true;
  };

  const saveProviderEdit = () => {
    if (editingProviderIndex.value !== null) {
      form.value.providers[editingProviderIndex.value] = { ...editingProviderData.value };
    }
    showProviderEditDialog.value = false;
    editingProviderIndex.value = null;
  };

  // ---- Cross-field rules ----

  // Routing profile is meaningless without auto-eligibility.
  watch(
    () => form.value.auto_eligible,
    (val) => {
      if (!val) {
        form.value.quality_tier = null;
        form.value.routing_assignments = null;
      }
    }
  );

  watch([() => form.value.icon_url, () => form.value.name], () => {
    iconPreviewFailed.value = false;
  });

  // ---- Null proxies for inputs that bind non-null strings ----

  // Textarea binds a non-null string; proxy null <-> empty for the description field.
  const descriptionModel = computed({
    get: () => form.value.description ?? "",
    set: (v: string) => {
      form.value.description = v || null;
    },
  });

  // Select uses a "none" sentinel; null otherwise.
  const statusModel = computed({
    get: () => form.value.status ?? "none",
    set: (v: string) => {
      form.value.status = v === "none" ? null : (v as ModelStatus);
    },
  });

  // Text inputs bind non-null strings; proxy null <-> empty for optional fields.
  const familyModel = computed({
    get: () => form.value.family ?? "",
    set: (v: string) => {
      form.value.family = v.trim() || null;
    },
  });
  const knowledgeModel = computed({
    get: () => form.value.knowledge ?? "",
    set: (v: string) => {
      form.value.knowledge = v || null;
    },
  });
  const releaseDateModel = computed({
    get: () => form.value.release_date ?? "",
    set: (v: string) => {
      form.value.release_date = v || null;
    },
  });

  // ---- Seeding ----

  /** Reset to a blank create draft with one empty provider row. */
  const openForCreate = () => {
    iconPreviewFailed.value = false;
    form.value = {
      ...emptyModelForm(),
      providers: [emptyProviderMapping()],
    };
    parameterOverrides.value = {};
  };

  /** Load an existing model into the draft for editing. */
  const openForEdit = (model: ModelRead) => {
    iconPreviewFailed.value = false;
    form.value = {
      ...emptyModelForm(),
      name: model.name,
      providers: model.providers?.length
        ? [...model.providers]
            .sort((a, b) => (b.priority || 0) - (a.priority || 0))
            .map((p) => ({
              provider_name: p.provider_name,
              priority: p.priority || 0,
              provider_model_name: p.provider_model_name,
              input_cost_per_1m: p.input_cost_per_1m ?? null,
              output_cost_per_1m: p.output_cost_per_1m ?? null,
              cached_read_cost_per_1m: p.cached_read_cost_per_1m ?? null,
              cached_write_cost_per_1m: p.cached_write_cost_per_1m ?? null,
              audio_input_cost_per_1m: p.audio_input_cost_per_1m ?? null,
              audio_output_cost_per_1m: p.audio_output_cost_per_1m ?? null,
              image_input_cost_per_1m: p.image_input_cost_per_1m ?? null,
              cost_per_image: p.cost_per_image ?? null,
              audio_cost_per_minute: p.audio_cost_per_minute ?? null,
              tts_cost_per_1m_chars: p.tts_cost_per_1m_chars ?? null,
              web_search_cost_per_1k: p.web_search_cost_per_1k ?? null,
              parameter_overrides: p.parameter_overrides ?? {},
            }))
        : [],
      input_cost_per_1m: model.input_cost_per_1m,
      output_cost_per_1m: model.output_cost_per_1m,
      cached_read_cost_per_1m: model.cached_read_cost_per_1m ?? null,
      cached_write_cost_per_1m: model.cached_write_cost_per_1m ?? null,
      audio_input_cost_per_1m: model.audio_input_cost_per_1m ?? null,
      audio_output_cost_per_1m: model.audio_output_cost_per_1m ?? null,
      image_input_cost_per_1m: model.image_input_cost_per_1m ?? null,
      cost_per_image: model.cost_per_image ?? null,
      audio_cost_per_minute: model.audio_cost_per_minute ?? null,
      tts_cost_per_1m_chars: model.tts_cost_per_1m_chars ?? null,
      web_search_cost_per_1k: model.web_search_cost_per_1k ?? null,
      icon_url: model.icon_url,
      parameter_overrides: model.parameter_overrides ?? null,
      auto_eligible: model.auto_eligible ?? false,
      quality_tier: model.quality_tier ?? null,
      supports_images: model.supports_images ?? false,
      supports_image_generation: model.supports_image_generation ?? false,
      supports_tts: model.supports_tts ?? false,
      supports_stt: model.supports_stt ?? false,
      supports_embedding: model.supports_embedding ?? false,
      supports_realtime: model.supports_realtime ?? false,
      attachment: model.attachment ?? false,
      reasoning: model.reasoning ?? false,
      tool_call: model.tool_call ?? false,
      structured_output: model.structured_output ?? false,
      temperature: model.temperature ?? false,
      experimental: model.experimental ?? false,
      open_weights: model.open_weights ?? false,
      status: model.status ?? null,
      family: model.family ?? null,
      knowledge: model.knowledge ?? null,
      release_date: model.release_date ?? null,
      max_output_tokens: model.max_output_tokens ?? null,
      routing_assignments: model.routing_assignments ?? null,
      description: model.description ?? null,
      homepage_url: model.homepage_url ?? null,
      context_length: model.context_length ?? null,
    };
    parameterOverrides.value = model.parameter_overrides ?? {};
  };

  // ---- Validation & payload ----

  /** First failing rule as an i18n key, or null when the draft is submittable. */
  function validate(): string | null {
    if (!form.value.name.trim()) return "models.nameRequired";
    if (form.value.providers.length === 0) return "models.atLeastOneProvider";
    if (form.value.providers.some((p) => !p.provider_name)) return "models.providerNameRequired";
    if (form.value.providers.some((p) => !p.provider_model_name?.trim()))
      return "models.providerModelNameRequired";
    if (form.value.auto_eligible && !form.value.quality_tier) return "models.qualityTierRequired";
    return null;
  }

  /**
   * Submission payload for both create and update. Normalizes the name
   * (trimmed), clears the routing profile when the model is not
   * auto-eligible, and strips provider rows down to the persisted fields.
   */
  function buildPayload(): ModelCreate {
    const data = { ...form.value, name: form.value.name.trim() };
    if (!data.auto_eligible) {
      data.quality_tier = null;
      data.routing_assignments = null;
    }
    return {
      ...data,
      quality_tier: data.quality_tier || null,
      status: data.status || null,
      providers: data.providers.map((p) => ({
        provider_name: p.provider_name,
        priority: p.priority,
        provider_model_name: p.provider_model_name,
        input_cost_per_1m: p.input_cost_per_1m ?? null,
        output_cost_per_1m: p.output_cost_per_1m ?? null,
        cached_read_cost_per_1m: p.cached_read_cost_per_1m ?? null,
        cached_write_cost_per_1m: p.cached_write_cost_per_1m ?? null,
        audio_input_cost_per_1m: p.audio_input_cost_per_1m ?? null,
        audio_output_cost_per_1m: p.audio_output_cost_per_1m ?? null,
        image_input_cost_per_1m: p.image_input_cost_per_1m ?? null,
        cost_per_image: p.cost_per_image ?? null,
        audio_cost_per_minute: p.audio_cost_per_minute ?? null,
        tts_cost_per_1m_chars: p.tts_cost_per_1m_chars ?? null,
        web_search_cost_per_1k: p.web_search_cost_per_1k ?? null,
        parameter_overrides: p.parameter_overrides ?? {},
      })),
      parameter_overrides:
        Object.keys(parameterOverrides.value).length > 0 ? parameterOverrides.value : null,
    };
  }

  return {
    form,
    parameterOverrides,
    iconPreviewFailed,
    getCapability,
    setCapability,
    showProviderEditDialog,
    editingProviderIndex,
    editingProviderData,
    addProvider,
    removeProvider,
    openProviderEditDialog,
    saveProviderEdit,
    descriptionModel,
    statusModel,
    familyModel,
    knowledgeModel,
    releaseDateModel,
    openForCreate,
    openForEdit,
    validate,
    buildPayload,
  };
}
