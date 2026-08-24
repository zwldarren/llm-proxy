import { shallowRef, ref, computed, type ComputedRef, type Ref } from "vue";

interface ResourceStoreOptions<T, CreateT, UpdateT> {
  name: string;
  fetchFn: () => Promise<T[]>;
  createFn: (data: CreateT) => Promise<T>;
  updateFn: (name: string, data: UpdateT) => Promise<T>;
  deleteFn: (name: string) => Promise<unknown>;
}

interface ResourceStore<T, CreateT, UpdateT> {
  items: Ref<T[]>;
  loading: Ref<boolean>;
  loaded: Ref<boolean>;
  error: Ref<string | null>;
  ready: ComputedRef<boolean>;
  fetchItems: (force?: boolean) => Promise<T[]>;
  prefetch: () => void;
  createItem: (data: CreateT) => Promise<T>;
  updateItem: (name: string, data: UpdateT) => Promise<T>;
  deleteItem: (name: string) => Promise<void>;
  reset: () => void;
}

/**
 * Factory for creating a standard CRUD Pinia store for a resource.
 *
 * Handles loading state, re-entrancy guard, caching, and error logging
 * consistently across all resource stores.
 */
export function createResourceStore<T, CreateT, UpdateT>(
  options: ResourceStoreOptions<T, CreateT, UpdateT>
): ResourceStore<T, CreateT, UpdateT> {
  const { name, fetchFn, createFn, updateFn, deleteFn } = options;

  const items = shallowRef<T[]>([]);
  const loading = ref(false);
  const loaded = ref(false);
  const error = ref<string | null>(null);

  const ready = computed(() => loaded.value);

  async function fetchItems(force = false): Promise<T[]> {
    if (loaded.value && !force) {
      return items.value;
    }
    if (loading.value) return items.value;
    loading.value = true;
    try {
      const res = await fetchFn();
      items.value = res;
      loaded.value = true;
      error.value = null;
      return res;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : `Failed to fetch ${name}s`;
      error.value = errorMsg;
      console.error(`Failed to fetch ${name}s:`, err);
      throw err;
    } finally {
      loading.value = false;
    }
  }

  function prefetch(): void {
    if (!loaded.value) {
      fetchItems().catch((err) => {
        const errorMsg = err instanceof Error ? err.message : `${name} prefetch failed`;
        error.value = errorMsg;
      });
    }
  }

  async function createItem(data: CreateT): Promise<T> {
    error.value = null;
    try {
      const res = await createFn(data);
      await fetchItems(true);
      return res;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : `Failed to create ${name}`;
      error.value = errorMsg;
      console.error(`Failed to create ${name}:`, err);
      throw err;
    }
  }

  async function updateItem(name: string, data: UpdateT): Promise<T> {
    error.value = null;
    try {
      const res = await updateFn(name, data);
      await fetchItems(true);
      return res;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : `Failed to update ${name}`;
      error.value = errorMsg;
      console.error(`Failed to update ${name}:`, err);
      throw err;
    }
  }

  async function deleteItem(name: string): Promise<void> {
    error.value = null;
    try {
      await deleteFn(name);
      await fetchItems(true);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : `Failed to delete ${name}`;
      error.value = errorMsg;
      console.error(`Failed to delete ${name}:`, err);
      throw err;
    }
  }

  function reset(): void {
    items.value = [];
    loading.value = false;
    loaded.value = false;
    error.value = null;
  }

  return {
    items,
    loading,
    loaded,
    error,
    ready,
    fetchItems,
    prefetch,
    createItem,
    updateItem,
    deleteItem,
    reset,
  };
}
