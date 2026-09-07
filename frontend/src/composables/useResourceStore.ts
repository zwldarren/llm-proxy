import { shallowRef, ref, computed, type ComputedRef, type Ref } from "vue";

interface ReadOnlyResourceStoreOptions<T> {
  name: string;
  fetchFn: () => Promise<T[]>;
}

interface ResourceStoreOptions<T, CreateT, UpdateT> extends ReadOnlyResourceStoreOptions<T> {
  createFn: (data: CreateT) => Promise<T>;
  updateFn: (name: string, data: UpdateT) => Promise<T>;
  deleteFn: (name: string) => Promise<unknown>;
}

interface ReadOnlyResourceStore<T> {
  items: Ref<T[]>;
  loading: Ref<boolean>;
  loaded: Ref<boolean>;
  error: Ref<string | null>;
  ready: ComputedRef<boolean>;
  fetchItems: (force?: boolean) => Promise<T[]>;
  prefetch: () => void;
  reset: () => void;
}

interface ResourceStore<T, CreateT, UpdateT> extends ReadOnlyResourceStore<T> {
  createItem: (data: CreateT) => Promise<T>;
  updateItem: (name: string, data: UpdateT) => Promise<T>;
  deleteItem: (name: string) => Promise<void>;
}

/**
 * Factory for creating a standard Pinia store for a resource.
 *
 * Handles loading state, re-entrancy guard, caching, and error logging
 * consistently across all resource stores. Passing only `fetchFn` yields a
 * read-only store (list + cache + prefetch) without the CRUD methods.
 */
export function createResourceStore<T, CreateT, UpdateT>(
  options: ResourceStoreOptions<T, CreateT, UpdateT>
): ResourceStore<T, CreateT, UpdateT>;
export function createResourceStore<T>(
  options: ReadOnlyResourceStoreOptions<T>
): ReadOnlyResourceStore<T>;
export function createResourceStore<T, CreateT, UpdateT>(
  options: ResourceStoreOptions<T, CreateT, UpdateT> | ReadOnlyResourceStoreOptions<T>
): ResourceStore<T, CreateT, UpdateT> | ReadOnlyResourceStore<T> {
  const { name, fetchFn } = options;

  const items = shallowRef<T[]>([]);
  const loading = ref(false);
  const loaded = ref(false);
  const error = ref<string | null>(null);

  const ready = computed(() => loaded.value);

  // Generation guard: a reset() during an in-flight fetch invalidates that
  // fetch — its completion must not repopulate the store with the previous
  // session's data.
  let generation = 0;

  async function fetchItems(force = false): Promise<T[]> {
    if (loaded.value && !force) {
      return items.value;
    }
    if (loading.value) return items.value;
    const gen = generation;
    loading.value = true;
    try {
      const res = await fetchFn();
      if (gen !== generation) return items.value;
      items.value = res;
      loaded.value = true;
      error.value = null;
      return res;
    } catch (err) {
      if (gen !== generation) throw err;
      const errorMsg = err instanceof Error ? err.message : `Failed to fetch ${name}s`;
      error.value = errorMsg;
      console.error(`Failed to fetch ${name}s:`, err);
      throw err;
    } finally {
      // A newer fetch may already be in flight after the reset; only the
      // fetch that still owns the current generation clears the flag.
      if (gen === generation) loading.value = false;
    }
  }

  function prefetch(): void {
    if (!loaded.value) {
      const gen = generation;
      fetchItems().catch((err) => {
        if (gen !== generation) return;
        const errorMsg = err instanceof Error ? err.message : `${name} prefetch failed`;
        error.value = errorMsg;
      });
    }
  }

  function reset(): void {
    generation++;
    items.value = [];
    loading.value = false;
    loaded.value = false;
    error.value = null;
  }

  const base: ReadOnlyResourceStore<T> = {
    items,
    loading,
    loaded,
    error,
    ready,
    fetchItems,
    prefetch,
    reset,
  };

  if (!("createFn" in options)) {
    return base;
  }

  const { createFn, updateFn, deleteFn } = options;

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

  return { ...base, createItem, updateItem, deleteItem };
}
