"""Tests for ThreadSafeRegistry base class."""

import asyncio
import threading

import pytest

from llm_proxy.core.registry_base import ThreadSafeRegistry


class TestThreadSafeRegistry:
    """Test suite for ThreadSafeRegistry."""

    def test_register_and_get(self):
        """Test basic register and get operations."""
        registry = ThreadSafeRegistry[str]()

        registry.register("key1", "value1")
        registry.register("key2", "value2")

        assert registry.get("key1") == "value1"
        assert registry.get("key2") == "value2"
        assert registry.get("nonexistent") is None

    def test_list_all(self):
        """Test listing all registered keys."""
        registry = ThreadSafeRegistry[int]()

        registry.register("a", 1)
        registry.register("b", 2)
        registry.register("c", 3)

        keys = registry.list_all()
        assert set(keys) == {"a", "b", "c"}
        assert len(keys) == 3

    def test_get_all(self):
        """Test getting all entries."""
        registry = ThreadSafeRegistry[str]()

        registry.register("a", "value_a")
        registry.register("b", "value_b")

        all_entries = registry.get_all()
        assert all_entries == {"a": "value_a", "b": "value_b"}

    def test_clear(self):
        """Test clearing all entries."""
        registry = ThreadSafeRegistry[str]()

        registry.register("a", "value_a")
        registry.register("b", "value_b")

        assert len(registry) == 2

        registry.clear()

        assert len(registry) == 0
        assert registry.list_all() == []

    def test_len(self):
        """Test __len__ method."""
        registry = ThreadSafeRegistry[str]()

        assert len(registry) == 0

        registry.register("a", "value_a")
        assert len(registry) == 1

        registry.register("b", "value_b")
        assert len(registry) == 2

    def test_concurrent_registration(self):
        """Test thread-safe concurrent registration."""
        registry = ThreadSafeRegistry[int]()
        num_threads = 100

        def register_items(start_idx: int):
            for i in range(start_idx, start_idx + 10):
                registry.register(f"key_{i}", i)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=register_items, args=(i * 10,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all items were registered
        assert len(registry) == num_threads * 10

        # Verify all values are correct
        for i in range(num_threads * 10):
            assert registry.get(f"key_{i}") == i

    def test_concurrent_read_write(self):
        """Test thread-safe concurrent read and write operations."""
        registry = ThreadSafeRegistry[str]()

        # Pre-populate with some data
        for i in range(10):
            registry.register(f"key_{i}", f"value_{i}")

        num_reads = 100
        num_writes = 50
        errors = []

        def read_operation():
            try:
                for _ in range(num_reads):
                    thread_id = threading.current_thread().ident or 0
                    key = f"key_{thread_id % 10}"
                    registry.get(key)
                    registry.list_all()
            except Exception as e:
                errors.append(e)

        def write_operation():
            try:
                for i in range(num_writes):
                    key = f"write_{threading.current_thread().ident}_{i}"
                    registry.register(key, f"value_{i}")
                    if i % 10 == 0:
                        pass
            except Exception as e:
                errors.append(e)

        threads = []
        # Start read threads
        for _ in range(5):
            thread = threading.Thread(target=read_operation)
            threads.append(thread)
            thread.start()

        # Start write threads
        for _ in range(3):
            thread = threading.Thread(target=write_operation)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_mixed_operations(self):
        """Test thread-safe mixed operations."""
        registry = ThreadSafeRegistry[int]()
        errors = []

        def mixed_operations(thread_id: int):
            try:
                for i in range(20):
                    registry.register(f"t{thread_id}_k{i}", i)
                    registry.get(f"t{thread_id}_k{i}")
                    registry.list_all()
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(10):
            thread = threading.Thread(target=mixed_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_race_condition_prevention(self):
        """Test that race conditions are prevented."""
        registry = ThreadSafeRegistry[int]()
        counter = [0]  # Use list for mutable counter in nested function
        lock = threading.Lock()

        def increment_and_register():
            with lock:
                my_id = counter[0]
                counter[0] += 1

            # Multiple threads trying to register with potentially conflicting keys
            # The registry should handle this safely
            for i in range(10):
                registry.register(f"key_{my_id}_{i}", my_id * 10 + i)

        threads = []
        for _ in range(20):
            thread = threading.Thread(target=increment_and_register)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all items are present and correct
        assert len(registry) == 20 * 10

    @pytest.mark.asyncio
    async def test_async_concurrent_operations(self):
        """Test concurrent operations using asyncio."""
        registry = ThreadSafeRegistry[str]()

        async def register_items(prefix: str, count: int):
            for i in range(count):
                registry.register(f"{prefix}_{i}", f"{prefix}_value_{i}")

        async def read_items(prefix: str, count: int):
            for i in range(count):
                registry.get(f"{prefix}_{i}")

        # Register items concurrently
        await asyncio.gather(
            register_items("a", 50),
            register_items("b", 50),
            register_items("c", 50),
        )

        assert len(registry) == 150

        # Read items concurrently
        await asyncio.gather(
            read_items("a", 50),
            read_items("b", 50),
            read_items("c", 50),
        )

    def test_overwrite_existing_key(self):
        """Test overwriting an existing key."""
        registry = ThreadSafeRegistry[str]()

        registry.register("key1", "value1")
        assert registry.get("key1") == "value1"

        registry.register("key1", "value2")
        assert registry.get("key1") == "value2"

    def test_empty_registry_operations(self):
        """Test operations on empty registry."""
        registry = ThreadSafeRegistry[str]()

        assert registry.get("nonexistent") is None
        assert registry.list_all() == []
        assert registry.get_all() == {}
        assert len(registry) == 0

    def test_get_all_returns_copy(self):
        """Test that get_all returns a copy, not the internal dict."""
        registry = ThreadSafeRegistry[str]()

        registry.register("key1", "value1")

        all_entries = registry.get_all()
        all_entries["key2"] = "value2"  # Modify the returned dict

        # Original registry should not be affected
        assert registry.get("key2") is None
        assert len(registry) == 1
