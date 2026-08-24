"""Unit tests for model allowlist semantics (user-level constraints)."""

from llm_proxy.api.middleware.model_restriction import check_model_restriction
from llm_proxy.security.model_acl import intersect_model_lists


class TestIntersectModelLists:
    def test_both_unrestricted(self):
        assert intersect_model_lists(None, None) is None

    def test_user_unrestricted_takes_key_list(self):
        assert intersect_model_lists(["a", "b"], None) == ["a", "b"]

    def test_key_unrestricted_takes_user_list(self):
        assert intersect_model_lists(None, ["a"]) == ["a"]

    def test_intersection(self):
        assert set(intersect_model_lists(["a", "b", "c"], ["b", "c", "d"])) == {"b", "c"}

    def test_disjoint_is_deny_all(self):
        assert intersect_model_lists(["a"], ["b"]) == []

    def test_empty_user_list_denies_everything(self):
        assert intersect_model_lists(["a"], []) == []
        assert intersect_model_lists(None, []) == []


class TestCheckModelRestrictionEmptyList:
    """Empty allowlist means deny-all (distinct from None = unrestricted)."""

    def test_none_is_unrestricted(self):
        allowed, _ = check_model_restriction("k", None, "any-model")
        assert allowed

    def test_empty_list_denies_model(self):
        allowed, msg = check_model_restriction("k", [], "any-model")
        assert not allowed
        assert "not authorized to access any model" in (msg or "")

    def test_empty_list_allows_modelless_requests(self):
        # Requests without a model field (e.g. some listing endpoints) pass.
        allowed, _ = check_model_restriction("k", [], None)
        assert allowed

    def test_listed_model_allowed(self):
        allowed, _ = check_model_restriction("k", ["m1"], "m1")
        assert allowed

    def test_unlisted_model_denied(self):
        allowed, msg = check_model_restriction("k", ["m1"], "m2")
        assert not allowed
        assert "m2" in (msg or "")
