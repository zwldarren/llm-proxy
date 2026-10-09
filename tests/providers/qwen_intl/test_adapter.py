"""The Qwen International adapter's own endpoints (DashScope Singapore).

The type's declared contract — base URL, branding, the native protocol set and
the Anthropic/Responses split between the bare site root and the
``/compatible-mode/v1`` alias — is asserted once for every provider of its
family in ``tests/providers/test_declared_providers.py``. What is specific to
this region is the image service: like the China-hosted ``qwen`` type it hangs
off the site root (the compatibility alias carries no image API), and the
region's keys are bound to those hosts.
"""

from llm_proxy.providers.qwen_intl import QwenIntlAdapter

SITE_ROOT = "https://dashscope-intl.aliyuncs.com"


def test_image_urls_hang_off_the_site_root():
    adapter = QwenIntlAdapter(api_key="test-key")

    assert adapter._image_generation_url() == (
        f"{SITE_ROOT}/api/v1/services/aigc/image-generation/generation"
    )
    assert adapter._image_task_url("t1") == f"{SITE_ROOT}/api/v1/tasks/t1"
