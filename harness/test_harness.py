"""Offline tests for the model client and the disk cache.

These tests replace ``litellm.completion`` with a fake response, so they
need no API key, no GPU, and no network. They run on every pull request.

From the repository root:

    python -m unittest discover -s harness -t . -p "test_*.py"
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness import cache
from harness.client import call
from harness.pricing import PRICING_PER_1M, compute_cost

HELLO = [{"role": "user", "content": "Reply with exactly: hello"}]


def fake_response(text="hello", prompt_tokens=6, completion_tokens=1):
    """Build an object shaped like a LiteLLM completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


class HarnessTestCase(unittest.TestCase):
    """Gives each test an empty cache folder and a clean environment."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        cache_patch = mock.patch.object(cache, "CACHE_DIR", Path(folder.name))
        cache_patch.start()
        self.addCleanup(cache_patch.stop)

        env_patch = mock.patch.dict(os.environ, {}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)

        completion_patch = mock.patch(
            "harness.client.litellm.completion", return_value=fake_response()
        )
        self.completion = completion_patch.start()
        self.addCleanup(completion_patch.stop)


class ClientTests(HarnessTestCase):
    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            call("gpt", HELLO)
        self.completion.assert_not_called()

    def test_gemini_requires_a_key(self):
        with self.assertRaises(RuntimeError):
            call("flash", HELLO)
        self.completion.assert_not_called()

    def test_flash_request(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        result = call("flash", HELLO)

        sent = self.completion.call_args.kwargs
        self.assertEqual(sent["model"], "gemini/gemini-3.5-flash")
        self.assertEqual(sent["temperature"], 0)
        self.assertEqual(sent["api_key"], "test-key")
        self.assertEqual(sent["messages"], HELLO)
        self.assertEqual(
            (result.text, result.prompt_tokens, result.completion_tokens, result.cached),
            ("hello", 6, 1, False),
        )

    def test_flash_lite_request(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        call("flash-lite", HELLO)
        self.assertEqual(
            self.completion.call_args.kwargs["model"], "gemini/gemini-3.5-flash-lite"
        )

    def test_qwen_requires_an_address(self):
        with self.assertRaises(RuntimeError):
            call("qwen", HELLO)
        self.completion.assert_not_called()

    def test_qwen_request(self):
        os.environ["QWEN_API_BASE"] = "http://qwen-host:8000/v1"
        call("qwen", HELLO)

        sent = self.completion.call_args.kwargs
        self.assertEqual(sent["model"], "openai/Qwen/Qwen3-8B-AWQ")
        self.assertEqual(sent["api_base"], "http://qwen-host:8000/v1")
        self.assertEqual(sent["temperature"], 0)
        self.assertEqual(
            sent["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}}
        )

    def test_qwen_address_is_read_on_each_call(self):
        os.environ["QWEN_API_BASE"] = "http://first:8000/v1"
        call("qwen", [{"role": "user", "content": "first"}])
        os.environ["QWEN_API_BASE"] = "http://second:8000/v1"
        call("qwen", [{"role": "user", "content": "second"}])
        self.assertEqual(
            self.completion.call_args.kwargs["api_base"], "http://second:8000/v1"
        )


class CacheTests(HarnessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["GEMINI_API_KEY"] = "test-key"

    def test_second_identical_call_is_served_from_cache(self):
        self.completion.return_value = fake_response(completion_tokens=81)
        first = call("flash", HELLO)
        second = call("flash", HELLO)

        self.assertEqual(self.completion.call_count, 1)
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(
            (second.text, second.prompt_tokens, second.completion_tokens),
            ("hello", 6, 81),
        )

    def test_each_model_has_its_own_entry(self):
        call("flash", HELLO)
        result = call("flash-lite", HELLO)
        self.assertEqual(self.completion.call_count, 2)
        self.assertFalse(result.cached)

    def test_different_messages_miss_the_cache(self):
        call("flash", HELLO)
        call("flash", [{"role": "user", "content": "Reply with exactly: bye"}])
        self.assertEqual(self.completion.call_count, 2)

    def test_key_ignores_dictionary_order(self):
        a = [{"role": "user", "content": "hi"}]
        b = [{"content": "hi", "role": "user"}]
        self.assertEqual(cache.cache_key("flash", a), cache.cache_key("flash", b))

class CostTests(HarnessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["GEMINI_API_KEY"] = "test-key"

    def test_flash_cost_comes_from_the_price_table(self):
        self.completion.return_value = fake_response(
            prompt_tokens=1000, completion_tokens=500
        )
        result = call("flash", HELLO)
        # 1000 * $1.50/1M + 500 * $9.00/1M
        self.assertAlmostEqual(result.cost, 0.006)

    def test_flash_lite_is_cheaper_than_flash(self):
        self.completion.return_value = fake_response(
            prompt_tokens=1000, completion_tokens=500
        )
        flash = call("flash", HELLO)
        lite = call("flash-lite", HELLO)
        # 1000 * $0.30/1M + 500 * $2.50/1M
        self.assertAlmostEqual(lite.cost, 0.00155)
        self.assertLess(lite.cost, flash.cost)

    def test_qwen_is_free(self):
        os.environ["QWEN_API_BASE"] = "http://qwen-host:8000/v1"
        self.assertEqual(call("qwen", HELLO).cost, 0.0)

    def test_cached_call_still_reports_cost(self):
        first = call("flash", HELLO)
        second = call("flash", HELLO)
        self.assertTrue(second.cached)
        self.assertAlmostEqual(second.cost, first.cost)
        self.assertGreater(second.cost, 0)

    def test_call_logs_tokens_and_cost(self):
        with self.assertLogs("harness.client", level="INFO") as logs:
            call("flash", HELLO)
        line = logs.output[0]
        self.assertIn("tokens_in=6", line)
        self.assertIn("tokens_out=1", line)
        self.assertIn("cost=$", line)

    def test_every_model_has_a_price(self):
        from harness.client import MODELS

        self.assertEqual(set(MODELS), set(PRICING_PER_1M))

    def test_unknown_model_has_no_price(self):
        with self.assertRaises(KeyError):
            compute_cost("gpt", 1, 1)


class RetryTests(HarnessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["GEMINI_API_KEY"] = "test-key"

    def test_retries_service_unavailable_then_succeeds(self):
        from litellm.exceptions import ServiceUnavailableError

        self.completion.side_effect = [
            ServiceUnavailableError(
                "busy",
                llm_provider="gemini",
                model="gemini/gemini-3.5-flash",
            ),
            fake_response(),
        ]
        with mock.patch("harness.client.time.sleep") as sleep:
            result = call("flash", HELLO)
        self.assertEqual(result.text, "hello")
        self.assertEqual(self.completion.call_count, 2)
        sleep.assert_called_once()

    def test_gives_up_after_max_retries(self):
        from litellm.exceptions import ServiceUnavailableError

        err = ServiceUnavailableError(
            "busy",
            llm_provider="gemini",
            model="gemini/gemini-3.5-flash",
        )
        self.completion.side_effect = err
        with mock.patch("harness.client.time.sleep"), self.assertRaises(ServiceUnavailableError):
            call("flash", HELLO)
        self.assertEqual(self.completion.call_count, 6)  # 1 + MAX_RETRIES


if __name__ == "__main__":
    unittest.main()
