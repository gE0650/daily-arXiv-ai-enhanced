import unittest

from runtime import build_chat_openai_kwargs, raise_if_processing_failed


class BuildChatOpenAIKwargsTests(unittest.TestCase):
    def test_openai_configuration_strips_key_and_omits_provider_extensions(self):
        """Catches accidentally sending a provider-specific `thinking` field to OpenAI."""
        kwargs = build_chat_openai_kwargs(
            model_name="gpt-4.1-mini",
            base_url="https://api.openai.com/v1/",
            api_key="  sk-test-key  ",
        )

        self.assertEqual("gpt-4.1-mini", kwargs["model"])
        self.assertEqual("https://api.openai.com/v1", kwargs["base_url"])
        self.assertEqual("sk-test-key", kwargs["api_key"])
        self.assertNotIn("extra_body", kwargs)

    def test_volcengine_configuration_disables_thinking_with_extra_body(self):
        """Catches losing the Ark-specific request option required by the current model."""
        kwargs = build_chat_openai_kwargs(
            model_name="doubao-seed-1.6-250615",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="ark-test-key",
        )

        self.assertEqual(
            {"thinking": {"type": "disabled"}},
            kwargs["extra_body"],
        )

    def test_deepseek_configuration_disables_thinking_with_extra_body(self):
        """Catches the 400 'Thinking mode does not support this tool_choice' failure."""
        for base_url in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
            with self.subTest(base_url=base_url):
                kwargs = build_chat_openai_kwargs(
                    model_name="deepseek-flash",
                    base_url=base_url,
                    api_key="sk-test-key",
                )

                self.assertEqual(
                    {"thinking": {"type": "disabled"}},
                    kwargs["extra_body"],
                )

    def test_lookalike_host_does_not_get_provider_extensions(self):
        """Catches matching on a bare suffix such as 'api.notdeepseek.com'."""
        kwargs = build_chat_openai_kwargs(
            model_name="deepseek-flash",
            base_url="https://api.notdeepseek.com/v1",
            api_key="sk-test-key",
        )

        self.assertNotIn("extra_body", kwargs)

    def test_widespread_failure_stops_the_batch(self):
        """Catches swallowing authentication failures and publishing fallback summaries."""
        with self.assertRaisesRegex(RuntimeError, r"513/513 paper\(s\) failed"):
            raise_if_processing_failed(["401 Unauthorized"] * 513, total=513)

    def test_isolated_failure_is_tolerated(self):
        """Catches discarding a whole day of papers because one request failed."""
        raise_if_processing_failed(["503 Service Unavailable"], total=513)

    def test_failure_above_the_tolerance_still_stops_the_batch(self):
        """Catches raising the tolerance so high that a broken provider slips through."""
        with self.assertRaisesRegex(RuntimeError, r"40/513"):
            raise_if_processing_failed(["500 Internal Server Error"] * 40, total=513)


if __name__ == "__main__":
    unittest.main()
