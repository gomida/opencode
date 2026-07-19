#!/usr/bin/env python3
"""Regression tests for the Gemini-native predecessor-letter adapter."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gemini_native_predecessor_letter_proxy import (
    State,
    extract_text,
    is_compaction_request,
    is_generation_path,
    merge_response,
    normalize_letter,
    render_progress,
)


class GeminiNativeProxyTest(unittest.TestCase):
    def test_review_text_excludes_native_thought_parts(self):
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "analysis"},
                            {"text": '{"status":"OK","letter":"Continue."}'},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(
            extract_text(payload), '{"status":"OK","letter":"Continue."}'
        )

    def test_native_generation_path_accepts_streaming_and_nonstreaming(self):
        self.assertTrue(
            is_generation_path(
                "/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse"
            )
        )
        self.assertTrue(
            is_generation_path("/v1beta/models/gemini-3.5-flash:generateContent")
        )
        self.assertFalse(
            is_generation_path("/v1beta/models/gemini-3.5-flash:countTokens")
        )

    def test_compaction_detection_scans_native_text_parts(self):
        self.assertTrue(
            is_compaction_request(
                {
                    "systemInstruction": {
                        "parts": [{"text": "You are an anchored context summarization assistant."}]
                    },
                    "contents": [],
                }
            )
        )
        self.assertFalse(
            is_compaction_request(
                {"contents": [{"role": "user", "parts": [{"text": "Fix the bug"}]}]}
            )
        )

    def test_sse_merge_keeps_native_signature_but_excludes_thought_from_progress(self):
        chunks = [
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"thought": True, "text": "hidden"},
                                {
                                    "functionCall": {"name": "read", "args": {"filePath": "a.py"}},
                                    "thoughtSignature": "signature",
                                },
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10},
            }
        ]
        body = ("data: " + json.dumps(chunks[0]) + "\n\n").encode()
        merged = merge_response(body, "text/event-stream")
        self.assertEqual(merged["native_parts"][1]["thoughtSignature"], "signature")
        self.assertEqual(
            merged["visible_parts"],
            [{"functionCall": {"name": "read", "args": {"filePath": "a.py"}}}],
        )

    def test_letter_injection_preserves_existing_native_contents(self):
        with TemporaryDirectory() as tmp:
            state = State("https://example.test/v1beta/", "model", "key", Path(tmp), 250, 2048)
            state.last_request = {"contents": [{"role": "user", "parts": [{"text": "before"}]}]}
            cycle = state.begin_compaction()
            cycle.letter = normalize_letter('{"status":"WARN","letter":"Check tests."}')
            original = {"contents": [{"role": "user", "parts": [{"text": "after"}]}]}
            mutated, event = state.inject_if_ready(original)
            self.assertEqual(event, 1)
            self.assertEqual(original["contents"][-1]["parts"][0]["text"], "after")
            self.assertIn("Check tests.", mutated["contents"][-1]["parts"][0]["text"])
            self.assertTrue(cycle.injected)

    def test_native_url_and_key_replacement_are_case_insensitive(self):
        with TemporaryDirectory() as tmp:
            state = State(
                "https://generativelanguage.googleapis.com/v1beta/",
                "gemini-3.5-flash",
                "secret-key",
                Path(tmp),
                250,
                2048,
            )
            self.assertEqual(
                state.upstream_url("/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse"),
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
            )
            self.assertEqual(
                state.outbound_headers(
                    {"X-Goog-Api-Key": "local", "Authorization": "local", "X-Test": "1"}
                ),
                {"X-Test": "1", "x-goog-api-key": "secret-key"},
            )

    def test_render_progress_contains_text_and_function_call(self):
        text = render_progress(
            [
                {
                    "role": "model",
                    "parts": [
                        {"text": "Inspecting."},
                        {"functionCall": {"name": "read", "args": {"filePath": "a.py"}}},
                    ],
                }
            ]
        )
        self.assertIn("Inspecting.", text)
        self.assertIn("TOOL CALL read", text)


if __name__ == "__main__":
    unittest.main()
