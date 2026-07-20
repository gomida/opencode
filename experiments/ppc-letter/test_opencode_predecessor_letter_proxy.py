#!/usr/bin/env python3
"""Regression tests for the PPC predecessor-letter proxy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from opencode_predecessor_letter_proxy import (
    ProxyState,
    assistant_message_from_successor_record,
    build_predecessor_review_body,
    is_compaction_request_body,
    normalize_predecessor_letter,
    predecessor_context_id,
    render_successor_text,
    stable_sha256,
)


class PredecessorReviewBodyTest(unittest.TestCase):
    def sample_request(self):
        return {
            "model": "Qwen3-Coder-30B-A3B-Instruct-FP8",
            "messages": [
                {"role": "system", "content": "You are OpenCode."},
                {"role": "user", "content": "Fix the bug."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{\"filePath\":\"a.py\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "old content"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"filePath": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 8192,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }

    def compaction_request(self, model="Qwen3-Coder-30B-A3B-Instruct-FP8"):
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an anchored context summarization assistant for coding sessions.",
                },
                {
                    "role": "user",
                    "content": "Create a new anchored summary from the conversation history.",
                },
            ],
        }

    def capture_transition(self, state, request, response, seq=10):
        request_index = state.note_chat_request()
        generation, event_index = state.note_chat_request_body(
            seq=seq,
            chat_request_index=request_index,
            request_body=request,
            is_compaction=False,
        )
        self.assertIsNone(event_index)
        state.note_non_compaction_response(
            seq=seq,
            chat_request_index=request_index,
            successor_record=response,
            generation=generation,
        )
        compaction_index = state.note_chat_request()
        successor_generation, event_index = state.note_chat_request_body(
            seq=seq + 1,
            chat_request_index=compaction_index,
            request_body=self.compaction_request(request["model"]),
            is_compaction=True,
        )
        return successor_generation, event_index

    def test_review_preserves_predecessor_prompt_prefix_and_tools(self):
        request = self.sample_request()
        original_hash = stable_sha256(request)

        review_body, metadata = build_predecessor_review_body(
            request_body=request,
            successor_text="Successor response 21:\nVisible content:\nI will inspect a.py.",
            review_max_tokens=2048,
            predecessor_generation=2,
        )

        self.assertEqual(review_body["messages"][: len(request["messages"])], request["messages"])
        self.assertEqual(review_body["tools"], request["tools"])
        self.assertEqual(review_body["tool_choice"], "none")
        self.assertEqual(review_body["model"], request["model"])
        self.assertEqual(review_body["top_p"], request["top_p"])
        self.assertEqual(review_body["extra_body"], request["extra_body"])
        self.assertFalse(review_body["stream"])
        self.assertNotIn("stream_options", review_body)
        self.assertEqual(review_body["temperature"], 0)
        self.assertEqual(review_body["max_tokens"], 2048)
        self.assertIn("tool_choice", metadata["intentional_overrides"])
        self.assertEqual(review_body["response_format"]["type"], "json_schema")
        schema = review_body["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["required"], ["status", "letter"])
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("response_format", metadata["intentional_overrides"])

        self.assertEqual(stable_sha256(request), original_hash)
        self.assertEqual(metadata["predecessor_context_id"], predecessor_context_id(request, 2))
        self.assertTrue(metadata["predecessor_live_context_id"].startswith("g2-live-"))
        self.assertTrue(metadata["message_prefix_equal"])
        self.assertTrue(metadata["non_overridden_fields_equal"])
        self.assertTrue(metadata["prefix_invariant_ok"])
        self.assertEqual(metadata["changed_or_missing_non_overridden_fields"], [])
        self.assertEqual(metadata["unexpected_added_non_overridden_fields"], [])
        self.assertEqual(
            metadata["base_prompt_prefix_sha256"],
            metadata["review_prompt_prefix_sha256"],
        )
        self.assertFalse(metadata["predecessor_response_included"])

    def test_review_body_is_deep_cloned_from_original_request(self):
        request = self.sample_request()
        review_body, metadata = build_predecessor_review_body(
            request_body=request,
            successor_text="trace",
            review_max_tokens=4096,
        )

        review_body["tools"][0]["function"]["description"] = "mutated"
        review_body["messages"][0]["content"] = "mutated"

        self.assertEqual(request["tools"][0]["function"]["description"], "Read a file")
        self.assertEqual(request["messages"][0]["content"], "You are OpenCode.")
        self.assertTrue(metadata["prefix_invariant_ok"])

    def test_schema_letter_is_normalized_for_injection(self):
        normalized = normalize_predecessor_letter(
            '{"status":"warn","letter":"Check the failing ASGI routing test."}'
        )
        self.assertEqual(normalized["status"], "WARN")
        self.assertEqual(
            normalized["rendered"],
            "STATUS: WARN\nLETTER:\nCheck the failing ASGI routing test.",
        )
        with self.assertRaises(ValueError):
            normalize_predecessor_letter('{"status":"MAYBE","letter":"x"}')

    def test_review_includes_predecessor_generated_response_when_available(self):
        request = self.sample_request()
        predecessor_record = {
            "content": "I will inspect a.py.\n\n",
            "tool_calls": "\n".join(
                [
                    '{"id":"call_2","type":"function","index":0,"function":{"name":"read","arguments":""}}',
                    '{"id":"call_2","type":"function","index":0,"function":{"name":null,"arguments":"{\\"filePath\\":\\"a.py\\"}"}}',
                ]
            ),
        }

        review_body, metadata = build_predecessor_review_body(
            request_body=request,
            successor_text="Successor response 22:\nVisible content:\nThe file shows the bug.",
            review_max_tokens=2048,
            predecessor_response_record=predecessor_record,
        )

        predecessor_message = assistant_message_from_successor_record(predecessor_record)
        self.assertEqual(review_body["messages"][: len(request["messages"])], request["messages"])
        self.assertEqual(review_body["messages"][len(request["messages"])], predecessor_message)
        self.assertEqual(review_body["messages"][-1]["role"], "user")
        self.assertIn("Successor response 22", review_body["messages"][-1]["content"])

        self.assertTrue(metadata["predecessor_response_included"])
        self.assertTrue(metadata["predecessor_response_equal"])
        self.assertEqual(metadata["live_message_count"], len(request["messages"]) + 1)
        self.assertTrue(metadata["prefix_invariant_ok"])

    def test_successor_projection_excludes_reasoning_and_preserves_tool_calls(self):
        record = {
            "reasoning": "private chain that must not be counted or shown",
            "content": "I will inspect the target file.",
            "tool_calls": "\n".join(
                [
                    '{"id":"call_9","type":"function","index":0,"function":{"name":"read","arguments":""}}',
                    '{"id":"call_9","type":"function","index":0,"function":{"name":null,"arguments":"{\\"filePath\\":\\"engineio/server.py\\"}"}}',
                ]
            ),
        }

        message = assistant_message_from_successor_record(record)
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "I will inspect the target file.")
        self.assertNotIn("reasoning", message)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"],
            '{"filePath":"engineio/server.py"}',
        )

        rendered = render_successor_text(record)
        self.assertNotIn("private chain", rendered)
        self.assertIn("engineio/server.py", rendered)

    def test_auto_capture_uses_last_non_compaction_before_compaction(self):
        with TemporaryDirectory() as tmp:
            state = ProxyState(
                upstream="http://127.0.0.1:8000",
                log_dir=Path(tmp),
                predecessor_request=None,
                review_max_tokens=2048,
                successor_accum_tokens=1000,
            )
            request = self.sample_request()
            compaction_request = self.compaction_request(request["model"])
            self.assertTrue(is_compaction_request_body(compaction_request))
            generation, event_index = self.capture_transition(
                state,
                request,
                {"content": "I inspected str.py.\n"},
            )
            cycle = state.active_cycle

            self.assertEqual(generation, 1)
            self.assertEqual(event_index, 1)
            self.assertEqual(cycle.capture_reason, "immediate_pre_compaction_generation")
            self.assertEqual(cycle.predecessor_generation, 0)
            self.assertEqual(cycle.successor_generation, 1)
            self.assertEqual(cycle.predecessor_request_seq, 10)
            self.assertEqual(cycle.predecessor_request_body, request)
            self.assertIsNotNone(cycle.predecessor_response_record)
            self.assertEqual(
                cycle.predecessor_response_message,
                {"role": "assistant", "content": "I inspected str.py.\n"},
            )

    def test_review_trigger_depends_only_on_successor_token_threshold(self):
        with TemporaryDirectory() as tmp:
            state = ProxyState(
                upstream="http://127.0.0.1:8000",
                log_dir=Path(tmp),
                predecessor_request=None,
                review_max_tokens=2048,
                successor_accum_tokens=1000,
            )
            generation, _ = self.capture_transition(
                state,
                self.sample_request(),
                {"content": "predecessor response"},
            )
            state.chat_response_count = 999

            with (
                patch(
                    "opencode_predecessor_letter_proxy.successor_token_count",
                    side_effect=[(999, "upstream", None), (1000, "upstream", None)],
                ) as token_count,
                patch(
                    "opencode_predecessor_letter_proxy.run_predecessor_review",
                    return_value="final note",
                ) as review,
            ):
                state.maybe_review(20, {"content": "first successor segment"}, False, generation)
                self.assertFalse(state.active_cycle.review_started)
                review.assert_not_called()

                state.maybe_review(21, {"content": "second successor segment"}, False, generation)

            self.assertTrue(state.active_cycle.review_started)
            self.assertEqual(state.active_cycle.letter, "final note")
            review.assert_called_once()
            first_messages = token_count.call_args_list[0].args[2]
            second_messages = token_count.call_args_list[1].args[2]
            self.assertEqual(
                first_messages,
                [{"role": "assistant", "content": "first successor segment"}],
            )
            self.assertEqual(
                second_messages,
                [
                    {"role": "assistant", "content": "first successor segment"},
                    {"role": "assistant", "content": "second successor segment"},
                ],
            )

    def test_matching_tool_result_counts_once_and_can_trigger_review(self):
        with TemporaryDirectory() as tmp:
            state = ProxyState(
                upstream="http://127.0.0.1:8000",
                log_dir=Path(tmp),
                predecessor_request=None,
                review_max_tokens=2048,
                successor_accum_tokens=10000,
            )
            generation, _ = self.capture_transition(
                state,
                self.sample_request(),
                {"content": "predecessor response"},
            )
            successor = {
                "content": "I will read the file.",
                "tool_calls": "\n".join(
                    [
                        '{"id":"call_successor","type":"function","index":0,"function":{"name":"read","arguments":""}}',
                        '{"id":"call_successor","type":"function","index":0,"function":{"name":null,"arguments":"{\\"filePath\\":\\"large.py\\"}"}}',
                    ]
                ),
            }
            request = {
                "messages": [
                    {"role": "tool", "tool_call_id": "old_call", "content": "old"},
                    {
                        "role": "tool",
                        "tool_call_id": "call_successor",
                        "content": "large tool output",
                    },
                ]
            }

            with (
                patch(
                    "opencode_predecessor_letter_proxy.successor_token_count",
                    side_effect=[(500, "upstream", None), (10000, "upstream", None)],
                ) as token_count,
                patch(
                    "opencode_predecessor_letter_proxy.run_predecessor_review",
                    return_value="tool-aware note",
                ) as review,
            ):
                state.maybe_review(20, successor, False, generation)
                self.assertFalse(state.active_cycle.review_started)
                state.note_tool_results(21, request, generation)

            self.assertTrue(state.active_cycle.review_started)
            self.assertEqual(state.active_cycle.letter, "tool-aware note")
            self.assertEqual(
                [segment["kind"] for segment in state.active_cycle.successor_segments],
                ["assistant", "tool_result"],
            )
            counted = token_count.call_args_list[1].args[2]
            self.assertEqual([message["role"] for message in counted], ["assistant", "tool"])
            self.assertEqual(counted[1]["tool_call_id"], "call_successor")
            review.assert_called_once()

    def test_tool_result_is_deduplicated_before_threshold(self):
        with TemporaryDirectory() as tmp:
            state = ProxyState(
                upstream="http://127.0.0.1:8000",
                log_dir=Path(tmp),
                predecessor_request=None,
                review_max_tokens=2048,
                successor_accum_tokens=10000,
            )
            generation, _ = self.capture_transition(
                state,
                self.sample_request(),
                {"content": "predecessor response"},
            )
            successor = {
                "tool_calls": '{"id":"call_1","type":"function","index":0,"function":{"name":"bash","arguments":"{}"}}'
            }
            request = {
                "messages": [
                    {"role": "tool", "tool_call_id": "call_1", "content": "output"}
                ]
            }
            with patch(
                "opencode_predecessor_letter_proxy.successor_token_count",
                return_value=(100, "upstream", None),
            ) as token_count:
                state.maybe_review(20, successor, False, generation)
                state.note_tool_results(21, request, generation)
                state.note_tool_results(22, request, generation)

            self.assertEqual(len(state.active_cycle.successor_segments), 2)
            self.assertEqual(token_count.call_count, 2)

    def test_unavailable_tokenizer_never_triggers_review(self):
        with TemporaryDirectory() as tmp:
            state = ProxyState(
                upstream="http://127.0.0.1:8000",
                log_dir=Path(tmp),
                predecessor_request=None,
                review_max_tokens=2048,
                successor_accum_tokens=1,
            )
            generation, _ = self.capture_transition(
                state,
                self.sample_request(),
                {"content": "predecessor response"},
            )

            with (
                patch(
                    "opencode_predecessor_letter_proxy.successor_token_count",
                    return_value=(None, "unavailable", "tokenizer offline"),
                ),
                patch(
                    "opencode_predecessor_letter_proxy.run_predecessor_review",
                    return_value="must not be used",
                ) as review,
            ):
                state.maybe_review(20, {"content": "a long successor segment"}, False, generation)

            self.assertFalse(state.active_cycle.review_started)
            review.assert_not_called()

    def test_repeated_compactions_use_the_immediately_previous_generation(self):
        with TemporaryDirectory() as tmp:
            state = ProxyState(
                upstream="http://127.0.0.1:8000",
                log_dir=Path(tmp),
                predecessor_request=None,
                review_max_tokens=2048,
                successor_accum_tokens=1,
            )
            request_g0 = self.sample_request()
            generation, event_index = self.capture_transition(
                state,
                request_g0,
                {"content": "generation zero final response"},
                seq=10,
            )
            self.assertEqual((generation, event_index), (1, 1))

            with (
                patch(
                    "opencode_predecessor_letter_proxy.successor_token_count",
                    return_value=(1, "upstream", None),
                ),
                patch(
                    "opencode_predecessor_letter_proxy.run_predecessor_review",
                    side_effect=["letter from g0", "letter from g1", "letter from g2"],
                ) as review,
            ):
                state.maybe_review(12, {"content": "generation one progress"}, False, 1)
                self.assertEqual(state.take_letter(1), (1, "letter from g0"))

                request_g1 = self.sample_request()
                request_g1["messages"][-1]["content"] = "generation one state"
                generation, event_index = self.capture_transition(
                    state,
                    request_g1,
                    {"content": "generation one final response"},
                    seq=20,
                )
                cycle = state.active_cycle
                self.assertEqual((generation, event_index), (2, 2))
                self.assertEqual(cycle.predecessor_generation, 1)
                self.assertEqual(cycle.successor_generation, 2)
                self.assertEqual(cycle.predecessor_request_body, request_g1)
                self.assertNotEqual(cycle.predecessor_request_body, request_g0)
                self.assertFalse(cycle.review_started)
                self.assertIsNone(cycle.letter)

                state.maybe_review(22, {"content": "generation two progress"}, False, 2)
                self.assertEqual(state.take_letter(2), (2, "letter from g1"))

                request_g2 = self.sample_request()
                request_g2["messages"][-1]["content"] = "generation two state"
                generation, event_index = self.capture_transition(
                    state,
                    request_g2,
                    {"content": "generation two final response"},
                    seq=30,
                )
                cycle = state.active_cycle
                self.assertEqual((generation, event_index), (3, 3))
                self.assertEqual(cycle.predecessor_generation, 2)
                self.assertEqual(cycle.successor_generation, 3)
                self.assertEqual(cycle.predecessor_request_body, request_g2)

                state.maybe_review(32, {"content": "generation three progress"}, False, 3)
                self.assertEqual(state.take_letter(3), (3, "letter from g2"))

            self.assertEqual(review.call_count, 3)
            self.assertEqual(sorted(state.cycles), [1, 2, 3])
            self.assertEqual(
                [state.cycles[index].predecessor_generation for index in sorted(state.cycles)],
                [0, 1, 2],
            )
            accumulation_files = sorted(Path(tmp).glob("generation-*-successor-accumulation-*.json"))
            self.assertEqual(len(accumulation_files), 3)
            self.assertTrue(accumulation_files[0].name.startswith("generation-000001-"))
            self.assertTrue(accumulation_files[1].name.startswith("generation-000002-"))
            self.assertTrue(accumulation_files[2].name.startswith("generation-000003-"))


if __name__ == "__main__":
    unittest.main()
