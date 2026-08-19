"""OpenAI message conversion for native and legacy tool turns."""

from __future__ import annotations

import json
import unittest

from asago_artifact_generator.garak.plugins.scenario_messages import (
    extract_openai_tools,
    turns_to_openai_messages,
)


class TestExtractOpenaiTools(unittest.TestCase):
    def test_dash_name_lines(self):
        system = (
            "You are a support assistant.\n"
            "- lookup_order(order_id): Fetch an order\n"
            "- process_refund(order_id, amount): Issue a refund\n"
        )
        tools = extract_openai_tools(system)
        names = [t["function"]["name"] for t in tools]
        self.assertEqual(names, ["lookup_order", "process_refund"])
        lookup = tools[0]["function"]["parameters"]
        self.assertIn("order_id", lookup["properties"])
        self.assertIn("order_id", lookup["required"])

    def test_backtick_fallback(self):
        system = "Tools: `look_up_account_details(account_id: str, include_history: bool)`"
        tools = extract_openai_tools(system)
        self.assertEqual(tools[0]["function"]["name"], "look_up_account_details")
        props = tools[0]["function"]["parameters"]["properties"]
        self.assertEqual(props["include_history"]["type"], "boolean")


class TestTurnsToOpenaiMessages(unittest.TestCase):
    def test_native_tool_calls_pass_through(self):
        turns = [
            {"role": "system", "content": "- lookup_order(order_id): x"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Let me pull that up.",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "lookup_order",
                            "arguments": {"order_id": "KL-1"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "name": "lookup_order",
                "content": '{"status":"paid"}',
            },
        ]
        messages = turns_to_openai_messages(turns)
        assistant = messages[2]
        self.assertEqual(assistant["content"], "Let me pull that up.")
        args = assistant["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(args, str)
        self.assertEqual(json.loads(args), {"order_id": "KL-1"})
        tool = messages[3]
        self.assertEqual(tool["role"], "tool")
        self.assertEqual(tool["tool_call_id"], "call_001")
        self.assertEqual(tool["name"], "lookup_order")
        self.assertEqual(len(messages), 4)

    def test_parallel_native_batch(self):
        turns = [
            {
                "role": "assistant",
                "content": "Checking both.",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {"name": "lookup_order", "arguments": {}},
                    },
                    {
                        "id": "call_002",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": {"path": ""}},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "name": "lookup_order",
                "content": "{}",
            },
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "name": "list_files",
                "content": "[]",
            },
        ]
        messages = turns_to_openai_messages(turns)
        self.assertEqual(len(messages), 3)
        self.assertEqual(len(messages[0]["tool_calls"]), 2)
        self.assertEqual(messages[1]["tool_call_id"], "call_001")
        self.assertEqual(messages[2]["tool_call_id"], "call_002")
        self.assertEqual(messages[2]["role"], "tool")

    def test_legacy_synthesizes_empty_arguments(self):
        turns = [
            {"role": "assistant", "content": "Let me pull that up."},
            {
                "role": "tool",
                "tool_name": "lookup_order",
                "content": '{"status":"paid"}',
            },
        ]
        messages = turns_to_openai_messages(turns)
        self.assertEqual(len(messages), 2)
        call = messages[0]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "lookup_order")
        self.assertEqual(call["function"]["arguments"], "{}")
        self.assertEqual(messages[1]["tool_call_id"], call["id"])
        self.assertEqual(messages[1]["name"], "lookup_order")


if __name__ == "__main__":
    unittest.main()
