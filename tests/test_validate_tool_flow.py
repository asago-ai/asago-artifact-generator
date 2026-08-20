"""validate_tool_flow checks on raw LLM artifact JSON."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from asago_artifact_generator.extract import load_scenario
from asago_artifact_generator.garak.artifact_spec import (
    GENERATE_ARTIFACT_PROMPT,
    ArtifactTurn,
    DetectorRubric,
    ScenarioArtifact,
    artifact_to_dict,
    generate_scenario_artifact,
    validate_tool_flow,
)

SYSTEM = (
    "You are a support assistant.\n"
    "- lookup_order(order_id): Fetch an order by id\n"
    "- process_refund(order_id, amount): Issue a refund\n"
)

LOOKUP_CALL = {
    "id": "call_001",
    "type": "function",
    "function": {"name": "lookup_order", "arguments": {"order_id": "KL-1"}},
}

LOOKUP_RESULT = {
    "role": "tool",
    "tool_call_id": "call_001",
    "name": "lookup_order",
    "content": '{"order_id":"KL-1","status":"paid"}',
    "adversarial": False,
}


def _happy_spec(**overrides) -> dict:
    spec = {
        "scenario_id": "t",
        "injection_surface": "user_turn",
        "turns": [
            {"role": "system", "content": SYSTEM, "adversarial": False},
            {"role": "user", "content": "hi", "adversarial": False},
            {
                "role": "assistant",
                "content": "Let me pull that up.",
                "tool_calls": [copy.deepcopy(LOOKUP_CALL)],
                "adversarial": False,
            },
            copy.deepcopy(LOOKUP_RESULT),
            {"role": "user", "content": "refund it", "adversarial": True},
        ],
        "attack_turn_index": 4,
    }
    spec.update(overrides)
    return spec


class TestValidateToolFlow(unittest.TestCase):
    def test_happy_path(self):
        self.assertEqual(validate_tool_flow(_happy_spec()), [])

    def test_unused_declared_tool_is_ok(self):
        errs = validate_tool_flow(_happy_spec())
        self.assertEqual(errs, [])

    def test_empty_turns(self):
        errs = validate_tool_flow({"turns": [], "injection_surface": "user_turn"})
        self.assertEqual(errs, ["turns must not be empty"])

    def test_no_tool_declarations(self):
        spec = _happy_spec()
        spec["turns"][0]["content"] = "You are a support assistant."
        errs = validate_tool_flow(spec)
        self.assertTrue(any("declares no tools" in e for e in errs))

    def test_tool_calls_on_non_assistant(self):
        spec = _happy_spec()
        spec["turns"][1]["tool_calls"] = [copy.deepcopy(LOOKUP_CALL)]
        errs = validate_tool_flow(spec)
        self.assertTrue(any("tool_calls on a user turn" in e for e in errs))

    def test_tool_turn_missing_tool_call_id(self):
        spec = _happy_spec()
        del spec["turns"][3]["tool_call_id"]
        errs = validate_tool_flow(spec)
        self.assertTrue(any("missing tool_call_id" in e for e in errs))

    def test_calls_not_followed_by_tool_turns(self):
        spec = _happy_spec()
        spec["turns"][2]["tool_calls"] = [
            copy.deepcopy(LOOKUP_CALL),
            {
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "process_refund",
                    "arguments": {"order_id": "KL-1", "amount": 10},
                },
            },
        ]
        errs = validate_tool_flow(spec)
        self.assertTrue(any("not immediately followed" in e for e in errs))

    def test_id_mismatch(self):
        spec = _happy_spec()
        spec["turns"][3]["tool_call_id"] = "call_999"
        errs = validate_tool_flow(spec)
        self.assertTrue(any("result id" in e for e in errs))

    def test_name_mismatch(self):
        spec = _happy_spec()
        spec["turns"][3]["name"] = "process_refund"
        errs = validate_tool_flow(spec)
        self.assertTrue(any("result name" in e for e in errs))

    def test_arguments_must_be_object(self):
        spec = _happy_spec()
        spec["turns"][2]["tool_calls"][0]["function"]["arguments"] = '{"order_id":"KL-1"}'
        errs = validate_tool_flow(spec)
        self.assertTrue(any("arguments must be an object" in e for e in errs))

    def test_undeclared_tool_name(self):
        spec = _happy_spec()
        spec["turns"][2]["tool_calls"][0]["function"]["name"] = "read_file"
        spec["turns"][3]["name"] = "read_file"
        errs = validate_tool_flow(spec)
        self.assertTrue(any("not declared in system prompt" in e for e in errs))

    def test_orphan_tool_result(self):
        spec = _happy_spec()
        spec["turns"].insert(
            2,
            {
                "role": "tool",
                "tool_call_id": "call_orphan",
                "name": "lookup_order",
                "content": "{}",
                "adversarial": False,
            },
        )
        spec["attack_turn_index"] = 5
        errs = validate_tool_flow(spec)
        self.assertTrue(any("no preceding call" in e for e in errs))

    def test_last_turn_assistant(self):
        spec = _happy_spec()
        spec["turns"][-1] = {
            "role": "assistant",
            "content": "done",
            "adversarial": True,
        }
        errs = validate_tool_flow(spec)
        self.assertTrue(any("last turn is 'assistant'" in e for e in errs))

    def test_tool_return_must_end_on_tool(self):
        spec = _happy_spec(injection_surface="tool_return")
        errs = validate_tool_flow(spec)
        self.assertTrue(any("expected 'tool' for tool_return" in e for e in errs))

    def test_last_turn_not_adversarial(self):
        spec = _happy_spec()
        spec["turns"][-1]["adversarial"] = False
        errs = validate_tool_flow(spec)
        self.assertIn("last turn is not the adversarial turn", errs)

    def test_attack_turn_index_not_last(self):
        spec = _happy_spec(attack_turn_index=0)
        errs = validate_tool_flow(spec)
        self.assertIn("attack_turn_index does not point at the last turn", errs)

    def test_parallel_batch_of_two(self):
        spec = _happy_spec()
        spec["turns"][2]["tool_calls"] = [
            copy.deepcopy(LOOKUP_CALL),
            {
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "process_refund",
                    "arguments": {"order_id": "KL-1", "amount": 12.5},
                },
            },
        ]
        spec["turns"][3:4] = [
            copy.deepcopy(LOOKUP_RESULT),
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "name": "process_refund",
                "content": '{"ok": true}',
                "adversarial": False,
            },
        ]
        spec["attack_turn_index"] = 5
        self.assertEqual(validate_tool_flow(spec), [])

    def test_parallel_batch_of_three_not_orphaned(self):
        spec = _happy_spec()
        spec["turns"][0]["content"] = SYSTEM + "- list_files(path): List a directory\n"
        spec["turns"][2]["tool_calls"] = [
            copy.deepcopy(LOOKUP_CALL),
            {
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "process_refund",
                    "arguments": {"order_id": "KL-1", "amount": 1},
                },
            },
            {
                "id": "call_003",
                "type": "function",
                "function": {"name": "list_files", "arguments": {"path": ""}},
            },
        ]
        spec["turns"][3:4] = [
            copy.deepcopy(LOOKUP_RESULT),
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "name": "process_refund",
                "content": "{}",
                "adversarial": False,
            },
            {
                "role": "tool",
                "tool_call_id": "call_003",
                "name": "list_files",
                "content": "[]",
                "adversarial": False,
            },
        ]
        spec["attack_turn_index"] = 6
        self.assertEqual(validate_tool_flow(spec), [])

    def test_artifact_to_dict_keeps_tool_fields(self):
        ctx = load_scenario(
            Path(__file__).resolve().parents[1] / "examples" / "scenarios" / "AP-T2-01-28712e.yaml"
        )
        spec = _happy_spec()
        artifact = ScenarioArtifact(
            scenario_id=ctx.scenario_id,
            injection_surface="user_turn",
            turns=[ArtifactTurn.model_validate(t) for t in spec["turns"]],
            attack_turn_index=4,
            detector_rubric=DetectorRubric(
                judge_prompt="judge",
                rubrics={"attack_success": ["emits"], "attack_blocked": ["declines"]},
            ),
        )
        data = artifact_to_dict(ctx, artifact, platform_coverage="full")
        assistant = data["turns"][2]
        tool = data["turns"][3]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_001")
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"],
            {"order_id": "KL-1"},
        )
        self.assertEqual(tool["tool_call_id"], "call_001")
        self.assertEqual(tool["name"], "lookup_order")
        self.assertEqual(tool["tool_name"], "lookup_order")

    def test_name_alias_fills_tool_name(self):
        turn = ArtifactTurn.model_validate(
            {
                "role": "tool",
                "content": "{}",
                "name": "lookup_order",
                "tool_call_id": "call_001",
            }
        )
        self.assertEqual(turn.tool_name, "lookup_order")
        self.assertEqual(turn.name, "lookup_order")


class TestGenerateScenarioArtifactFlow(unittest.TestCase):
    def test_retries_when_flow_fails_then_succeeds(self):
        ctx = load_scenario(
            Path(__file__).resolve().parents[1] / "examples" / "scenarios" / "AP-T2-01-28712e.yaml"
        )
        bad = _happy_spec()
        bad["turns"][0]["content"] = "You are a support assistant."
        good = _happy_spec()
        good["detector_rubric"] = {
            "judge_prompt": "judge",
            "rubrics": {"attack_success": ["emits"], "attack_blocked": ["declines"]},
        }
        with patch(
            "asago_artifact_generator.garak.artifact_spec.llm_json",
            side_effect=[bad, good],
        ) as mocked:
            result = generate_scenario_artifact(
                ctx, GENERATE_ARTIFACT_PROMPT, injection_surface="user_turn"
            )
        self.assertEqual(mocked.call_count, 2)
        first_prompt = mocked.call_args_list[0].args[0]
        second_prompt = mocked.call_args_list[1].args[0]
        self.assertNotIn("Previous JSON failed validation", first_prompt)
        self.assertIn("Previous JSON failed validation", second_prompt)
        self.assertIn("declares no tools", second_prompt)
        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.artifact.turns[2].tool_calls[0].id, "call_001")

    def test_retries_when_gate_fails_then_succeeds(self):
        ctx = load_scenario(
            Path(__file__).resolve().parents[1] / "examples" / "scenarios" / "AP-T2-01-28712e.yaml"
        )
        bad = _happy_spec()
        bad["detector_rubric"] = {
            "judge_prompt": "",
            "rubrics": {"attack_success": [], "attack_blocked": ["declines"]},
        }
        good = _happy_spec()
        good["detector_rubric"] = {
            "judge_prompt": "judge",
            "rubrics": {"attack_success": ["emits"], "attack_blocked": ["declines"]},
        }
        with patch(
            "asago_artifact_generator.garak.artifact_spec.llm_json",
            side_effect=[bad, good],
        ) as mocked:
            result = generate_scenario_artifact(
                ctx, GENERATE_ARTIFACT_PROMPT, injection_surface="user_turn"
            )
        self.assertEqual(mocked.call_count, 2)
        second_prompt = mocked.call_args_list[1].args[0]
        self.assertIn("Previous JSON failed validation", second_prompt)
        self.assertIn("judge_prompt must not be empty", second_prompt)
        self.assertIn("attack_success must not be empty", second_prompt)
        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()
