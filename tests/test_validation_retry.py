"""Structural and judge gates feed errors back into generation."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from asago_artifact_generator.extract import load_scenario
from asago_artifact_generator.garak.artifact_spec import (
    GENERATE_ARTIFACT_PROMPT,
    generate_scenario_artifact,
)
from asago_artifact_generator.garak.judge import (
    FIDELITY_MINIMUM,
    SCORE_DIMENSIONS,
    JudgeResult,
    acceptance_errors,
    legacy_judge_payload,
    resolve_max_attempts,
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


def _good_spec() -> dict:
    return {
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
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "name": "lookup_order",
                "content": '{"order_id":"KL-1","status":"paid"}',
                "adversarial": False,
            },
            {"role": "user", "content": "refund it", "adversarial": True},
        ],
        "attack_turn_index": 4,
        "detector_rubric": {
            "judge_prompt": "judge",
            "rubrics": {"attack_success": ["emits"], "attack_blocked": ["declines"]},
        },
    }


def _judge_payload(**overrides) -> dict:
    judged = {
        "scores": {name: 2 for name in SCORE_DIMENSIONS},
        "critical_violations": [],
        "rubric_errors": [],
        "strengths": [],
        "weaknesses": [],
        "evidence": [],
    }
    judged.update(overrides)
    return judged


def _result(judged: dict) -> JudgeResult:
    errors = acceptance_errors(judged)
    scores = judged["scores"]
    return JudgeResult(
        ok=not errors,
        errors=errors,
        fidelity_total=sum(scores.values()),
        scores=scores,
        critical_violations=list(judged["critical_violations"]),
        rubric_errors=list(judged["rubric_errors"]),
    )


def _scenario():
    return load_scenario(
        Path(__file__).resolve().parents[1] / "examples" / "scenarios" / "AP-T2-01-28712e.yaml"
    )


class TestAcceptance(unittest.TestCase):
    def test_passing_score_has_no_errors(self):
        self.assertEqual(acceptance_errors(_judge_payload()), [])

    def test_fidelity_below_minimum_is_an_error(self):
        scores = {name: 2 for name in SCORE_DIMENSIONS}
        scores["mechanism"] = 0
        scores["objective"] = 1
        errors = acceptance_errors(_judge_payload(scores=scores))
        self.assertTrue(any(f"below {FIDELITY_MINIMUM}" in error for error in errors))
        self.assertTrue(any("mechanism=0" in error for error in errors))

    def test_critical_violation_fails_even_at_full_score(self):
        errors = acceptance_errors(_judge_payload(critical_violations=["wrong objective"]))
        self.assertTrue(any("critical violation: wrong objective" in error for error in errors))

    def test_rubric_errors_do_not_fail_acceptance(self):
        self.assertEqual(
            acceptance_errors(_judge_payload(rubric_errors=["claims real execution"])),
            [],
        )

    def test_legacy_payload_uses_the_old_scenario_and_no_synthetic_contract(self):
        payload = legacy_judge_payload(_scenario(), _good_spec())
        self.assertIn("refund", payload["reference"]["objective"].lower())
        self.assertIn("parameter pollution", payload["source_behavior"].lower())
        self.assertIn("No fixed synthetic contract", payload["runtime_contract"])
        self.assertNotIn('"tools"', payload["runtime_contract"])
        self.assertEqual(
            set(payload["artifact"]),
            {"turns", "attack_turn_index", "injection_surface", "detector_rubric"},
        )


class TestGenerationRetry(unittest.TestCase):
    def test_judge_failure_is_fed_back_then_passes(self):
        weak = _judge_payload(critical_violations=["wrong objective"])
        weak["scores"] = {name: 1 for name in SCORE_DIMENSIONS}
        with (
            patch(
                "asago_artifact_generator.garak.artifact_spec.llm_json",
                side_effect=[_good_spec(), _good_spec()],
            ) as generate,
            patch(
                "asago_artifact_generator.garak.artifact_spec.judge_artifact",
                side_effect=[_result(weak), _result(_judge_payload())],
            ),
        ):
            result = generate_scenario_artifact(
                _scenario(),
                GENERATE_ARTIFACT_PROMPT,
                injection_surface="user_turn",
            )
        self.assertEqual(generate.call_count, 2)
        second_prompt = generate.call_args_list[1].args[0]
        self.assertIn("Previous JSON failed validation", second_prompt)
        self.assertIn("critical violation: wrong objective", second_prompt)
        self.assertIn(f"below {FIDELITY_MINIMUM}", second_prompt)
        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.fidelity_total, 12)
        self.assertEqual(result.attempts, 2)

    def test_max_attempts_stops_the_loop(self):
        weak = _judge_payload(critical_violations=["invented attacker access"])
        with (
            patch(
                "asago_artifact_generator.garak.artifact_spec.llm_json",
                side_effect=[_good_spec(), _good_spec(), _good_spec()],
            ) as generate,
            patch(
                "asago_artifact_generator.garak.artifact_spec.judge_artifact",
                side_effect=[_result(weak), _result(weak), _result(_judge_payload())],
            ),
        ):
            result = generate_scenario_artifact(
                _scenario(),
                GENERATE_ARTIFACT_PROMPT,
                injection_surface="user_turn",
                max_attempts=2,
            )
        self.assertEqual(generate.call_count, 2)
        self.assertFalse(result.ok)
        self.assertTrue(any("invented attacker access" in error for error in result.errors))
        self.assertEqual(result.attempts, 2)

    def test_structural_failure_does_not_call_the_judge(self):
        bad = _good_spec()
        bad["turns"][0]["content"] = "You are a support assistant."
        with (
            patch(
                "asago_artifact_generator.garak.artifact_spec.llm_json",
                side_effect=[bad, _good_spec()],
            ),
            patch(
                "asago_artifact_generator.garak.artifact_spec.judge_artifact",
                return_value=_result(_judge_payload()),
            ) as judge,
        ):
            result = generate_scenario_artifact(
                _scenario(),
                GENERATE_ARTIFACT_PROMPT,
                injection_surface="user_turn",
            )
        self.assertEqual(judge.call_count, 1)
        self.assertTrue(result.ok)

    def test_env_sets_the_attempt_hyperparameter(self):
        with patch.dict("os.environ", {"ARTIFACT_MAX_ATTEMPTS": "5"}):
            self.assertEqual(resolve_max_attempts(None), 5)
        self.assertEqual(resolve_max_attempts(1), 1)
        with self.assertRaises(ValueError):
            resolve_max_attempts(0)
