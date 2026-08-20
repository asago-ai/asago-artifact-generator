"""Tests for Garak gate classification."""

from __future__ import annotations

import unittest
from pathlib import Path

from asago_artifact_generator.extract import extract_scenario, load_scenario
from asago_artifact_generator.garak.artifact_spec import skip_to_artifact_dict
from asago_artifact_generator.garak.classify import (
    classify_garak,
    lookup_surface,
    surface_skip_reason,
)

SCENARIOS = Path(__file__).resolve().parents[1] / "examples" / "scenarios"


class TestClassify(unittest.TestCase):
    def test_entry_point_input_is_user_turn(self):
        ctx = load_scenario(SCENARIOS / "AP-T2-01-28712e.yaml")
        self.assertIn("(input)", ctx.entry_point)
        self.assertEqual(lookup_surface(ctx), "user_turn")

    def test_entry_point_tool_execution_is_tool_return(self):
        ctx = load_scenario(SCENARIOS / "AP-T15-01-832854.yaml")
        self.assertIn("(tool_execution)", ctx.entry_point)
        self.assertEqual(lookup_surface(ctx), "tool_return")
        ctx_family = load_scenario(SCENARIOS / "AP-T2-03-7108ff.yaml")
        self.assertEqual(lookup_surface(ctx_family), "tool_return")

    def test_supply_chain_no_coverage(self):
        for name in ("AP-T17-01-00d278.yaml", "AP-T17-02-8e8e3e.yaml"):
            with self.subTest(name=name):
                ctx = load_scenario(SCENARIOS / name)
                self.assertIn("supply chain", ctx.threat_name.lower())
                self.assertEqual(lookup_surface(ctx), "none")
                skip = surface_skip_reason(ctx)
                self.assertIsNotNone(skip)
                self.assertIn("no coverage", skip.lower())
                self.assertIn("supply chain", skip.lower())
                result, reason = classify_garak(
                    ctx,
                    trigger_tool="process_refund",
                    tools_grounded=True,
                )
                self.assertEqual(result, "skip")
                self.assertIn("supply chain", reason.lower())

    def test_supply_chain_threat_overrides_input_entry_point(self):
        ctx = extract_scenario(
            {
                "scenario_id": "synthetic-sc",
                "scenario_seed_metadata": {
                    "seed_id": "AP-T2-01",
                    "threat_name": "Supply Chain Compromise",
                },
                "narrative": {
                    "entry_point": "natural language customer queries (input)",
                    "summary": "",
                },
            }
        )
        self.assertEqual(lookup_surface(ctx), "none")

    def test_t2_01_full_with_grounded_trigger(self):
        ctx = load_scenario(SCENARIOS / "AP-T2-01-28712e.yaml")
        self.assertIn("process_refund", ctx.quoted_tools)
        result, reason = classify_garak(
            ctx,
            trigger_tool="process_refund",
            tools_grounded=True,
        )
        self.assertEqual(result, "full")
        self.assertIn("supported", reason.lower())

    def test_t2_06_partial_inferred_tools(self):
        ctx = load_scenario(SCENARIOS / "AP-T2-06-85bd56.yaml")
        result, _ = classify_garak(
            ctx,
            trigger_tool="process_refund",
            tools_grounded=False,
        )
        self.assertEqual(result, "partial")

    def test_skip_artifact_has_none_surface(self):
        ctx = load_scenario(SCENARIOS / "AP-T17-02-8e8e3e.yaml")
        data = skip_to_artifact_dict(ctx)
        self.assertEqual(data["injection_surface"], "none")
        self.assertIsNone(data["platform_coverage"])


if __name__ == "__main__":
    unittest.main()
