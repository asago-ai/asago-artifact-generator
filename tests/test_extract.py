"""Tests for scenario extraction."""

from __future__ import annotations

import unittest
from pathlib import Path

from asago_artifact_generator.extract import behavior_spec_text, load_scenario

SCENARIOS = Path(__file__).resolve().parents[1] / "examples" / "scenarios"


class TestExtract(unittest.TestCase):
    def test_t2_06_metadata(self):
        ctx = load_scenario(SCENARIOS / "AP-T2-06-85bd56.yaml")
        self.assertEqual(ctx.seed_id, "AP-T2-06")
        self.assertEqual(ctx.threat_id, "T2")
        self.assertTrue(ctx.narrative_summary)
        self.assertIn("tool_execution", ctx.zone_sequence)

    def test_t17_02_quoted_tools(self):
        ctx = load_scenario(SCENARIOS / "AP-T17-02-8e8e3e.yaml")
        self.assertIn("process_refund", ctx.quoted_tools)
        self.assertIn("modify_payment", ctx.quoted_tools)


class TestBehaviorSpecText(unittest.TestCase):
    def test_legacy_string_passthrough(self):
        text = "@id:AP-T2-01\nFeature: Legacy\n  Background:\n    Given a system"
        self.assertEqual(behavior_spec_text(text), text)

    def test_dict_returns_gherkin_text(self):
        spec = {
            "gherkin_text": "Feature: Structured\n  Background:\n    Given a system",
            "actions": [{"gherkin_keyword": "Given", "text": "a system"}],
        }
        self.assertEqual(behavior_spec_text(spec), spec["gherkin_text"])

    def test_missing_or_empty_spec(self):
        self.assertEqual(behavior_spec_text(None), "")
        self.assertEqual(behavior_spec_text(""), "")

    def test_dict_without_gherkin_text_raises(self):
        spec = {"actions": [{"gherkin_keyword": "Given", "text": "a system"}]}
        with self.assertRaises(ValueError):
            behavior_spec_text(spec)

    def test_dict_with_blank_gherkin_text_raises(self):
        with self.assertRaises(ValueError):
            behavior_spec_text({"gherkin_text": "   ", "actions": []})

    def test_dict_with_non_string_gherkin_text_raises(self):
        with self.assertRaises(ValueError):
            behavior_spec_text({"gherkin_text": 42})

    def test_empty_dict_raises(self):
        with self.assertRaises(ValueError):
            behavior_spec_text({})

    def test_unknown_shapes_raise(self):
        for shape in ([], ["unexpected"], 42, 3.5):
            with self.assertRaises(ValueError):
                behavior_spec_text(shape)


if __name__ == "__main__":
    unittest.main()
