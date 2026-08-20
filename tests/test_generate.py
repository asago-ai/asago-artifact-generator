"""generate_artifact success/failure contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asago_artifact_generator.extract import load_scenario
from asago_artifact_generator.garak.gen import generate_artifact
from asago_artifact_generator.garak.probe_spec import (
    DetectorRubric,
    ProbeBuildResult,
    ProbeTurn,
    ScenarioProbe,
)


def _failed_probe(ctx) -> ProbeBuildResult:
    probe = ScenarioProbe(
        scenario_id=ctx.scenario_id,
        probe_name="test",
        injection_surface="user_turn",
        turns=[
            ProbeTurn(role="system", content="- lookup_order(order_id): lookup"),
            ProbeTurn(role="user", content="hi"),
            ProbeTurn(role="assistant", content="done", adversarial=True),
        ],
        attack_turn_index=2,
        detector_rubric=DetectorRubric(
            judge_prompt="j",
            rubrics={"attack_success": ["s"], "attack_blocked": ["b"]},
        ),
    )
    return ProbeBuildResult(
        probe=probe,
        ok=False,
        errors=[
            "turn 5: 1 call(s) not immediately followed by 1 tool turn(s)",
            "last turn is 'assistant', expected 'user' for user_turn",
        ],
    )


class TestGenerateArtifactForce(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = load_scenario(
            Path(__file__).resolve().parents[1] / "examples" / "scenarios" / "AP-T2-01-28712e.yaml"
        )
        self.failed = _failed_probe(self.ctx)

    def test_force_writes_artifact_but_coverage_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "asago_artifact_generator.garak.gen.generate_scenario_probe",
                return_value=self.failed,
            ):
                result = generate_artifact(self.ctx, output_dir=Path(tmp), force=True)
            self.assertFalse(result.ok)
            self.assertEqual(result.gate, "full")
            self.assertEqual(result.gate_reason, "fully supported")
            self.assertIsNotNone(result.artifact_path)
            artifact = Path(result.artifact_path)
            self.assertTrue(artifact.is_file())
            data = json.loads(artifact.read_text())
            self.assertEqual(data["platform_coverage"], "full")
            validation = json.loads((artifact.parent / "validation.json").read_text())
            self.assertFalse(validation["ok"])

    def test_without_force_keeps_coverage_and_does_not_write_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "asago_artifact_generator.garak.gen.generate_scenario_probe",
                return_value=self.failed,
            ):
                result = generate_artifact(self.ctx, output_dir=Path(tmp), force=False)
            self.assertFalse(result.ok)
            self.assertEqual(result.gate, "full")
            self.assertEqual(result.gate_reason, "fully supported")
            self.assertIsNone(result.artifact_path)
            run_dir = Path(tmp) / self.ctx.scenario_id
            self.assertFalse((run_dir / f"{self.ctx.scenario_id}-garak.json").exists())
            validation = json.loads((run_dir / "validation.json").read_text())
            self.assertFalse(validation["ok"])

    def test_no_llm_keeps_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_artifact(self.ctx, output_dir=Path(tmp), use_llm=False)
        self.assertFalse(result.ok)
        self.assertEqual(result.gate, "full")
        self.assertIsNone(result.artifact_path)
        self.assertTrue(any("LLM" in e for e in (result.errors or [])))


if __name__ == "__main__":
    unittest.main()
