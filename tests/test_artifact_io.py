"""Artifact path and serialization tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import asago_artifact_generator.llm as llm_mod
from asago_artifact_generator.extract import ScenarioContext, load_scenario
from asago_artifact_generator.garak.artifact_spec import (
    ARTIFACT_DISCLOSURE,
    ARTIFACT_VALIDATION_CHECKS,
    ArtifactTurn,
    DetectorRubric,
    ScenarioArtifact,
    artifact_to_dict,
    gate_artifact_errors,
    skip_to_artifact_dict,
)
from asago_artifact_generator.garak.spec_io import (
    garak_artifact_path,
    save_validation,
    scenario_id_from_garak_path,
    validation_path,
)


class TestGarakArtifact(unittest.TestCase):
    def _minimal_ctx(self) -> ScenarioContext:
        return load_scenario(
            Path(__file__).resolve().parents[1] / "examples" / "scenarios" / "AP-T2-01-28712e.yaml"
        )

    def test_garak_artifact_path(self):
        p = garak_artifact_path("AP-T7-03-f2bbd2", Path("runs"))
        self.assertEqual(p, Path("runs/AP-T7-03-f2bbd2/AP-T7-03-f2bbd2-garak.json"))

    def test_validation_path(self):
        p = validation_path("AP-T7-03-f2bbd2", Path("runs"))
        self.assertEqual(p, Path("runs/AP-T7-03-f2bbd2/validation.json"))

    def test_scenario_id_from_garak_path(self):
        self.assertEqual(
            scenario_id_from_garak_path(Path("runs/AP-T7-03-f2bbd2/AP-T7-03-f2bbd2-garak.json")),
            "AP-T7-03-f2bbd2",
        )
        self.assertEqual(
            scenario_id_from_garak_path(Path("runs/AP-T7-03-f2bbd2/garak.json")),
            "AP-T7-03-f2bbd2",
        )
        self.assertEqual(
            scenario_id_from_garak_path(Path("runs/AP-T7-03-f2bbd2-garak.json")),
            "AP-T7-03-f2bbd2",
        )

    def test_save_validation_includes_checks(self):
        path = save_validation(
            "AP-T2-01-test",
            True,
            [],
            Path("runs"),
            checks=ARTIFACT_VALIDATION_CHECKS,
        )
        data = json.loads(path.read_text())
        self.assertTrue(data["ok"])
        self.assertIn("detector_rubric", data["checks"])
        self.assertEqual(data["errors"], [])

    def test_skip_to_artifact_dict(self):
        ctx = load_scenario(
            Path(__file__).resolve().parents[1]
            / "examples"
            / "scenarios"
            / "AP-T17-02-8e8e3e.yaml"
        )
        data = skip_to_artifact_dict(ctx)
        self.assertEqual(data["scenario_id"], ctx.scenario_id)
        self.assertEqual(data["injection_surface"], "none")
        self.assertIsNone(data["platform_coverage"])
        self.assertEqual(data["disclosure"], ARTIFACT_DISCLOSURE)
        self.assertIsNone(data["model"])
        self.assertIn("timestamp", data)
        self.assertIn("summary", data["narrative"])
        self.assertNotIn("turns", data)
        self.assertNotIn("detector_rubric", data)

    def test_load_skip_artifact_without_turns(self):
        import tempfile

        from asago_artifact_generator.garak.spec_io import load_garak_artifact, save_garak_artifact

        ctx = load_scenario(
            Path(__file__).resolve().parents[1]
            / "examples"
            / "scenarios"
            / "AP-T17-01-00d278.yaml"
        )
        data = skip_to_artifact_dict(ctx)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_garak_artifact(ctx.scenario_id, data, Path(tmp))
            loaded = load_garak_artifact(path)
        self.assertEqual(loaded["injection_surface"], "none")
        self.assertIsNone(loaded["platform_coverage"])
        self.assertNotIn("turns", loaded)

    def test_load_rejects_non_artifact(self):
        import tempfile

        from asago_artifact_generator.garak.spec_io import load_garak_artifact

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_garak_artifact(path)

    def test_scenario_to_yaml_strips_narrative(self):
        ctx = self._minimal_ctx()
        from asago_artifact_generator.garak.artifact_spec import scenario_to_yaml

        prompt_yaml = scenario_to_yaml(ctx)
        self.assertNotIn("narrative:", prompt_yaml)
        full_yaml = scenario_to_yaml(ctx, include_narrative=True)
        self.assertIn("narrative:", full_yaml)

    def test_artifact_to_dict_shape(self):
        ctx = self._minimal_ctx()
        artifact = ScenarioArtifact(
            scenario_id=ctx.scenario_id,
            injection_surface="user_turn",
            turns=[
                ArtifactTurn(role="system", content="sys"),
                ArtifactTurn(role="user", content="attack", adversarial=True),
            ],
            attack_turn_index=1,
            detector_rubric=DetectorRubric(
                judge_prompt="judge",
                rubrics={"attack_success": ["emits refund"], "attack_blocked": ["declines"]},
            ),
        )
        data = artifact_to_dict(ctx, artifact, platform_coverage="full")
        self.assertEqual(data["scenario_id"], ctx.scenario_id)
        self.assertEqual(data["injection_surface"], "user_turn")
        self.assertEqual(list(data["narrative"].keys()), ["summary"])
        self.assertIn("summary", data["narrative"])
        self.assertEqual(data["platform_coverage"], "full")
        self.assertEqual(data["disclosure"], ARTIFACT_DISCLOSURE)
        self.assertEqual(data["model"], llm_mod.MODEL)
        self.assertRegex(data["timestamp"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        self.assertNotIn("probe_name", data)
        self.assertNotIn("metadata", data)

    def test_artifact_to_dict_omits_empty_tool_name(self):
        ctx = self._minimal_ctx()
        artifact = ScenarioArtifact(
            scenario_id="AP-T2-01-test",
            injection_surface="user_turn",
            turns=[
                ArtifactTurn(role="system", content="sys"),
                ArtifactTurn(role="user", content="attack", adversarial=True),
            ],
            attack_turn_index=1,
            detector_rubric=DetectorRubric(
                judge_prompt="judge",
                rubrics={"attack_success": ["emits refund"], "attack_blocked": ["declines"]},
            ),
        )
        data = artifact_to_dict(ctx, artifact, platform_coverage="partial")
        self.assertNotIn("tool_name", data["turns"][0])
        self.assertEqual(data["attack_turn_index"], 1)
        self.assertIn("detector_rubric", data)

    def test_artifact_to_dict_uses_live_configured_model(self):
        from asago_artifact_generator.llm import configure_llm

        prev = (llm_mod.PROVIDER, llm_mod.BASE_URL, llm_mod.MODEL)
        self.addCleanup(lambda: configure_llm(provider=prev[0], base_url=prev[1], model=prev[2]))
        configure_llm(provider="openai", model="gemma-4-26b")
        ctx = self._minimal_ctx()
        artifact = ScenarioArtifact(
            scenario_id=ctx.scenario_id,
            injection_surface="user_turn",
            turns=[
                ArtifactTurn(role="system", content="sys"),
                ArtifactTurn(role="user", content="attack", adversarial=True),
            ],
            attack_turn_index=1,
            detector_rubric=DetectorRubric(
                judge_prompt="judge",
                rubrics={"attack_success": ["emits refund"], "attack_blocked": ["declines"]},
            ),
        )
        data = artifact_to_dict(ctx, artifact, platform_coverage="full")
        self.assertEqual(data["model"], "gemma-4-26b")
        self.assertEqual(data["model"], llm_mod.MODEL)

    def test_gate_rejects_surface_mismatch(self):
        artifact = ScenarioArtifact(
            scenario_id="AP-T11-02-test",
            injection_surface="user_turn",
            turns=[
                ArtifactTurn(role="system", content="sys"),
                ArtifactTurn(role="user", content="attack", adversarial=True),
            ],
            attack_turn_index=1,
            detector_rubric=DetectorRubric(
                judge_prompt="judge",
                rubrics={"attack_success": ["emits"], "attack_blocked": ["declines"]},
            ),
        )
        errors = gate_artifact_errors(artifact, expected_surface="tool_return")
        self.assertTrue(any("injection_surface" in e for e in errors))

    def test_gate_requires_tool_role_for_tool_return(self):
        artifact = ScenarioArtifact(
            scenario_id="AP-T11-02-test",
            injection_surface="tool_return",
            turns=[
                ArtifactTurn(role="system", content="sys"),
                ArtifactTurn(role="user", content="attack", adversarial=True),
            ],
            attack_turn_index=1,
            detector_rubric=DetectorRubric(
                judge_prompt="judge",
                rubrics={"attack_success": ["emits"], "attack_blocked": ["declines"]},
            ),
        )
        errors = gate_artifact_errors(artifact, expected_surface="tool_return")
        self.assertIn("tool_return artifact must end with role=tool", errors)


if __name__ == "__main__":
    unittest.main()
