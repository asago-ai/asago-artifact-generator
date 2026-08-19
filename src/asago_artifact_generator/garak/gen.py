"""Garak artifact generation: scenario YAML → runs/{scenario_id}/{scenario_id}-garak.json."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..extract import ScenarioContext
from .classify import lookup_surface, surface_skip_reason
from .gate import gate_from_context
from .probe_spec import (
    GENERATE_ARTIFACT_PROMPT,
    PROBE_VALIDATION_CHECKS,
    SKIP_VALIDATION_CHECKS,
    generate_scenario_probe,
    probe_to_artifact_dict,
    skip_to_artifact_dict,
)
from .spec_io import (
    runs_dir,
    save_garak_artifact,
    save_validation,
)

log = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "scenarios"


@dataclass
class GenResult:
    scenario_id: str
    ok: bool
    gate: str
    gate_reason: str = ""
    artifact_path: str | None = None
    errors: list[str] | None = None


def generate_artifact(
    ctx: ScenarioContext,
    *,
    prompt_path: Path | None = None,
    output_dir: Path | None = None,
    use_llm: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> GenResult:
    """Scenario YAML → runs/{scenario_id}/{scenario_id}-garak.json."""
    prompt = prompt_path or GENERATE_ARTIFACT_PROMPT
    out_base = runs_dir(output_dir)

    skip_reason = surface_skip_reason(ctx)
    if skip_reason:
        log.info("SKIP %s — %s", ctx.scenario_id, skip_reason)
        artifact_path: str | None = None
        if not dry_run:
            save_validation(
                ctx.scenario_id,
                True,
                [],
                out_base,
                checks=SKIP_VALIDATION_CHECKS,
            )
            path = save_garak_artifact(ctx.scenario_id, skip_to_artifact_dict(ctx), out_base)
            artifact_path = str(path)
            log.info("DONE %s → %s (no coverage)", ctx.scenario_id, path.name)
        return GenResult(
            scenario_id=ctx.scenario_id,
            ok=True,
            gate="skip",
            gate_reason=skip_reason,
            artifact_path=artifact_path,
        )

    if not use_llm:
        return GenResult(
            scenario_id=ctx.scenario_id,
            ok=False,
            gate="error",
            errors=["Artifact generation requires LLM (omit --no-llm)"],
        )

    injection_surface = lookup_surface(ctx.seed_id)
    log.info(
        "Classified %s injection_surface=%s",
        ctx.scenario_id,
        injection_surface,
    )

    probe_result = generate_scenario_probe(ctx, prompt, injection_surface=injection_surface)
    save_validation(
        ctx.scenario_id,
        probe_result.ok,
        probe_result.errors,
        out_base,
        checks=PROBE_VALIDATION_CHECKS,
    )

    trigger = ""
    for name in ctx.quoted_tools:
        if any(k in name for k in ("refund", "payment", "modify")):
            trigger = name
            break
    if not trigger and ctx.quoted_tools:
        trigger = ctx.quoted_tools[0]
    gate_result, gate_reason = gate_from_context(
        ctx,
        surface=injection_surface,
        trigger_tool=trigger,
        oracle_kind="forbidden_call" if trigger else "output_string",
        tools_grounded=bool(ctx.quoted_tools),
    )
    if not probe_result.ok:
        detail = "; ".join(probe_result.errors)
        gate_result = "partial" if force else "error"
        gate_reason = f"{gate_reason}; {detail}" if detail else gate_reason

    if not probe_result.ok and not force:
        log.warning("Probe gate failed for %s: %s", ctx.scenario_id, probe_result.errors)
        return GenResult(
            scenario_id=ctx.scenario_id,
            ok=False,
            gate="error",
            gate_reason=gate_reason,
            errors=probe_result.errors,
        )

    if dry_run:
        log.info("DRY-RUN %s — gate=%s (%s)", ctx.scenario_id, gate_result, gate_reason)
        return GenResult(
            scenario_id=ctx.scenario_id,
            ok=True,
            gate=gate_result,
            gate_reason=gate_reason,
        )

    artifact_path = save_garak_artifact(
        ctx.scenario_id,
        probe_to_artifact_dict(ctx, probe_result.probe, platform_coverage=gate_result),
        out_base,
    )
    log.info("DONE %s → %s", ctx.scenario_id, artifact_path.name)
    return GenResult(
        scenario_id=ctx.scenario_id,
        ok=True,
        gate=gate_result,
        gate_reason=gate_reason,
        artifact_path=str(artifact_path),
        errors=probe_result.errors if probe_result.errors else None,
    )


def list_scenario_files() -> list[Path]:
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(EXAMPLES_DIR.glob("AP-*.yaml"))
