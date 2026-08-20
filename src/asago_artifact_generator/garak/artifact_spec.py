"""One-shot Garak artifact generation from full scenario YAML (generate_artifact.md)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .. import llm as llm_mod
from ..extract import ScenarioContext
from ..llm import llm_json
from .classify import lookup_surface

log = logging.getLogger(__name__)

GENERATE_ARTIFACT_PROMPT = Path(__file__).resolve().parent / "prompts" / "generate_artifact.md"

TOOL_DECL = re.compile(r"^- (\w+)\(", re.M)
MAX_GENERATION_ATTEMPTS = 3

ARTIFACT_VALIDATION_CHECKS = (
    "Artifact structural gate after LLM generation: validate_tool_flow checks Chat "
    "Completions tool pairing (tool_calls immediately followed by matching role=tool "
    "turns), '- name(params):' tool declarations on the system turn, called names "
    "subseteq declared names, last-turn injection role, adversarial last turn, and "
    "attack_turn_index; detector_rubric has non-empty judge_prompt, attack_success, "
    "and attack_blocked lists; injection_surface matches the classified surface; "
    "Pydantic schema keeps tool_calls, tool_call_id, and name on turns."
)

SKIP_VALIDATION_CHECKS = (
    "Pre-plan gate: classified injection surface must be writable by Garak "
    "(user_turn or tool_return). Supply chain threats and other unwritable "
    "surfaces skip LLM generation (no coverage)."
)

ARTIFACT_DISCLOSURE = "This artifact contains AI generated content"

_LLM_SYSTEM = (
    "You convert threat scenarios into Garak artifact JSON "
    "(transcript turns plus detector_rubric). Respond with valid JSON only."
)


def validate_tool_flow(spec: dict) -> list[str]:
    """Structural checks on raw LLM artifact JSON, before metadata merge."""
    if not isinstance(spec, dict):
        return ["artifact must be an object"]
    turns = spec.get("turns")
    if not isinstance(turns, list) or not turns:
        return ["turns must not be empty"]

    errs: list[str] = []
    first = turns[0] if isinstance(turns[0], dict) else {}
    declared = set(TOOL_DECL.findall(str(first.get("content") or "")))
    if not declared:
        errs.append("system prompt declares no tools in '- name(params):' form")

    seen_ids: set[str] = set()
    i = 0
    while i < len(turns):
        t = turns[i] if isinstance(turns[i], dict) else {}
        calls = t.get("tool_calls") or []
        role = t.get("role")

        if calls and role != "assistant":
            errs.append(f"turn {i}: tool_calls on a {role} turn")
        if role == "tool" and not t.get("tool_call_id"):
            errs.append(f"turn {i}: tool turn missing tool_call_id")

        if calls:
            results = turns[i + 1 : i + 1 + len(calls)]
            if len(results) < len(calls) or any(
                not isinstance(r, dict) or r.get("role") != "tool" for r in results
            ):
                errs.append(
                    f"turn {i}: {len(calls)} call(s) not immediately followed "
                    f"by {len(calls)} tool turn(s)"
                )
            else:
                for c, r in zip(calls, results, strict=False):
                    try:
                        cid, name = c["id"], c["function"]["name"]
                    except (KeyError, TypeError):
                        errs.append(f"turn {i}: malformed tool_call")
                        continue
                    if cid in seen_ids:
                        errs.append(f"turn {i}: duplicate tool_call_id {cid}")
                    seen_ids.add(cid)
                    if not r.get("tool_call_id"):
                        errs.append(f"turn {i}: tool turn missing tool_call_id")
                    elif r.get("tool_call_id") != cid:
                        errs.append(f"turn {i}: result id {r.get('tool_call_id')} != {cid}")
                    if r.get("name") != name:
                        errs.append(f"turn {i}: result name {r.get('name')} != {name}")
                    if name not in declared:
                        errs.append(f"turn {i}: '{name}' not declared in system prompt")
                    fn = c.get("function") if isinstance(c, dict) else None
                    args = fn.get("arguments") if isinstance(fn, dict) else None
                    if not isinstance(args, dict):
                        errs.append(f"turn {i}: arguments must be an object")
                i += len(calls)
        i += 1

    for j, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        if t.get("role") == "tool" and t.get("tool_call_id") not in seen_ids:
            errs.append(f"turn {j}: tool result with no preceding call")

    last = turns[-1] if isinstance(turns[-1], dict) else {}
    surface = spec.get("injection_surface")
    expected = "tool" if surface == "tool_return" else "user"
    if last.get("role") != expected:
        errs.append(f"last turn is '{last.get('role')}', expected '{expected}' for {surface}")
    if not last.get("adversarial"):
        errs.append("last turn is not the adversarial turn")
    if spec.get("attack_turn_index") != len(turns) - 1:
        errs.append("attack_turn_index does not point at the last turn")

    return errs


class ToolCallFunction(BaseModel):
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ArtifactTurn(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_name: str = ""
    name: str = ""
    tool_call_id: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    adversarial: bool = False

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _coerce_tool_calls(cls, value: Any) -> Any:
        return value or []

    @model_validator(mode="after")
    def _alias_tool_identity(self) -> ArtifactTurn:
        if self.role == "tool":
            if self.name and not self.tool_name:
                self.tool_name = self.name
            elif self.tool_name and not self.name:
                self.name = self.tool_name
        return self


class DetectorRubric(BaseModel):
    judge_prompt: str = ""
    context: str = ""
    rubrics: dict[str, list[str]] = Field(default_factory=dict)


class ScenarioArtifact(BaseModel):
    scenario_id: str = ""
    injection_surface: Literal["user_turn", "tool_return", "system_prompt"] = "user_turn"
    turns: list[ArtifactTurn] = Field(default_factory=list)
    attack_turn_index: int = Field(ge=0)
    detector_rubric: DetectorRubric = Field(default_factory=DetectorRubric)

    @model_validator(mode="after")
    def _validate_transcript(self) -> ScenarioArtifact:
        if not self.turns:
            raise ValueError("turns must not be empty")
        for i, turn in enumerate(self.turns):
            if turn.role == "tool" and not (turn.name or turn.tool_name):
                raise ValueError(f"tool turn at index {i} missing name")
            if turn.role != "tool" and turn.tool_name:
                raise ValueError(f"non-tool turn at index {i} must not set tool_name")
        return self


@dataclass
class ArtifactBuildResult:
    artifact: ScenarioArtifact
    ok: bool
    errors: list[str]


def scenario_to_yaml(ctx: ScenarioContext, *, include_narrative: bool = False) -> str:
    """Serialize scenario for the LLM prompt (narrative stripped by default)."""
    data = dict(ctx.raw)
    if not include_narrative:
        data.pop("narrative", None)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _artifact_header(
    ctx: ScenarioContext,
    *,
    injection_surface: str,
    platform_coverage: str | None,
    model: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    narrative = ctx.raw.get("narrative") or {}
    return {
        "scenario_id": ctx.scenario_id,
        "injection_surface": injection_surface,
        "narrative": {"summary": str(narrative.get("summary", ""))},
        "platform_coverage": platform_coverage,
        "disclosure": ARTIFACT_DISCLOSURE,
        "model": model,
        "timestamp": timestamp or _utc_timestamp(),
    }


def skip_to_artifact_dict(ctx: ScenarioContext) -> dict[str, Any]:
    """Minimal artifact for scenarios Garak cannot cover (no LLM)."""
    return _artifact_header(
        ctx,
        injection_surface=lookup_surface(ctx),
        platform_coverage=None,
        model=None,
    )


def _turn_to_artifact(t: ArtifactTurn) -> dict[str, Any]:
    turn: dict[str, Any] = {
        "role": t.role,
        "content": t.content,
        "adversarial": t.adversarial,
    }
    if t.tool_calls:
        turn["tool_calls"] = [tc.model_dump() for tc in t.tool_calls]
    if t.role == "tool":
        name = t.name or t.tool_name
        if name:
            turn["name"] = name
            turn["tool_name"] = name
        if t.tool_call_id:
            turn["tool_call_id"] = t.tool_call_id
    return turn


def artifact_to_dict(
    ctx: ScenarioContext,
    artifact: ScenarioArtifact,
    *,
    platform_coverage: str,
) -> dict[str, Any]:
    """Serialize the generated spec to the on-disk Garak artifact dict."""
    header = _artifact_header(
        ctx,
        injection_surface=artifact.injection_surface,
        platform_coverage=platform_coverage,
        model=llm_mod.MODEL,
    )
    return {
        **header,
        "turns": [_turn_to_artifact(t) for t in artifact.turns],
        "attack_turn_index": artifact.attack_turn_index,
        "detector_rubric": artifact.detector_rubric.model_dump(),
    }


def _fill(template: str, **slots: str) -> str:
    out = template
    for key, value in slots.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def gate_artifact_errors(
    artifact: ScenarioArtifact,
    *,
    expected_surface: str | None = None,
) -> list[str]:
    errors: list[str] = []
    dr = artifact.detector_rubric
    if not dr.judge_prompt.strip():
        errors.append("detector_rubric.judge_prompt must not be empty")
    success = dr.rubrics.get("attack_success") or []
    blocked = dr.rubrics.get("attack_blocked") or []
    if not success:
        errors.append("detector_rubric.rubrics.attack_success must not be empty")
    if not blocked:
        errors.append("detector_rubric.rubrics.attack_blocked must not be empty")
    if expected_surface and artifact.injection_surface != expected_surface:
        errors.append(
            f"injection_surface must be {expected_surface!r}, got {artifact.injection_surface!r}"
        )
    last = artifact.turns[-1]
    if expected_surface == "user_turn" and last.role != "user":
        errors.append("user_turn artifact must end with role=user")
    if expected_surface == "tool_return" and last.role != "tool":
        errors.append("tool_return artifact must end with role=tool")
    if expected_surface == "system_prompt" and last.role != "system":
        errors.append("system_prompt artifact must end with role=system")
    return errors


def _feedback_block(errors: list[str]) -> str:
    lines = "\n".join(f"- {e}" for e in errors)
    return (
        "\n\nPrevious JSON failed validation. Fix these issues and emit a "
        "complete new JSON object:\n" + lines
    )


def generate_scenario_artifact(
    ctx: ScenarioContext,
    prompt_path: Path,
    *,
    injection_surface: str | None = None,
) -> ArtifactBuildResult:
    surface = injection_surface or lookup_surface(ctx)
    template = prompt_path.read_text(encoding="utf-8")
    base_prompt = _fill(
        template,
        scenario_yaml=scenario_to_yaml(ctx),
        injection_surface=surface,
    )
    log.info(
        "Generating scenario artifact with %s (injection_surface=%s)",
        prompt_path,
        surface,
    )

    artifact: ScenarioArtifact | None = None
    last_error: Exception | None = None
    last_data: dict[str, Any] | None = None
    last_errors: list[str] = []
    feedback = ""

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        user_prompt = base_prompt + feedback
        try:
            data = llm_json(user_prompt, _LLM_SYSTEM)
            if not isinstance(data, dict):
                raise ValueError("LLM did not return a JSON object")
            last_data = data
            errors = validate_tool_flow(data)
            parsed: ScenarioArtifact | None = None
            if not errors:
                parsed = ScenarioArtifact.model_validate(data)
                errors = gate_artifact_errors(parsed, expected_surface=surface)
            last_errors = errors
            if errors:
                log.warning(
                    "Artifact validation attempt %d failed: %s",
                    attempt + 1,
                    "; ".join(errors),
                )
                feedback = _feedback_block(errors)
                if parsed is not None:
                    artifact = parsed
                continue
            artifact = parsed
            last_errors = []
            break
        except Exception as e:
            last_error = e
            log.warning("Artifact generation attempt %d failed: %s", attempt + 1, e)
            feedback = _feedback_block([str(e)])

    if artifact is None:
        if last_data is not None:
            try:
                artifact = ScenarioArtifact.model_validate(last_data)
                last_errors = validate_tool_flow(last_data) + gate_artifact_errors(
                    artifact, expected_surface=surface
                )
            except Exception as e:
                raise last_error or e from e
        else:
            raise last_error or RuntimeError("Artifact generation failed") from None

    if last_errors:
        log.warning("Artifact validation failed: %s", "; ".join(last_errors))
    return ArtifactBuildResult(artifact=artifact, ok=not last_errors, errors=last_errors)
