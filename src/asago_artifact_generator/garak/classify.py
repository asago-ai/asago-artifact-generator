"""Injection-surface classification from scenario YAML + Garak gate."""

from __future__ import annotations

import re
from typing import Literal

from ..extract import ScenarioContext, name_in_source

GateResult = Literal["full", "partial", "skip"]
InjectionSurface = Literal[
    "user_turn",
    "tool_return",
    "system_prompt",
    "tool_definition",
    "data_store",
    "none",
]

GARAK_WRITABLE = {"user_turn", "tool_return", "system_prompt"}

_ENTRY_ZONE = re.compile(r"\(([^)]+)\)\s*$")

_ZONE_TO_SURFACE: dict[str, InjectionSurface] = {
    "input": "user_turn",
    "tool_execution": "tool_return",
}


def is_supply_chain(threat_name: str) -> bool:
    return "supply chain" in threat_name.lower()


def entry_zone(entry_point: str) -> str | None:
    match = _ENTRY_ZONE.search(entry_point.strip())
    if match:
        return match.group(1).strip()
    return None


def lookup_surface(ctx: ScenarioContext) -> InjectionSurface:
    """Map YAML narrative zone to a Garak injection surface.

    Supply-chain threats have no Garak coverage (`none`). Otherwise the
    parenthetical zone on `narrative.entry_point` selects the chat role:
    `input` → `user_turn`, `tool_execution` → `tool_return`.
    """
    if is_supply_chain(ctx.threat_name):
        return "none"
    zone = entry_zone(ctx.entry_point)
    if zone in _ZONE_TO_SURFACE:
        return _ZONE_TO_SURFACE[zone]
    return "user_turn"


def first_injection_step(ctx: ScenarioContext) -> int | None:
    narrative = ctx.raw.get("narrative") or {}
    for step in narrative.get("steps") or []:
        zone = str(step.get("zone", ""))
        if zone in ("input", "tool_execution"):
            return int(step.get("step_number", 0)) or None
    return None


def surface_skip_reason(ctx: ScenarioContext, *, surface: str | None = None) -> str | None:
    """Deterministic pre-plan gate: return skip reason, or None to proceed."""
    sid = ctx.seed_id or ctx.scenario_id
    surf = surface or lookup_surface(ctx)

    if is_supply_chain(ctx.threat_name):
        return f"Garak has no coverage for supply chain threats ({sid})"
    if surf == "none":
        return f"Garak has no coverage ({sid})"
    if surf == "data_store":
        return f"unclassified or data_store surface for {sid}"
    if surf == "tool_definition":
        return f"Garak cannot write to tool_definition surface ({sid})"
    if surf not in GARAK_WRITABLE:
        return f"Garak cannot write to {surf} ({sid})"
    return None


def classify_garak(
    ctx: ScenarioContext,
    *,
    surface: str | None = None,
    trigger_tool: str = "",
    oracle_kind: str = "forbidden_call",
    tools_grounded: bool = True,
) -> tuple[GateResult, str]:
    """Return (gate_result, reason) after probe generation — full / partial / skip."""
    sid = ctx.seed_id or ctx.scenario_id
    surf = surface or lookup_surface(ctx)

    skip = surface_skip_reason(ctx, surface=surf)
    if skip:
        return "skip", skip
    if oracle_kind == "state_vs_log":
        return "skip", f"Garak cannot observe state_vs_log oracle ({sid})"
    observable_kinds = ("forbidden_call", "forbidden_argument", "output_string")
    if oracle_kind not in observable_kinds:
        return "skip", f"Garak cannot observe {oracle_kind} ({sid})"

    if oracle_kind == "output_string":
        return "partial", f"output-oracle attack ({sid})"

    if oracle_kind in ("forbidden_call", "forbidden_argument") and not trigger_tool:
        return "skip", f"no grounded trigger tool for {sid}"

    if not tools_grounded:
        return "partial", f"tools inferred from prose, not quoted in scenario ({sid})"

    if trigger_tool and not name_in_source(trigger_tool, ctx.grounding_text):
        return "partial", f"trigger tool {trigger_tool!r} not quoted in scenario ({sid})"

    if surf in GARAK_WRITABLE and first_injection_step(ctx) is None and surf != "system_prompt":
        return "partial", f"no clear injection step in narrative ({sid})"

    return "full", "fully supported"
