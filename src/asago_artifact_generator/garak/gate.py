"""Post-generation Garak coverage gate."""

from __future__ import annotations

from ..extract import ScenarioContext
from .classify import GateResult, classify_garak


def gate_from_context(
    ctx: ScenarioContext,
    *,
    surface: str | None = None,
    trigger_tool: str = "",
    oracle_kind: str = "forbidden_call",
    tools_grounded: bool = True,
) -> tuple[GateResult, str]:
    return classify_garak(
        ctx,
        surface=surface,
        trigger_tool=trigger_tool,
        oracle_kind=oracle_kind,
        tools_grounded=tools_grounded,
    )
