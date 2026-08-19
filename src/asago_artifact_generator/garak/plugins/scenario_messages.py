"""Convert ArtifactGen turns into OpenAI chat-completions messages + tool schemas.

Kept free of garak imports so the notebook can use it before plugins are installed.
"""

from __future__ import annotations

import json
import re

_TOOL_LINE_RE = re.compile(
    r"^- ([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)",
    re.M,
)
_TOOL_SIG_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)\s*\(([^`]*)\)`")
_PARAM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*([A-Za-z_][A-Za-z0-9_\[\]| ]*))?")


def _iter_tool_decls(system_content: str):
    seen: set[str] = set()
    text = system_content or ""
    for pattern in (_TOOL_LINE_RE, _TOOL_SIG_RE):
        for name, raw_args in pattern.findall(text):
            if name in seen:
                continue
            seen.add(name)
            yield name, raw_args


def _stringify_arguments(arguments) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments if arguments is not None else {})


def _normalize_tool_calls(tool_calls: list) -> list[dict]:
    out: list[dict] = []
    for i, tc in enumerate(tool_calls or []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        out.append(
            {
                "id": tc.get("id") or f"call_{i:03d}",
                "type": tc.get("type") or "function",
                "function": {
                    "name": fn.get("name") or "",
                    "arguments": _stringify_arguments(fn.get("arguments")),
                },
            }
        )
    return out


def _schema_from_params(raw_args: str) -> dict:
    properties: dict[str, dict] = {}
    required: list[str] = []
    for pname, ptype in _PARAM_RE.findall(raw_args or ""):
        if pname in {"str", "int", "float", "bool", "Optional"}:
            continue
        json_type = "string"
        ptl = (ptype or "").lower()
        if "int" in ptl:
            json_type = "integer"
        elif "float" in ptl or "number" in ptl:
            json_type = "number"
        elif "bool" in ptl:
            json_type = "boolean"
        properties[pname] = {"type": json_type}
        if "optional" not in ptl and "none" not in ptl:
            required.append(pname)
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def extract_openai_tools(system_content: str) -> list[dict]:
    """Best-effort OpenAI tool schemas from a system prompt that names ``fn(args)``."""
    tools: list[dict] = []
    for name, raw_args in _iter_tool_decls(system_content):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool advertised by the scenario system prompt.",
                    "parameters": _schema_from_params(raw_args),
                },
            }
        )
    return tools


def _synthetic_tool_call(call_id: str, tool_name: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": tool_name, "arguments": "{}"},
    }


def turns_to_openai_messages(turns: list[dict]) -> list[dict]:
    """Convert artifact turns into chat-completions messages ToolChat can replay."""
    messages: list[dict] = []
    for i, turn in enumerate(turns or []):
        role = str(turn.get("role") or "user")
        content = turn.get("content") if turn.get("content") is not None else ""
        if role != "tool":
            msg: dict = {"role": role, "content": content}
            raw_calls = turn.get("tool_calls") or []
            if raw_calls:
                msg["tool_calls"] = _normalize_tool_calls(raw_calls)
            messages.append(msg)
            continue

        tool_name = str(turn.get("name") or turn.get("tool_name") or "unknown_tool")
        call_id = str(turn.get("tool_call_id") or f"call_{i:03d}")
        prev = messages[-1] if messages else None
        if prev and prev.get("role") == "assistant":
            if not prev.get("tool_calls"):
                prev["tool_calls"] = [_synthetic_tool_call(call_id, tool_name)]
                if prev.get("content") is None:
                    prev["content"] = ""
        elif prev and prev.get("role") == "tool":
            pass
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_synthetic_tool_call(call_id, tool_name)],
                }
            )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": content,
            }
        )
    return messages
