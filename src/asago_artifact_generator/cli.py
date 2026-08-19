"""CLI: scenario YAML → runs/{scenario_id}/{scenario_id}-garak.json (one-shot LLM)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from .extract import load_scenario
from .garak.gen import generate_artifact, list_scenario_files
from .garak.spec_io import MANIFEST_FILE, runs_dir

app = typer.Typer(
    help="Policy-driven agentic red-teaming artifact generator.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Policy-driven agentic red-teaming artifact generator."""


@app.command()
def generate(
    scenarios: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Scenario YAML file(s). If omitted, processes all in examples/scenarios/",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Override runs/ output directory (default: runs/)"),
    ] = None,
    prompt: Annotated[
        Path | None,
        typer.Option(
            "--prompt",
            help="Override probe prompt (default: prompts/generate_artifact.md)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Classify + LLM + validate only — do not write garak JSON"),
    ] = False,
    no_llm: Annotated[
        bool,
        typer.Option("--no-llm", help="Refuse LLM (only useful for skip check)"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Write garak JSON even if probe validation fails"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Verbose logging"),
    ] = False,
) -> None:
    """Generate Garak probe artifacts from scenario YAMLs."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-5s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    prompt_path = prompt
    paths = list(scenarios) if scenarios else list_scenario_files()
    if not paths:
        typer.echo("No scenario files found.", err=True)
        raise typer.Exit(1)

    manifest: list[dict] = []
    counts = {"full": 0, "partial": 0, "skip": 0, "error": 0}

    for path in paths:
        try:
            ctx = load_scenario(path)
            result = generate_artifact(
                ctx,
                prompt_path=prompt_path,
                output_dir=output_dir,
                use_llm=not no_llm,
                force=force,
                dry_run=dry_run,
            )
            entry = {
                "scenario_id": result.scenario_id,
                "gate_result": result.gate,
                "gate_reason": result.gate_reason,
                "artifact_path": result.artifact_path,
            }
            if result.errors:
                entry["errors"] = result.errors
            manifest.append(entry)
            counts[result.gate] += 1
        except Exception as e:
            log.error("ERROR processing %s: %s", path.name, e)
            manifest.append(
                {
                    "scenario_id": path.stem,
                    "gate_result": "error",
                    "error": str(e),
                }
            )
            counts["error"] += 1

    if not dry_run:
        out = runs_dir(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / MANIFEST_FILE
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log.info("Manifest written to %s", manifest_path)

    print(f"\n{'=' * 50}")
    print("Garak artifact generation summary")
    print(f"{'=' * 50}")
    print(f"Total scenarios: {len(manifest)}")
    print(f"  Full:    {counts['full']}")
    print(f"  Partial: {counts['partial']}")
    print(f"  Skip:    {counts['skip']}")
    print(f"  Error:   {counts['error']}")

    print(f"\n{'Scenario':<25} {'Gate':<8} {'Reason'}")
    print("-" * 70)
    for e in manifest:
        sid = e.get("scenario_id", "?")
        gr = e.get("gate_result", "?")
        reason = e.get("gate_reason", e.get("error", ""))
        print(f"{sid:<25} {gr:<8} {reason}")

    if counts["error"] > 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
