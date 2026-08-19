# Policy-Driven Agentic Red Teaming

Takes pre-built **scenario** YAMLs, classifies their injection surface, and generates red-teaming artifacts that can be run on downstream evaluation platforms.

```
examples/scenarios/*.yaml
  → classify (injection surface + platform coverage)
  → generate a platform-specific artifact
  → runs/{scenario_id}/
```

## Setup

Asago Artifact Generator requires Python 3.11 or newer. The lock file is the
authoritative development environment.

```bash
uv sync --locked
cp .env.example .env   # set GEMINI_API_KEY or configure Ollama
```

The installed command is `asago-artifact-generator` and the Python package is
`asago_artifact_generator`.

Supported LLM backends: **Gemini** (default when `GEMINI_API_KEY` is set), **OpenAI**, **Ollama**, Hugging Face, or OpenRouter.

## How it works

1. **Classify** the injection surface from the scenario seed (`user_turn`, `tool_return`, `system_prompt`, or skip).
2. **Skip** surfaces the target platform cannot express.
3. **Generate** a red-teaming artifact for that platform (transcript, probe, or equivalent).
4. **Validate** structural gates so the artifact is runnable downstream.
5. **Gate** platform coverage: `full`, `partial`, or `skip`.

## Supported platforms

| Platform | Status | Details |
|----------|--------|---------|
| [Garak](https://github.com/NVIDIA/garak) | Supported | See `src/asago_artifact_generator/garak/` |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) | Planned | — |
| [PyRIT](https://github.com/Azure/PyRIT) | Planned | — |

Each platform generator lives in its own subpackage under
`src/asago_artifact_generator/` and writes artifacts under `runs/`.

## Generate artifacts

```bash
# One scenario
asago-artifact-generator generate examples/scenarios/AP-T2-01-28712e.yaml --force -v

# All scenarios in examples/scenarios/
asago-artifact-generator generate -v
```

| Flag | Effect |
|------|--------|
| `--force` | Write garak JSON even when structural validation fails |
| `--dry-run` | Classify + LLM + validate only — no files written |
| `--no-llm` | Skip LLM (useful to test pre-plan surface skips) |
| `--output-dir DIR` | Override default `runs/` output directory |
| `-v` | Verbose logging |

### Pipeline

1. **Classify** injection surface from scenario seed (`user_turn`, `tool_return`, `system_prompt`, or skip).
2. **Skip** unwritable surfaces (e.g. `tool_definition`) — writes a minimal artifact without calling the LLM.
3. **Generate** probe via one-shot LLM (`prompts/generate_artifact.md`).
4. **Validate** structural gates (rubric completeness, surface/turn alignment, schema).
5. **Gate** platform coverage: `full`, `partial`, or `skip`.

## Output layout

Each scenario gets its own directory under `runs/`:

```
runs/
  manifest.json
  AP-T2-01-28712e/
    AP-T2-01-28712e-garak.json    # probe artifact
    validation.json               # structural gate result
```

**`{scenario_id}-garak.json`** — probe artifact:

- `scenario_id`, `injection_surface`, `platform_coverage` (`full` | `partial` | `null` for skips)
- `narrative.summary`
- `model` — LLM used for generation (`null` on skip)
- `timestamp` — UTC ISO time when the artifact was written
- `turns[]` with adversarial attack turn
- `detector_rubric` (judge prompt + success/blocked rubrics)

**`validation.json`** — sidecar from the structural gate:

```json
{
  "ok": true,
  "checks": "Probe structural gate after LLM generation: ...",
  "errors": []
}
```

Skipped scenarios (unwritable surfaces) get a pre-plan `checks` string and no LLM call.

**`manifest.json`** — batch summary (`gate_result`, `gate_reason`, `artifact_path` per scenario).

## Interactive demo

End-to-end Jupyter walkthrough (API key → scenario YAML → artifact → Garak `toolchat.ToolChat` attack):

```bash
uv sync --locked
pip install ipywidgets
jupyter notebook examples/demo/demo.ipynb
```

## Development

```bash
./scripts/quality.sh
uv run pytest tests/ -q
```

The unit test suite is deterministic and does not require an LLM endpoint.

## Project structure

```
├── src/asago_artifact_generator/    # shared models, LLM client, CLI
│   ├── garak/                        # Garak platform generator
│   │   ├── plugins/                  # probe + detector sources
│   │   └── prompts/                  # generation prompt
│   └── data/                         # exploit styles catalog
├── tests/                            # unit tests
├── examples/
│   ├── scenarios/                    # input scenario YAMLs
│   └── demo/                         # Jupyter walkthrough
└── runs/                             # generated artifacts (gitignored)
```

## Modules

| Module | Role |
|--------|------|
| `cli.py` | `typer` CLI — orchestrates classify → generate → validate → save |
| `garak/gen.py` | Core generation logic for Garak artifacts |
| `garak/probe_spec.py` | `ScenarioProbe` schema, LLM call, `gate_probe_errors`, artifact dicts |
| `garak/spec_io.py` | Paths and I/O for `runs/{id}/{id}-garak.json` and `validation.json` |
| `garak/classify.py` | Injection-surface table, pre-plan skip, `platform_coverage` gates |
| `extract.py` | Load scenario YAML into `ScenarioContext` |
| `garak/gate.py` | `gate_from_context` — full vs partial vs skip after probe |
| `llm.py` | Provider-agnostic completion (Gemini / Ollama / OpenAI / HF / OpenRouter) |
| `garak/prompts/generate_artifact.md` | One-shot probe generation prompt |

## License

Apache 2.0 — see [LICENSE](LICENSE).
