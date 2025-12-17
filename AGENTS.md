# Repository Guidelines

## Project Structure & Modules
- Core package lives in `llmwalk/` (`cli.py` for CLI wiring and output rendering, `llm.py` for search logic, `__main__.py` for entrypoint).
- Prompt examples sit in `prompts/` and feed snapshot tests; update or add `.txt` files there to cover new cases.
- Tests and snapshots are under `tests/` with Syrupy fixtures in `tests/__snapshots__/`.
- Top-level assets: `README.md` for user-facing usage, `example1.gif` for demo, `pyproject.toml` for deps/metadata.

## Build, Run, and Dev Commands
- Run the tool with dependencies resolved via uv: `uv run llmwalk -p "Your prompt"`.
- Export structured results: append `--format json` or `--format csv`.

## Coding Style & Naming
- Python 3.10+, 4-space indentation, type hints throughout; prefer dataclasses for config/data carriers (`SearchConfig`, `Branch`).
- Keep CLI arguments descriptive and validated early (see `parse_args`); guard user-facing output with clear defaults.
- Follow existing naming: `PromptTreeSearch`, `render_*` for display helpers
- Rich output colors are hex strings; keep ASCII for code/doc unless the runtime already emits symbols.

## Testing Guidelines
- Primary suite uses `pytest` plus Syrupy snapshots: `uv run pytest`.
- Update snapshots intentionally with `uv run pytest --snapshot-update`; ensure prompt fixtures in `prompts/` match expected outputs.
- Tests currently focus on deterministic JSON output; add new prompts or behaviors alongside snapshot coverage where possible.

## Commit & PR Practices
- Commit messages in history are short, imperative, and scoped (e.g., “Allow -p to be a file path.”); mirror that style and keep changesets focused.
- For PRs, describe model/runtime expectations (model id, `--top-k`, `--top-p`, `--temperature`) and note any snapshot updates or offline-mode considerations.
- Include reproducible commands (`uvx`/`uv run pytest`) and mention any HuggingFace downloads or cache needs.***
