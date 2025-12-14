## Project notes for Codex/agents

This repo is a single-file `uv` inline script.

- Entry point: `llmtree.py`
- Dependencies are declared in the `# /// script` block at the top of `llmtree.py`
- Prefer running via `uv` so dependencies resolve correctly:
  - `uv run llmtree.py -- --help`
  - `uv run llmtree.py -- -p "Your prompt here"`

Avoid using `pip install` directly for this repo unless you have a specific reason; the intended workflow is `uv run` with the inline dependency block.

