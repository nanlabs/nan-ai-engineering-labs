# AGENTS

## Scope

This file defines agent behavior for this repository.

## Source of Truth

1. Follow this file first for agent behavior.
1. Follow `docs/MODULE-LAB-STANDARD.md` for learning module structure and README headings.
1. Follow `.github/skills/lab-module-standardization/SKILL.md` for implementation and validation flow.
1. If instructions conflict: AGENTS.md wins for agent behavior, the standard file wins for content schema.

## Repository Conventions

- Portfolio scope:
  - Core track: `modules/*`
  - Advanced track: `trends-extras/*`
- Canonical lab module structure:
  - `README.md`
  - `STATUS.md`
  - `theory/README.md`
  - `examples/README.md`
  - `practices/README.md`
  - `mini-project/README.md`
  - `evaluation/README.md`
  - `notes/README.md`
- Language policy:
  - All repository content must be English.
  - This includes README text, docs, Python comments/docstrings/messages, and file names.
  - If a legacy file is not yet migrated, migrate it before expanding it.
- Quality policy:
  - Every change should pass pre-commit hooks.
  - Module contract validation must run before commit.
  - Keep changes scoped and avoid broad rewrites without request.

## Validation Commands

Use the project venv explicitly when available:

```bash
PYTHON=/media/nquiroga/SSDedo/Documents/projects/NanLabs/labs/.venv/bin/python
$PYTHON scripts/validate_learning_labs.py
$PYTHON scripts/validate_learning_labs.py --report-json
```

For hook execution:

```bash
pre-commit run --all-files
```

## Safety

- Do not run destructive git commands (`reset --hard`, `checkout --`) unless explicitly requested.
- Do not revert unrelated user changes.
- Never commit secrets or credentials.
- Keep docs and standards aligned when changing structure rules.
