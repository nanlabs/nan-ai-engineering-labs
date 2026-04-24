______________________________________________________________________

## name: nan-ai-english-standards description: Enforce English-only repository content and naming standards in nan-ai-engineering-labs, including heading contracts and pre-commit validations.

# Nan AI English Standards

Use this skill when you are migrating, editing, or reviewing content in
`nan-ai-engineering-labs`.

## Goals

- Keep all repository content in English.
- Keep Python module filenames snake_case and lint-compliant.
- Keep learning-lab heading contracts aligned with English section titles.

## Required Workflow

1. Read impacted README/docs/scripts before editing.
1. Keep file names and Python module names in snake_case.
1. Use English-only wording in README/docs text, Python comments/docstrings, and user-facing messages.
1. Run contract validation and quality gates before finalizing:

```bash
PYTHON=/media/nquiroga/SSDedo/Documents/projects/NanLabs/labs/.venv/bin/python
$PYTHON scripts/validate_learning_labs.py --strict-core --strict-headings
pre-commit run --all-files
```

## Validation Rules

- `scripts/validate_learning_labs.py --strict-headings` is the source of truth for required English headings.
- Pre-commit hook `ruff module naming (N999)` enforces snake_case module names.
- Pre-commit hook `validate-english-content` enforces English-only governance files.

## Definition of Done

- Changes are in English.
- Module and script names are snake_case.
- Validation and pre-commit checks pass.

## Safety

- Do not remove user-authored content unless explicitly requested.
- Do not run destructive git commands unless explicitly requested.
- Keep changes scoped; avoid unrelated rewrites.
