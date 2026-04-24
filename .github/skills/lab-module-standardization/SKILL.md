______________________________________________________________________

## name: lab-module-standardization description: Standardize nan-ai-engineering-labs learning units to a single lab format, enforce English README heading schema, and validate completeness for core modules and trends extras

# Lab Module Standardization

Use this skill when the task involves module structure consistency, README normalization, lab completeness checks, or migration of `trends-extras` to the same model as `modules`.

## Inputs

- Target scope:
  - `modules/*`
  - `trends-extras/*`
- Reference schema:
  - `docs/MODULE-LAB-STANDARD.md`
- Agent contract:
  - `AGENTS.md`

## Required Workflow

1. Read the current unit README and all section readmes.
1. Compare against required folder layout and heading schema.
1. Apply minimal edits to align headings and missing sections.
1. Use English-only for all newly created or updated assets from this point:

- File names
- README explanations and body text
- Python code comments, docstrings, and messages

1. Keep required heading schema compatibility with the English contract used by validators.
1. Run `scripts/validate_learning_labs.py`.
1. Run pre-commit checks before finalizing.

## Validation Rules

- Core modules (`modules/*`) are strict by default.
- Extras (`trends-extras/*`) are warning-only until full migration.
- Any missing required file in core is a failure.
- Heading mismatches are failures in strict mode.

## Definition of Done

- Unit follows canonical folder layout.
- Required README headings are present.
- Unit has practices, project, and evaluation content.
- Validator returns success for strict core checks.
- Pre-commit hooks pass.

## Safety

- Do not remove user-authored content unless explicitly requested.
- Preserve existing links and references when renaming headings.
- Avoid formatting-only changes outside touched files.
