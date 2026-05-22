# Contributing

Thanks for contributing. This repo is a **learning system**, not a production runtime. Quality, clarity and portability matter more than feature volume.

## Before you start

1. Read [`AGENTS.md`](AGENTS.md). It is the primary contract.
2. Read [`docs/CHARTER.md`](docs/CHARTER.md) for the upstream NaNLABS Lab Charter v1.
3. Run `python scripts/validate_learning_labs.py` to confirm CI is green on `main`.

## Branching and commits

- Branch from `main`: `feat/<module>-<topic>` or `fix/<scope>`.
- Commit messages: imperative, English, scoped: `feat(01-programming-math): add numpy exercises`.
- Squash on merge.

## Adding new content

1. Choose the target module directory under `modules/` or `trends-extras/`.
2. Follow the canonical structure defined in `docs/MODULE-LAB-STANDARD.md`:
   - `README.md`, `STATUS.md`, `theory/README.md`, `examples/README.md`
   - `practices/README.md`, `mini-project/README.md`, `evaluation/README.md`, `notes/README.md`
3. Write examples that are runnable — a learner must be able to execute them.
4. Add real references and citations where applicable.
5. Run `python scripts/validate_learning_labs.py` until green.

## Editing existing content

- Preserve module numbers. Renaming requires updating docs and the README hub table.
- Do not modify other contributors' personal notes directories.
- Keep all content in English per language policy.

## Quality bar

Every module must:

- Be **runnable** (examples a learner can execute).
- Be **measurable** (evaluation checklist present).
- Be **self-contained** (theory + examples + practices + evaluation).
- Cite authoritative external references where applicable.

## Pull request

- Fill the PR template.
- Link related modules.
- Confirm `validate_learning_labs.py` and `validate_english_content.py` pass.
- Add a screenshot or short example when relevant.

## Code of conduct

Be kind, be specific, prefer examples to opinions.
