# Module Lab Standard

This document defines the canonical model that all learning units must follow in this repository.

## Scope

- Applies to `modules/*`.
- Migration target for `trends-extras/*`.

## Canonical Folder Layout

Each learning unit must contain:

```text
<unit>/
  README.md
  STATUS.md
  theory/
    README.md
  examples/
    README.md
  practices/
    README.md
  mini-project/
    README.md
  evaluation/
    README.md
  notes/
    README.md
```

## Required README Headings

### Unit `README.md`

1. `# <Module Title>`
1. `## Objective of the module`
1. `## What are you going to achieve?
1. `## Structure interna`
1. `## Level path (L1-L4)`
1. `## Recommended plan (by progress, not by weeks)`
1. `## Module criteria completed`

### `theory/README.md`

1. `# Theory - <Module Title>`
1. `## Why does this module matter?
1. `## Final compression checklist`

### `examples/README.md`

1. `# Examples - <Module Title>`
1. `## Examples disponibles`
1. `## Como use estos Examples`
1. `## Proximos steps`

### `practices/README.md`

1. `# Practices - <Module Title>`
1. `## Practicas`
1. `## Approval criteria`

### `mini-project/README.md`

1. `# Mini-project - <Module Title>`
1. `## Project`
1. `## Objective`
1. `## Alcance`
1. `## Entregables`
1. `## Acceptance criteria`

### `evaluation/README.md`

1. `# Evaluation - <Module Title>`
1. `## Ponderacion`
1. `## Rubrica`
1. `## Evaluation final`
1. `## Approval criteria`

### `notes/README.md`

1. `# Notes - <Module Title>`
1. `## Como use this carpeta`

### `STATUS.md`

1. `# Status - <Module Title>`
1. `## Progreso actual`
1. `## Checklist`
1. `## Bloqueos`
1. `## Proximos steps`

## Lab Completeness Definition

A unit is considered lab-complete when:

1. It is self-contained: all required folders and readmes exist.
1. It includes guided and autonomous practice instructions.
1. It includes an evaluation rubric and completion threshold.
1. It has status tracking (`STATUS.md`).
1. It passes automated contract checks.

## Executable Example Standard

This repository prioritizes depth and clarity over raw example count.

### Target Density

- Core modules (`modules/*`): 6 to 8 executable examples per module.
- Advanced modules (`trends-extras/*`): 4 to 6 executable examples per unit.

### Deep Example Requirements

Each example should be executable with minimal setup and include:

1. Problem statement and objective.
1. Input data definition (or dataset source link when needed).
1. Expected output or success criteria.
1. A small variation task (parameter/model/data change).
1. Common failure modes and quick debugging hints.

### Difficulty Progression (L1 to L4)

Examples should progressively map to the module level model:

1. L1: Guided, fully reproducible baseline.
1. L2: Controlled variation with 2 to 3 key parameters.
1. L3: Realistic scenario with metrics and interpretation.
1. L4: Extension focused on maintainability or production constraints.

### Definition Of Clear Example

An example is considered clear when:

1. A learner can run it end-to-end without guessing missing steps.
1. The README explains why each key step exists.
1. The learner can compare observed output against expected output.
1. The example includes one explicit next challenge.

## Testing Strategy (Incremental)

- Phase 1: Contract tests (structure + headings + required sections).
- Phase 2: Add smoke tests per unit for executable assets where present.
- Phase 3: Add stricter validation for `trends-extras/*` until parity with `modules/*`.

## Enforcement

- Local: pre-commit hook runs `scripts/validate_learning_labs.py`.
- Manual: `python scripts/validate_learning_labs.py --strict-core`.
- CI (future): run strict mode for both core and extras after migration.
