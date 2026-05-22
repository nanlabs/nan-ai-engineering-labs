# Getting Started — AI Engineering Labs

## What this program is

A self-guided, progressive learning system that takes you from zero AI knowledge to implementing production systems. 12 core modules covering the full AI/ML engineering stack plus 6 advanced trend modules.

## Prerequisites

- `git` installed
- One of:
  - **DevContainer-capable IDE** (VS Code, Cursor, GitHub Codespaces) — recommended
  - **Python 3.13+** for local setup
- Familiarity with basic Python (variables, loops, functions)
- High school level math (algebra, graphs)

## Setup

### Option A — DevContainer (recommended)

```bash
git clone git@github.com:nanlabs/nan-ai-engineering-labs.git
cd nan-ai-engineering-labs
code .
# Command Palette -> "Dev Containers: Reopen in Container"
```

The DevContainer pre-installs Python, common ML libraries, and pre-commit hooks.

### Option B — Local

```bash
git clone git@github.com:nanlabs/nan-ai-engineering-labs.git
cd nan-ai-engineering-labs

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate            # Windows

pip install numpy pandas matplotlib scikit-learn jupyter
pip install pre-commit
pre-commit install
```

## First run — validators

```bash
python scripts/validate_learning_labs.py
python scripts/validate_english_content.py
```

Both should exit 0 on a clean checkout.

## First module

```bash
cd modules/01-programming-math-for-ml
cat README.md
```

Study flow within each module:

```text
theory/ -> examples/ -> practices/ -> mini-project/ -> evaluation/
```

## Checkpoints

End of each phase is a checkpoint:

- **Phase 1** (Modules 1-4): Implement an image classifier with CNN.
- **Phase 2** (Modules 5-8): Fine-tune an LLM and deploy a chatbot.
- **Phase 3** (Modules 9-12): Deploy a complete system with CI/CD.

## Help

- Read [`AGENTS.md`](AGENTS.md) before asking AI to modify the repo.
- Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR.
- File issues for content gaps; PRs welcome for new modules and improvements.
