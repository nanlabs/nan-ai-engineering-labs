#!/usr/bin/env bash
set -euo pipefail

echo "==> Creating virtualenv and installing deps"
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn jupyter

echo "==> Installing pre-commit hooks"
pip install pre-commit
pre-commit install || true

echo "==> Done. Start with: cd modules/01-programming-math-for-ml"
