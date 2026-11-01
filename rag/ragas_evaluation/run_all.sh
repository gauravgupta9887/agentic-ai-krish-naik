#!/usr/bin/env bash
# Full experiment: validate QA set -> generate (both variants) -> judge (both) -> compare.
# Usage:  ./run_all.sh            (from this folder; uses ./.venv)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export TOKENIZERS_PARALLELISM=false

$PY validate_ground_truth.py
for v in naive reranked; do $PY generate.py --variant "$v"; done
for v in naive reranked; do $PY evaluate_ragas.py --variant "$v"; done
$PY compare.py
