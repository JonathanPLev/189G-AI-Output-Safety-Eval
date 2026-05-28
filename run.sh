#!/usr/bin/env bash
set -e

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

source .venv/bin/activate

SCRIPT=${1:-wildguard_llm_judge_eval.py}

echo "Running $SCRIPT..."
python "$SCRIPT"
