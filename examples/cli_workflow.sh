#!/bin/bash
# CLI workflow: run, report, resume, and list experiments.
#
# Prerequisites:
#   uv sync --extra all --extra dev
#
# This script demonstrates the full CLI lifecycle using the DemoAdapter
# (Branin function). Run from the repository root.

set -euo pipefail

DB="demo.db"
ID="demo-1"
ADAPTER="examples.demo_adapter:DemoAdapter"

# Clean up from any previous run
rm -f "$DB"

echo "=== Step 1: Run a new experiment (10 steps) ==="
uv run causal-optimizer run \
    --adapter "$ADAPTER" \
    --budget 10 \
    --db "$DB" \
    --id "$ID" \
    --seed 42

echo ""
echo "=== Step 2: Print a table report ==="
uv run causal-optimizer report --id "$ID" --db "$DB"

echo ""
echo "=== Step 3: Resume the experiment for 5 more steps ==="
uv run causal-optimizer resume \
    --adapter "$ADAPTER" \
    --id "$ID" \
    --db "$DB" \
    --budget 5

echo ""
echo "=== Step 4: Print a JSON report ==="
uv run causal-optimizer report --id "$ID" --db "$DB" --format json

echo ""
echo "=== Step 5: List all experiments ==="
uv run causal-optimizer list --db "$DB"

# Clean up
rm -f "$DB"
echo ""
echo "Done."
