#!/bin/bash

PYTHON_SCRIPT="/home/tyudha/skripsi/inference/golden_infer.py"

for SEQ_COUNT in {2..5}
do
    echo "=== Running inference for seq_count = $SEQ_COUNT ==="
    python3 "$PYTHON_SCRIPT" --seq $SEQ_COUNT
done

echo "=== All done ==="