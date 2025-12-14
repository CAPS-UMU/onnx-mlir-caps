#!/usr/bin/env bash

set -e


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/input.onnx"
    exit 1
fi

INPUT_MODEL="$1"
DIR="$(dirname "$INPUT_MODEL")"
FILENAME="$(basename "$INPUT_MODEL")"
BASENAME="${FILENAME%.onnx}"


OUTPUT_MODEL_interm="${DIR}/${BASENAME}_t_interm.onnx"
OUTPUT_MODEL="${DIR}/${BASENAME}_t.onnx"


python transpose_perm_c1.py "$INPUT_MODEL" "$OUTPUT_MODEL_interm"
python transpose_perm_c2.py "$OUTPUT_MODEL_interm" "$OUTPUT_MODEL"

echo "Output written to: $OUTPUT_MODEL"
