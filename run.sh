#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/input data/output

python3 src/generate_dataset.py --output data/input --count 200 --width 256 --height 256

make

./bin/batch_image_processor data/input data/output

echo "Run complete. See data/output/run_log.txt"
