#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$ROOT_DIR/data/hybridqa}"

mkdir -p "$DATA_DIR"

if [ ! -d "$DATA_DIR/HybridQA" ]; then
  git clone --depth 1 --filter=blob:none https://github.com/wenhuchen/HybridQA "$DATA_DIR/HybridQA"
fi

if [ ! -d "$DATA_DIR/WikiTables-WithLinks" ]; then
  git clone --depth 1 --filter=blob:none https://github.com/wenhuchen/WikiTables-WithLinks "$DATA_DIR/WikiTables-WithLinks"
fi

echo "HybridQA data downloaded to: $DATA_DIR"
echo "Set HYBRIDQA_DATA_DIR=$DATA_DIR or HYBRIDQA_DATA_DIR=$DATA_DIR/WikiTables-WithLinks"
