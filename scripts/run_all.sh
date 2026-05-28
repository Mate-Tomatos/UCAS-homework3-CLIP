#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/kxh/miniconda3/envs/trl/bin/python}"
MODEL_NAME="${MODEL_NAME:-openai/clip-vit-base-patch32}"
MAX_IMAGES="${MAX_IMAGES:-0}"

cd "$(dirname "$0")/.."

"${PYTHON_BIN}" scripts/prepare_datasets.py --dataset all --max-images "${MAX_IMAGES}"

"${PYTHON_BIN}" scripts/evaluate_retrieval.py \
  --annotations-jsonl datasets/flickr30k/annotations.jsonl \
  --model-name "${MODEL_NAME}" \
  --output-json results/clip_flickr30k_metrics.json

"${PYTHON_BIN}" scripts/evaluate_retrieval.py \
  --annotations-jsonl datasets/coco/annotations.jsonl \
  --model-name "${MODEL_NAME}" \
  --output-json results/clip_coco_metrics.json

"${PYTHON_BIN}" scripts/evaluate_catdog.py \
  --test-dir datasets/catdog/test \
  --model-name "${MODEL_NAME}" \
  --output-json results/clip_catdog_metrics.json
