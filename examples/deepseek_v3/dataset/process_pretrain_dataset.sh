#!/bin/bash

set -ex

SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"
MEGATRON_LM_DIR="$( cd ${SCRIPT_DIR}/../../.. && pwd )"

PYTHON=python3
DEEPSEEK_HF_MODEL_DIR="${SCRIPT_DIR}/../DeepSeek-V3"
DATASET_JSONL_PATH=/ssd/kurt/dataset/enwiki-100k.jsonl
TARGET_DIR="${SCRIPT_DIR}/enwiki-100k"
DATA_FILE_PREFIX=enwiki-100k

mkdir -p "${TARGET_DIR}"

export TRUST_REMOTE_CODE=1

${PYTHON} ${MEGATRON_LM_DIR}/tools/preprocess_data.py \
    --input ${DATASET_JSONL_PATH} \
    --output-prefix ${TARGET_DIR}/${DATA_FILE_PREFIX} \
    --append-eod \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${DEEPSEEK_HF_MODEL_DIR} \
    --workers 64
