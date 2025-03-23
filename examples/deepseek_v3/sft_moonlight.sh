#!/bin/bash

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")
MEGATRON_LM_ROOT=$(realpath "$(dirname "$0")"/../../)

# ----Example Config----
export HL_LAUNCHER_TYPE="torchrun"
#export HL_LAUNCHER_TYPE="mpirun"
export HL_HOSTSFILE="${MEGATRON_LM_ROOT}/examples/hostsfile"
export HL_DATA_DIR_ROOT="${SCRIPT_DIR}/dataset"
export HL_DATA_FILE_PREFIX="alpaca_zh-qwen-train.json"
export HL_TOKENIZER_MODEL="${SCRIPT_DIR}/DeepSeek-V3"

## 1 node, Drop
export HL_SEQ_LEN=$((1*1024))
export HL_NUM_NODES=1
export HL_TP=1
export HL_PP=1
export HL_FISRT_PP_STAGE_LAYERS=0
export HL_LAST_PP_STAGE_LAYERS=0
export HL_DP=8
export HL_EP=8
export HL_SEQ_PARALLEL=0
export HL_MICRO_BATCH=1
export HL_GBS=32
export HL_USE_DISTRIBUTED_OPTIMIZER=1
export HL_CKP_ACT=1
export HL_USE_FUSED_SDPA=1
export HL_USE_FUSED_SDPA_WITH_RECOMPUTE=1
export HL_MOE_LAYER_RECOMPUTE=0
export HL_NUM_LAYERS=27
export HL_TOKEN_DROP=1
export HL_TOKEN_DISPATCHER_TYPE=alltoall
export HL_MOE_MEMORY_OPTIMIZATION=2
export HL_MOE_SKIP_FIRST_LAYERS=1

## common
#export HL_TRANSFORMER_IMPL="transformer_engine"
#export HL_USE_MOE_GROUPED_GEMM=1
export HL_ENABLE_PARAM_GATHER_OVERLAP=0
export HL_ENABLE_GRAD_REDUCE_OVERLAP=0
export HL_ENABLE_SHARED_EXPERT_OVERLAP=0
export HL_DETERMINISTIC_MODE=0
#export NCCL_ALGO="Ring"
#export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HL_SAVE=0
#export HL_SAVE_INTERVAL=200
export HL_USE_DIST_CKPT=1
export HL_DIST_CKPT_FORMAT="zarr"
export HL_FP8=0
export HL_NUM_WORKERS=0

export HL_LOG_INTERVAL=1
export HL_REDIRECT_LOGS=1
export HL_EXIT_INTERVAL=1500
export HL_LOAD_DIR="/ssd/libo/deepseek/MG_models/moonlight-16B-A3B-hf-to-mcore-tp1-pp1/"

# debug
#export HL_PROFILE=1
#export HL_PROFILE_PYTORCH=1
#export HL_PROFILE_STEP_START=7
#export HL_PROFILE_STEP_END=8
#export HL_PROFILE_RANKS="0 8 16 24"
#export NVTE_DEBUG=1
#export NVTE_DEBUG_LEVEL=2
##export NVTE_FUSED_ATTN_BACKEND=1
# ----------------------

# GPU specific settings
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Distributed training variables
LAUNCHER_TYPE=${HL_LAUNCHER_TYPE:-mpirun}
DATA_DIR=${HL_DATA_DIR_ROOT:-/data/datasets/red_pajama}
DATA_CACHE_DIR=${HL_DATA_CACHE_DIR:-}
DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-redpajama}
TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-DeepSeekV2Tokenizer}
TOKENIZER_MODEL=${HL_TOKENIZER_MODEL:-}
TRANSFORMER_IMPL=${HL_TRANSFORMER_IMPL:-local}
# Parallelism variables
NUM_NODES=${HL_NUM_NODES:-1}
TP=${HL_TP:-2}
PP=${HL_PP:-2}
DP=${HL_DP:-2}
EP=${HL_EP:-2}
MICRO_BATCH_SIZE=${HL_MICRO_BATCH:-1}
# TODO implement gradually increasing batch size mode
GLOBAL_BATCH_SIZE=${HL_GBS:-3072} # 15360 after 469B tokens
SEQ_LEN=${HL_SEQ_LEN:-4096}
TRAIN_ITERS=${HL_TRAIN_ITERS:-500000} # 1_085_677_083
LR_DECAY_ITER=${HL_LR_DECAY_ITER:-320000} # 773_177_083
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
OUTPUT_DIR_PREFIX=${HL_RESULTS_DIR_PREFIX:-.}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-10000}
DIST_CKPT_FORMAT=${HL_DIST_CKPT_FORMAT:-torch_dist}
USE_DISTRIBUTED_OPTIMIZER=${HL_USE_DISTRIBUTED_OPTIMIZER:-1}
USE_DIST_CKPT=${HL_USE_DIST_CKPT:-0}
LOAD_DIR=${HL_LOAD_DIR:-}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
HOSTSFILE=${HL_HOSTSFILE:-}
CKP_ACT=${HL_CKP_ACT:-0}
RECOMPUTE_NUM_LAYERS=${HL_RECOMPUTE_NUM_LAYERS:-1}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
DEVICES_PER_NODE=${HL_DEVICES_PER_NODE:-8}
SEQ_PARALLEL=${HL_SEQ_PARALLEL:-1}
OPTIMIZER=${HL_OPTIMIZER:-adam}
DROPOUT=${HL_DROPOUT:-0.0}
EVAL_ITERS=${HL_EVAL_ITERS:-10}
EVAL_INTERVAL=${HL_EVAL_INTERVAL:-1000}
PROFILE=${HL_PROFILE:-0}
PROFILE_PYTORCH=${HL_PROFILE_PYTORCH:-0}
PROFILE_STEP_START=${HL_PROFILE_STEP_START:-3}
PROFILE_STEP_END=${HL_PROFILE_STEP_END:-4}
PROFILE_RANKS=${HL_PROFILE_RANKS:-}
REDIRECT_LOGS=${HL_REDIRECT_LOGS:-0}
DETERMINISTIC_MODE=${HL_DETERMINISTIC_MODE:-1}
FP8=${HL_FP8:-0}
NUM_WORKERS=${HL_NUM_WORKERS:-2}
ENABLE_PARAM_GATHER_OVERLAP=${HL_ENABLE_PARAM_GATHER_OVERLAP:-1}
ENABLE_GRAD_REDUCE_OVERLAP=${HL_ENABLE_GRAD_REDUCE_OVERLAP:-1}
# MoE
TOKEN_DROP=${HL_TOKEN_DROP:-0}
TOKEN_DISPATCHER_TYPE=${HL_TOKEN_DISPATCHER_TYPE:-allgather}
MOE_LAYER_RECOMPUTE=${HL_MOE_LAYER_RECOMPUTE:-0}
ENABLE_SHARED_EXPERT_OVERLAP=${HL_ENABLE_SHARED_EXPERT_OVERLAP:-1}
USE_MOE_GROUPED_GEMM=${HL_USE_MOE_GROUPED_GEMM:-0}
# first 3 layers dense
MOE_SKIP_FIRST_LAYERS=${HL_MOE_SKIP_FIRST_LAYERS:-3}
# PP
FISRT_PP_STAGE_LAYERS=${HL_FISRT_PP_STAGE_LAYERS:-0}
LAST_PP_STAGE_LAYERS=${HL_LAST_PP_STAGE_LAYERS:-0}

if [[ ${NUM_NODES} -gt 1 ]] && [[ "${LAUNCHER_TYPE}" = "torchrun" ]] && [[ -z ${HL_TORCHRUN_NODE_RANK} || -z $HL_TORCHRUN_MASTER_ADDR ]]
then
    echo "HL_TORCHRUN_NODE_RANK and HL_TORCHRUN_MASTER_ADDR must be set for torchrun mode"
    exit 1
fi
TORCHRUN_NODE_RANK=${HL_TORCHRUN_NODE_RANK:-0}
TORCHRUN_MASTER_ADDR=${HL_TORCHRUN_MASTER_ADDR:-0}

if [[ -z "${MEGATRON_LM_ROOT}" ]]; then
    MEGATRON_LM_ROOT=$(realpath "$(dirname "$0")"/../../)
fi

if [[ $((NUM_NODES*DEVICES_PER_NODE)) -ne $((DP*TP*PP)) ]]; then
    echo "NUM_NODES*DEVICES_PER_NODE != DP*TP*PP"
    exit 1
fi

if [[ "${TRANSFORMER_IMPL}" = "local" && ${FP8} -eq 1 ]]; then
    echo "fp8 is not supported with local transformer implementation"
    exit 1
fi

# Model size variables
MAX_SEQ_LEN=${SEQ_LEN}
EXTRA_VOCAB_SIZE=0

HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=11264
EXPERT_FFN_HIDDEN_SIZE=1408
SHARED_EXPERT_FFN_HIDDEN_SIZE=2816 # EXPERT_FFN_HIDDEN_SIZE * NUM_SHARED_EXPERTS
NUM_HEADS=16
NUM_KV_HEADS=16
Q_LORA_RANK=0
KV_LORA_RANK=512
QK_HEAD_DIM=128
QK_POS_EMB_HEAD_DIM=64
V_HEAD_DIM=128
NUM_EXPERTS=64
NUM_SHARED_EXPERTS=2
TOPK=6
NUM_LAYERS=${HL_NUM_LAYERS:-27}

LR=2.2e-4
MIN_LR=2.2e-5
MIN_LR_1=7.3e-6
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR_WARMUP_ITERS=2000
ROTARY_BASE=10000
RMSNORM_EPS=1e-5
INIT_STD=6e-3

AUX_LOSS_COEFF=1e-4

if [[ ${FISRT_PP_STAGE_LAYERS} -eq 0 && ${LAST_PP_STAGE_LAYERS} -eq 0 && $(( NUM_LAYERS % PP )) -ne 0 ]]; then
    echo 'HL_NUM_LAYERS must be divisible by PP'
    exit 1
fi

# Paths
SRC_PATH="${MEGATRON_LM_ROOT}/train_sft.py"
DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}

if [[ -z "${TOKENIZER_MODEL}" ]]; then
    TOKENIZER_MODEL="${DATA_DIR}/tokenizer.model"
fi

NUM_DEVICES=$((DEVICES_PER_NODE*NUM_NODES))

RUNTIME=$(date +"%Y%m%d_%H%M.%S.%N")
# Experiment name
if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="default"
fi
# output paths
if [[ -z "${OUTPUT_DIR}" ]]; then
    data_type="bf16"
    if [[ "${FP8}" -eq 1 ]]; then
        data_type="fp8"
    fi
    OUTPUT_DIR=${OUTPUT_DIR_PREFIX}/out/moonlight_16b_a3b/${data_type}_${TRANSFORMER_IMPL}_${EXP_NAME}_nl${NUM_LAYERS}_gb${GLOBAL_BATCH_SIZE}_mb${MICRO_BATCH_SIZE}_sp${SEQ_PARALLEL}_D${DP}_T${TP}_P${PP}_E${EP}_devices${NUM_DEVICES}_${RUNTIME}
fi
if [[ -z "${CHECKPOINTS_DIR}" ]]; then
    CHECKPOINTS_DIR=${OUTPUT_DIR}/checkpoints
fi
if [[ -z "${LOAD_DIR}" ]]; then
    LOAD_DIR=${CHECKPOINTS_DIR}
fi

if [[ -z "${TENSORBOARD_DIR}" ]]; then
    TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
fi
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINTS_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

if [[ "${NUM_NODES}" -ne "1" ]] && [[ -z "${HOSTSFILE}" ]] && [[ "${LAUNCHER_TYPE}" = "mpirun" ]]; then
    HOSTSFILE=${MEGATRON_LM_ROOT}/examples/hostsfile
    if [[ -f "${HOSTSFILE}" ]]; then
        cat /dev/null > "${HOSTSFILE}"
    fi
    etc_mpi_hostfile="/etc/mpi/hostfile"
    if [[ ! -f ${etc_mpi_hostfile} ]]; then
        echo "${etc_mpi_hostfile} not available, set HL_HOSTSFILE"
        exit 1
    fi
    cat ${etc_mpi_hostfile} | xargs -I{} echo {} slots=8 >> "${HOSTSFILE}"
fi

# Set training command
CMD=""
if [[ "${LAUNCHER_TYPE}" = "mpirun" ]]; then
    CMD="${CMD} mpirun"
    CMD="${CMD} --allow-run-as-root"
    CMD="${CMD} -n ${NUM_DEVICES}"
    CMD="${CMD} --bind-to none"
    if [[ "${NUM_NODES}" -ne "1" ]]; then
        CMD="${CMD} -hostfile ${HOSTSFILE}"
        CMD="${CMD} -x MASTER_ADDR=$(head -n 1 "${HOSTSFILE}" | sed -n s/[[:space:]]slots.*//p)"
    else
        CMD="${CMD} -x MASTER_ADDR=localhost"
    fi
    CMD="${CMD} -x MASTER_PORT=12345"
elif [[ "${LAUNCHER_TYPE}" = "torchrun" ]]; then
    #if [[ "${NUM_NODES}" -ne "1" ]]; then
    #    echo "NUM_NODES greater than 1 not supported by torchrun"
    #    exit 1
    #fi
    CMD="${CMD} torchrun"
    CMD="${CMD} --nnodes ${NUM_NODES}"
    CMD="${CMD} --nproc-per-node ${DEVICES_PER_NODE}"
    CMD="${CMD} --no-python"
    CMD="${CMD} --node-rank $TORCHRUN_NODE_RANK"
    CMD="${CMD} --master-addr $TORCHRUN_MASTER_ADDR"
    CMD="${CMD} --master-port 12345"
else
    echo "Unsupported launcher type = ${LAUNCHER_TYPE}"
    exit 2
fi

# prepare MoE args
MOE_ARGS="--num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${TOPK} \
    --expert-model-parallel-size ${EP} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-bias-update-rate 0.001 \
    --moe-router-score-function sigmoid \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-num-groups 1 \
    --moe-router-group-topk 1 \
    --moe-router-enable-expert-bias \
    --moe-aux-loss-coeff ${AUX_LOSS_COEFF} \
    --moe-token-dispatcher-type ${TOKEN_DISPATCHER_TYPE} \
    --moe-ffn-hidden-size ${EXPERT_FFN_HIDDEN_SIZE} \
    --moe-shared-expert-intermediate-size ${SHARED_EXPERT_FFN_HIDDEN_SIZE}"

if [[ "${TOKEN_DISPATCHER_TYPE}" = "alltoall" && ${ENABLE_SHARED_EXPERT_OVERLAP} -eq 1 ]]; then
    MOE_ARGS="${MOE_ARGS} \
        --moe-shared-expert-overlap"
fi

if [[ ${TOKEN_DROP} -eq 1 ]]; then
    MOE_ARGS="${MOE_ARGS} \
        --moe-expert-capacity-factor 1.0 \
        --moe-pad-expert-input-to-capacity \
        --moe-token-drop-policy probs"
fi

if [[ ${MOE_LAYER_RECOMPUTE} -eq 1 ]]; then
    MOE_ARGS="${MOE_ARGS} \
        --moe-layer-recompute"
fi

if [[ ${MOE_SKIP_FIRST_LAYERS} -gt 0 ]]; then
    NUM_MOE_LAYERS=$((NUM_LAYERS - MOE_SKIP_FIRST_LAYERS))
    MOE_ARGS="${MOE_ARGS} \
       --moe-layer-freq ([0]*${MOE_SKIP_FIRST_LAYERS}+[1]*${NUM_MOE_LAYERS})"
fi

if [[ ${USE_MOE_GROUPED_GEMM} -eq 1 ]]; then
    MOE_ARGS="${MOE_ARGS} \
        --moe-grouped-gemm"
fi

FISRT_PP_STAGE_LAYERS_ARG=""
if [[ ${FISRT_PP_STAGE_LAYERS} -gt 0 ]]; then
    FISRT_PP_STAGE_LAYERS_ARG="--decoder-first-pipeline-num-layers ${FISRT_PP_STAGE_LAYERS}"
fi

LAST_PP_STAGE_LAYERS_ARG=""
if [[ ${LAST_PP_STAGE_LAYERS} -gt 0 ]]; then
    LAST_PP_STAGE_LAYERS_ARG="--decoder-last-pipeline-num-layers ${LAST_PP_STAGE_LAYERS}"
fi

MLA_ARGS="--multi-latent-attention \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_HEAD_DIM} \
    --qk-layernorm \
    --qk-pos-emb-head-dim ${QK_POS_EMB_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM}"

if [[ ${Q_LORA_RANK} -gt 0 ]]; then
    MLA_ARGS="${MLA_ARGS} \
        --q-lora-rank ${Q_LORA_RANK}"
fi

CMD="${CMD} \
    python ${SRC_PATH} \
    --transformer-impl ${TRANSFORMER_IMPL} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    ${FISRT_PP_STAGE_LAYERS_ARG} \
    ${LAST_PP_STAGE_LAYERS_ARG} \
    --distributed-backend nccl \
    --seq-length ${SEQ_LEN} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --num-query-groups ${NUM_KV_HEADS} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --position-embedding-type rope \
    --no-rope-fusion \
    --rotary-base ${ROTARY_BASE} \
    --max-position-embeddings ${MAX_SEQ_LEN} \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --attention-dropout ${DROPOUT} \
    --hidden-dropout ${DROPOUT} \
    ${MOE_ARGS} \
    ${MLA_ARGS} \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 ${ADAM_BETA1}\
    --adam-beta2 ${ADAM_BETA2} \
    --adam-eps ${ADAM_EPS} \
    --lr ${LR} \
    --lr-decay-style cosine \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --min-lr ${MIN_LR} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-iters ${LR_DECAY_ITER} \
    --log-interval ${LOG_INTERVAL} \
    --log-throughput \
    --disable-bias-linear \
    --optimizer ${OPTIMIZER} \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --use-mcore-models \
    --bf16 \
    --exit-interval ${EXIT_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --load ${LOAD_DIR} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --data-path ${DATA_PATH} \
    --init-method-std ${INIT_STD} \
    --no-check-for-nan-in-loss-and-grad \
    --num-workers ${NUM_WORKERS} \
    --finetune \
    --seed 1111 \
    "
# --log-memory-to-tensorboard

if [[ ${ENABLE_PARAM_GATHER_OVERLAP} -eq 1 ]]; then
    CMD="${CMD} --overlap-param-gather"
fi

if [[ ${ENABLE_GRAD_REDUCE_OVERLAP} -eq 1 ]]; then
    CMD="${CMD} --overlap-grad-reduce"
fi

if [[ "${SEQ_PARALLEL}" -eq 1 ]]; then
    CMD="${CMD} --sequence-parallel"
fi

if [[ "${CKP_ACT}" -eq 1 ]]; then
    CMD="${CMD} --recompute-granularity=full"
    CMD="${CMD} --recompute-method uniform"
    CMD="${CMD} --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
elif [[ "${CKP_ACT}" -eq 2 ]]; then
    CMD="${CMD} --recompute-granularity selective"
fi

if [[ "${USE_DISTRIBUTED_OPTIMIZER}" -eq 1 ]]; then
    CMD="${CMD} --use-distributed-optimizer"
fi

if [[ "${DETERMINISTIC_MODE}" -eq 1 ]]; then
    CMD="${CMD} --deterministic-mode"
fi

# profile
if [[ "${PROFILE}" -eq 1 ]]; then
    CMD="${CMD} --profile"
    CMD="${CMD} --profile-step-start ${PROFILE_STEP_START}"
    CMD="${CMD} --profile-step-end ${PROFILE_STEP_END}"
    if [ -n "${PROFILE_RANKS}" ]; then
        CMD="${CMD} --profile-ranks ${PROFILE_RANKS}"
    fi
    if [[ "${PROFILE_PYTORCH}" -eq 1 ]]; then
        CMD="${CMD} --use-pytorch-profiler"
    fi
fi

if [[ "${CHECKPOINT_SAVE}" -eq 1 ]]; then
    CMD="${CMD} --save ${CHECKPOINTS_DIR}"
    CMD="${CMD} --save-interval ${SAVE_INTERVAL}"
    CMD="${CMD} --dist-ckpt-format ${DIST_CKPT_FORMAT}"
fi

if [[ "${USE_DIST_CKPT}" -eq 1 ]]; then
    CMD="${CMD} --use-dist-ckpt"
fi

if [[ "${TOKENIZER_TYPE}" = "HuggingFaceTokenizer" || "${TOKENIZER_TYPE}" = "GPTSentencePieceTokenizer" || "${TOKENIZER_TYPE}" = "Llama2Tokenizer" || "${TOKENIZER_TYPE}" = "Llama3Tokenizer" || "${TOKENIZER_TYPE}" = "TikTokenizer" ]]; then
    CMD="${CMD} --tokenizer-type ${TOKENIZER_TYPE}"
    CMD="${CMD} --tokenizer-model ${TOKENIZER_MODEL}"
elif [[ "${TOKENIZER_TYPE}" = "DeepSeekV2Tokenizer" ]]; then
    CMD="${CMD} --tokenizer-type ${TOKENIZER_TYPE}"
    CMD="${CMD} --tokenizer-model ${TOKENIZER_MODEL}"
    if [[ "${EXTRA_VOCAB_SIZE}" -gt 0 ]]; then
        CMD="${CMD} --extra-vocab-size ${EXTRA_VOCAB_SIZE}"
    fi
elif [[ "${TOKENIZER_TYPE}" = "GPT2BPETokenizer" ]]; then
    CMD="${CMD} --tokenizer-type GPT2BPETokenizer"
    CMD="${CMD} --vocab-file ${DATA_DIR}/gpt2-vocab.json"
    CMD="${CMD} --merge-file ${DATA_DIR}/gpt2-merges.txt"
else
    echo "incorrect HL_TOKENIZER_TYPE=${TOKENIZER_TYPE} is set"
    exit 1
fi

if [[ -n "${DATA_CACHE_DIR}" ]]; then
    CMD="${CMD} --data-cache-path ${DATA_CACHE_DIR}"
fi

if [[ "${REDIRECT_LOGS}" -eq 1 ]]; then
    ${CMD} 2>&1 | tee "${OUTPUT_DIR}"/log_"${EXP_NAME}"_"${RUNTIME}".txt
else
    ${CMD}
fi
