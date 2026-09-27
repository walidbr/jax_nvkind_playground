#!/usr/bin/env bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${DIR}/.venv/bin/python"

NUM_RANKS=${1:-4}
COORDINATOR_PORT=${2:-12360}
COORDINATOR_ADDR="127.0.0.1:${COORDINATOR_PORT}"
SCRIPT="${3:-${DIR}/02_sharding_and_placement.py}"

echo "=== Launching ${NUM_RANKS}-rank simulated GPU cluster on RTX 5090 ==="
echo "Coordinator: ${COORDINATOR_ADDR}"
echo "Script: ${SCRIPT}"

# Dynamic memory allocation with modest fraction
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.04
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=262144
export NCCL_MAX_NCHANNELS=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN

PIDS=()
# Launch rank 0 first
JAX_RANK=0 \
JAX_WORLD_SIZE=$NUM_RANKS \
JAX_COORDINATOR_ADDRESS=$COORDINATOR_ADDR \
CUDA_VISIBLE_DEVICES=0 \
$PYTHON "$SCRIPT" > "${DIR}/rank_0.log" 2>&1 &
PIDS+=($!)
echo "Spawned Rank 0 (PID ${PIDS[0]})"

sleep 1.2

for (( rank=1; rank<NUM_RANKS; rank++ )); do
    JAX_RANK=$rank \
    JAX_WORLD_SIZE=$NUM_RANKS \
    JAX_COORDINATOR_ADDRESS=$COORDINATOR_ADDR \
    CUDA_VISIBLE_DEVICES=0 \
    $PYTHON "$SCRIPT" > "${DIR}/rank_${rank}.log" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "Spawned Rank $rank (PID $PID)"
    sleep 0.4
done

echo "Waiting for all ranks to complete..."
STATUS=0
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        STATUS=1
    fi
done

echo ""
echo "=== Results from Ranks ==="
for (( rank=0; rank<NUM_RANKS; rank++ )); do
    echo "--- [Log: Rank ${rank}] ---"
    cat "${DIR}/rank_${rank}.log"
    echo ""
done

exit $STATUS
