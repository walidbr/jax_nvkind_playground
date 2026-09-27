"""
Test 05: Explicit Collective Scatter, Gather, and Reduction with Flow-Traceable Markers
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

def main():
    rank = int(os.environ.get("JAX_RANK", 0))
    world_size = int(os.environ.get("JAX_WORLD_SIZE", 4))
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "127.0.0.1:12388")
    base_trace_dir = os.environ.get("JAX_TRACE_DIR", "/home/walid/repo/jax_cluster_lab/traces")
    rank_trace_dir = os.path.join(base_trace_dir, f"rank_{rank}")

    jax.distributed.initialize(
        coordinator_address=coordinator_addr,
        num_processes=world_size,
        process_id=rank,
        local_device_ids=[0]
    )

    if rank == 0:
        print("="*65)
        print("  SCATTER-GATHER & REDUCTION COLLECTIVES WITH FLOW TRACING")
        print("="*65)

    # Initial array
    val = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) * (rank + 1)

    # Warmup
    _ = multihost_utils.process_allgather(val)

    # Start profiling on this rank
    os.makedirs(rank_trace_dir, exist_ok=True)
    jax.profiler.start_trace(rank_trace_dir)

    for step in range(3):
        # 1. SCATTER / BROADCAST PHASE (GPU 0 -> All GPUs)
        with jax.profiler.TraceAnnotation(f"OP_SCATTER_STEP_{step}"):
            scattered_chunk = multihost_utils.broadcast_one_to_all(val, is_source=(rank == 0))

        # 2. LOCAL COMPUTE KERNEL ON EACH GPU
        with jax.profiler.TraceAnnotation(f"OP_LOCAL_COMPUTE_STEP_{step}"):
            local_processed = jnp.dot(scattered_chunk, scattered_chunk.T) + (rank * 10.0)
            local_processed.block_until_ready()

        # 3. GATHER / ALL-GATHER PHASE (All GPUs exchange slices)
        with jax.profiler.TraceAnnotation(f"OP_GATHER_STEP_{step}"):
            gathered_all = multihost_utils.process_allgather(local_processed)

        # 4. REDUCTION PHASE (Sum aggregation)
        with jax.profiler.TraceAnnotation(f"OP_REDUCE_STEP_{step}"):
            reduced = jnp.sum(gathered_all, axis=0)

    jax.profiler.stop_trace()
    jax.distributed.shutdown()

if __name__ == "__main__":
    main()
