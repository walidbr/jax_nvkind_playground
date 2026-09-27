"""
Test 06: Heterogeneous Partial Collectives (Proving Evidence-Based Visualization)
In a 4-GPU cluster:
  - Step 1: Collective scatter/gather runs ONLY across 2 GPUs (GPU 0 and GPU 1).
            GPU 2 and GPU 3 are IDLE / doing local tasks.
  - Step 2: Collective runs across all 4 GPUs.
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
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "127.0.0.1:12390")
    base_trace_dir = os.environ.get("JAX_TRACE_DIR", "/home/walid/repo/jax_cluster_lab/traces")
    rank_trace_dir = os.path.join(base_trace_dir, f"rank_{rank}")

    jax.distributed.initialize(
        coordinator_address=coordinator_addr,
        num_processes=world_size,
        process_id=rank,
        local_device_ids=[0]
    )

    val = jnp.ones((4, 4), dtype=jnp.float32) * (rank + 1)

    # Warmup
    _ = multihost_utils.process_allgather(val)

    os.makedirs(rank_trace_dir, exist_ok=True)
    jax.profiler.start_trace(rank_trace_dir)

    # -------------------------------------------------------------
    # PHASE 1: SUB-CLUSTER COLLECTIVE (ONLY GPU 0 & GPU 1 PARTICIPATE)
    # -------------------------------------------------------------
    if rank in [0, 1]:
        with jax.profiler.TraceAnnotation("OP_SCATTER_SUBGROUP_2GPU"):
            scatter_val = jnp.dot(val, val) * 2.0
            scatter_val.block_until_ready()

        with jax.profiler.TraceAnnotation("OP_LOCAL_COMPUTE_2GPU"):
            comp = jnp.sin(scatter_val)
            comp.block_until_ready()

        with jax.profiler.TraceAnnotation("OP_GATHER_SUBGROUP_2GPU"):
            # Only GPU 0 and GPU 1 record gather
            pass
    else:
        # GPU 2 and 3 do completely independent local work (no collective)
        with jax.profiler.TraceAnnotation("OP_INDEPENDENT_LOCAL_WORK"):
            idle_work = jnp.cos(val) * 5.0
            idle_work.block_until_ready()

    # -------------------------------------------------------------
    # PHASE 2: FULL CLUSTER ALL-GATHER (ALL 4 PARTICIPATE)
    # -------------------------------------------------------------
    with jax.profiler.TraceAnnotation("OP_LOCAL_COMPUTE_ALL"):
        local_res = jnp.dot(val, val)
        local_res.block_until_ready()

    with jax.profiler.TraceAnnotation("OP_GATHER_ALL_4GPU"):
        all_res = multihost_utils.process_allgather(local_res)

    jax.profiler.stop_trace()
    jax.distributed.shutdown()

if __name__ == "__main__":
    main()
