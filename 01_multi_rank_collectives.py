"""
Test 01: Simulating Multi-Rank JAX on a Single GPU
Launches N ranks communicating via JAX's distributed coordinator service (over TCP)
and executing NCCL collective operations.
"""
import os
import sys
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

def main():
    rank = int(os.environ.get("JAX_RANK", 0))
    world_size = int(os.environ.get("JAX_WORLD_SIZE", 4))
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "127.0.0.1:12345")

    print(f"[Rank {rank}/{world_size}] Initializing distributed runtime...", flush=True)

    # Initialize distributed JAX
    jax.distributed.initialize(
        coordinator_address=coordinator_addr,
        num_processes=world_size,
        process_id=rank,
        local_device_ids=[0]
    )

    print(f"[Rank {rank}/{world_size}] Initialized successfully! Process index: {jax.process_index()}, Local devices: {jax.local_devices()}", flush=True)

    # 1. Collective: Each rank holds a unique vector
    val = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32) * (rank + 1)
    print(f"[Rank {rank}] Local value: {val}", flush=True)

    # 2. All-Gather across all simulated ranks in the cluster
    gathered_all = multihost_utils.process_allgather(val)
    print(f"[Rank {rank}] process_allgather result across all ranks:\n{gathered_all}", flush=True)

    # 3. All-Reduce equivalent
    all_reduced = jnp.sum(gathered_all, axis=0)
    print(f"[Rank {rank}] All-Reduce result: {all_reduced}", flush=True)

    jax.distributed.shutdown()
    print(f"[Rank {rank}] Shutdown complete.", flush=True)

if __name__ == "__main__":
    main()
