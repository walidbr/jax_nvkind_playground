"""
Test 04: Multi-GPU Cluster Trace Generator
Every rank captures its local GPU & Host execution spans to its own folder:
  traces/rank_<id>/
A merger then synthesizes them into a single unified timeline:
  traces/unified_4gpu_cluster.trace.json.gz
which shows 4 discrete GPU tracks side-by-side in Perfetto!
"""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

def main():
    rank = int(os.environ.get("JAX_RANK", 0))
    world_size = int(os.environ.get("JAX_WORLD_SIZE", 4))
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "127.0.0.1:12385")
    base_trace_dir = os.environ.get("JAX_TRACE_DIR", "/home/walid/repo/jax_cluster_lab/traces")
    rank_trace_dir = os.path.join(base_trace_dir, f"rank_{rank}")

    jax.distributed.initialize(
        coordinator_address=coordinator_addr,
        num_processes=world_size,
        process_id=rank,
        local_device_ids=[0]
    )

    devices = jax.devices()
    mesh_devices = np.array(devices).reshape(2, 2)
    mesh = Mesh(mesh_devices, axis_names=('rack', 'gpu'))

    if rank == 0:
        print("="*65)
        print("  CAPTURING ALL 4 SIMULATED GPUS FOR PERFETTO")
        print("="*65)

    with jax.set_mesh(mesh):
        # Megatron-style sharded layer:
        # X sharded over rack, W sharded over gpu
        x_sharding = NamedSharding(mesh, P('rack', None))
        w1_sharding = NamedSharding(mesh, P(None, 'gpu'))
        w2_sharding = NamedSharding(mesh, P('gpu', None))

        @jax.jit
        def transformer_block(x, w1, w2):
            # Column Parallel (local MatMul + GELU)
            h = jax.nn.gelu(jnp.matmul(x, w1))
            # Row Parallel MatMul -> Triggers All-Reduce collective across 'gpu'
            out = jnp.matmul(h, w2)
            return out

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(rank), 3)
        x_global = jax.random.normal(k1, (1024, 2048), dtype=jnp.float32)
        w1_global = jax.random.normal(k2, (2048, 4096), dtype=jnp.float32)
        w2_global = jax.random.normal(k3, (4096, 2048), dtype=jnp.float32)

        x = jax.device_put(x_global, x_sharding)
        w1 = jax.device_put(w1_global, w1_sharding)
        w2 = jax.device_put(w2_global, w2_sharding)

        # Warmup
        out = transformer_block(x, w1, w2)
        out.block_until_ready()

        # Start profiler on THIS rank
        os.makedirs(rank_trace_dir, exist_ok=True)
        jax.profiler.start_trace(rank_trace_dir)

        # Run 5 synchronized distributed steps with named span annotations
        for step in range(5):
            with jax.profiler.TraceAnnotation(f"distributed_step_{step}_rank_{rank}"):
                out = transformer_block(x, w1, w2)
                out.block_until_ready()

        # Stop profiler
        jax.profiler.stop_trace()

    jax.distributed.shutdown()

if __name__ == "__main__":
    main()
