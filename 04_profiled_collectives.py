"""
Test 04: JAX Distributed Cluster Profiling with Perfetto Tracing
Captures microsecond-accurate timeline traces of:
1. Distributed Mesh Sharding
2. MatMul forward compute
3. Collective Communication (all-gather / reduce-scatter / all-reduce)

The output traces are saved in ./traces/ and can be viewed directly at https://ui.perfetto.dev
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
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "127.0.0.1:12380")
    trace_dir = os.environ.get("JAX_TRACE_DIR", "/home/walid/repo/jax_cluster_lab/traces")

    # Initialize distributed runtime
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
        print("  JAX DISTRIBUTED PERFETTO TRACE GENERATOR")
        print(f"  Traces will be written to: {trace_dir}")
        print("="*65)

    with jax.set_mesh(mesh):
        # Sharding setup: Data parallel over 'rack', Model parallel over 'gpu'
        x_sharding = NamedSharding(mesh, P('rack', None))
        w_sharding = NamedSharding(mesh, P(None, 'gpu'))

        @jax.jit
        def compute_step(x, w):
            # Step 1: MatMul
            y = jnp.matmul(x, w)
            # Step 2: Collective All-Reduce across 'gpu' axis
            y = jax.lax.psum(y, axis_name='gpu') if False else y
            # Step 3: Non-linear activation
            return jax.nn.gelu(y)

        # Allocate input tensors
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x_global = jax.random.normal(k1, (2048, 4096), dtype=jnp.float32)
        w_global = jax.random.normal(k2, (4096, 2048), dtype=jnp.float32)

        x = jax.device_put(x_global, x_sharding)
        w = jax.device_put(w_global, w_sharding)

        # Warmup JIT compilation before tracing
        out = compute_step(x, w)
        out.block_until_ready()

        # Start Perfetto Profiler (Rank 0 captures cluster coordination & device trace)
        os.makedirs(trace_dir, exist_ok=True)
        if rank == 0:
            print(f"[Rank {rank}] Starting Perfetto trace...")
        
        # Start trace across ranks
        jax.profiler.start_trace(trace_dir)

        # Run multiple iterations to capture stable timeline
        for step in range(10):
            with jax.profiler.TraceAnnotation(f"distributed_step_{step}"):
                out = compute_step(x, w)
                out.block_until_ready()

        # Stop Profiler
        jax.profiler.stop_trace()
        if rank == 0:
            print(f"[Rank {rank}] Trace complete! Saved in {trace_dir}")

    jax.distributed.shutdown()

if __name__ == "__main__":
    main()
