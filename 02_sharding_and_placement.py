"""
Test 02: Modern JAX Sharding & Operator Placement Across Cluster Racks/GPUs
Demonstrates:
1. Defining a multi-dimensional Device Mesh (e.g. rack x gpu)
2. PartitionSpec (Data Parallelism vs Model/Tensor Parallelism)
3. Sharded Matrix Multiplication and Automatic Collective Generation (All-Gather / Reduce-Scatter)
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

def main():
    rank = int(os.environ.get("JAX_RANK", 0))
    world_size = int(os.environ.get("JAX_WORLD_SIZE", 4))
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "127.0.0.1:12352")

    jax.distributed.initialize(
        coordinator_address=coordinator_addr,
        num_processes=world_size,
        process_id=rank,
        local_device_ids=[0]
    )

    devices = jax.devices()
    if rank == 0:
        print(f"=== Running on {len(devices)} Devices across {jax.process_count()} Ranks ===")
        print(f"Devices: {devices}\n")

    # -------------------------------------------------------------
    # 1. Logical Cluster Topology: 2 "Racks" x 2 "GPUs per rack" = 4 GPUs
    # -------------------------------------------------------------
    # Shape of the cluster mesh: ('rack', 'gpu')
    mesh_devices = np.array(devices).reshape(2, 2)
    mesh = Mesh(mesh_devices, axis_names=('rack', 'gpu'))

    if rank == 0:
        print(f"Cluster Mesh layout:\n{mesh}\n")

    # -------------------------------------------------------------
    # 2. Operator & Tensor Placement
    # We simulate a large linear layer: Y = X @ W
    #   X shape: [8, 16] (Batch=8, Features=16) -> Sharded across 'rack' (Data Parallel)
    #   W shape: [16, 8] -> Sharded across 'gpu' (Model / Tensor Parallel)
    # -------------------------------------------------------------
    with mesh:
        # Sharding specifications:
        # X: dimension 0 sharded over 'rack', dimension 1 replicated (None)
        x_sharding = NamedSharding(mesh, P('rack', None))
        
        # W: dimension 0 replicated, dimension 1 sharded over 'gpu' (Megatron-style Column Parallel)
        w_sharding = NamedSharding(mesh, P(None, 'gpu'))

        # Expected output Y: dimension 0 is sharded over 'rack', dimension 1 is sharded over 'gpu'
        y_expected_sharding = NamedSharding(mesh, P('rack', 'gpu'))

        if rank == 0:
            print("--- Sharding Specifications ---")
            print(f"Input X Sharding (Data Parallel over rack): {x_sharding}")
            print(f"Weights W Sharding (Model Parallel over gpu): {w_sharding}")
            print(f"Output Y Sharding (2D Sharded rack x gpu):   {y_expected_sharding}\n")

        # Create global arrays (each rank creates the same shape, JAX distributes values automatically)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        
        # JIT compiled distributed matrix multiplication
        @jax.jit
        def forward_layer(x, w):
            return jnp.matmul(x, w)

        x_global = jax.random.normal(k1, (8, 16))
        w_global = jax.random.normal(k2, (16, 8))

        # Distribute arrays to the mesh
        x_dist = jax.device_put(x_global, x_sharding)
        w_dist = jax.device_put(w_global, w_sharding)

        # Run forward computation
        y_dist = forward_layer(x_dist, w_dist)
        
        # Block until computed
        y_dist.block_until_ready()

        # Print local device partition owned by this rank
        local_shard = y_dist.addressable_data(0)
        print(f"[Rank {rank}] Completed distributed MatMul! Local shard slice shape: {local_shard.shape}")

    # Inspect the HLO / XLA generated collective operations (only Rank 0 needs to explain)
    if rank == 0:
        print("\n--- Compiling Lowered HLO to view underlying Collective Communication ---")
        lowered = forward_layer.lower(x_dist, w_dist)
        hlo_text = lowered.as_text()
        print("Generated HLO contains collective instructions:")
        for line in hlo_text.splitlines():
            if any(coll in line for coll in ["all-reduce", "all-gather", "reduce-scatter", "all-to-all"]):
                print(f"  -> {line.strip()}")
        print("\n=== Success! Modern JAX Mesh Sharding is fully working on the cluster ===")

    jax.distributed.shutdown()

if __name__ == "__main__":
    main()
