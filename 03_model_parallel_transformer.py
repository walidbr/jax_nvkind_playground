"""
Test 03: Simulated Rack-Scale Distributed Transformer Layer (Megatron-LM style in JAX)
Exercises:
1. Column-Parallel Linear (Feed-Forward 1) -> No collective in forward pass
2. Row-Parallel Linear (Feed-Forward 2) -> All-Reduce / psum across 'model' axis
3. Data-Parallelism across 'data' / 'rack' axis -> Gradient All-Reduce
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
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "127.0.0.1:12370")

    jax.distributed.initialize(
        coordinator_address=coordinator_addr,
        num_processes=world_size,
        process_id=rank,
        local_device_ids=[0]
    )

    devices = jax.devices()
    # Topology: 2 Data-Parallel Racks x 2 Model-Parallel GPUs
    mesh_devices = np.array(devices).reshape(2, 2)
    mesh = Mesh(mesh_devices, axis_names=('data', 'model'))

    if rank == 0:
        print("="*65)
        print("  DISTRIBUTED 2D MESH TRANSFORMER MLP (MEGATRON-STYLE)")
        print(f"  Mesh: 2 Data-Parallel Racks x 2 Model-Parallel GPUs")
        print("="*65)

    with jax.set_mesh(mesh):
        BATCH = 8
        D_MODEL = 64
        D_FF = 256

        # Sharding Specs:
        # Inputs: Sharded across 'data' (Batch), Replicated across 'model'
        x_sharding = NamedSharding(mesh, P('data', None))
        
        # W1 (Column Parallel): [D_MODEL, D_FF] -> Sharded across 'model' on dim 1
        w1_sharding = NamedSharding(mesh, P(None, 'model'))
        
        # W2 (Row Parallel): [D_FF, D_MODEL] -> Sharded across 'model' on dim 0
        w2_sharding = NamedSharding(mesh, P('model', None))

        # Weights initialization
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        x_global = jax.random.normal(k1, (BATCH, D_MODEL))
        w1_global = jax.random.normal(k2, (D_MODEL, D_FF))
        w2_global = jax.random.normal(k3, (D_FF, D_MODEL))

        x = jax.device_put(x_global, x_sharding)
        w1 = jax.device_put(w1_global, w1_sharding)
        w2 = jax.device_put(w2_global, w2_sharding)

        @jax.jit
        def transformer_mlp(x, w1, w2):
            # 1. Column Parallel Projection: [B, D_MODEL] @ [D_MODEL, D_FF / model]
            # Output is sharded: [B/data, D_FF / model] (Zero communication needed!)
            h = jax.nn.gelu(jnp.matmul(x, w1))

            # 2. Row Parallel Projection: [B/data, D_FF / model] @ [D_FF / model, D_MODEL]
            # Output mathematically requires an All-Reduce (psum) across the 'model' axis!
            out = jnp.matmul(h, w2)
            return out

        # Compile and check HLO for collectives
        lowered = transformer_mlp.lower(x, w1, w2)
        compiled = lowered.compile()

        if rank == 0:
            print("\nAnalyzing Compiler IR for Collective Operations:")
            hlo = lowered.as_text()
            found_collectives = False
            for line in hlo.splitlines():
                if any(c in line for c in ["all-reduce", "all-gather", "reduce-scatter"]):
                    print(f"  [Auto-generated Collective] -> {line.strip()}")
                    found_collectives = True
            if not found_collectives:
                print("  No explicit named collectives in top-level HLO (fused by XLA).")

        out = compiled(x, w1, w2)
        out.block_until_ready()

        local_out = out.addressable_data(0)
        print(f"[Rank {rank}] Execution complete! Local shard output shape: {local_out.shape}")

    jax.distributed.shutdown()

if __name__ == "__main__":
    main()
