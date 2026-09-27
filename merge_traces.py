"""
Clean Evidence-Based Multi-GPU Trace Synthesizer

Architectural Design:
  Lane 100: JAX Framework
    - Contains clean Python execution scopes:
      - STEP_<step>_DISPATCH
      - OP_SCATTER_STEP_<step>
      - OP_GATHER_STEP_<step>
      - High-level PJRT / XLA compile & dispatch calls
    - Stripped of Python internal noise ($builtins len, isinstance, getattr, etc.)
  
  Lanes 200..203: Real Hardware GPUs (GPU 0, 1, 2, 3)
    - Contain ONLY true GPU device activities (CUPTI CUDA streams):
      - MatMul kernels (gemm)
      - Elementwise compute kernels
      - NCCL Collective kernels (ncclDevKernel_AllGather, ncclDevKernel_AllReduce)
      - Hardware Memcpy (MemcpyH2D / D2H)

Flow Arrows:
  1. Framework Dispatch -> Pointing FORWARD to the GPU hardware kernel that executes it.
  2. Scatter: GPU 0 hardware kernel -> Pointing FORWARD to GPU 1, 2, 3 hardware kernels.
  3. Gather: Each GPU's hardware compute kernel completion -> Pointing FORWARD to the NCCL Gather kernel!
"""
import os
import glob
import gzip
import json
import sys

def merge(trace_base_dir, output_file):
    merged_events = []
    
    rank_dirs = sorted(glob.glob(os.path.join(trace_base_dir, "rank_*")))
    if not rank_dirs:
        print(f"No rank trace directories found in {trace_base_dir}")
        return

    num_discovered_ranks = len(rank_dirs)

    # 1. Register Lanes
    merged_events.append({
        "name": "process_name", "ph": "M", "pid": 100,
        "args": {"name": "JAX Framework [Host Runtime & Python Dispatch]"}
    })
    merged_events.append({
        "name": "process_sort_index", "ph": "M", "pid": 100,
        "args": {"sort_index": 0}
    })

    for r_idx in range(num_discovered_ranks):
        gpu_pid = 200 + r_idx
        merged_events.append({
            "name": "process_name", "ph": "M", "pid": gpu_pid,
            "args": {"name": f"GPU {r_idx} [Hardware Stream & Kernels]"}
        })
        merged_events.append({
            "name": "process_sort_index", "ph": "M", "pid": gpu_pid,
            "args": {"sort_index": r_idx + 1}
        })

    # Noise filter for Python builtins
    NOISE_PREFIXES = ("$builtins", "$typing", "$inspect", "$contextlib", "$linecache", "$abc", "$warnings")

    framework_ops = []  # High-level operations in Framework lane
    gpu_kernels_by_rank = {r: [] for r in range(num_discovered_ranks)}

    for r_idx, r_dir in enumerate(rank_dirs):
        gz_files = glob.glob(os.path.join(r_dir, "plugins/profile/*/*.trace.json.gz"))
        if not gz_files:
            continue
        
        gz_path = gz_files[0]
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
            events = data.get("traceEvents", [])
            gpu_pid = 200 + r_idx

            for ev in events:
                name = ev.get("name", "")
                if name in ["process_name", "process_sort_index"]:
                    continue

                orig_pid = ev.get("pid", 0)
                new_ev = dict(ev)

                # A. TRUE HARDWARE GPU EVENTS (pid 1) -> Go to GPU Lane
                if orig_pid == 1:
                    new_ev["pid"] = gpu_pid
                    merged_events.append(new_ev)
                    gpu_kernels_by_rank[r_idx].append(new_ev)

                # B. FRAMEWORK HOST EVENTS -> Go to Framework Lane (Take from Rank 0 only)
                elif r_idx == 0:
                    # Filter out micro Python interpreter noise like $builtins len
                    if name.startswith(NOISE_PREFIXES):
                        continue
                    
                    # Only keep significant framework spans (user annotations, PJRT executes, collectives)
                    if (name.startswith("OP_") or 
                        name.startswith("PjitFunction") or 
                        name.startswith("CommonPjRtLoadedExecutable") or 
                        "broadcast_one_to_all" in name or 
                        "process_allgather" in name or
                        name in ["MemcpyH2D", "H2D Dispatch"]):
                        
                        new_ev["pid"] = 100
                        merged_events.append(new_ev)

                        if name.startswith("OP_"):
                            framework_ops.append(new_ev)

    # -------------------------------------------------------------
    # SYNTHESIZE FORWARD CAUSAL ARROWS
    # -------------------------------------------------------------
    flow_id = 95000

    # 1. Connect JAX Framework OP_* spans down to the first GPU kernel executed in that span
    for f_op in framework_ops:
        op_name = f_op.get("name", "")
        f_start = f_op.get("ts", 0)
        f_end = f_start + f_op.get("dur", 10)

        # Find the earliest GPU kernel on GPU 0 that falls within or immediately following this framework span
        matching_gpu_kernels = [
            k for k in gpu_kernels_by_rank[0] 
            if f_start <= k.get("ts", 0) <= f_end + 5000
        ]

        if matching_gpu_kernels:
            first_gpu_k = matching_gpu_kernels[0]
            flow_id += 1
            # Arrow: Framework OP span -> First GPU hardware kernel
            merged_events.append({
                "name": f"Framework Launch -> {first_gpu_k.get('name')}",
                "ph": "s", "cat": "flow", "id": flow_id,
                "pid": 100, "tid": f_op.get("tid", 0), "ts": f_start + 5
            })
            merged_events.append({
                "name": f"Framework Launch -> {first_gpu_k.get('name')}",
                "ph": "f", "cat": "flow", "id": flow_id,
                "pid": 200, "tid": first_gpu_k.get("tid", 0), "ts": first_gpu_k.get("ts", 0)
            })

    # 2. Inter-GPU Collectives (NCCL AllGather / AllReduce)
    # When NCCL kernels execute across GPUs around the same timestamp, connect them!
    all_nccl_kernels = []
    for r in range(num_discovered_ranks):
        for k in gpu_kernels_by_rank[r]:
            kname = k.get("name", "")
            if "nccl" in kname.lower() or "allreduce" in kname.lower() or "allgather" in kname.lower():
                all_nccl_kernels.append((r, k))

    # Sort all collective kernels by timestamp
    all_nccl_kernels.sort(key=lambda x: x[1].get("ts", 0))

    # Group NCCL kernels that execute concurrently (within 200 microseconds of each other)
    collective_clusters = []
    current_cluster = []

    for r, k in all_nccl_kernels:
        k_ts = k.get("ts", 0)
        if not current_cluster:
            current_cluster.append((r, k))
        else:
            first_ts = current_cluster[0][1].get("ts", 0)
            if abs(k_ts - first_ts) < 500:  # Within 500us
                current_cluster.append((r, k))
            else:
                collective_clusters.append(current_cluster)
                current_cluster = [(r, k)]
    if current_cluster:
        collective_clusters.append(current_cluster)

    # For each collective cluster, draw forward flow arrows between the participating GPUs
    for cluster in collective_clusters:
        if len(cluster) >= 2:
            root_rank, root_k = cluster[0]
            root_ts = root_k.get("ts", 0)

            for peer_rank, peer_k in cluster[1:]:
                peer_ts = peer_k.get("ts", 0)
                # Strict forward in time
                arr_start = root_ts
                arr_end = max(peer_ts, arr_start + 1)

                flow_id += 1
                merged_events.append({
                    "name": f"NCCL Sync: GPU {root_rank} -> GPU {peer_rank}",
                    "ph": "s", "cat": "flow", "id": flow_id,
                    "pid": 200 + root_rank, "tid": root_k.get("tid", 0), "ts": arr_start
                })
                merged_events.append({
                    "name": f"NCCL Sync: GPU {root_rank} -> GPU {peer_rank}",
                    "ph": "f", "cat": "flow", "id": flow_id,
                    "pid": 200 + peer_rank, "tid": peer_k.get("tid", 0), "ts": arr_end
                })

    output_data = {"traceEvents": merged_events}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"\nUnified clean trace generated: {output_file}")
    print(f"Total causal flow arrows synthesized: {flow_id - 95000}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/walid/repo/jax_cluster_lab/traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/walid/repo/jax_cluster_lab/traces/unified_4gpu_cluster.trace.json.gz"
    merge(base, out)
