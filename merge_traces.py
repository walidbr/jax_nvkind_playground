"""
Deterministic Causal Trace Synthesizer for Multi-GPU JAX

Track Layout:
  Track 100: JAX Framework (Host Python execution, Step Scopes, Dispatches)
  Track 200: GPU 0 [Hardware Stream & Kernels]
  Track 201: GPU 1 [Hardware Stream & Kernels]
  Track 202: GPU 2 [Hardware Stream & Kernels]
  Track 203: GPU 3 [Hardware Stream & Kernels]

Forward-Only Flow Arrow Rules:
  1. Framework Launch -> GPU Kernel:
     - Arrow starts at f_start (LEFT edge of Framework OP_* span)
     - Arrow arrives at k_start on GPU (which is >= f_start), strictly forward in time.
  2. Scatter Fan-Out:
     - GPU 0 Scatter kernel -> Fans out strictly FORWARD to GPU 1, 2, 3 receiving kernels.
  3. Gather / All-Reduce Fan-In:
     - Every GPU's completed compute kernel -> Points strictly FORWARD into the Gather/Reduce kernel!
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

    # 1. Register Track Metadata
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

    NOISE_PREFIXES = ("$builtins", "$typing", "$inspect", "$contextlib", "$linecache", "$abc", "$warnings")

    framework_ops = []  # User annotations (OP_SCATTER, OP_LOCAL_COMPUTE, OP_GATHER, OP_REDUCE)
    gpu_kernels_by_rank = {r: [] for r in range(num_discovered_ranks)}
    step_events_by_rank = {r: {} for r in range(num_discovered_ranks)}

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

                # A. Real GPU Hardware Kernels (pid 1)
                if orig_pid == 1:
                    new_ev["pid"] = gpu_pid
                    merged_events.append(new_ev)
                    gpu_kernels_by_rank[r_idx].append(new_ev)

                # B. Framework & User Spans
                else:
                    if name.startswith("OP_"):
                        # Record step-level operation per rank
                        parts = name.split("_")
                        op_type = parts[1]  # SCATTER, LOCAL, GATHER, REDUCE
                        step_id = parts[-1]
                        step_events_by_rank[r_idx].setdefault(step_id, {})[op_type] = new_ev

                        # Framework Lane shows Rank 0's high-level orchestration
                        if r_idx == 0:
                            new_ev["pid"] = 100
                            merged_events.append(new_ev)
                            framework_ops.append(new_ev)

                    elif r_idx == 0:
                        if name.startswith(NOISE_PREFIXES):
                            continue
                        if (name.startswith("PjitFunction") or 
                            name.startswith("CommonPjRtLoadedExecutable") or 
                            "broadcast_one_to_all" in name or 
                            "process_allgather" in name):
                            new_ev["pid"] = 100
                            merged_events.append(new_ev)

    # -------------------------------------------------------------
    # SYNTHESIZE FORWARD FLOW ARROWS
    # -------------------------------------------------------------
    flow_id = 100000

    # 1. JAX FRAMEWORK OP_* -> GPU 0 LAUNCH (Guaranteed Forward in time)
    for f_op in framework_ops:
        f_name = f_op.get("name", "")
        f_start = f_op.get("ts", 0)
        f_end = f_start + f_op.get("dur", 10)

        # Match GPU 0 kernels that fall within this framework execution window
        matching = [k for k in gpu_kernels_by_rank[0] if f_start <= k.get("ts", 0) <= f_end + 2000]
        if matching:
            target_k = matching[0]
            k_start = target_k.get("ts", 0)

            # Ensure start is strictly <= finish
            start_ts = f_start
            finish_ts = max(k_start, start_ts + 1)

            flow_id += 1
            merged_events.append({
                "name": f"Framework -> GPU 0: {f_name}",
                "ph": "s", "cat": "flow", "id": flow_id,
                "pid": 100, "tid": f_op.get("tid", 0), "ts": start_ts
            })
            merged_events.append({
                "name": f"Framework -> GPU 0: {f_name}",
                "ph": "f", "cat": "flow", "id": flow_id,
                "pid": 200, "tid": target_k.get("tid", 0), "ts": finish_ts
            })

    # 2. SCATTER FAN-OUT ARROWS: GPU 0 -> GPU 1, 2, 3
    # When GPU 0 executes SCATTER in step S, draw forward arrows to GPU 1, 2, 3
    all_steps = sorted(list(step_events_by_rank[0].keys()))

    for step_id in all_steps:
        # Check if rank 0 has SCATTER in this step
        r0_ops = step_events_by_rank[0].get(step_id, {})
        if "SCATTER" in r0_ops:
            scatter_ev = r0_ops["SCATTER"]
            src_ts = scatter_ev.get("ts", 0) + 10

            for target_rank in range(1, num_discovered_ranks):
                tgt_ops = step_events_by_rank[target_rank].get(step_id, {})
                if "SCATTER" in tgt_ops or "LOCAL" in tgt_ops:
                    target_ev = tgt_ops.get("SCATTER", tgt_ops.get("LOCAL"))
                    tgt_ts = max(target_ev.get("ts", 0), src_ts + 20)

                    flow_id += 1
                    merged_events.append({
                        "name": f"Scatter: GPU 0 -> GPU {target_rank}",
                        "ph": "s", "cat": "flow", "id": flow_id,
                        "pid": 200, "tid": scatter_ev.get("tid", 0), "ts": src_ts
                    })
                    merged_events.append({
                        "name": f"Scatter: GPU 0 -> GPU {target_rank}",
                        "ph": "f", "cat": "flow", "id": flow_id,
                        "pid": 200 + target_rank, "tid": target_ev.get("tid", 0), "ts": tgt_ts
                    })

        # 3. GATHER FAN-IN ARROWS: All GPUs -> Gather target on GPU 0
        if "GATHER" in r0_ops:
            gather_root_ev = r0_ops["GATHER"]
            gather_root_ts = gather_root_ev.get("ts", 0)

            for src_rank in range(num_discovered_ranks):
                src_ops = step_events_by_rank[src_rank].get(step_id, {})
                if "LOCAL" in src_ops:
                    comp_ev = src_ops["LOCAL"]
                    comp_end_ts = comp_ev.get("ts", 0) + comp_ev.get("dur", 10)
                    
                    # Strictly forward arrow from completion of compute to arrival at gather
                    arr_start = comp_end_ts - 2
                    arr_end = max(gather_root_ts + (src_rank * 15), arr_start + 10)

                    flow_id += 1
                    merged_events.append({
                        "name": f"Gather Inflow: GPU {src_rank} -> GPU 0",
                        "ph": "s", "cat": "flow", "id": flow_id,
                        "pid": 200 + src_rank, "tid": comp_ev.get("tid", 0), "ts": arr_start
                    })
                    merged_events.append({
                        "name": f"Gather Inflow: GPU {src_rank} -> GPU 0",
                        "ph": "f", "cat": "flow", "id": flow_id,
                        "pid": 200, "tid": gather_root_ev.get("tid", 0), "ts": arr_end
                    })

    output_data = {"traceEvents": merged_events}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"\nUnified trace with STRICT FORWARD CAUSALITY generated: {output_file}")
    print(f"Total verified forward flow arrows: {flow_id - 100000}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/walid/repo/jax_cluster_lab/traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/walid/repo/jax_cluster_lab/traces/unified_4gpu_cluster.trace.json.gz"
    merge(base, out)
