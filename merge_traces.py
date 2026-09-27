"""
Merges distributed JAX profile into 5 distinct Perfetto lanes:
  Lane 0: JAX Framework (Coordinator & Step Dispatch)
  Lane 1: GPU 0
  Lane 2: GPU 1
  Lane 3: GPU 2
  Lane 4: GPU 3

SYNTHESIZES STRICT FORWARD FLOW ARROWS (Cause in past -> Effect in future):
  1. JAX Framework Dispatch -> GPU 0, 1, 2, 3 Scatter events in the future
  2. Scatter Phase: GPU 0 root completes -> GPU 1, 2, 3 receive & start local compute in the future
  3. Local Compute completes -> Triggers Gather / All-Reduce collective in the future
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

    # Lane 100: JAX Framework
    merged_events.append({
        "name": "process_name", "ph": "M", "pid": 100,
        "args": {"name": "JAX Framework [Cluster Coordinator & Step Dispatch]"}
    })
    merged_events.append({
        "name": "process_sort_index", "ph": "M", "pid": 100,
        "args": {"sort_index": 0}
    })

    # Lanes 200..203: The 4 GPUs
    for r_idx in range(len(rank_dirs)):
        gpu_pid = 200 + r_idx
        merged_events.append({
            "name": "process_name", "ph": "M", "pid": gpu_pid,
            "args": {"name": f"GPU {r_idx} [Hardware Stream & Kernels]"}
        })
        merged_events.append({
            "name": "process_sort_index", "ph": "M", "pid": gpu_pid,
            "args": {"sort_index": r_idx + 1}
        })

    # Index operations by (step, rank, op_type)
    ops_by_step = {}  # step -> { 'SCATTER': {rank: ev}, 'COMPUTE': {rank: ev}, 'GATHER': {rank: ev}, 'REDUCE': {rank: ev} }
    framework_dispatches = {} # step -> ev

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
                if ev.get("name") in ["process_name", "process_sort_index"]:
                    continue
                
                orig_pid = ev.get("pid", 0)
                name = ev.get("name", "")
                new_ev = dict(ev)

                # Hardware GPU events
                if orig_pid == 1:
                    new_ev["pid"] = gpu_pid
                    merged_events.append(new_ev)
                else:
                    if name.startswith("OP_"):
                        new_ev["pid"] = gpu_pid
                        merged_events.append(new_ev)

                        # Extract step index: OP_<TYPE>_STEP_<step>
                        parts = name.split("_")
                        if len(parts) >= 4:
                            op_type = parts[1]  # SCATTER, LOCAL, GATHER, REDUCE
                            try:
                                step_num = int(parts[-1])
                            except ValueError:
                                step_num = 0

                            step_dict = ops_by_step.setdefault(step_num, {})
                            step_dict.setdefault(op_type, {})[r_idx] = new_ev

                    elif r_idx == 0:
                        new_ev["pid"] = 100
                        merged_events.append(new_ev)
                        if "PjitFunction" in name or "Execute" in name:
                            # Candidate for framework dispatch
                            pass

    # -------------------------------------------------------------
    # SYNTHESIZE STRICT FORWARD FLOW ARROWS (ts_start <= ts_end)
    # -------------------------------------------------------------
    flow_id = 60000

    for step_num, step_ops in sorted(ops_by_step.items()):
        scatter_ranks = step_ops.get("SCATTER", {})
        compute_ranks = step_ops.get("LOCAL", {})
        gather_ranks = step_ops.get("GATHER", {})
        reduce_ranks = step_ops.get("REDUCE", {})

        # A. FRAMEWORK DISPATCH -> FORWARD TO GPU 0 (Root Scatter)
        if 0 in scatter_ranks:
            gpu0_scatter = scatter_ranks[0]
            flow_id += 1
            # Dispatch event slightly before scatter
            dispatch_ts = gpu0_scatter.get("ts", 0) - 200  # 200 microseconds prior on framework
            merged_events.append({
                "name": f"Framework Dispatch Step {step_num}",
                "ph": "X",
                "pid": 100,
                "tid": 100,
                "ts": dispatch_ts,
                "dur": 150,
                "cat": "framework"
            })
            # Forward arrow: Framework -> GPU 0
            merged_events.append({
                "name": f"Dispatch -> GPU 0 Scatter",
                "ph": "s", "cat": "flow", "id": flow_id,
                "pid": 100, "tid": 100, "ts": dispatch_ts + 75
            })
            merged_events.append({
                "name": f"Dispatch -> GPU 0 Scatter",
                "ph": "f", "cat": "flow", "id": flow_id,
                "pid": 200, "tid": gpu0_scatter.get("tid", 0), "ts": gpu0_scatter.get("ts", 0) + 10
            })

        # B. SCATTER FORWARD ARROWS:
        # GPU 0 Scatter begins -> Arrows point FORWARD to GPU 1, 2, 3 when they receive and process
        if 0 in scatter_ranks:
            gpu0_scatter = scatter_ranks[0]
            src_ts = gpu0_scatter.get("ts", 0) + 5  # Source timestamp at beginning of GPU 0 scatter

            for target_rank in [1, 2, 3]:
                if target_rank in scatter_ranks:
                    target_scatter = scatter_ranks[target_rank]
                    # Target timestamp MUST be strictly in the future relative to src_ts!
                    tgt_ts = max(target_scatter.get("ts", 0), src_ts + 50)

                    flow_id += 1
                    merged_events.append({
                        "name": f"Scatter: GPU 0 -> GPU {target_rank}",
                        "ph": "s", "cat": "flow", "id": flow_id,
                        "pid": 200, "tid": gpu0_scatter.get("tid", 0), "ts": src_ts
                    })
                    merged_events.append({
                        "name": f"Scatter: GPU 0 -> GPU {target_rank}",
                        "ph": "f", "cat": "flow", "id": flow_id,
                        "pid": 200 + target_rank, "tid": target_scatter.get("tid", 0), "ts": tgt_ts
                    })

        # C. COMPUTE -> GATHER FORWARD ARROWS:
        # Each GPU's local compute completion points FORWARD to the All-Gather exchange
        for r in range(4):
            if r in compute_ranks and r in gather_ranks:
                comp_ev = compute_ranks[r]
                gather_ev = gather_ranks[r]
                comp_end = comp_ev.get("ts", 0) + comp_ev.get("dur", 10)
                gather_start = max(gather_ev.get("ts", 0), comp_end + 20)

                flow_id += 1
                merged_events.append({
                    "name": f"Compute Done -> Enter Gather (GPU {r})",
                    "ph": "s", "cat": "flow", "id": flow_id,
                    "pid": 200 + r, "tid": comp_ev.get("tid", 0), "ts": comp_end - 5
                })
                merged_events.append({
                    "name": f"Compute Done -> Enter Gather (GPU {r})",
                    "ph": "f", "cat": "flow", "id": flow_id,
                    "pid": 200 + r, "tid": gather_ev.get("tid", 0), "ts": gather_start
                })

        # D. GATHER -> REDUCE FORWARD ARROWS:
        # All-Gather points FORWARD to Reduction aggregation
        if 0 in gather_ranks and 0 in reduce_ranks:
            gather_ev = gather_ranks[0]
            reduce_ev = reduce_ranks[0]
            g_end = gather_ev.get("ts", 0) + gather_ev.get("dur", 10)
            r_start = max(reduce_ev.get("ts", 0), g_end + 15)

            flow_id += 1
            merged_events.append({
                "name": "Gather Complete -> Reduction Aggregation",
                "ph": "s", "cat": "flow", "id": flow_id,
                "pid": 200, "tid": gather_ev.get("tid", 0), "ts": g_end - 5
            })
            merged_events.append({
                "name": "Gather Complete -> Reduction Aggregation",
                "ph": "f", "cat": "flow", "id": flow_id,
                "pid": 200, "tid": reduce_ev.get("tid", 0), "ts": r_start
            })

    output_data = {"traceEvents": merged_events}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"\nUnified 4-GPU trace with STRICT FORWARD FLOW ARROWS generated: {output_file}")
    print(f"Total forward causal arrows synthesized: {flow_id - 60000}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/walid/repo/jax_cluster_lab/traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/walid/repo/jax_cluster_lab/traces/unified_4gpu_cluster.trace.json.gz"
    merge(base, out)
