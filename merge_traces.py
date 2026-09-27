"""
Merges distributed JAX profile into 5 distinct Perfetto lanes:
  Lane 0: JAX Framework (Coordinator & Step Dispatch)
  Lane 1: GPU 0
  Lane 2: GPU 1
  Lane 3: GPU 2
  Lane 4: GPU 3

AND synthesizes interactive Perfetto Flow Arrows:
  - SCATTER Arrows: Flow starts on GPU 0 and fans out with arrows to GPU 1, 2, 3!
  - GATHER / ALL-GATHER Arrows: Flows start on every GPU and converge with arrows to all ranks!
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

    # Data structures to index collective spans across ranks to build flow arrows
    # Key: (op_name, step) -> { rank: event }
    scatter_spans = {}
    gather_spans = {}

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
                    # Also map high-level OP_ annotations to the GPU lane so flow arrows connect visibly
                    if name.startswith("OP_"):
                        new_ev["pid"] = gpu_pid
                        merged_events.append(new_ev)

                        # Record for flow arrows
                        if "SCATTER" in name:
                            scatter_spans.setdefault(name, {})[r_idx] = new_ev
                        elif "GATHER" in name:
                            gather_spans.setdefault(name, {})[r_idx] = new_ev

                    elif r_idx == 0:
                        # General framework events on Lane 100
                        new_ev["pid"] = 100
                        merged_events.append(new_ev)

    # -------------------------------------------------------------
    # SYNTHESIZE PERFETTO FLOW ARROWS
    # -------------------------------------------------------------
    flow_id = 50000

    # 1. SCATTER FLOW ARROWS (Root GPU 0 -> Fan-out to GPU 1, 2, 3)
    for op_name, rank_map in scatter_spans.items():
        if 0 in rank_map:
            root_ev = rank_map[0]
            for target_rank in [1, 2, 3]:
                if target_rank in rank_map:
                    target_ev = rank_map[target_rank]
                    flow_id += 1

                    # Flow Start on GPU 0
                    merged_events.append({
                        "name": f"Scatter: GPU 0 -> GPU {target_rank}",
                        "ph": "s",
                        "cat": "flow",
                        "id": flow_id,
                        "pid": 200,
                        "tid": root_ev.get("tid", 0),
                        "ts": root_ev.get("ts", 0),
                        "bp": "e"
                    })

                    # Flow Finish on target GPU
                    merged_events.append({
                        "name": f"Scatter: GPU 0 -> GPU {target_rank}",
                        "ph": "f",
                        "cat": "flow",
                        "id": flow_id,
                        "pid": 200 + target_rank,
                        "tid": target_ev.get("tid", 0),
                        "ts": target_ev.get("ts", 0) + target_ev.get("dur", 10),
                        "bp": "e"
                    })

    # 2. GATHER / ALL-GATHER FLOW ARROWS (All GPUs cross-exchange)
    for op_name, rank_map in gather_spans.items():
        # Every rank sends flow arrows to every other rank
        for src_rank in rank_map:
            for dst_rank in rank_map:
                if src_rank != dst_rank:
                    src_ev = rank_map[src_rank]
                    dst_ev = rank_map[dst_rank]
                    flow_id += 1

                    # Flow Start on Source GPU
                    merged_events.append({
                        "name": f"Gather: GPU {src_rank} -> GPU {dst_rank}",
                        "ph": "s",
                        "cat": "flow",
                        "id": flow_id,
                        "pid": 200 + src_rank,
                        "tid": src_ev.get("tid", 0),
                        "ts": src_ev.get("ts", 0),
                        "bp": "e"
                    })

                    # Flow Finish on Destination GPU
                    merged_events.append({
                        "name": f"Gather: GPU {src_rank} -> GPU {dst_rank}",
                        "ph": "f",
                        "cat": "flow",
                        "id": flow_id,
                        "pid": 200 + dst_rank,
                        "tid": dst_ev.get("tid", 0),
                        "ts": dst_ev.get("ts", 0) + dst_ev.get("dur", 10),
                        "bp": "e"
                    })

    output_data = {"traceEvents": merged_events}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"\nUnified 4-GPU trace with Flow Arrows generated: {output_file}")
    print(f"Total flow arrows synthesized: {flow_id - 50000}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/walid/repo/jax_cluster_lab/traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/walid/repo/jax_cluster_lab/traces/unified_4gpu_cluster.trace.json.gz"
    merge(base, out)
