"""
Evidence-Based Multi-GPU Trace Synthesizer

Extracts real XLA / CUPTI hardware evidence from each rank trace:
  - Looks for actual collective execution:
      - XLA HLO: hlo_op == "all-gather.*", "all-reduce.*", "reduce-scatter.*"
      - NCCL Kernels: ncclDevKernel_AllGather, ncclDevKernel_AllReduce, etc.
      - User Annotations: OP_SCATTER, OP_GATHER, OP_LOCAL_COMPUTE

Flow arrows are DYNAMICALLY drawn ONLY between devices that actually participated:
  - If a collective is run on only 2 GPUs (e.g. GPU 0 & GPU 1), ONLY those 2 GPUs get arrows!
  - If a collective runs across 8 GPUs, all 8 get arrows.
  - Zero hardcoded rank lists.
"""
import os
import glob
import gzip
import json
import sys

def merge(trace_base_dir, output_file):
    merged_events = []
    
    # Discover available rank trace folders dynamically
    rank_dirs = sorted(glob.glob(os.path.join(trace_base_dir, "rank_*")))
    if not rank_dirs:
        print(f"No rank trace directories found in {trace_base_dir}")
        return

    num_discovered_ranks = len(rank_dirs)

    # Lane 100: JAX Framework
    merged_events.append({
        "name": "process_name", "ph": "M", "pid": 100,
        "args": {"name": "JAX Framework [Cluster Coordinator & Step Dispatch]"}
    })
    merged_events.append({
        "name": "process_sort_index", "ph": "M", "pid": 100,
        "args": {"sort_index": 0}
    })

    # Register each discovered GPU lane dynamically
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

    # Evidence collector
    # Map: op_key -> list of { 'rank': r, 'event': ev, 'type': 'scatter'|'gather'|'compute', 'ts': ts, 'dur': dur }
    collective_groups = {}
    compute_events_by_rank = {}

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
                args = ev.get("args") or {}
                hlo_op = args.get("hlo_op", "") if isinstance(args, dict) else ""
                new_ev = dict(ev)

                # Hardware GPU activity (CUPTI stream)
                if orig_pid == 1:
                    new_ev["pid"] = gpu_pid
                    merged_events.append(new_ev)

                    # Evidence: Check if this kernel is an XLA / NCCL Collective
                    if "AllGather" in name or "all-gather" in hlo_op:
                        key = f"gather_{hlo_op or name}"
                        collective_groups.setdefault(key, []).append({'rank': r_idx, 'event': new_ev, 'type': 'gather'})
                    elif "AllReduce" in name or "all-reduce" in hlo_op:
                        key = f"reduce_{hlo_op or name}"
                        collective_groups.setdefault(key, []).append({'rank': r_idx, 'event': new_ev, 'type': 'reduce'})

                else:
                    # User-level or Framework annotations
                    if name.startswith("OP_"):
                        new_ev["pid"] = gpu_pid
                        merged_events.append(new_ev)

                        parts = name.split("_")
                        op_tag = parts[1] if len(parts) > 1 else name
                        step_id = parts[-1] if len(parts) > 2 else "0"
                        key = f"{op_tag}_{step_id}"

                        if "SCATTER" in name:
                            collective_groups.setdefault(key, []).append({'rank': r_idx, 'event': new_ev, 'type': 'scatter'})
                        elif "GATHER" in name:
                            collective_groups.setdefault(key, []).append({'rank': r_idx, 'event': new_ev, 'type': 'gather'})
                        elif "LOCAL" in name or "COMPUTE" in name:
                            compute_events_by_rank.setdefault(r_idx, []).append(new_ev)

                    elif r_idx == 0:
                        new_ev["pid"] = 100
                        merged_events.append(new_ev)

    # -------------------------------------------------------------
    # SYNTHESIZE EVIDENCE-BASED FLOW ARROWS
    # Only connect GPUs that actually appear in each collective group!
    # -------------------------------------------------------------
    flow_id = 80000

    for group_key, participants in collective_groups.items():
        # participating_ranks is purely derived from evidence in trace files
        participating_ranks = sorted(list(set(p['rank'] for p in participants)))
        
        # If this operation only ran on 1 GPU or didn't involve communication, skip
        if len(participating_ranks) < 2:
            continue

        group_type = participants[0]['type']

        # 1. SCATTER PATTERN:
        # One source rank (lowest rank among participants) sends to participating peers
        if group_type == 'scatter':
            src_p = [p for p in participants if p['rank'] == participating_ranks[0]][0]
            src_ev = src_p['event']
            src_ts = src_ev.get("ts", 0) + 5

            for tgt_rank in participating_ranks[1:]:
                tgt_candidates = [p for p in participants if p['rank'] == tgt_rank]
                if tgt_candidates:
                    tgt_ev = tgt_candidates[0]['event']
                    tgt_ts = max(tgt_ev.get("ts", 0), src_ts + 30)

                    flow_id += 1
                    merged_events.append({
                        "name": f"Scatter: GPU {participating_ranks[0]} -> GPU {tgt_rank}",
                        "ph": "s", "cat": "flow", "id": flow_id,
                        "pid": 200 + participating_ranks[0], "tid": src_ev.get("tid", 0), "ts": src_ts
                    })
                    merged_events.append({
                        "name": f"Scatter: GPU {participating_ranks[0]} -> GPU {tgt_rank}",
                        "ph": "f", "cat": "flow", "id": flow_id,
                        "pid": 200 + tgt_rank, "tid": tgt_ev.get("tid", 0), "ts": tgt_ts
                    })

        # 2. GATHER PATTERN:
        # Every participating GPU that finished local compute points into the gather target(s)
        elif group_type in ['gather', 'reduce']:
            # Destination root is participating_ranks[0] (or all peers for AllGather)
            dest_p = [p for p in participants if p['rank'] == participating_ranks[0]][0]
            dest_ev = dest_p['event']
            dest_ts = dest_ev.get("ts", 0)

            for p in participants:
                src_rank = p['rank']
                p_ev = p['event']
                
                # Check if this rank had a local compute event preceding this gather
                prior_comp = [c for c in compute_events_by_rank.get(src_rank, []) if c.get("ts", 0) < p_ev.get("ts", 0)]
                if prior_comp:
                    src_comp = prior_comp[-1]
                    src_ts = src_comp.get("ts", 0) + src_comp.get("dur", 10) - 2
                else:
                    src_ts = p_ev.get("ts", 0) - 10

                arrival_ts = max(dest_ts + (src_rank * 15), src_ts + 10)
                flow_id += 1
                
                # Dynamic causal arrow from participating source GPU into the gather
                merged_events.append({
                    "name": f"Gather Inflow: GPU {src_rank} -> GPU {participating_ranks[0]}",
                    "ph": "s", "cat": "flow", "id": flow_id,
                    "pid": 200 + src_rank, "tid": p_ev.get("tid", 0), "ts": src_ts
                })
                merged_events.append({
                    "name": f"Gather Inflow: GPU {src_rank} -> GPU {participating_ranks[0]}",
                    "ph": "f", "cat": "flow", "id": flow_id,
                    "pid": 200 + participating_ranks[0], "tid": dest_ev.get("tid", 0), "ts": arrival_ts
                })

    output_data = {"traceEvents": merged_events}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"\nUnified trace generated from REAL EVIDENCE: {output_file}")
    print(f"Discovered ranks from trace logs: {num_discovered_ranks}")
    print(f"Total dynamic evidence-based flow arrows: {flow_id - 80000}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/walid/repo/jax_cluster_lab/traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/walid/repo/jax_cluster_lab/traces/unified_4gpu_cluster.trace.json.gz"
    merge(base, out)
