"""
Evidence-Based Multi-GPU Trace Synthesizer

Extracts real XLA / CUPTI hardware evidence from each rank trace:
  - JAX Framework Track (Lane 100): Shows Python runtime, Pjit dispatches, and step scopes
  - GPU Tracks (Lanes 200..203): Shows actual GPU CUDA streams and NCCL collective kernels

Generates bidirectional, forward-causal flow arrows:
  1. Framework -> GPU: From JAX dispatch / PjitFunction / broadcast_one_to_all to the receiving GPU(s)
  2. Scatter Phase: Root GPU -> Participating GPUs
  3. Compute -> Gather: Completed local compute -> Converging into Gather
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

    # Lane 100: JAX Framework
    merged_events.append({
        "name": "process_name", "ph": "M", "pid": 100,
        "args": {"name": "JAX Framework [Host Runtime & Dispatch Engine]"}
    })
    merged_events.append({
        "name": "process_sort_index", "ph": "M", "pid": 100,
        "args": {"sort_index": 0}
    })

    # Lanes 200..203: The GPUs
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

    collective_groups = {}
    compute_events_by_rank = {}
    framework_dispatches = []

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

                    if "AllGather" in name or "all-gather" in hlo_op:
                        key = f"gather_{hlo_op or name}"
                        collective_groups.setdefault(key, []).append({'rank': r_idx, 'event': new_ev, 'type': 'gather'})
                    elif "AllReduce" in name or "all-reduce" in hlo_op:
                        key = f"reduce_{hlo_op or name}"
                        collective_groups.setdefault(key, []).append({'rank': r_idx, 'event': new_ev, 'type': 'reduce'})

                else:
                    # User-level operation annotations
                    if name.startswith("OP_"):
                        # Keep high-level annotation visible on the GPU lane
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
                        # Rank 0 Host CPU events populate the JAX Framework Lane
                        new_ev["pid"] = 100
                        merged_events.append(new_ev)

                        # Capture key framework dispatch spans to connect to GPUs
                        if any(kw in name for kw in ["broadcast_one_to_all", "process_allgather", "PjitFunction", "Execute", "H2D Dispatch"]):
                            if new_ev.get("dur", 0) > 10:
                                framework_dispatches.append(new_ev)

    # -------------------------------------------------------------
    # SYNTHESIZE FORWARD FLOW ARROWS
    # -------------------------------------------------------------
    flow_id = 90000

    # 1. JAX FRAMEWORK -> GPU DISPATCH ARROWS
    # Links each significant framework dispatch event to the corresponding GPU kernel/operation that begins immediately after
    for f_ev in framework_dispatches:
        f_ts = f_ev.get("ts", 0)
        f_name = f_ev.get("name", "Dispatch")
        f_end = f_ts + f_ev.get("dur", 10)

        # Find the earliest GPU collective/scatter event that occurs at or shortly after this dispatch
        target_gpu_events = []
        for grp_key, grp_parts in collective_groups.items():
            for p in grp_parts:
                p_ev = p['event']
                p_ts = p_ev.get("ts", 0)
                # If GPU event starts within a reasonable causal window [f_ts - 50, f_end + 50000]
                if f_ts - 100 <= p_ts <= f_end + 30000:
                    target_gpu_events.append((p['rank'], p_ev))

        # Draw dispatch arrow to participating GPU(s)
        seen_ranks = set()
        for r, tgt_ev in target_gpu_events[:4]:
            if r not in seen_ranks:
                seen_ranks.add(r)
                flow_id += 1
                merged_events.append({
                    "name": f"JAX Framework Launch -> GPU {r}",
                    "ph": "s", "cat": "flow", "id": flow_id,
                    "pid": 100, "tid": f_ev.get("tid", 0), "ts": f_ts + (f_ev.get("dur", 10) / 2)
                })
                merged_events.append({
                    "name": f"JAX Framework Launch -> GPU {r}",
                    "ph": "f", "cat": "flow", "id": flow_id,
                    "pid": 200 + r, "tid": tgt_ev.get("tid", 0), "ts": max(tgt_ev.get("ts", 0), f_ts + 20)
                })

    # 2. SCATTER FLOW ARROWS (GPU -> GPU)
    for group_key, participants in collective_groups.items():
        participating_ranks = sorted(list(set(p['rank'] for p in participants)))
        if len(participating_ranks) < 2:
            continue

        group_type = participants[0]['type']

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

        # 3. GATHER FLOW ARROWS (Participating GPUs -> Gather Target)
        elif group_type in ['gather', 'reduce']:
            dest_p = [p for p in participants if p['rank'] == participating_ranks[0]][0]
            dest_ev = dest_p['event']
            dest_ts = dest_ev.get("ts", 0)

            for p in participants:
                src_rank = p['rank']
                p_ev = p['event']
                
                prior_comp = [c for c in compute_events_by_rank.get(src_rank, []) if c.get("ts", 0) < p_ev.get("ts", 0)]
                if prior_comp:
                    src_comp = prior_comp[-1]
                    src_ts = src_comp.get("ts", 0) + src_comp.get("dur", 10) - 2
                else:
                    src_ts = p_ev.get("ts", 0) - 10

                arrival_ts = max(dest_ts + (src_rank * 15), src_ts + 10)
                flow_id += 1
                
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

    print(f"\nUnified trace with JAX Framework -> GPU flow links generated: {output_file}")
    print(f"Total causal flow arrows synthesized: {flow_id - 90000}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/walid/repo/jax_cluster_lab/traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/walid/repo/jax_cluster_lab/traces/unified_4gpu_cluster.trace.json.gz"
    merge(base, out)
