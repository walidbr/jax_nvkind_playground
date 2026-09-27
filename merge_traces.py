"""
Deterministic Causal Trace Synthesizer for Multi-GPU JAX

Clear Visual Architecture:
  Track 100: JAX Framework
    - High-level Python execution, Pjit, and Step scopes
  Track 200..203: GPU 0..3
    - User operations visible on every GPU: OP_SCATTER, OP_LOCAL_COMPUTE, OP_GATHER, OP_REDUCE
    - Real hardware CUDA streams & kernels: MemcpyH2D, gemm, ncclDevKernel

Interactive Flow Arrows:
  1. Framework Dispatch -> Fan-out arrows to OP_SCATTER on all participating GPUs
  2. Scatter: OP_SCATTER on GPU 0 -> Fan-out arrows to OP_LOCAL_COMPUTE on GPU 1, 2, 3
  3. Gather: OP_LOCAL_COMPUTE on every GPU -> Fan-in arrows to OP_GATHER on GPU 0
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

    # 1. Track Setup
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

    # Indexed operations: rank -> step_id -> op_type -> event
    gpu_op_spans = {r: {} for r in range(num_discovered_ranks)}
    framework_step_dispatches = {}

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

                # A. Real GPU Hardware Kernels (pid 1) -> Stay in GPU lane
                if orig_pid == 1:
                    new_ev["pid"] = gpu_pid
                    merged_events.append(new_ev)

                # B. User OP Spans (OP_SCATTER, OP_LOCAL_COMPUTE, etc.)
                # Visible on EACH GPU track so you see what that GPU is doing!
                elif name.startswith("OP_"):
                    new_ev["pid"] = gpu_pid
                    merged_events.append(new_ev)

                    parts = name.split("_")
                    op_type = parts[1]  # SCATTER, LOCAL, GATHER, REDUCE
                    step_id = parts[-1]
                    gpu_op_spans[r_idx].setdefault(step_id, {})[op_type] = new_ev

                    # Also create a high-level Step Dispatch in Framework lane for Rank 0
                    if r_idx == 0 and op_type == "SCATTER":
                        framework_step_dispatches[step_id] = {
                            "name": f"Framework Step {step_id} Dispatch",
                            "ph": "X",
                            "pid": 100,
                            "tid": 100,
                            "ts": max(0, new_ev.get("ts", 0) - 150),
                            "dur": 120,
                            "cat": "framework"
                        }
                        merged_events.append(framework_step_dispatches[step_id])

                # C. High-level runtime orchestration in Framework lane
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
    # SYNTHESIZE FORWARD-ONLY FLOW ARROWS
    # -------------------------------------------------------------
    flow_id = 110000

    all_steps = sorted(list(framework_step_dispatches.keys()))

    for step_id in all_steps:
        disp_ev = framework_step_dispatches[step_id]
        disp_ts = disp_ev["ts"] + 20

        # 1. FRAMEWORK DISPATCH -> GPU 0 SCATTER (Forward)
        r0_ops = gpu_op_spans[0].get(step_id, {})
        if "SCATTER" in r0_ops:
            scatter_0 = r0_ops["SCATTER"]
            s0_start = scatter_0["ts"]
            
            flow_id += 1
            merged_events.append({
                "name": f"Dispatch -> GPU 0 Scatter",
                "ph": "s", "cat": "flow", "id": flow_id,
                "pid": 100, "tid": disp_ev["tid"], "ts": disp_ts
            })
            merged_events.append({
                "name": f"Dispatch -> GPU 0 Scatter",
                "ph": "f", "cat": "flow", "id": flow_id,
                "pid": 200, "tid": scatter_0["tid"], "ts": max(s0_start, disp_ts + 5)
            })

            # 2. SCATTER FAN-OUT ARROWS: GPU 0 Scatter -> GPU 1, 2, 3 Compute
            # Arrows originate from GPU 0's OP_SCATTER span!
            for peer_rank in range(1, num_discovered_ranks):
                peer_ops = gpu_op_spans[peer_rank].get(step_id, {})
                target_span = peer_ops.get("LOCAL", peer_ops.get("SCATTER"))
                if target_span:
                    tgt_ts = max(target_span["ts"], s0_start + 25)
                    flow_id += 1
                    merged_events.append({
                        "name": f"Scatter: GPU 0 -> GPU {peer_rank}",
                        "ph": "s", "cat": "flow", "id": flow_id,
                        "pid": 200, "tid": scatter_0["tid"], "ts": s0_start + 10
                    })
                    merged_events.append({
                        "name": f"Scatter: GPU 0 -> GPU {peer_rank}",
                        "ph": "f", "cat": "flow", "id": flow_id,
                        "pid": 200 + peer_rank, "tid": target_span["tid"], "ts": tgt_ts
                    })

        # 3. GATHER FAN-IN ARROWS: All GPUs -> GPU 0 OP_GATHER
        # Every GPU finishes OP_LOCAL_COMPUTE and sends an arrow into OP_GATHER on GPU 0
        if "GATHER" in r0_ops:
            gather_0 = r0_ops["GATHER"]
            g0_start = gather_0["ts"]

            for src_rank in range(num_discovered_ranks):
                src_ops = gpu_op_spans[src_rank].get(step_id, {})
                if "LOCAL" in src_ops:
                    comp_span = src_ops["LOCAL"]
                    comp_end = comp_span["ts"] + comp_span.get("dur", 10)
                    
                    arr_start = comp_end - 2
                    arr_end = max(g0_start + (src_rank * 15), arr_start + 10)

                    flow_id += 1
                    merged_events.append({
                        "name": f"Gather Inflow: GPU {src_rank} -> GPU 0",
                        "ph": "s", "cat": "flow", "id": flow_id,
                        "pid": 200 + src_rank, "tid": comp_span["tid"], "ts": arr_start
                    })
                    merged_events.append({
                        "name": f"Gather Inflow: GPU {src_rank} -> GPU 0",
                        "ph": "f", "cat": "flow", "id": flow_id,
                        "pid": 200, "tid": gather_0["tid"], "ts": arr_end
                    })

    output_data = {"traceEvents": merged_events}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"\nUnified trace generated: {output_file}")
    print(f"Total flow arrows synthesized: {flow_id - 110000}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/walid/repo/jax_cluster_lab/traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/walid/repo/jax_cluster_lab/traces/unified_4gpu_cluster.trace.json.gz"
    merge(base, out)
