"""
Merges distributed JAX profile into 5 distinct Perfetto lanes:
  Lane 0: JAX Framework (Coordinator, Global Dispatch, Step Annotations)
  Lane 1: GPU 0 (Rank 0 Device Activity)
  Lane 2: GPU 1 (Rank 1 Device Activity)
  Lane 3: GPU 2 (Rank 2 Device Activity)
  Lane 4: GPU 3 (Rank 3 Device Activity)
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

    # Track 100: JAX Framework Lane
    merged_events.append({
        "name": "process_name",
        "ph": "M",
        "pid": 100,
        "args": {"name": "JAX Framework [Cluster Coordinator & Step Dispatch]"}
    })
    merged_events.append({
        "name": "process_sort_index",
        "ph": "M",
        "pid": 100,
        "args": {"sort_index": 0}
    })

    # Create 4 discrete GPU device tracks: PIDs 200, 201, 202, 203
    for r_idx in range(len(rank_dirs)):
        gpu_pid = 200 + r_idx
        merged_events.append({
            "name": "process_name",
            "ph": "M",
            "pid": gpu_pid,
            "args": {"name": f"GPU {r_idx} [Hardware Stream & Kernels]"}
        })
        merged_events.append({
            "name": "process_sort_index",
            "ph": "M",
            "pid": gpu_pid,
            "args": {"sort_index": r_idx + 1}
        })

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
                new_ev = dict(ev)

                # orig_pid == 1 represents the GPU Stream / CUDA device activity in JAX xplane
                if orig_pid == 1:
                    new_ev["pid"] = gpu_pid
                    merged_events.append(new_ev)
                else:
                    # Host CPU / Framework traces
                    # Only take Rank 0's framework trace so we have 1 clean JAX Framework timeline
                    if r_idx == 0:
                        new_ev["pid"] = 100
                        merged_events.append(new_ev)

    output_data = {"traceEvents": merged_events}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"Synthesized exactly 5 tracks into {output_file}")
    print("  -> Track 0: JAX Framework (Coordinator & Step Dispatch)")
    print("  -> Track 1: GPU 0")
    print("  -> Track 2: GPU 1")
    print("  -> Track 3: GPU 2")
    print("  -> Track 4: GPU 3")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/walid/repo/jax_cluster_lab/traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/walid/repo/jax_cluster_lab/traces/unified_4gpu_cluster.trace.json.gz"
    merge(base, out)
