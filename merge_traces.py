"""
Merges individual rank Perfetto traces into a single unified multi-GPU cluster trace.
Each rank gets its own distinct Process & Thread track in Perfetto:
  - GPU 0 (Rank 0) / Host 0
  - GPU 1 (Rank 1) / Host 1
  - GPU 2 (Rank 2) / Host 2
  - GPU 3 (Rank 3) / Host 3
"""
import os
import glob
import gzip
import json
import sys

def merge(trace_base_dir, output_file):
    merged_events = []
    
    # Locate all rank trace directories
    rank_dirs = sorted(glob.glob(os.path.join(trace_base_dir, "rank_*")))
    if not rank_dirs:
        print(f"No rank trace directories found in {trace_base_dir}")
        return

    print(f"Found {len(rank_dirs)} rank traces to merge into unified 4-GPU view:")

    for r_idx, r_dir in enumerate(rank_dirs):
        gz_files = glob.glob(os.path.join(r_dir, "plugins/profile/*/*.trace.json.gz"))
        if not gz_files:
            continue
        
        gz_path = gz_files[0]
        print(f"  Processing Rank {r_idx}: {gz_path}")

        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
            events = data.get("traceEvents", [])

            # Offset PIDs so each rank occupies distinct tracks in Perfetto
            # Rank 0: GPU PID 1000, Host PID 2000
            # Rank 1: GPU PID 1001, Host PID 2001, etc.
            gpu_pid = 1000 + r_idx
            host_pid = 2000 + r_idx

            # Track metadata overrides
            merged_events.append({
                "name": "process_name",
                "ph": "M",
                "pid": gpu_pid,
                "args": {"name": f"GPU {r_idx} [Simulated Device {r_idx}]"}
            })
            merged_events.append({
                "name": "process_sort_index",
                "ph": "M",
                "pid": gpu_pid,
                "args": {"sort_index": r_idx * 2}
            })
            merged_events.append({
                "name": "process_name",
                "ph": "M",
                "pid": host_pid,
                "args": {"name": f"Host {r_idx} Engine [Rank {r_idx}]"}
            })
            merged_events.append({
                "name": "process_sort_index",
                "ph": "M",
                "pid": host_pid,
                "args": {"sort_index": r_idx * 2 + 1}
            })

            for ev in events:
                if ev.get("name") in ["process_name", "process_sort_index"]:
                    continue  # We set custom distinct names above
                
                # Copy event
                new_ev = dict(ev)
                orig_pid = ev.get("pid", 0)

                # Remap PID: 1 is /device:GPU:0 in JAX xplane, other pids are Host CPU
                if orig_pid == 1:
                    new_ev["pid"] = gpu_pid
                else:
                    new_ev["pid"] = host_pid

                merged_events.append(new_ev)

    output_data = {"traceEvents": merged_events}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"\nUnified 4-GPU trace generated: {output_file}")
    print(f"Total events across all 4 GPUs: {len(merged_events)}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/walid/repo/jax_cluster_lab/traces"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/walid/repo/jax_cluster_lab/traces/unified_4gpu_cluster.trace.json.gz"
    merge(base, out)
