"""Grades starter.py. Prints PASS when every case is within 1%."""
import sys
import starter as s

GB = 1e9
cases = [
    ("all-reduce 1 MiB over 8 GPUs", lambda: s.ring_allreduce_bytes_per_gpu(2**20, 8), 2 * 7 / 8 * 2**20),
    ("all-reduce on 1 GPU is free", lambda: s.ring_allreduce_bytes_per_gpu(2**20, 1), 0.0),
    # Llama-3-70B: hidden 8192, 80 layers. Decode step with 64 sequences, TP=8, BF16.
    ("70B decode b=64 TP=8 traffic", lambda: s.tp_bytes_per_step(64, 8192, 80, 8), 160 * 2 * 7 / 8 * 64 * 8192 * 2),
    ("70B prefill 8K tokens TP=8", lambda: s.tp_bytes_per_step(8192, 8192, 80, 8), 160 * 2 * 7 / 8 * 8192 * 8192 * 2),
    ("comm time 293.6 MB @ 450 GB/s", lambda: s.comm_time_s(293_601_280, 450), 293_601_280 / (450 * GB)),
    ("bubble p=4 m=4", lambda: s.pp_bubble_fraction(4, 4), 3 / 7),
    ("bubble p=4 m=32", lambda: s.pp_bubble_fraction(4, 32), 3 / 35),
    # DeepSeek-V3-like: hidden 7168, 8 routed experts per token, EP=32, 4096 tokens per GPU, BF16.
    ("MoE all-to-all EP=32", lambda: s.moe_alltoall_bytes_per_gpu(4096, 7168, 8, 32), 2 * 4096 * 8 * 7168 * 2 * 31 / 32),
    ("MoE all-to-all EP=1 is local", lambda: s.moe_alltoall_bytes_per_gpu(4096, 7168, 8, 1), 0.0),
    ("70B BF16 weights at TP=4 PP=2", lambda: s.weight_bytes_per_gpu(70e9, 2, 4, 2), 70e9 * 2 / 8),
]
fails = []
for name, fn, exp in cases:
    try:
        got = fn()
    except NotImplementedError:
        fails.append(f"  {name}: not implemented"); continue
    except Exception as e:  # noqa: BLE001
        fails.append(f"  {name}: raised {type(e).__name__}: {e}"); continue
    if abs(got - exp) > 0.01 * abs(exp) + 1e-9:
        fails.append(f"  {name}: got {got:.6g}, expected {exp:.6g}")
if fails:
    print(f"FAIL ({len(fails)}/{len(cases)})"); print("\n".join(fails)); sys.exit(1)
print(f"PASS ({len(cases)}/{len(cases)})")
