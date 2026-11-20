"""Grades starter.py. Prints PASS when every case is within tolerance."""
import math, sys
import starter as s

GB = 1e9
cases = []

def case(name, fn, expected, rel=0.01):
    cases.append((name, fn, expected, rel))

# Llama-3-70B: 80 layers, 8 KV heads (GQA), head_dim 128.  Llama-2-7B: 32 layers, 32 KV heads.
case("70B GQA KV per token, FP16", lambda: s.kv_bytes_per_token(80, 8, 128, 2), 327_680)
case("7B MHA KV per token, FP16", lambda: s.kv_bytes_per_token(32, 32, 128, 2), 524_288)
case("Q3: 70B, 32K ctx, 100 users, FP16 (bytes)",
     lambda: s.kv_cache_bytes(80, 8, 128, 2, 32_768, 100), 327_680 * 32_768 * 100)
case("Q3 again in FP8 KV", lambda: s.kv_cache_bytes(80, 8, 128, 1, 32_768, 100), 163_840 * 32_768 * 100)
# 8B BF16 on H100: 16 GB weights, 3,350 GB/s -> ~4.8 ms floor at tiny batch
case("decode floor, 8B bf16, H100, no KV", lambda: s.decode_step_floor_s(16 * GB, 0, 3350), 16 / 3350)
case("decode floor with 10 GB of KV read", lambda: s.decode_step_floor_s(16 * GB, 10 * GB, 3350), 26 / 3350)
# 8B bf16 on an 80 GB GPU, 4K-token sequences with GQA-8 (32 layers): 131,072 B/token
kv_seq = 32 * 8 * 128 * 2 * 2 * 4096
case("max seqs, 8B on 80 GB, 4K ctx", lambda: s.max_concurrent_seqs(80, 16 * GB, kv_seq), math.floor((80 * GB * 0.9 - 16 * GB) / kv_seq), rel=0)
case("weights do not fit -> 0", lambda: s.max_concurrent_seqs(24, 140 * GB, kv_seq), 0, rel=0)
case("spec E[tokens] a=0.8 g=4", lambda: s.spec_expected_tokens(0.8, 4), (1 - 0.8 ** 5) / 0.2)
case("spec E[tokens] a=1.0 g=4", lambda: s.spec_expected_tokens(1.0, 4), 5.0)
case("spec speedup a=0.8 g=4 c=0.05", lambda: s.spec_speedup(0.8, 4, 0.05), (1 - 0.8 ** 5) / 0.2 / 1.2)
case("spec speedup a=0.4 g=4 c=0.05 (weak draft)", lambda: s.spec_speedup(0.4, 4, 0.05), (1 - 0.4 ** 5) / 0.6 / 1.2)
case("best gamma a=0.8 c=0.05", lambda: s.best_gamma(0.8, 0.05), 8, rel=0)
case("best gamma a=0.6 c=0.2", lambda: s.best_gamma(0.6, 0.2), 2, rel=0)

fails = []
for name, fn, exp, rel in cases:
    try:
        got = fn()
    except NotImplementedError:
        fails.append(f"  {name}: not implemented"); continue
    except Exception as e:  # noqa: BLE001 — report any student error
        fails.append(f"  {name}: raised {type(e).__name__}: {e}"); continue
    ok = got == exp if rel == 0 else abs(got - exp) <= rel * abs(exp)
    if not ok:
        fails.append(f"  {name}: got {got}, expected {exp}")
if fails:
    print(f"FAIL ({len(fails)}/{len(cases)})"); print("\n".join(fails)); sys.exit(1)
print(f"PASS ({len(cases)}/{len(cases)})")
