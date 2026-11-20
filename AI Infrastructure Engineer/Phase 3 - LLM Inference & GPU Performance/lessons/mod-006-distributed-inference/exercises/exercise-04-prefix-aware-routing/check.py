"""Grades policies.py against the simulator. Prints PASS when every policy meets its bar."""
import sys
import sim
import policies as p

reqs = sim.workload()
results, fails = {}, []
for name in ["round_robin", "least_loaded", "prefix_hash", "prefix_aware"]:
    try:
        results[name] = sim.simulate(getattr(p, name), reqs)
    except NotImplementedError:
        fails.append(f"  {name}: not implemented")
    except Exception as e:  # noqa: BLE001 — surface student errors
        fails.append(f"  {name}: raised {type(e).__name__}: {e}")

def need(cond, msg):
    if not cond:
        fails.append("  " + msg)

r = results
if "round_robin" in r:
    need(r["round_robin"]["imbalance"] < 1.1, f"round_robin imbalance {r['round_robin']['imbalance']:.2f} should be < 1.1")
if "least_loaded" in r:
    need(r["least_loaded"]["imbalance"] < 1.05, f"least_loaded imbalance {r['least_loaded']['imbalance']:.2f} should be < 1.05")
if "prefix_hash" in r:
    need(r["prefix_hash"]["hit_rate"] >= 0.95, f"prefix_hash hit rate {r['prefix_hash']['hit_rate']:.2f} should be >= 0.95")
if "prefix_aware" in r:
    pa = r["prefix_aware"]
    need(pa["hit_rate"] >= 0.90, f"prefix_aware hit rate {pa['hit_rate']:.2f} should be >= 0.90")
    need(pa["imbalance"] <= 1.25, f"prefix_aware imbalance {pa['imbalance']:.2f} should be <= 1.25")
    if "round_robin" in r:
        need(pa["total_work"] <= 0.5 * r["round_robin"]["total_work"],
             "prefix_aware total work should be <= half of round_robin's (cache hits skip prefill)")

for name, m in results.items():
    print(f"{name:13s} hit_rate={m['hit_rate']:.3f} imbalance={m['imbalance']:.2f} total_work={m['total_work']:.3g}")
if fails:
    print(f"FAIL ({len(fails)})"); print("\n".join(fails)); sys.exit(1)
print("PASS")
