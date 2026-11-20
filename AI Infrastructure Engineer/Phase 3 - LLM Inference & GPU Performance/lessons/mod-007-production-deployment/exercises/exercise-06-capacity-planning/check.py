"""Grades starter.py against the worked design in Lecture 06. Prints PASS when all cases match."""
import sys
import starter as s

AVG = 5e6 / 86400
RSEC = 2 * 70e9 * 1500 / (2 * 1979e12 * 0.4) + 400 * 0.020 / 64
CAP = 0.8 / RSEC
cases = [
    ("average rps, 1M DAU x 5", lambda: s.average_rps(1e6, 5), AVG, 0.001),
    ("peak rps at 3x", lambda: s.peak_rps(AVG, 3), AVG * 3, 0.001),
    ("replica-seconds per request", lambda: s.replica_seconds_per_request(70e9, 1500, 400, 0.020, 64, 2 * 1979e12, 0.4), RSEC, 0.001),
    ("replica-seconds with 50% prefix hits",
     lambda: s.replica_seconds_per_request(70e9, 1500, 400, 0.020, 64, 2 * 1979e12, 0.4, 0.5),
     0.5 * 2 * 70e9 * 1500 / (2 * 1979e12 * 0.4) + 400 * 0.020 / 64, 0.001),
    ("planned rps per replica at 80%", lambda: s.planned_rps_per_replica(RSEC, 0.8), CAP, 0.001),
    ("replicas for peak", lambda: s.replicas_for(AVG * 3, CAP), 56, 0),
    ("replicas for peak + 10% headroom", lambda: s.replicas_for(AVG * 3, CAP, 0.10), 62, 0),
    ("daily cost, autoscaled at 1.25x average", lambda: s.daily_cost(AVG, CAP, 2, 1.25, 2.50),
     AVG / CAP * 1.25 * 2 * 24 * 2.50, 0.001),
    ("$ per 1M output tokens", lambda: s.cost_per_million_tokens(2800, 2e9), 1.40, 0.001),
    ("autoscale x FP8 levers", lambda: s.combined_cost_factor([1.25 / 3.3, 0.63]), 1.25 / 3.3 * 0.63, 0.001),
    ("no levers", lambda: s.combined_cost_factor([]), 1.0, 0.0),
]
fails = []
for name, fn, exp, rel in cases:
    try:
        got = fn()
    except NotImplementedError:
        fails.append(f"  {name}: not implemented"); continue
    except Exception as e:  # noqa: BLE001
        fails.append(f"  {name}: raised {type(e).__name__}: {e}"); continue
    ok = got == exp if rel == 0 else abs(got - exp) <= rel * abs(exp) + 1e-12
    if not ok:
        fails.append(f"  {name}: got {got:.6g}, expected {exp:.6g}")
if fails:
    print(f"FAIL ({len(fails)}/{len(cases)})"); print("\n".join(fails)); sys.exit(1)
print(f"PASS ({len(cases)}/{len(cases)})")
