# Exercise 09: Incident Response Game Day

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** All prior exercises (you'll use everything you've built)

## Objective

Run a structured Game Day: an exercise in which you deliberately break a production-like environment and practice the incident-response workflow. You'll measure MTTD (mean time to detect), MTTR (mean time to recover), and produce a postmortem that's specific and actionable.

## Why this matters

Real incidents are rare and high-stress. Game Days build muscle memory and surface gaps in your observability without paging anyone. Teams that run quarterly Game Days have 30-50% shorter MTTR in real incidents.

## Requirements

You'll inject 3 incidents into your running iris-api stack (with monitoring, alerts, tracing, logs from prior exercises). For each:
1. Inject the fault (described below).
2. Time how long until your alerts fire (MTTD).
3. Time how long until you identify the root cause (TTI — time to identification).
4. Time how long until you restore service (MTTR).
5. Write a postmortem (template below).

## The three incidents

### Incident A: Latency regression after deploy
**Inject:** Deploy a new image (`iris-api:0.3-slow`) that adds a 300ms artificial sleep to every prediction.

**Expected detection:** `SlowResponses` warning alert fires within ~10 minutes.

**Skills exercised:** identify a recent deploy as the cause; rollback procedure.

### Incident B: Cascading failure from downstream
**Inject:** Block the feature-store's egress port via NetworkPolicy. The backend now times out on feature lookup, returning 504s.

**Expected detection:** `HighErrorRate` critical alert.

**Skills exercised:** trace correlation (you should be able to see in a single trace where the time went), distinguish "we're broken" from "downstream is broken".

### Incident C: Resource exhaustion
**Inject:** Lower the iris-api memory limit to 256Mi while serving traffic. Pods OOM-killed within ~1 minute.

**Expected detection:** `TargetDown` + restart counter alerts.

**Skills exercised:** read OOMKilled status, understand the difference between pod-level and node-level failure, decide whether to bump limits or fix the leak.

## Step-by-step

### Step 1 — Prep (30 min)
- Make sure your stack is healthy: prom, grafana, loki, tempo, alertmanager, iris-api all green.
- Have your runbooks open.
- Designate a recorder (you, with a timer) and a responder (you, debugging).

### Step 2 — Incident A: latency regression (45 min)
1. Note current time. Deploy `iris-api:0.3-slow`.
2. Stop timer when alert fires; record as MTTD.
3. Identify cause: open Grafana → latency dashboard → see the spike → correlate with deploy events → see which pod (rollout history) → read its logs → confirm hypothesis with a trace. Stop timer; record as TTI.
4. Recover: `kubectl rollout undo`. Wait for latency to normalize. Stop timer; record as MTTR.

### Step 3 — Incident B: downstream block (45 min)
1. Apply the NetworkPolicy.
2. Watch for `HighErrorRate` alert.
3. Open Tempo, find a failing trace, see the 504 from feature-store. Identify that feature-store is the upstream cause.
4. Verify with a `kubectl exec` from iris-api pod: `curl -v http://feature-store/health` → connection refused.
5. Remove the NetworkPolicy. Verify recovery.

### Step 4 — Incident C: resource exhaustion (45 min)
1. Patch deployment with `resources.limits.memory: 256Mi`.
2. Watch pods restart.
3. `kubectl get pod -o yaml` → see `OOMKilled` in lastState.
4. Decide: bump limit back or investigate the leak? For the lab, bump back to 1Gi and confirm recovery.

### Step 5 — Write postmortems (30 min each)
For each incident, fill in:
```markdown
## Incident A: Latency regression from v0.3 deploy

**Date:** 2026-MM-DD
**Duration:** [MTTR minutes]
**Severity:** 2 (degraded service)

### Summary (1 sentence)
A deploy of iris-api v0.3 introduced an unintentional 300ms artificial delay in
every prediction, causing latency SLO violation.

### Timeline
- 14:00 — v0.3 deployed
- 14:08 — `SlowResponses` warning fires (MTTD 8 min)
- 14:09 — Engineer acknowledges, opens dashboard
- 14:14 — Identifies recent deploy as cause via rollout history (TTI 14 min)
- 14:15 — Initiates rollback to v0.2
- 14:18 — Rollout complete, latency normalized (MTTR 18 min)

### Root Cause
v0.3 inadvertently included a debug `time.sleep(0.3)` left over from local
load-test simulation.

### Detection
- What worked: latency dashboard + recent-deploy annotation overlay
- What didn't: no canary deploy that would have caught this at 1% rollout

### Recovery
- Rollback via `kubectl rollout undo` — standard procedure, worked as expected

### Action items (with owner + date)
- AI-1 (alice, by 2026-MM-DD): Enforce canary deploy via Argo Rollouts; new code
  goes to 10% of traffic for 5 minutes with auto-rollback on SLO breach.
- AI-2 (bob, by 2026-MM-DD): Add a pre-commit hook that grep -i "sleep" in
  inference paths and warns.
- AI-3 (alice, by 2026-MM-DD): Reduce MTTD by lowering `SlowResponses` for
  clause from 10m to 3m.
```

## Deliverables

1. Stopwatch numbers for MTTD/TTI/MTTR for each incident.
2. 3 postmortems following the template above.
3. `GAMEDAY_SUMMARY.md`: top 5 improvements identified across all incidents, prioritized.
4. A 1-pager `RUNBOOK_UPDATES.md` of specific runbook changes generated by the game day.

## Validation

- [ ] All 3 incidents detected by your monitoring within target MTTD.
- [ ] Each TTI < 15 min (otherwise observability has a gap to close).
- [ ] All postmortems have specific action items with owners and dates.
- [ ] You executed at least one action item before declaring the exercise done.

## Stretch goals

- Add **Incident D: A complete cloud-region failure** (simulate by NetworkPolicy isolating an entire AZ).
- **Game Day in production-shape staging** with a real on-call rotation.
- Use **Chaos Mesh** or **Gremlin** to schedule recurring small disruptions.

## Common pitfalls

- **Engineering perfectionism** — During the game day, you'll see lots of things to fix. Capture them; don't try to fix during the exercise.
- **Skipping the postmortem** — Postmortems are 70% of the value. A game day without one is just an outage.
- **Blame culture creeping in** — Postmortems must be blameless. "Alice broke prod" → "the deploy process allowed an untested code path through".
- **Action items without owners** — Vague action items don't get done. Always assign + date.
