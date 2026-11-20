"""Ex 04 — routing policies. Each policy is called once per request:

    policy(req: sim.Request, replicas: list[sim.Replica], state: dict) -> replica index

`state` is a dict you may use to remember things between calls (it starts empty).
A replica exposes `.work` (total cost assigned so far) and `.has_prefix(prefix_id)`.
Do not call `replica.serve()` yourself — the simulator does that.
"""
from __future__ import annotations


def round_robin(req, replicas, state) -> int:
    """Requests go to replicas 0, 1, 2, ... in turn."""
    raise NotImplementedError


def least_loaded(req, replicas, state) -> int:
    """The replica with the least work so far."""
    raise NotImplementedError


def prefix_hash(req, replicas, state) -> int:
    """Same prefix -> same replica, always (consistent hashing on prefix_id)."""
    raise NotImplementedError


def prefix_aware(req, replicas, state) -> int:
    """Balance cache hits against load, e.g.
        score = (tokens saved by a hit on this replica) - lam * (this replica's excess load)
    and pick the best score. Tune it until check.py passes; explain your choice in ANSWERS.md."""
    raise NotImplementedError
