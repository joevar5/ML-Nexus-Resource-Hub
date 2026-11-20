"""Replica + workload simulator for Ex 04. You do not need to edit this file.

Model (deliberately simple, so policies are easy to compare):
- Each replica keeps an LRU cache of whole prefixes (system prompts) — a stand-in for vLLM's prefix cache.
- Serving a request costs prefill tokens (the prefix only on a cache MISS, plus the user suffix)
  plus decode tokens. A replica's `work` is the total cost it has been given so far.
- Hit rate = fraction of requests whose prefix was already cached on the chosen replica.
- Imbalance = max replica work / mean replica work (1.0 = perfectly even).
"""
from __future__ import annotations
import random
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    rid: int
    prefix_id: int
    prefix_len: int
    suffix_len: int
    out_len: int


class Replica:
    def __init__(self, cache_size: int):
        self.cache: OrderedDict[int, None] = OrderedDict()
        self.cache_size = cache_size
        self.work = 0.0

    def has_prefix(self, prefix_id: int) -> bool:
        return prefix_id in self.cache

    def serve(self, req: Request) -> bool:
        hit = req.prefix_id in self.cache
        if hit:
            self.cache.move_to_end(req.prefix_id)
        else:
            self.cache[req.prefix_id] = None
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.work += (0 if hit else req.prefix_len) + req.suffix_len + req.out_len
        return hit


def workload(n: int = 20_000, num_prefixes: int = 200, zipf_s: float = 1.1, seed: int = 0) -> list[Request]:
    """Shared system prompts with Zipf popularity: a few prefixes are very hot (realistic for agents/RAG apps)."""
    rng = random.Random(seed)
    weights = [1 / (i + 1) ** zipf_s for i in range(num_prefixes)]
    prefix_len = {p: rng.randint(500, 3000) for p in range(num_prefixes)}
    reqs = []
    for rid in range(n):
        p = rng.choices(range(num_prefixes), weights)[0]
        reqs.append(Request(rid, p, prefix_len[p], rng.randint(20, 200), rng.randint(50, 400)))
    return reqs


def simulate(policy, requests: list[Request], n_replicas: int = 8, cache_size: int = 30) -> dict:
    replicas = [Replica(cache_size) for _ in range(n_replicas)]
    state: dict = {}
    hits = 0
    for req in requests:
        i = policy(req, replicas, state)
        if not 0 <= i < n_replicas:
            raise ValueError(f"policy returned replica {i}")
        hits += replicas[i].serve(req)
    works = [r.work for r in replicas]
    return {"hit_rate": hits / len(requests),
            "imbalance": max(works) / (sum(works) / len(works)),
            "total_work": sum(works)}
