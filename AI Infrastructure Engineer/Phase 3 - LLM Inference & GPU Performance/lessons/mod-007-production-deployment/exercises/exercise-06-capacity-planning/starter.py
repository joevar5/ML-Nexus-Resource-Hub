"""Ex 06 — capacity and cost model (CPU-only). Fill in every function, then `python check.py`.
Follow Lecture 06; the check uses its numbers. Units: seconds, requests, tokens, dollars."""
from __future__ import annotations


def average_rps(daily_active_users: float, requests_per_user_per_day: float) -> float:
    raise NotImplementedError


def peak_rps(avg_rps: float, peak_to_average: float) -> float:
    raise NotImplementedError


def replica_seconds_per_request(params: float, prompt_tokens: float, output_tokens: float,
                                decode_step_s: float, decode_batch: int,
                                replica_peak_flops: float, mfu: float,
                                prefix_hit_rate: float = 0.0) -> float:
    """Replica time one request consumes: its prefill (2·params FLOPs per *uncached* prompt token,
    at peak·MFU) plus its share of decode (each step serves `decode_batch` sequences)."""
    raise NotImplementedError


def planned_rps_per_replica(replica_seconds: float, target_utilization: float) -> float:
    raise NotImplementedError


def replicas_for(peak: float, rps_per_replica: float, headroom_frac: float = 0.0) -> int:
    """Whole replicas to serve `peak`, then add headroom: ceil(ceil(peak / cap) * (1 + headroom))."""
    raise NotImplementedError


def daily_cost(avg_rps: float, rps_per_replica: float, gpus_per_replica: int,
               overprovision: float, gpu_hourly_usd: float) -> float:
    """Autoscaled cost: you pay for (average replicas × overprovision) all day."""
    raise NotImplementedError


def cost_per_million_tokens(daily_cost_usd: float, tokens_per_day: float) -> float:
    raise NotImplementedError


def combined_cost_factor(factors: list[float]) -> float:
    """Independent levers multiply. Returns the remaining fraction of cost (0.3 = a 70% cut)."""
    raise NotImplementedError
