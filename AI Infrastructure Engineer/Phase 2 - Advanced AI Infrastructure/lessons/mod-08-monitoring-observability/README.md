# Module 08: Monitoring & Observability

**Duration:** ~30–35 hours · **Difficulty:** Intermediate to Advanced · **Prerequisites:** Modules 03–04, basic networking

## Overview

When something breaks in production at 2am, the difference between a five-minute fix and a five-hour outage is whether you built observability in ahead of time. This module covers the three pillars — metrics, logs, and traces — and, specifically, what's different about monitoring *ML* systems: model drift, prediction quality, and GPU utilization on top of the usual infrastructure health.

## What you'll walk away with

- Prometheus collecting real metrics, queried with PromQL you actually understand
- Grafana dashboards people would want to look at, not a wall of default panels
- Centralized logging (ELK/EFK) so you're not SSHing into pods to `tail -f`
- Distributed tracing that shows you exactly where a slow request lost its time
- Alerts tuned enough that people don't start ignoring them
- Model-specific monitoring: drift detection, prediction quality, A/B metrics

## Lessons

| # | Lesson |
|---|--------|
| 01 | [Introduction to Observability](./01-introduction-observability.md) — the three pillars, OpenTelemetry |
| 02 | [Prometheus Metrics](./02-prometheus-metrics.md) — architecture, metric types, PromQL, recording rules |
| 03 | [Grafana Visualization](./03-grafana-visualization.md) — dashboard design, panels, variables, alerting |
| 04 | [Logging: ELK & Loki](./04-logging-elk-loki.md) — centralized logging, structured logs, log queries |
| 05 | [Distributed Tracing](./05-distributed-tracing.md) — Jaeger/Zipkin, instrumenting services, trace analysis |
| 06 | [Alerting Strategies](./06-alerting-strategies.md) — AlertManager, on-call workflows, incident response |
| 07 | [ML Model Observability](./07-ml-model-observability.md) — performance metrics, drift detection, rollbacks |
| 08 | [Best Practices & Culture](./08-best-practices-culture.md) — SLIs/SLOs, error budgets, capacity planning |

## Hands-on

Five labs in [`labs/`](./labs/): stand up Prometheus + Grafana, centralize logs with Loki, add distributed tracing, build a model-monitoring dashboard, and combine it all into one production observability stack.

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 25 questions
- **Practical** — design an observability strategy for a given ML system
- **Capstone** — implement the complete monitoring stack end to end

## Tooling

```bash
pip install prometheus-client opentelemetry-api opentelemetry-sdk python-json-logger
```
Prometheus, Grafana, and Alertmanager binaries or Docker images; Elasticsearch/Kibana or Loki for logs; Jaeger for tracing.

## What's next

With visibility into your infrastructure, move to **Module 09: Infrastructure as Code** to make sure this whole monitoring stack is reproducible, not hand-built.

More reading in [`resources.md`](./resources.md).
