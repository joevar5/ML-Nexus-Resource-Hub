# Module 05: Data Pipelines and Orchestration

> Trimmed: Airflow, Spark, and Kafka are not used by the Phase 2 projects. Only data quality validation is kept.

**Duration:** ~45 hours · **Difficulty:** Intermediate · **Prerequisites:** Modules 02–04, Python, SQL basics

## Overview

Models are only as good as the data reaching them, and getting data from "raw and messy" to "clean and ready for training" reliably, on a schedule, at scale — is its own discipline. This module covers the tools that make that automatic: Airflow for orchestration, DVC for data versioning, Spark for processing data too big to fit on one machine, and Kafka for the stuff that never stops arriving.

## What you'll walk away with

- Airflow DAGs you'd actually trust to run unattended, with retries and alerts wired up
- Datasets and pipelines versioned with DVC the same way you version code
- Spark jobs that process real-scale data on Kubernetes, not toy examples
- A working feel for when to reach for streaming (Kafka) vs. batch
- Data quality checks that catch drift and bad data before it reaches training

## Lessons

| # | Lesson |
|---|--------|
| 01 | [Data Quality & Validation](./01-data-quality-validation.md) — Great Expectations, Pandera, drift detection |

## Hands-on

1. **Airflow pipeline** — pull from multiple sources, transform, validate, alert on failure
2. **Data versioning** — DVC on a 10GB+ dataset with full lineage tracking
3. **Spark pipeline** — feature engineering over a 100GB+ dataset, running on Kubernetes
4. **End-to-end pipeline** — Airflow + DVC + Spark + data quality checks, triggering model training

Full write-ups in [`labs/`](./labs/).

## Assessment

- **Practical** — build a production data pipeline with monitoring, submitted for review

## Tooling

Airflow 2.7+, DVC 3.0+, Spark 3.4+/PySpark, Kafka 3.5+, Great Expectations or Pandera for validation — all running on the Docker/Kubernetes setup from earlier modules.

## What's next

Once your pipeline is producing clean, versioned data, move to **Module 06: MLOps** — where you'll track experiments and register the models that pipeline feeds.

More reading in [`resources.md`](./resources.md).
