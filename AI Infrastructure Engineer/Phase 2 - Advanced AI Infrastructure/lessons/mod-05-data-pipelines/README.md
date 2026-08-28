# Module 05: Data Pipelines and Orchestration

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
| 01 | [Data Pipeline Architecture](./01-data-pipeline-architecture.md) — patterns, batch vs. streaming, design principles |
| 02 | [Apache Airflow Fundamentals](./02-apache-airflow-fundamentals.md) — architecture, running on Kubernetes, your first DAG |
| 03 | [Advanced Airflow for ML](./03-advanced-airflow-ml.md) — TaskFlow API, XComs, dynamic DAGs, ML operators |
| 04 | [Data Versioning with DVC](./04-data-versioning-dvc.md) — tracking datasets/models, remote storage, data pipelines |
| 05 | [Data Processing with Spark](./05-data-processing-spark.md) — PySpark, running on Kubernetes, optimization |
| 06 | [Streaming Data with Kafka](./06-streaming-data-kafka.md) — producers/consumers, real-time feature engineering |
| 07 | [Data Quality & Validation](./07-data-quality-validation.md) — Great Expectations, Pandera, drift detection |
| 08 | [Pipeline Monitoring & Error Handling](./08-pipeline-monitoring-errors.md) — alerting, retries, SLA monitoring |

## Hands-on

1. **Airflow pipeline** — pull from multiple sources, transform, validate, alert on failure
2. **Data versioning** — DVC on a 10GB+ dataset with full lineage tracking
3. **Spark pipeline** — feature engineering over a 100GB+ dataset, running on Kubernetes
4. **End-to-end pipeline** — Airflow + DVC + Spark + data quality checks, triggering model training

Full write-ups in [`labs/`](./labs/).

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 25 questions on Airflow, DVC, and Spark
- **Practical** — build a production data pipeline with monitoring, submitted for review

## Tooling

Airflow 2.7+, DVC 3.0+, Spark 3.4+/PySpark, Kafka 3.5+, Great Expectations or Pandera for validation — all running on the Docker/Kubernetes setup from earlier modules.

## What's next

Once your pipeline is producing clean, versioned data, move to **Module 06: MLOps** — where you'll track experiments and register the models that pipeline feeds.

More reading in [`resources.md`](./resources.md).
