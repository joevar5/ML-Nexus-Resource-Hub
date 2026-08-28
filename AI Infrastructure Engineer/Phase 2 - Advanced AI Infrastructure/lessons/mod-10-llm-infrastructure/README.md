# Module 10: LLM Infrastructure

**Duration:** ~80–100 hours · **Difficulty:** Advanced · **Prerequisites:** Modules 03, 04, 08

## Overview

The final module — and the one where everything else in this curriculum comes together. LLMs break the usual ML-infra playbook: models are 7B–70B+ parameters, GPU memory is the constant constraint, and the deployment patterns (RAG, fine-tuning, multi-model serving) are their own discipline. This module takes you from serving your first open-source model with vLLM to designing a full multi-model LLM platform.

## What you'll walk away with

- LLMs served through vLLM with an OpenAI-compatible API — and an understanding of *why* vLLM is fast
- A RAG system built from scratch: chunking, retrieval, ranking, evaluation
- A vector database (Qdrant/Weaviate/Chroma) deployed and tuned, not just imported
- Fine-tuning infrastructure using LoRA/QLoRA, without needing a dozen GPUs
- Inference optimized through quantization and batching — genuinely lower cost, not just fewer tokens
- A design for a multi-model platform: routing, caching, observability, all wired together

## Lessons

| # | Lesson |
|---|--------|
| 01 | [Introduction to LLM Infrastructure](./01-introduction-llm-infrastructure.md) — what's different about LLMs vs. traditional ML infra |
| 02 | [vLLM Deployment](./02-vllm-deployment.md) — architecture, OpenAI-compatible serving, production deployment |
| 03 | [RAG Systems](./03-rag-systems.md) — chunking, retrieval, ranking, evaluation |
| 04 | [Vector Databases](./04-vector-databases.md) — Qdrant/Weaviate/Chroma, deployment and tuning |
| 05 | [LLM Fine-Tuning Infrastructure](./05-llm-fine-tuning-infrastructure.md) — LoRA, QLoRA, distributed fine-tuning |
| 06 | [LLM Serving Optimization](./06-llm-serving-optimization.md) — quantization, Flash Attention, batching |
| 07 | [LLM Platform Architecture](./07-llm-platform-architecture.md) — multi-model serving, routing, caching |
| 08 | [Production LLM Best Practices](./08-production-llm-best-practices.md) — HA, security, compliance, operations |

## Hands-on

Eight exercises in [`labs/`](./labs/), progressing from a first vLLM deployment through a full RAG pipeline, vector DB setup, LoRA fine-tuning, streaming responses, cost monitoring, prompt-injection defense, and multi-model serving.

Most exercises target 7B-parameter models — a single consumer/cloud GPU (T4, A10G, or better) with 16GB+ system RAM covers nearly everything here.

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 25 multiple-choice + 3 scenario questions
- **Capstone** — deploy a production LLM API with monitoring and auto-scaling; you should be able to cut inference cost 50%+ through quantization and batching by the end

## Before you start

Make sure you're solid on Docker, Kubernetes, and Prometheus/Grafana (Modules 03/04/08) — this module assumes them and moves fast.

## What's next

This is the last module in Phase 2. From here: build **Project 103 (LLM Deployment Platform)** to put everything in this curriculum together, or go deeper into a specific area (fine-tuning, serving optimization, platform design) that matches where you want to specialize.

More reading in [`resources.md`](./resources.md).
