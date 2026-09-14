# Module 10: LLM Infrastructure

## Overview

This is the module where everything else in the curriculum comes together. LLMs don't follow the usual ML-infra playbook — models run 7B to 70B+ parameters, GPU memory is the constant constraint, and patterns like retrieval, fine-tuning, and agent orchestration each bring their own operational demands. You'll go from serving your first open-source model with vLLM to designing a full multi-model platform that can support both chat and agentic workloads.

## What you'll walk away with

- LLMs served through vLLM with an OpenAI-compatible API, and a working understanding of *why* vLLM is fast
- A retrieval pipeline (chunking, ranking, evaluation) built well enough to know when it's the right tool and when it isn't
- A vector database (Qdrant/Weaviate/Chroma) deployed and tuned, not just imported
- Fine-tuning infrastructure using LoRA/QLoRA, without needing a dozen GPUs
- Inference optimized through quantization and batching — genuinely lower cost, not just fewer tokens
- The infrastructure agentic systems need on top of a completion API: tool calling, state, and multi-step orchestration
- A design for a multi-model platform: routing, caching, observability, all wired together

## Lessons

| # | Lesson | Hours |
|---|--------|-------|
| 01 | [Introduction to LLM Infrastructure](./01-introduction-llm-infrastructure.md) — what's different about LLMs vs. traditional ML infra | 8 |
| 02 | [vLLM Deployment](./02-vllm-deployment.md) — architecture, OpenAI-compatible serving, production deployment | 10 |
| 03 | [RAG Systems](./03-rag-systems.md) — chunking, retrieval, ranking, evaluation | 10 |
| 04 | [Vector Databases](./04-vector-databases.md) — Qdrant/Weaviate/Chroma, deployment and tuning | 10 |
| 05 | [LLM Fine-Tuning Infrastructure](./05-llm-fine-tuning-infrastructure.md) — LoRA, QLoRA, distributed fine-tuning | 12 |
| 06 | [LLM Serving Optimization](./06-llm-serving-optimization.md) — quantization, Flash Attention, batching | 10 |
| 07 | [LLM Platform Architecture](./07-llm-platform-architecture.md) — multi-model serving, routing, caching | 10 |
| 08 | [LLM Infrastructure for Agentic Systems](./08-llm-infrastructure-for-agents.md) — tool calling, state/memory, multi-step orchestration | 10 |
| 09 | [Production LLM Best Practices](./09-production-llm-best-practices.md) — HA, security, compliance, operations | 10 |

RAG (Lessons 03–04) is covered as one retrieval pattern among several you'll use — the same vector-DB infrastructure shows up again in Lesson 08 as agent memory, so it's worth building well the first time even if grounding chat responses isn't your main use case.

## Hands-on

Eight exercises in [`labs/`](./labs/), progressing from a first vLLM deployment through a full RAG pipeline, vector DB setup, LoRA fine-tuning, streaming responses, cost monitoring, prompt-injection defense, and multi-model serving.

Most exercises target 7B-parameter models — a single consumer/cloud GPU (T4, A10G, or better) with 16GB+ system RAM covers nearly everything here. No free tier covers this reliably: budget for on-demand or spot GPU hours (Lambda Labs, RunPod, or your cloud's spot instances) and set a spend alert before you start.

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 25 multiple-choice + 3 scenario questions
- **Capstone** — deploy a production LLM API with monitoring and auto-scaling; you should be able to cut inference cost 50%+ through quantization and batching by the end

## Before you start

Make sure you're solid on Docker, Kubernetes, and Prometheus/Grafana (Modules 03/04/08) — this module assumes them and moves fast.

## What's next

This is the last module in Phase 2. From here: build **Project 103 (LLM Deployment Platform)** to put everything in this curriculum together, or go deeper into a specific area (fine-tuning, serving optimization, agentic platform design) that matches where you want to specialize.

More reading in [`resources.md`](./resources.md).
