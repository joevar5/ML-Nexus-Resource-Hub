# Module 02: Cloud Computing for ML Infrastructure

## Overview

Every real ML system eventually has to live somewhere that isn't your laptop. This module is about learning that terrain across the three clouds you'll actually run into on the job — AWS, GCP, and Azure — so you can architect, deploy, and (critically) not blow the budget on ML infrastructure in production.

## What you'll walk away with

- The judgment to pick the right cloud and services for a given ML workload, not just "the one I know"
- Working ML deployments on AWS (EC2/EKS/SageMaker), GCP (GKE/Vertex AI), and Azure (AKS/Azure ML)
- Storage and networking choices that don't quietly cost you a fortune
- A real instinct for cloud cost optimization — reserved instances, spot instances, and where the money actually goes

## Lessons

| # | Lesson | Hours |
|---|--------|-------|
| 01 | [Cloud Architecture for ML](./01-cloud-architecture.md) — patterns, cost modeling, designing an ML system | 6 |
| 02 | [AWS for ML Infrastructure](./02-aws-ml-infrastructure.md) — EC2, S3, EKS, SageMaker, IAM | 7 |
| 03 | [GCP for ML Infrastructure](./03-gcp-ml-infrastructure.md) — Compute Engine, GKE, Vertex AI, TPUs | 7 |
| 04 | [Azure for ML Infrastructure](./04-azure-ml-infrastructure.md) — VMs, AKS, Azure ML, Azure OpenAI | 6 |
| 05 | [Cloud Storage for ML](./05-cloud-storage.md) — object/block/file storage, data lakes, caching | 6 |
| 06 | [Cloud Networking for ML](./06-cloud-networking.md) — VPCs, load balancers, CDN, service mesh | 6 |
| 07 | [Managed ML Services](./07-managed-ml-services.md) — SageMaker vs. Vertex AI vs. Azure ML | 6 |
| 08 | [Multi-Cloud & Cost Optimization](./08-multi-cloud-cost-optimization.md) — reserved/spot instances, cost monitoring | 6 |

## Hands-on

Each lesson ends with a deploy-something exercise, building toward a capstone: **deploy a production-ready ML system** that's multi-region, auto-scales, stays under $50/month, and hits 99.9% availability.

## Before you start

You'll want free-tier accounts on all three clouds — set billing alerts immediately, this module is designed to fit inside free-tier limits if you're careful:

- **AWS**: [free tier](https://aws.amazon.com/free/) — 750 hrs/month t2.micro, 5GB S3
- **GCP**: [$300 credit](https://cloud.google.com/free) for 90 days
- **Azure**: [$200 credit](https://azure.microsoft.com/free/) for 30 days

```bash
pip install awscli azure-cli
# GCP SDK: https://cloud.google.com/sdk/docs/install
```

## Assessment

- **Quiz** — [`quizzes/final-quiz.md`](./quizzes/final-quiz.md), 50 questions covering all 8 lessons, 80% to pass
- Three practical exercises: multi-cloud deployment, a cost-optimization challenge, and a network architecture design

## What's next

Pass the quiz, ship the capstone, then move on to **Module 03: Containerization** — the images you build there are what you'll actually be deploying onto this cloud infrastructure.

More reading in [`resources.md`](./resources.md).
