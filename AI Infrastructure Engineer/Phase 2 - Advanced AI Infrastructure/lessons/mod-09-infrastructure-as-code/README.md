# Module 09: Infrastructure as Code (IaC)

**Duration:** ~32–41 hours · **Difficulty:** Intermediate · **Prerequisites:** Modules 02–04

## Overview

Clicking through a cloud console to build infrastructure doesn't scale, isn't reviewable, and isn't reproducible when you need to rebuild a GPU cluster at 3am after something dies. This module is about defining infrastructure as code instead — mainly Terraform, with a look at Pulumi for when you'd rather write Python than HCL — so your infrastructure changes go through Git like everything else you build.

## What you'll walk away with

- Terraform you'd trust with real cloud spend: state managed remotely, changes reviewed via plan/apply
- GPU instances, Kubernetes clusters, and storage provisioned from code, not clicked into existence
- A Pulumi alternative in your toolkit for when Python beats HCL
- A GitOps workflow — PRs that plan infrastructure changes before they apply
- Enough security sense to keep credentials out of your Terraform state and Git history

## Lessons

| # | Lesson |
|---|--------|
| 01 | [Terraform Fundamentals](./01-terraform-fundamentals.md) — HCL, providers, resources, your first project |
| 02 | [Terraform State Management](./02-terraform-state-management.md) — remote backends, locking, workspaces |
| 03 | [Building AI Infrastructure with Terraform](./03-building-ai-infrastructure-terraform.md) — GPU instances, EKS/GKE/AKS |
| 04 | [GitOps & IaC CI/CD](./04-gitops-iac-cicd.md) — PR workflows, automated plan/apply, policy as code |
| 05 | [Security Best Practices](./05-security-best-practices.md) — secrets management, scanning, cost estimation |

## Hands-on

Eight exercises in [`labs/`](./labs/) — from deploying a single EC2 instance up to building reusable Terraform modules, standing up a GKE cluster, deploying with Pulumi, and wiring a GitOps CI/CD workflow.

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 23 questions, 70% to pass
- **Practical** — design and deploy a complete ML infrastructure stack (GPU training instance, K8s inference cluster, S3 storage, monitoring) entirely via Terraform with remote state

## Tooling

Terraform 1.5+, Pulumi (Python SDK), plus `tflint`, `tfsec`, and `terraform-docs` if you want the full production setup. AWS/GCP/Azure CLIs from Module 02.

## What's next

With infrastructure defined as code, move to **Module 10: LLM Infrastructure** — where you'll provision the GPU clusters LLM serving actually needs.

More reading in [`resources.md`](./resources.md).
