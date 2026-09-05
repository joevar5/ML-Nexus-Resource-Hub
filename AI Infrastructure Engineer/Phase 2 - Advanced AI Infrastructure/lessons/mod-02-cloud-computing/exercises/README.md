# Module 02 Exercises: Cloud Computing for ML

## Overview

These hands-on exercises reinforce the concepts learned in Module 02. Each exercise builds practical skills in deploying and managing ML infrastructure across cloud platforms — from cost analysis and multi-cloud deployment through disaster recovery, networking, managed services, security, and FinOps automation.

## Exercise List

### Exercise 01: Multi-Cloud Cost Analyzer
**Difficulty:** Intermediate
**Folder:** [`exercise-01-multi-cloud-cost-analyzer/`](./exercise-01-multi-cloud-cost-analyzer/)

Build a cost analysis tool that compares pricing across AWS, GCP, and Azure using their billing APIs, and generates cost optimization recommendations and interactive dashboards.

### Exercise 02: Cloud ML Infrastructure Deployment
**Difficulty:** Intermediate
**Folder:** [`exercise-02-cloud-ml-infrastructure/`](./exercise-02-cloud-ml-infrastructure/)

Deploy identical ML infrastructure to AWS, GCP, and Azure using Terraform, and compare performance and cost across all three.

### Exercise 03: Cloud Disaster Recovery System
**Difficulty:** Intermediate
**Folder:** [`exercise-03-disaster-recovery/`](./exercise-03-disaster-recovery/)

Design and build an automated backup, multi-region replication, and failover system, then test it against real RTO/RPO targets.

### Exercise 04: Cross-Region Replication for ML Artifacts
**Duration:** 3 hours · **Difficulty:** Intermediate
**Prerequisites:** Exercises 01–03
**Folder:** [`exercise-04-cross-region-replication/`](./exercise-04-cross-region-replication/)

Build an `artifact-replicator` tool that keeps model artifacts and training datasets synchronized across two cloud regions, with conflict detection, bandwidth control, and integrity verification.

### Exercise 05: Cloud Networking for ML Workloads
**Duration:** 3 hours · **Difficulty:** Intermediate
**Folder:** [`exercise-05-cloud-networking-for-ml/`](./exercise-05-cloud-networking-for-ml/)

Provision an ML-aware VPC with three subnet tiers across 3 AZs, an isolated GPU node pool, restricted NAT egress, and VPC Endpoints for S3/ECR/CloudWatch — then validate it with a connectivity test matrix.

### Exercise 06: Managed ML Services Comparison
**Duration:** 3 hours · **Difficulty:** Intermediate
**Prerequisites:** Exercises 01–02
**Folder:** [`exercise-06-managed-ml-services-comparison/`](./exercise-06-managed-ml-services-comparison/)

Deploy the same model to SageMaker, Vertex AI, and Azure ML managed endpoints and produce a structured trade-off matrix for the "managed vs. roll-your-own" decision.

### Exercise 07: Multi-Account / Multi-Project Security Architecture
**Duration:** 3 hours · **Difficulty:** Intermediate+
**Prerequisites:** AWS Organizations or GCP Folders, admin access
**Folder:** [`exercise-07-multi-account-security/`](./exercise-07-multi-account-security/)

Design a multi-account (or multi-project) security architecture: separate prod/staging/dev/sandbox environments, central audit logging, org-level guardrails, cross-account IAM, and a least-privilege CI deployment role.

### Exercise 08: FinOps Automation for ML Infrastructure
**Duration:** 3 hours · **Difficulty:** Intermediate
**Prerequisites:** Exercises 01–07, AWS Cost Explorer access
**Folder:** [`exercise-08-finops-automation/`](./exercise-08-finops-automation/)

Build an `mlfinops` CLI that collects daily cloud spend per team/workload, attributes idle GPU/VM cost, surfaces top waste, and sends a weekly digest to Slack.

## Prerequisites

Before starting these exercises:

- [ ] Completed all Module 02 lessons
- [ ] Active accounts on AWS, GCP, and Azure
- [ ] Installed and configured cloud CLI tools (aws, gcloud, az)
- [ ] Basic understanding of networking concepts
- [ ] Familiarity with Terraform or another IaC tool (helpful but not required)

## Setup Instructions

### 1. Verify Cloud Accounts

```bash
# Verify AWS CLI
aws sts get-caller-identity

# Verify GCP CLI
gcloud auth list

# Verify Azure CLI
az account show
```

### 2. Set Billing Alerts

**AWS:**
```bash
# Create billing alarm for $50
aws cloudwatch put-metric-alarm \
  --alarm-name "BillingAlarm" \
  --alarm-description "Alert if billing exceeds $50" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 21600 \
  --evaluation-periods 1 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold
```

**GCP:**
```bash
# Set budget alert in GCP Console
# Navigation: Billing → Budgets & Alerts → Create Budget
```

**Azure:**
```bash
# Set budget alert
az consumption budget create \
  --budget-name "ML-Budget" \
  --amount 50 \
  --time-grain Monthly
```

## Exercise Guidelines

### Best Practices

1. **Document Everything:** Keep notes on commands, errors, and solutions
2. **Version Control:** Commit progress frequently to Git
3. **Cost Awareness:** Shut down resources when not in use
4. **Security:** Never commit credentials or API keys
5. **Clean Up:** Delete resources after completing exercises

### Submission Format

For each exercise, submit:

1. **Code:** All configuration files, scripts, and IaC code
2. **Documentation:** README with setup instructions and architecture diagrams
3. **Results:** Screenshots, logs, or output demonstrating successful completion
4. **Reflection:** Brief write-up (200-300 words) on:
   - What you learned
   - Challenges encountered
   - How you would improve the solution

### Getting Help

If you get stuck:

1. Review the relevant lesson material
2. Check cloud provider documentation
3. Search for error messages in Stack Overflow
4. Ask questions in GitHub Discussions
5. Review solution hints (provided in each exercise's README)

## Completion Criteria

You've successfully completed Module 02 exercises when you can:

- [ ] Compare and optimize cloud costs across providers
- [ ] Deploy ML infrastructure identically across AWS, GCP, and Azure
- [ ] Design and test a disaster recovery plan against real RTO/RPO targets
- [ ] Replicate ML artifacts across regions with integrity guarantees
- [ ] Design an ML-aware, cost-conscious network architecture
- [ ] Evaluate and use managed ML services from all three clouds
- [ ] Design a least-privilege multi-account security architecture
- [ ] Automate cloud cost attribution and reporting

## Cost Estimates

All exercises can be completed within free tier limits if you:
- Use smallest instance types (t2.micro, e2-micro, B1S)
- Clean up resources immediately after completion
- Stay within free tier hours (750 hours/month per service)
- Use spot instances for non-critical workloads

Expect a few dollars per exercise if free tier is exhausted — clean up promptly to avoid surprises, especially in Exercises 03, 05, and 07 where multi-region or multi-account resources are easy to forget.

## Additional Resources

### Troubleshooting Guides
- AWS Troubleshooting: https://aws.amazon.com/premiumsupport/knowledge-center/
- GCP Troubleshooting: https://cloud.google.com/support/docs
- Azure Troubleshooting: https://docs.microsoft.com/en-us/troubleshoot/azure/

### Cost Calculators
- AWS Pricing Calculator: https://calculator.aws/
- GCP Pricing Calculator: https://cloud.google.com/products/calculator
- Azure Pricing Calculator: https://azure.microsoft.com/pricing/calculator/

### Community Forums
- r/aws: https://reddit.com/r/aws
- r/googlecloud: https://reddit.com/r/googlecloud
- r/azure: https://reddit.com/r/azure

## Next Steps

After completing all exercises:
1. Review your solutions and compare with the provided solution hints
2. Complete the Module 02 final quiz (`../quizzes/final-quiz.md`)
3. Work on the Module 02 capstone project (optional but recommended)
4. Proceed to Module 03: Containerization with Docker

---

**Questions?** Open an issue in the GitHub repository or post in Discussions.
