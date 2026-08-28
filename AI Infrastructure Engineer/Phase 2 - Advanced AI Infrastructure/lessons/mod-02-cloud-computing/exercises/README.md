# Module 02 Exercises: Cloud Computing for ML

## Overview

These hands-on exercises reinforce the concepts learned in Module 02. Each exercise builds practical skills in deploying and managing ML infrastructure across cloud platforms.

## Exercise List

### Exercise 01: Multi-Cloud ML Deployment
**Duration:** 3-4 hours
**Difficulty:** Intermediate
**File:** `exercise-01-multi-cloud.md`

Deploy the same ML model across AWS, GCP, and Azure, comparing:
- Deployment complexity
- Performance characteristics
- Cost differences
- Management overhead

### Exercise 02: Cloud Cost Optimization
**Duration:** 2-3 hours
**Difficulty:** Intermediate
**File:** `exercise-02-cost-optimization.md`

Analyze and optimize cloud spending for an ML workload:
- Identify cost drivers
- Implement cost-saving strategies
- Set up billing alerts
- Compare reserved vs on-demand vs spot instances

### Exercise 03: Network Architecture Design
**Duration:** 2-3 hours
**Difficulty:** Intermediate
**File:** `exercise-03-networking.md`

Design and implement secure network architecture:
- VPC design with public and private subnets
- Security groups and firewall rules
- Load balancer configuration
- VPN or private connectivity setup

### Exercise 04: Cloud Storage Pipeline
**Duration:** 2-3 hours
**Difficulty:** Beginner-Intermediate
**File:** `exercise-04-storage-pipeline.md`

Build a data pipeline using cloud storage services:
- Upload datasets to object storage
- Implement data versioning
- Set up caching layers
- Optimize data transfer costs

### Exercise 05: Managed ML Service Comparison
**Duration:** 3-4 hours
**Difficulty:** Intermediate
**File:** `exercise-05-managed-services.md`

Deploy the same model using managed services from each cloud:
- AWS SageMaker
- GCP Vertex AI
- Azure Machine Learning
- Compare features, ease of use, and costs

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

### 3. Clone Exercise Templates

```bash
cd ~/ai-infrastructure-learning
git clone <your-fork> exercises-module-02
cd exercises-module-02
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
5. Review solution hints (provided for each exercise)

## Completion Criteria

You've successfully completed Module 02 exercises when you can:

- [ ] Deploy ML models across multiple cloud platforms
- [ ] Implement cost optimization strategies
- [ ] Design secure network architectures
- [ ] Build efficient data pipelines using cloud storage
- [ ] Evaluate and use managed ML services
- [ ] Troubleshoot common cloud deployment issues
- [ ] Estimate and monitor cloud costs effectively

## Time Estimates

| Exercise | Estimated Time | Complexity |
|----------|---------------|------------|
| 01 - Multi-Cloud | 3-4 hours | Intermediate |
| 02 - Cost Optimization | 2-3 hours | Intermediate |
| 03 - Networking | 2-3 hours | Intermediate |
| 04 - Storage Pipeline | 2-3 hours | Beginner-Intermediate |
| 05 - Managed Services | 3-4 hours | Intermediate |
| **Total** | **12-17 hours** | - |

## Cost Estimates

All exercises can be completed within free tier limits if you:
- Use smallest instance types (t2.micro, e2-micro, B1S)
- Clean up resources immediately after completion
- Stay within free tier hours (750 hours/month per service)
- Use spot instances for non-critical workloads

**Expected costs (if free tier exhausted):**
- Exercise 01: $2-5
- Exercise 02: $1-3
- Exercise 03: $1-2
- Exercise 04: $1-2
- Exercise 05: $3-5

**Total: ~$8-17 if free tier is exhausted**

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
1. Review your solutions and compare with provided solutions
2. Complete the Module 02 quiz
3. Work on the Module 02 capstone project (optional but recommended)
4. Proceed to Module 03: Containerization with Docker

---

**Questions?** Open an issue in the GitHub repository or post in Discussions.

**Need solutions?** Solution files are available in the `solutions/` branch after honest attempt at exercises.
