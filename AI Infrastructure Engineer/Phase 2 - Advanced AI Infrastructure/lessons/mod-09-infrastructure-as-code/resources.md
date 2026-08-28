# Module 09: Infrastructure as Code - Resources

Comprehensive collection of resources for learning Infrastructure as Code, Terraform, Pulumi, and IaC best practices for AI/ML infrastructure.

---

## Official Documentation

### Terraform

- **[Terraform Documentation](https://www.terraform.io/docs)** - Official Terraform docs
- **[Terraform Registry](https://registry.terraform.io/)** - Browse providers and modules
- **[Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)** - AWS provider documentation
- **[Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)** - Google Cloud provider
- **[Terraform Azure Provider](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)** - Azure provider
- **[HCL Syntax](https://www.terraform.io/language/syntax/configuration)** - HashiCorp Configuration Language guide
- **[Terraform CLI Commands](https://www.terraform.io/cli/commands)** - Complete CLI reference

### Pulumi

- **[Pulumi Documentation](https://www.pulumi.com/docs/)** - Official Pulumi docs
- **[Pulumi Python SDK](https://www.pulumi.com/docs/languages-sdks/python/)** - Python-specific documentation
- **[Pulumi AWS Package](https://www.pulumi.com/registry/packages/aws/)** - AWS resources in Python
- **[Pulumi Examples](https://github.com/pulumi/examples)** - 500+ example programs
- **[Pulumi vs Terraform](https://www.pulumi.com/docs/concepts/vs/terraform/)** - Official comparison

### AWS

- **[AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)** - Best practices
- **[AWS EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)** - Choose right instance for ML
- **[AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/)** - Pre-configured ML environments
- **[AWS EKS Documentation](https://docs.aws.amazon.com/eks/)** - Kubernetes on AWS

### Google Cloud

- **[GCP Best Practices](https://cloud.google.com/docs/terraform/best-practices-for-terraform)** - GCP Terraform guide
- **[GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)** - Google Kubernetes Engine
- **[GCP GPU Instances](https://cloud.google.com/compute/docs/gpus)** - GPU configuration

---

## Books

### Infrastructure as Code

1. **"Terraform: Up & Running" by Yevgeniy Brikman**
   - Publisher: O'Reilly Media
   - Best comprehensive Terraform book
   - Covers basics to advanced patterns
   - Includes team collaboration strategies
   - [Amazon Link](https://www.amazon.com/Terraform-Running-Writing-Infrastructure-Code/dp/1098116747)

2. **"Infrastructure as Code" by Kief Morris**
   - Publisher: O'Reilly Media
   - Language-agnostic IaC principles
   - Patterns and anti-patterns
   - DevOps best practices
   - [Amazon Link](https://www.amazon.com/Infrastructure-Code-Dynamic-Systems-Cloud/dp/1098114671)

3. **"The Terraform Book" by James Turnbull**
   - Practical, hands-on approach
   - Beginner-friendly
   - Step-by-step tutorials
   - [Amazon Link](https://terraformbook.com/)

4. **"Pulumi in Action" (Manning - Early Access)**
   - Comprehensive Pulumi guide
   - Multi-cloud infrastructure
   - Real-world examples
   - [Manning Link](https://www.manning.com/books/pulumi-in-action)

### Cloud & DevOps

5. **"Cloud Native DevOps with Kubernetes" by John Arundel & Justin Domingus**
   - O'Reilly Media
   - Kubernetes infrastructure patterns
   - Great complement to IaC
   - [Amazon Link](https://www.amazon.com/Cloud-Native-DevOps-Kubernetes-Applications/dp/1492040762)

6. **"Site Reliability Engineering" by Google**
   - Free online at [sre.google](https://sre.google/books/)
   - Infrastructure automation at scale
   - Google's production practices

---

## Online Courses

### Free Courses

1. **HashiCorp Learn - Terraform**
   - URL: https://learn.hashicorp.com/terraform
   - Official HashiCorp tutorials
   - Beginner to advanced
   - Hands-on labs
   - **Topics**: AWS, Azure, GCP, Kubernetes, Cloud Development Kit

2. **Pulumi Getting Started**
   - URL: https://www.pulumi.com/docs/get-started/
   - Interactive tutorials
   - Multiple languages (Python, TypeScript, Go)
   - Cloud provider walkthroughs

3. **A Cloud Guru - Terraform Free Content**
   - URL: https://acloudguru.com/
   - Monthly free courses
   - Hands-on labs in sandbox environments

4. **freeCodeCamp - Terraform Course (YouTube)**
   - URL: https://www.youtube.com/watch?v=SLB_c_ayRMo
   - 2+ hour comprehensive course
   - Beginner-friendly
   - AWS-focused

### Paid Courses

5. **"Learn DevOps: Infrastructure Automation With Terraform"**
   - Platform: Udemy
   - Instructor: Edward Viaene
   - 8+ hours of content
   - Hands-on projects
   - ~$20-50 (frequent sales)

6. **"Terraform on AWS with SRE & IaC DevOps"**
   - Platform: Udemy
   - Instructor: Kalyan Reddy
   - Complete AWS + Terraform bootcamp
   - 20+ hours
   - Includes EKS, VPC, IAM

7. **"Advanced Terraform"**
   - Platform: Pluralsight
   - Instructor: Ned Bellavance
   - Advanced patterns and practices
   - Enterprise-focused
   - Subscription required (~$29/month)

8. **"Pulumi: Infrastructure as Code"**
   - Platform: Pluralsight
   - Python, TypeScript, Go coverage
   - Multi-cloud examples

---

## Interactive Learning & Labs

### Terraform

1. **Terraform Katacoda**
   - URL: https://www.katacoda.com/terraform
   - Browser-based interactive labs
   - No setup required
   - Free

2. **Instruqt - HashiCorp Tracks**
   - URL: https://play.instruqt.com/hashicorp
   - Interactive Terraform tutorials
   - Real cloud environments
   - Free tier available

3. **killercoda - Terraform Playground**
   - URL: https://killercoda.com/terraform
   - Practice environment
   - Pre-configured scenarios

### Cloud Sandboxes

4. **AWS Free Tier**
   - URL: https://aws.amazon.com/free/
   - 12 months free tier
   - Practice Terraform on real AWS
   - 750 hours/month of EC2 t2.micro

5. **GCP Free Tier**
   - URL: https://cloud.google.com/free
   - $300 credit for 90 days
   - Always-free tier afterwards
   - Good for Terraform practice

6. **Azure Free Tier**
   - URL: https://azure.microsoft.com/free/
   - $200 credit for 30 days
   - Free services for 12 months

---

## Tools & Utilities

### Terraform Tools

1. **tfenv** - Terraform version manager
   - URL: https://github.com/tfutils/tfenv
   - Manage multiple Terraform versions
   - Similar to pyenv, nvm

2. **tflint** - Terraform linter
   - URL: https://github.com/terraform-linters/tflint
   - Find errors and enforce conventions
   - Plugin system for cloud providers

3. **terraform-docs** - Documentation generator
   - URL: https://github.com/terraform-docs/terraform-docs
   - Auto-generate module documentation
   - Markdown, JSON, YAML outputs

4. **tfsec** - Security scanner
   - URL: https://github.com/aquasecurity/tfsec
   - Static analysis for security issues
   - Hundreds of security checks
   - CI/CD integration

5. **Checkov** - Policy as Code scanner
   - URL: https://github.com/bridgecrewio/checkov
   - 1000+ built-in policies
   - AWS, Azure, GCP, Kubernetes
   - Terraform, CloudFormation, Kubernetes

6. **Terrascan** - Security scanner
   - URL: https://github.com/tenable/terrascan
   - 500+ policies
   - Compliance frameworks (PCI-DSS, HIPAA)

7. **Infracost** - Cost estimation
   - URL: https://www.infracost.io/
   - Estimate infrastructure costs
   - GitHub integration
   - Free for individuals

8. **Atlantis** - Terraform automation
   - URL: https://www.runatlantis.io/
   - GitOps for Terraform
   - Automated plan/apply via PRs
   - Self-hosted

9. **env0** - Terraform Cloud alternative
   - URL: https://www.env0.com/
   - Managed Terraform platform
   - Cost tracking
   - Policy enforcement

10. **Spacelift** - Advanced IaC platform
    - URL: https://spacelift.io/
    - Multi-tool support (Terraform, Pulumi, etc.)
    - Advanced workflows

### Pulumi Tools

11. **Pulumi CLI**
    - URL: https://www.pulumi.com/docs/install/
    - Core Pulumi command-line tool

12. **Pulumi Service**
    - URL: https://app.pulumi.com/
    - State management
    - Team collaboration
    - Free tier available

### VS Code Extensions

13. **HashiCorp Terraform**
    - Syntax highlighting
    - Auto-completion
    - Formatting support
    - Validation

14. **Terraform Autocomplete**
    - Advanced auto-completion
    - Resource documentation lookup

---

## GitHub Repositories & Examples

### Terraform Examples

1. **terraform-aws-modules**
   - URL: https://github.com/terraform-aws-modules
   - Comprehensive AWS module collection
   - Production-ready modules
   - VPC, EKS, RDS, S3, and more

2. **terraform-best-practices**
   - URL: https://github.com/antonbabenko/terraform-best-practices
   - Best practices guide
   - Naming conventions
   - Code structure

3. **awesome-terraform**
   - URL: https://github.com/shuaibiyy/awesome-terraform
   - Curated list of Terraform resources
   - Tools, modules, tutorials

4. **terraform-google-modules**
   - URL: https://github.com/terraform-google-modules
   - Official GCP modules
   - GKE, VPC, IAM, etc.

### ML Infrastructure Examples

5. **terraform-aws-eks**
   - URL: https://github.com/terraform-aws-modules/terraform-aws-eks
   - Production-grade EKS clusters
   - GPU node groups support
   - 10k+ stars

6. **kubeflow-terraform**
   - URL: https://github.com/kubeflow/terraform
   - Deploy Kubeflow with Terraform
   - ML platform infrastructure

7. **mlops-terraform**
   - URL: https://github.com/GoogleCloudPlatform/mlops-on-gcp
   - ML infrastructure patterns on GCP
   - Vertex AI integration

### Pulumi Examples

8. **pulumi/examples**
   - URL: https://github.com/pulumi/examples
   - 500+ example programs
   - All major clouds
   - Multiple languages

---

## Blogs & Articles

### HashiCorp Blog

- **[HashiCorp Blog](https://www.hashicorp.com/blog)** - Official updates and guides
- **[Terraform Best Practices](https://www.terraform-best-practices.com/)** - Community guide

### Cloud Provider Blogs

- **[AWS Terraform Blog Posts](https://aws.amazon.com/blogs/apn/tag/terraform/)** - AWS + Terraform
- **[GCP Terraform Guides](https://cloud.google.com/docs/terraform)** - Google Cloud Terraform

### Community Blogs

- **[Gruntwork Blog](https://blog.gruntwork.io/)** - Terraform expertise
- **[Spacelift Blog](https://spacelift.io/blog)** - IaC best practices
- **[env0 Blog](https://www.env0.com/blog)** - Terraform workflows

---

## YouTube Channels

1. **HashiCorp**
   - URL: https://www.youtube.com/c/HashiCorp
   - Official tutorials
   - HashiConf talks
   - Product updates

2. **Pulumi**
   - URL: https://www.youtube.com/c/PulumiTV
   - Pulumi workshops
   - Cloud engineering patterns

3. **TechWorld with Nana**
   - URL: https://www.youtube.com/c/TechWorldwithNana
   - Excellent Terraform tutorials
   - DevOps content

4. **freeCodeCamp.org**
   - Long-form Terraform courses
   - Beginner-friendly

5. **AWS Online Tech Talks**
   - AWS infrastructure automation
   - Terraform on AWS

---

## Podcasts

1. **Screaming in the Cloud**
   - Host: Corey Quinn
   - Cloud economics and IaC

2. **Arrested DevOps**
   - DevOps practices
   - Infrastructure automation

3. **The Cloudcast**
   - Cloud infrastructure topics
   - Industry trends

---

## Certification & Career

### Terraform Certifications

1. **HashiCorp Certified: Terraform Associate**
   - URL: https://www.hashicorp.com/certification/terraform-associate
   - Official Terraform certification
   - Multiple choice exam (57 questions, 60 minutes)
   - $70.50 USD
   - Valid for 2 years
   - **Study Guide**: https://learn.hashicorp.com/terraform/certification/terraform-associate

2. **Terraform Associate Exam Prep (Udemy)**
   - Practice exams
   - ~$20-50

### Cloud Certifications (Complement IaC Skills)

3. **AWS Certified Solutions Architect - Associate**
   - Includes infrastructure design
   - Terraform knowledge helpful

4. **Google Cloud Professional Cloud Architect**
   - Infrastructure design patterns
   - IaC best practices

---

## Community & Support

### Forums & Communities

1. **HashiCorp Discuss**
   - URL: https://discuss.hashicorp.com/
   - Official Terraform forum
   - Get help from experts

2. **Terraform on Reddit**
   - URL: https://www.reddit.com/r/Terraform/
   - Community discussions
   - 50k+ members

3. **Pulumi Community Slack**
   - URL: https://slack.pulumi.com/
   - Real-time help
   - Community support

4. **Stack Overflow**
   - Tag: [terraform]
   - Tag: [pulumi]
   - 80k+ Terraform questions

### Meetups & Events

5. **HashiConf**
   - Annual HashiCorp conference
   - Terraform talks and workshops
   - Virtual and in-person

6. **Local HashiCorp User Groups**
   - URL: https://www.meetup.com/pro/hugs/
   - Meetups worldwide

7. **PulumiUP**
   - Annual Pulumi conference
   - Cloud engineering content

---

## Practice Projects for ML/AI Infrastructure

### Beginner Projects

1. **Single GPU Training Server**
   - 1x EC2 p3.2xlarge instance
   - S3 bucket for datasets
   - Security group configuration
   - IAM role for S3 access

2. **Multi-Environment Setup**
   - Dev, staging, prod environments
   - Use workspaces or directories
   - Environment-specific instance types

3. **Auto-Scaling Inference Cluster**
   - Auto Scaling Group
   - Load balancer
   - CloudWatch alarms

### Intermediate Projects

4. **Complete ML Training Platform**
   - EKS cluster with GPU nodes
   - S3 for data/models
   - ECR for Docker images
   - VPC with public/private subnets
   - Monitoring with Prometheus/Grafana

5. **Multi-Cloud Deployment**
   - Same ML workload on AWS and GCP
   - Use Terraform or Pulumi
   - Cloud-agnostic module design

6. **GitOps Workflow**
   - GitHub Actions for Terraform
   - Automated plan on PR
   - Automated apply on merge
   - tfsec security scanning
   - Infracost integration

### Advanced Projects

7. **Enterprise ML Platform**
   - Multi-account AWS setup
   - Network hub-and-spoke design
   - Centralized logging/monitoring
   - Cost allocation tags
   - Compliance policies (OPA)

8. **Kubernetes-Native ML Infrastructure**
   - EKS with Kubeflow
   - GPU node pools
   - Autoscaling
   - Service mesh (Istio)
   - GitOps with ArgoCD

---

## Terraform Module Registry

### AWS Modules

- **[terraform-aws-vpc](https://registry.terraform.io/modules/terraform-aws-modules/vpc/aws/)** - VPC creation
- **[terraform-aws-eks](https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/)** - EKS clusters
- **[terraform-aws-rds](https://registry.terraform.io/modules/terraform-aws-modules/rds/aws/)** - RDS databases
- **[terraform-aws-s3-bucket](https://registry.terraform.io/modules/terraform-aws-modules/s3-bucket/aws/)** - S3 buckets
- **[terraform-aws-iam](https://registry.terraform.io/modules/terraform-aws-modules/iam/aws/)** - IAM resources

### GCP Modules

- **[terraform-google-network](https://registry.terraform.io/modules/terraform-google-modules/network/google/)** - VPC
- **[terraform-google-kubernetes-engine](https://registry.terraform.io/modules/terraform-google-modules/kubernetes-engine/google/)** - GKE

### Azure Modules

- **[terraform-azurerm-aks](https://registry.terraform.io/modules/Azure/aks/azurerm/)** - AKS clusters
- **[terraform-azurerm-vnet](https://registry.terraform.io/modules/Azure/vnet/azurerm/)** - Virtual networks

---

## Security Resources

1. **[tfsec Documentation](https://aquasecurity.github.io/tfsec/)** - Security checks
2. **[Checkov Policies](https://www.checkov.io/5.Policy%20Index/terraform.html)** - Policy index
3. **[AWS Security Best Practices](https://docs.aws.amazon.com/security/)** - AWS security
4. **[Terraform Security Best Practices](https://blog.gitguardian.com/terraform-security/)** - GitGuardian guide

---

## Cheat Sheets & Quick References

1. **[Terraform Cheat Sheet by Gruntwork](https://blog.gruntwork.io/a-comprehensive-guide-to-terraform-b3d32832baca)**
2. **[HCL Syntax Quick Reference](https://www.terraform.io/language/syntax)**
3. **[Terraform CLI Commands](https://www.terraform.io/cli/commands)**
4. **[AWS Instance Types Cheat Sheet](https://instances.vantage.sh/)** - Compare prices

---

## Staying Updated

### Newsletters

1. **[Terraform Weekly](https://weekly.tf/)** - Weekly Terraform news
2. **[Last Week in AWS](https://www.lastweekinaws.com/)** - Cloud news

### Release Notes

3. **[Terraform Releases](https://github.com/hashicorp/terraform/releases)** - Latest versions
4. **[AWS Provider Releases](https://github.com/hashicorp/terraform-provider-aws/releases)**

---

## Recommended Learning Path

### Week 1-2: Foundations
- Read Lessons 01-02
- Complete HashiCorp Learn intro tutorials
- Deploy first resources (S3, EC2)

### Week 3-4: Terraform Deep Dive
- Read Lessons 03-04
- Practice state management
- Build complete ML infrastructure

### Week 5: Pulumi
- Read Lesson 05
- Try Python-based infrastructure
- Compare with Terraform

### Week 6: Advanced Patterns
- Read Lessons 06-07
- Create reusable modules
- Set up GitOps workflow

### Week 7: Security & Production
- Read Lesson 08
- Implement security scanning
- Practice disaster recovery

### Week 8: Certification Prep
- Review all lessons
- Take practice exams
- Consider HashiCorp Terraform Associate cert

---

**Continue your learning journey with Module 10: LLM Infrastructure!**
