# Module 02 Resources: Cloud Computing for ML

## Official Documentation

### AWS
- [AWS Documentation](https://docs.aws.amazon.com/)
- [AWS Machine Learning](https://aws.amazon.com/machine-learning/)
- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Amazon S3 Documentation](https://docs.aws.amazon.com/s3/)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [AWS Pricing Calculator](https://calculator.aws/)
- [AWS Well-Architected Framework - Machine Learning Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/)

### Google Cloud Platform
- [GCP Documentation](https://cloud.google.com/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine/docs)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [Compute Engine Documentation](https://cloud.google.com/compute/docs)
- [TPU Documentation](https://cloud.google.com/tpu/docs)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
- [ML Best Practices on GCP](https://cloud.google.com/architecture/ml-on-gcp-best-practices)

### Azure
- [Azure Documentation](https://docs.microsoft.com/azure/)
- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/)
- [Azure Kubernetes Service (AKS)](https://docs.microsoft.com/azure/aks/)
- [Azure Blob Storage](https://docs.microsoft.com/azure/storage/blobs/)
- [Azure Virtual Machines](https://docs.microsoft.com/azure/virtual-machines/)
- [Azure OpenAI Service](https://learn.microsoft.com/azure/ai-services/openai/)
- [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
- [Azure Architecture Center - AI/ML](https://docs.microsoft.com/azure/architecture/ai-ml/)

---

## Books

### Cloud Computing Fundamentals
1. **"Architecting the Cloud: Design Decisions for Cloud Computing Service Models"** by Michael J. Kavis
   - Comprehensive guide to cloud architecture patterns
   - Focus on SaaS, PaaS, IaaS design decisions

2. **"Cloud Native Transformation" by Pini Reznik, Jamie Dobson, and Michelle Gienow**
   - Practical patterns for cloud-native systems
   - Real-world case studies from enterprises

### AWS Specific
3. **"AWS Certified Solutions Architect Study Guide"** by Ben Piper and David Clinton
   - Covers core AWS services in depth
   - Great for understanding AWS architecture patterns

4. **"Machine Learning on AWS"** by Janine Garzik and Kevin Huddy
   - Focused on AWS ML services (SageMaker, etc.)
   - Hands-on examples and best practices

### GCP Specific
5. **"Google Cloud Platform in Action"** by JJ Geewax
   - Comprehensive intro to GCP services
   - Practical examples and architecture patterns

6. **"Building Machine Learning Pipelines on Google Cloud Platform"** by Hannes Hapke and Catherine Nelson
   - End-to-end ML workflows on GCP
   - Focus on Vertex AI and TFX

### Azure Specific
7. **"Microsoft Azure for Dummies"** by Jack Hyman
   - Beginner-friendly intro to Azure
   - Covers core services and concepts

8. **"Azure Machine Learning Engineering"** by Sina Fakhraee
   - Comprehensive guide to Azure ML
   - Production ML best practices on Azure

### Cost Optimization
9. **"Cloud FinOps"** by J.R. Storment and Mike Fuller
   - Financial operations for cloud
   - Cost optimization strategies and culture

---

## Online Courses

### Multi-Cloud and General
- **[Full Stack Deep Learning](https://fullstackdeeplearning.com/)** (Free)
  - Comprehensive course on ML infrastructure
  - Covers cloud deployments and MLOps

- **[Andrew Ng's Machine Learning Engineering for Production (MLOps)](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)** (Coursera)
  - MLOps fundamentals
  - Cloud deployment strategies

### AWS Courses
- **[AWS Training and Certification](https://aws.amazon.com/training/)** (Free and Paid)
  - Official AWS training paths
  - ML-specific courses available

- **[AWS Machine Learning Specialty Exam Prep](https://www.udemy.com/course/aws-machine-learning/)** (Udemy)
  - Comprehensive AWS ML services coverage
  - Hands-on labs

- **[A Cloud Guru: AWS Certified Solutions Architect](https://acloudguru.com/course/aws-certified-solutions-architect-associate-saa-c03)** (Subscription)
  - Architecture fundamentals
  - Hands-on labs

### GCP Courses
- **[Google Cloud Skills Boost](https://www.cloudskillsboost.google/)** (Free and Paid)
  - Interactive labs and courses
  - ML and data engineering paths

- **[Coursera: Machine Learning with TensorFlow on GCP](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp)**
  - Official Google Cloud training
  - End-to-end ML on GCP

- **[Linux Academy: Google Cloud Certified Professional Cloud Architect](https://linuxacademy.com/)** (Subscription)
  - Architecture best practices
  - Hands-on scenarios

### Azure Courses
- **[Microsoft Learn](https://docs.microsoft.com/learn/)** (Free)
  - Official Microsoft training
  - AI and ML learning paths

- **[Coursera: Microsoft Azure Data Scientist Associate](https://www.coursera.org/professional-certificates/azure-data-scientist)**
  - Azure ML platform in-depth
  - Hands-on projects

- **[Pluralsight: Microsoft Azure Solutions Architect](https://www.pluralsight.com/paths/microsoft-azure-solutions-architect)** (Subscription)
  - Enterprise architecture on Azure
  - Security and compliance

### Cost Optimization
- **[FinOps Foundation Training](https://www.finops.org/training/)** (Free and Paid)
  - Cloud financial management
  - Cost optimization strategies

---

## Tutorials and Hands-On Labs

### Interactive Labs
- **[Katacoda](https://www.katacoda.com/)** - Interactive cloud labs (free)
- **[Qwiklabs](https://www.qwiklabs.com/)** - Google Cloud hands-on labs
- **[AWS Workshops](https://workshops.aws/)** - Free AWS hands-on workshops
- **[Microsoft Learn Sandbox](https://docs.microsoft.com/learn/)** - Free Azure sandbox environments

### Video Tutorials
- **YouTube Channels:**
  - [AWS Online Tech Talks](https://www.youtube.com/user/AWSwebinars)
  - [Google Cloud Tech](https://www.youtube.com/c/GoogleCloudTech)
  - [Microsoft Azure](https://www.youtube.com/c/MicrosoftAzure)
  - [freeCodeCamp.org](https://www.youtube.com/c/Freecodecamp) - Full cloud courses

### Blog Posts and Articles
- **[AWS Architecture Blog](https://aws.amazon.com/blogs/architecture/)**
- **[GCP Blog](https://cloud.google.com/blog/)**
- **[Azure Blog](https://azure.microsoft.com/blog/)**
- **[Towards Data Science - Cloud ML](https://towardsdatascience.com/tagged/cloud-computing)**
- **[The New Stack - Cloud Native](https://thenewstack.io/category/cloud-native/)**

---

## Tools and Software

### CLI Tools
```bash
# AWS CLI
pip install awscli
aws configure

# GCP SDK
curl https://sdk.cloud.google.com | bash
gcloud init

# Azure CLI
pip install azure-cli
az login
```

### Infrastructure as Code
- **[Terraform](https://www.terraform.io/)** - Multi-cloud IaC
- **[Pulumi](https://www.pulumi.com/)** - Modern IaC with Python/TypeScript
- **[CloudFormation](https://aws.amazon.com/cloudformation/)** - AWS native IaC
- **[Google Cloud Deployment Manager](https://cloud.google.com/deployment-manager)** - GCP native IaC
- **[Azure Resource Manager (ARM)](https://docs.microsoft.com/azure/azure-resource-manager/)** - Azure native IaC

### Cost Management Tools
- **[AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/)**
- **[GCP Cost Management](https://cloud.google.com/cost-management)**
- **[Azure Cost Management](https://azure.microsoft.com/services/cost-management/)**
- **[CloudHealth by VMware](https://www.cloudhealthtech.com/)** - Multi-cloud cost management
- **[Spot.io](https://spot.io/)** - Cloud cost optimization
- **[Infracost](https://www.infracost.io/)** - Cost estimates for Terraform

### Monitoring and Observability
- **[CloudWatch (AWS)](https://aws.amazon.com/cloudwatch/)**
- **[Cloud Monitoring (GCP)](https://cloud.google.com/monitoring)**
- **[Azure Monitor](https://azure.microsoft.com/services/monitor/)**
- **[Datadog](https://www.datadoghq.com/)** - Multi-cloud monitoring
- **[New Relic](https://newrelic.com/)** - Application performance monitoring

---

## Community and Forums

### Discussion Forums
- **[r/aws](https://reddit.com/r/aws)** - AWS Reddit community
- **[r/googlecloud](https://reddit.com/r/googlecloud)** - GCP Reddit community
- **[r/azure](https://reddit.com/r/azure)** - Azure Reddit community
- **[Stack Overflow - aws](https://stackoverflow.com/questions/tagged/amazon-web-services)**
- **[Stack Overflow - gcp](https://stackoverflow.com/questions/tagged/google-cloud-platform)**
- **[Stack Overflow - azure](https://stackoverflow.com/questions/tagged/azure)**

### Slack Communities
- **[AWS Community Slack](https://aws-community.slack.com/)**
- **[GCP Community Slack](https://googlecloud-community.slack.com/)**
- **[Azure Community Slack](https://azurecommunity.slack.com/)**
- **[MLOps Community Slack](https://mlops-community.slack.com/)**

### LinkedIn Groups
- AWS User Groups
- Google Cloud Certified Professionals
- Azure Community
- MLOps Community

### Twitter Follows
- **[@awscloud](https://twitter.com/awscloud)** - AWS official
- **[@GCPcloud](https://twitter.com/gcpcloud)** - GCP official
- **[@Azure](https://twitter.com/azure)** - Azure official
- **[@QuinnyPig](https://twitter.com/QuinnyPig)** - Corey Quinn, AWS cost optimization expert
- **[@kelseyhightower](https://twitter.com/kelseyhightower)** - Kubernetes and cloud-native advocate

---

## Certifications

### AWS
- **AWS Certified Cloud Practitioner** (Foundational)
- **AWS Certified Solutions Architect – Associate** (Recommended)
- **AWS Certified Machine Learning – Specialty** (ML focus)
- **AWS Certified Solutions Architect – Professional** (Advanced)

### GCP
- **Associate Cloud Engineer** (Foundational)
- **Professional Cloud Architect** (Recommended)
- **Professional Machine Learning Engineer** (ML focus)
- **Professional Cloud Developer** (Development focus)

### Azure
- **Microsoft Certified: Azure Fundamentals** (Foundational)
- **Microsoft Certified: Azure Administrator Associate** (Operations)
- **Microsoft Certified: Azure Solutions Architect Expert** (Recommended)
- **Microsoft Certified: Azure AI Engineer Associate** (ML focus)

### Multi-Cloud
- **FinOps Certified Practitioner** - Cloud cost management
- **Certified Kubernetes Administrator (CKA)** - Kubernetes across clouds

---

## Podcasts

- **[AWS Podcast](https://aws.amazon.com/podcasts/aws-podcast/)** - Official AWS podcast
- **[Google Cloud Podcast](https://www.gcppodcast.com/)** - Weekly GCP updates
- **[Azure Friday](https://azure.microsoft.com/resources/videos/azure-friday/)** - Azure video series
- **[Screaming in the Cloud](https://www.lastweekinaws.com/podcast/screaming-in-the-cloud/)** - Cloud rants and discussions by Corey Quinn
- **[The Cloudcast](https://www.thecloudcast.net/)** - Cloud computing news and trends

---

## Newsletters

- **[Last Week in AWS](https://www.lastweekinaws.com/)** - AWS news with humor (Corey Quinn)
- **[Google Cloud Weekly Newsletter](https://cloud.google.com/newsletter)**
- **[Azure Weekly](https://azureweekly.info/)**
- **[The Register - Cloud](https://www.theregister.com/data_centre/cloud/)**
- **[Cloud Native Computing Foundation (CNCF) Newsletter](https://www.cncf.io/newsletter/)**

---

## GitHub Repositories

### Example Projects
- **[AWS Samples](https://github.com/aws-samples)** - Official AWS code examples
- **[GoogleCloudPlatform](https://github.com/GoogleCloudPlatform)** - Official GCP samples
- **[Azure-Samples](https://github.com/Azure-Samples)** - Official Azure samples

### Terraform Modules
- **[Terraform AWS Modules](https://github.com/terraform-aws-modules)**
- **[Terraform GCP Modules](https://github.com/terraform-google-modules)**
- **[Terraform Azure Modules](https://github.com/Azure/terraform-azurerm-modules)**

### MLOps Projects
- **[Made With ML](https://github.com/GokuMohandas/Made-With-ML)** - Production ML systems
- **[MLOps Template](https://github.com/fmind/mlops-template)** - End-to-end MLOps
- **[Awesome MLOps](https://github.com/visenger/awesome-mlops)** - Curated MLOps resources

---

## Cheat Sheets

- **[AWS Services Overview](https://d1.awsstatic.com/whitepapers/aws-overview.pdf)** - PDF
- **[GCP Services Comparison](https://cloud.google.com/free/docs/aws-azure-gcp-service-comparison)** - AWS/Azure/GCP equivalent services
- **[Azure Services Comparison](https://docs.microsoft.com/azure/architecture/aws-professional/)** - AWS to Azure mapping
- **[Terraform Cheat Sheet](https://github.com/scraly/terraform-cheat-sheet)**
- **[Kubernetes Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)**

---

## Price Comparison Tools

- **[Cloud Pricing Comparison](https://cloudpricingcomparison.com/)** - Compare VM/storage prices
- **[EC2Instances.info](https://instances.vantage.sh/)** - AWS EC2 instance comparison
- **[GCP Machine Types](https://gcpinstances.doit-intl.com/)** - GCP instance comparison
- **[Azure VM Comparison](https://azureprice.net/)** - Azure VM pricing

---

## Additional Learning Paths

### After Module 02

**Next Module:**
- **Module 03: Containerization with Docker** - Build on cloud knowledge with containers

**Alternative Paths:**
- **Deep Dive: AWS** - Get AWS Solutions Architect Associate certified
- **Deep Dive: GCP** - Explore TPUs and advanced GCP ML services
- **Deep Dive: Cost Optimization** - Focus on FinOps and cloud economics
- **Multi-Cloud Strategy** - Design applications that work across providers

---

## Keep Learning

- **Set up billing alerts immediately** - Never get surprised by cloud costs
- **Join cloud community Slacks** - Network and learn from practitioners
- **Follow cloud blogs** - Stay current with new services and features
- **Build projects on free tiers** - Hands-on practice is essential
- **Experiment and break things** - Learn by doing in sandboxed environments

---

**Questions or suggestions for this resource list?** Open an issue on GitHub!

**Want to contribute?** Submit a PR with additional high-quality resources!
