# Module 04 Resources: Kubernetes Fundamentals

## Official Documentation

### Kubernetes Core
- [Kubernetes Documentation](https://kubernetes.io/docs/) - Complete official docs
- [Kubernetes Concepts](https://kubernetes.io/docs/concepts/) - Architecture and fundamentals
- [kubectl Reference](https://kubernetes.io/docs/reference/kubectl/) - Command-line tool documentation
- [Kubernetes API Reference](https://kubernetes.io/docs/reference/kubernetes-api/) - Complete API documentation
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/) - Common commands

### Helm
- [Helm Documentation](https://helm.sh/docs/) - Package manager for Kubernetes
- [Helm Charts Repository](https://artifacthub.io/) - Public Helm charts
- [Helm Best Practices](https://helm.sh/docs/chart_best_practices/) - Chart development guide

### GPU Support
- [Kubernetes Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)
- [GPU Scheduling in Kubernetes](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

---

## Books

### Kubernetes Fundamentals
1. **"Kubernetes in Action" by Marko Luksa**
   - Comprehensive introduction to K8s
   - Hands-on examples throughout
   - Excellent for beginners to intermediate

2. **"Kubernetes: Up and Running" by Brendan Burns, Joe Beda, and Kelsey Hightower**
   - Written by K8s creators
   - Practical guide to running applications
   - Best practices from Google engineers

3. **"The Kubernetes Book" by Nigel Poulton**
   - Clear, concise explanations
   - Good for quick learning
   - Regularly updated

### Advanced Kubernetes
4. **"Production Kubernetes" by Josh Rosso, Rich Lander, Alex Brand, John Harris**
   - Building production-grade clusters
   - Security, networking, monitoring
   - Real-world scenarios

5. **"Programming Kubernetes" by Michael Hausenblas and Stefan Schimanski**
   - Extending Kubernetes with custom resources
   - Building operators
   - For advanced users

### ML on Kubernetes
6. **"Kubeflow for Machine Learning" by Trevor Grant et al.**
   - ML workflows on Kubernetes
   - Kubeflow platform deep dive
   - End-to-end ML pipelines

---

## Online Courses

### Kubernetes Basics
- **[Kubernetes for Beginners](https://www.youtube.com/watch?v=X48VuDVv0do)** (FreeCodeCamp, YouTube, Free)
  - 4-hour comprehensive introduction
  - Hands-on demos
  - Perfect starting point

- **[Kubernetes Certified Application Developer (CKAD)](https://www.udemy.com/course/certified-kubernetes-application-developer/)** (Udemy)
  - Mumshad Mannambeth's popular course
  - Hands-on labs
  - Certification preparation

- **[Kubernetes: Getting Started](https://www.pluralsight.com/courses/kubernetes-getting-started)** (Pluralsight, Subscription)
  - Nigel Poulton's course
  - Clear explanations
  - Practical examples

### Advanced Kubernetes
- **[Kubernetes the Hard Way](https://github.com/kelseyhightower/kubernetes-the-hard-way)** (Free, GitHub)
  - By Kelsey Hightower
  - Manual K8s cluster setup
  - Deep understanding of components

- **[Advanced Kubernetes on GCP](https://www.coursera.org/learn/advanced-kubernetes-on-gcp)** (Coursera)
  - Advanced GKE concepts
  - Security, networking, monitoring
  - Google Cloud official training

### ML on Kubernetes
- **[Machine Learning with Kubeflow](https://www.linkedin.com/learning/machine-learning-with-kubeflow)** (LinkedIn Learning)
  - Kubeflow platform
  - ML workflows on K8s
  - Hands-on projects

---

## Tutorials and Guides

### Interactive Learning
- **[Play with Kubernetes](https://labs.play-with-k8s.com/)** - Free browser-based K8s playground
- **[Katacoda Kubernetes Scenarios](https://www.katacoda.com/courses/kubernetes)** - Interactive tutorials
- **[Kubernetes by Example](https://kubernetesbyexample.com/)** - Hands-on examples

### Blog Posts and Articles

#### Kubernetes Basics
- [Understanding Kubernetes Objects](https://kubernetes.io/docs/concepts/overview/working-with-objects/kubernetes-objects/)
- [A Beginner's Guide to Kubernetes](https://www.cncf.io/blog/2019/08/19/how-kubernetes-became-the-solution-for-migrating-legacy-applications/)
- [Kubernetes Networking Explained](https://sookocheff.com/post/kubernetes/understanding-kubernetes-networking-model/)

#### Best Practices
- [Kubernetes Best Practices (Google)](https://cloud.google.com/blog/products/containers-kubernetes/your-guide-kubernetes-best-practices)
- [Production Best Practices](https://learnk8s.io/production-best-practices)
- [Resource Requests and Limits](https://sysdig.com/blog/kubernetes-limits-requests/)

#### ML on Kubernetes
- [Deploying ML Models on Kubernetes](https://towardsdatascience.com/deploying-machine-learning-models-on-kubernetes-a-complete-guide-8d5b5e8d8e62)
- [GPU Scheduling in K8s](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [MLOps with Kubernetes](https://www.infoq.com/articles/mlops-kubernetes/)

---

## Tools and Software

### Kubernetes Distributions

#### Local Development
- **[minikube](https://minikube.sigs.k8s.io/)** - Local K8s cluster (most popular)
- **[kind](https://kind.sigs.k8s.io/)** - Kubernetes in Docker (fast, lightweight)
- **[k3s](https://k3s.io/)** - Lightweight K8s (perfect for edge/IoT)
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** - Includes K8s cluster (easy setup)
- **[MicroK8s](https://microk8s.io/)** - Ubuntu's minimal K8s

#### Cloud-Managed Kubernetes
- **[AWS EKS](https://aws.amazon.com/eks/)** - Elastic Kubernetes Service
- **[Google GKE](https://cloud.google.com/kubernetes-engine)** - Google Kubernetes Engine
- **[Azure AKS](https://azure.microsoft.com/en-us/services/kubernetes-service/)** - Azure Kubernetes Service
- **[DigitalOcean DOKS](https://www.digitalocean.com/products/kubernetes/)** - Affordable managed K8s

### Essential CLI Tools
```bash
# kubectl - Kubernetes CLI (required)
# Install: https://kubernetes.io/docs/tasks/tools/

# kubectx - Switch between clusters
# kubens - Switch between namespaces
brew install kubectx

# k9s - Terminal-based K8s UI
brew install k9s

# stern - Multi-pod log tailing
brew install stern

# Helm - Package manager
brew install helm
```

### GUI Tools
- **[Lens](https://k8slens.dev/)** - Kubernetes IDE (free, excellent)
- **[Octant](https://octant.dev/)** - Web-based K8s dashboard (VMware)
- **[K9s](https://k9scli.io/)** - Terminal-based UI
- **[Kubernetes Dashboard](https://kubernetes.io/docs/tasks/access-application-cluster/web-ui-dashboard/)** - Official web UI

### Monitoring and Observability
- **[Prometheus](https://prometheus.io/)** - Metrics collection
- **[Grafana](https://grafana.com/)** - Visualization
- **[Jaeger](https://www.jaegertracing.io/)** - Distributed tracing
- **[Kiali](https://kiali.io/)** - Service mesh observability

### CI/CD
- **[ArgoCD](https://argoproj.github.io/cd/)** - GitOps continuous delivery
- **[Flux](https://fluxcd.io/)** - GitOps toolkit
- **[Tekton](https://tekton.dev/)** - Cloud-native CI/CD
- **[Jenkins X](https://jenkins-x.io/)** - CI/CD for K8s

### Security
- **[Falco](https://falco.org/)** - Runtime security
- **[OPA (Open Policy Agent)](https://www.openpolicyagent.org/)** - Policy enforcement
- **[KubeSec](https://kubesec.io/)** - Security risk analysis
- **[Trivy](https://aquasecurity.github.io/trivy/)** - Container vulnerability scanning

---

## Kubernetes for ML

### ML Platforms on Kubernetes
- **[Kubeflow](https://www.kubeflow.org/)** - End-to-end ML platform
- **[MLflow on K8s](https://mlflow.org/)** - Experiment tracking
- **[Seldon Core](https://www.seldon.io/)** - ML deployment platform
- **[KServe (KFServing)](https://kserve.github.io/website/)** - Serverless model serving
- **[Ray on K8s](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)** - Distributed computing

### GPU Scheduling
- **[NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)** - Automate GPU setup
- **[GPU Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)** - NVIDIA GPU device plugin
- **[GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/)** - High-performance data loading

---

## Community and Forums

### Official Communities
- **[Kubernetes Slack](https://kubernetes.slack.com/)** - Official Slack (very active)
- **[Kubernetes Discourse](https://discuss.kubernetes.io/)** - Forum discussions
- **[Kubernetes GitHub](https://github.com/kubernetes/kubernetes)** - Source code and issues
- **[CNCF Slack](https://cloud-native.slack.com/)** - Cloud Native Computing Foundation

### Discussion Forums
- **[r/kubernetes](https://reddit.com/r/kubernetes)** - Kubernetes Reddit community
- **[Stack Overflow - kubernetes](https://stackoverflow.com/questions/tagged/kubernetes)** - Q&A
- **[Server Fault - kubernetes](https://serverfault.com/questions/tagged/kubernetes)** - Ops-focused Q&A

### Social Media
- **[@kubernetesio](https://twitter.com/kubernetesio)** - Official Kubernetes Twitter
- **[@kelseyhightower](https://twitter.com/kelseyhightower)** - Kelsey Hightower, K8s advocate
- **[@thockin](https://twitter.com/thockin)** - Tim Hockin, K8s co-founder
- **[@brendandburns](https://twitter.com/brendandburns)** - Brendan Burns, K8s co-founder

### Meetups and Conferences
- **[KubeCon + CloudNativeCon](https://www.cncf.io/kubecon-cloudnativecon-events/)** - Premier K8s conference
- **[Kubernetes Community Days](https://www.cncf.io/kubernetes-community-days/)** - Local community events
- **[Meetup.com - Kubernetes](https://www.meetup.com/topics/kubernetes/)** - Local K8s meetups

---

## Cheat Sheets and Reference

- **[kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)** - Official kubectl commands
- **[Kubernetes API Cheat Sheet](https://unofficial-kubernetes.readthedocs.io/en/latest/api-reference/v1/definitions/)** - API objects reference
- **[YAML Templates](https://k8syaml.com/)** - Common K8s YAML examples
- **[Kubernetes Patterns](https://k8spatterns.io/)** - Design patterns for K8s

---

## Certifications

### CNCF Certifications
- **[Certified Kubernetes Application Developer (CKAD)](https://www.cncf.io/certification/ckad/)** - Application deployment focus (Recommended)
- **[Certified Kubernetes Administrator (CKA)](https://www.cncf.io/certification/cka/)** - Cluster administration
- **[Certified Kubernetes Security Specialist (CKS)](https://www.cncf.io/certification/cks/)** - Security focus

### Cloud-Specific
- **[Google Cloud Professional Cloud Developer](https://cloud.google.com/certification/cloud-developer)** - Includes GKE
- **[AWS Certified Solutions Architect](https://aws.amazon.com/certification/certified-solutions-architect-professional/)** - Includes EKS
- **[Microsoft Certified: Azure Administrator](https://docs.microsoft.com/en-us/learn/certifications/azure-administrator/)** - Includes AKS

---

## YouTube Channels

- **[Kubernetes](https://www.youtube.com/c/KubernetesCommunity)** - Official K8s channel
- **[CNCF](https://www.youtube.com/c/cloudnativefdn)** - Cloud Native Computing Foundation
- **[TechWorld with Nana](https://www.youtube.com/c/TechWorldwithNana)** - K8s tutorials
- **[That DevOps Guy](https://www.youtube.com/c/MarcelDempers)** - K8s and DevOps
- **[Just me and Opensource](https://www.youtube.com/c/wenkatn-justmeandopensource)** - K8s deep dives

---

## Podcasts

- **[Kubernetes Podcast from Google](https://kubernetespodcast.com/)** - Weekly K8s news and interviews
- **[The Podlets](https://thepodlets.io/)** - Cloud-native discussions
- **[Software Engineering Daily - Kubernetes Episodes](https://softwareengineeringdaily.com/?s=kubernetes)** - K8s interviews

---

## GitHub Repositories

### Learning Resources
- **[Awesome Kubernetes](https://github.com/ramitsurana/awesome-kubernetes)** - Curated K8s resources
- **[Kubernetes Examples](https://github.com/kubernetes/examples)** - Official example applications
- **[Kubernetes Failure Stories](https://github.com/hjacobs/kubernetes-failure-stories)** - Learn from production incidents

### Tools and Projects
- **[Helm Charts](https://github.com/helm/charts)** - Helm chart repository
- **[Kubernetes Device Plugins](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/resource-management/device-plugin.md)** - Device plugin spec
- **[KubeEdge](https://github.com/kubeedge/kubeedge)** - K8s for edge computing

---

## Best Practices Guides

### Production Kubernetes
- [Production Best Practices](https://learnk8s.io/production-best-practices)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/security-best-practices/)
- [Resource Management Best Practices](https://cloud.google.com/blog/products/containers-kubernetes/kubernetes-best-practices-resource-requests-and-limits)

### ML on Kubernetes
- [ML Inference at Scale on K8s](https://developer.nvidia.com/blog/deploying-machine-learning-models-in-production-with-kubernetes/)
- [GPU Best Practices](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
- [Kubeflow Best Practices](https://www.kubeflow.org/docs/started/installing-kubeflow/)

---

## Additional Learning Paths

### After Module 04

**Next Module:**
- **Module 05: Data Pipelines and Orchestration** - Build ML data workflows

**Deep Dive Options:**
- **Advanced Kubernetes Networking** - Service mesh, network policies
- **Kubernetes Operators** - Build custom controllers
- **Multi-Cluster Management** - Federation, multi-cluster deployments
- **Kubernetes Security** - RBAC, pod security, network policies

---

## Troubleshooting Resources

### Common Issues
- [Kubernetes Troubleshooting Guide](https://kubernetes.io/docs/tasks/debug/)
- [Debug Pods](https://kubernetes.io/docs/tasks/debug/debug-application/debug-pods/)
- [Debug Services](https://kubernetes.io/docs/tasks/debug/debug-application/debug-service/)
- [Network Troubleshooting](https://kubernetes.io/docs/tasks/debug/debug-cluster/network-troubleshooting/)

### Debugging Tools
- `kubectl describe` - Detailed resource information
- `kubectl logs` - Container logs
- `kubectl exec` - Execute commands in containers
- `kubectl port-forward` - Forward ports for debugging
- `kubectl top` - Resource usage (requires Metrics Server)

---

## Practice Environments

### Free Kubernetes Clusters
- **[Play with Kubernetes](https://labs.play-with-k8s.com/)** - 4-hour sessions, free
- **[Killercoda (formerly Katacoda)](https://killercoda.com/playgrounds/course/kubernetes)** - Interactive scenarios
- **[GCP Free Tier GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-zonal-cluster#free-tier-autopilot-cluster)** - Free Autopilot cluster

### Sandboxes
- **Minikube** - Local cluster, free
- **kind** - Docker-based K8s, free
- **Docker Desktop K8s** - Built-in cluster, free

---

## Keep Learning

- **Practice daily with kubectl** - Master the CLI
- **Deploy real applications** - Learn by doing
- **Study official docs** - They're excellent
- **Join K8s communities** - Slack, Reddit, forums
- **Follow K8s on Twitter** - Stay updated
- **Contribute to projects** - Open source contribution
- **Attend KubeCon** - Premier K8s conference (virtual options available)
- **Get certified** - CKAD or CKA validates skills
- **Build operators** - Next-level K8s expertise
- **Share your knowledge** - Blog, speak, mentor

---

**Questions or suggestions for this resource list?** Open an issue on GitHub!

**Want to contribute?** Submit a PR with additional high-quality resources!
