# Module 03 Resources: Containerization with Docker

## Official Documentation

### Docker Documentation
- [Docker Documentation](https://docs.docker.com/) - Complete official docs
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/) - All Dockerfile instructions
- [Docker CLI Reference](https://docs.docker.com/engine/reference/commandline/cli/) - Command-line reference
- [Docker Compose Documentation](https://docs.docker.com/compose/) - Multi-container applications
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/) - Security best practices

### NVIDIA GPU Support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/) - Pre-built GPU containers
- [CUDA Container Images](https://hub.docker.com/r/nvidia/cuda)

---

## Books

### Docker Fundamentals
1. **"Docker Deep Dive" by Nigel Poulton**
   - Comprehensive Docker guide
   - Covers fundamentals to advanced topics
   - Updated regularly for new Docker features

2. **"Docker in Action" by Jeff Nickoloff and Stephen Kuenzli**
   - Practical Docker patterns
   - Real-world examples
   - Good for intermediate learners

3. **"Docker for Data Science" by Joshua Cook**
   - Docker specifically for data science and ML
   - Jupyter notebooks in Docker
   - Reproducible research workflows

### Container Best Practices
4. **"Container Security" by Liz Rice**
   - Security considerations for containers
   - Scanning, runtime security, compliance
   - Essential for production deployments

5. **"Kubernetes Patterns" by Bilgin Ibryam and Roland Huß**
   - While focused on K8s, covers container design patterns
   - Excellent for understanding cloud-native architecture

---

## Online Courses

### Docker Basics
- **[Docker for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo)** (FreeCodeCamp, YouTube, Free)
  - 2-hour comprehensive intro
  - Hands-on examples
  - Perfect for beginners

- **[Docker Mastery](https://www.udemy.com/course/docker-mastery/)** (Udemy, ~$15)
  - Bret Fisher's comprehensive course
  - Covers Docker, Compose, Swarm
  - Highly rated, regularly updated

- **[Docker Certified Associate Exam Prep](https://acloudguru.com/course/docker-certified-associate-dca)** (A Cloud Guru, Subscription)
  - Certification preparation
  - In-depth Docker knowledge
  - Hands-on labs

### ML-Specific Docker
- **[MLOps with Docker](https://www.coursera.org/learn/mlops-docker-kubernetes)** (Coursera)
  - Docker for ML workflows
  - Integration with ML tools
  - Production deployment patterns

- **[Full Stack Deep Learning](https://fullstackdeeplearning.com/)** (Free)
  - Module on containerization for ML
  - Production ML systems
  - Best practices

---

## Tutorials and Guides

### Interactive Tutorials
- **[Play with Docker](https://labs.play-with-docker.com/)** - Free browser-based Docker playground
- **[Katacoda Docker Scenarios](https://www.katacoda.com/courses/docker)** - Interactive Docker tutorials
- **[Docker 101 Tutorial](https://www.docker.com/101-tutorial/)** - Official Docker getting started

### Blog Posts and Articles

#### Docker Basics
- [Intro to Docker for Data Scientists](https://towardsdatascience.com/docker-for-data-scientists-5732501f0ba4)
- [Docker Best Practices for Data Science](https://dagshub.com/blog/docker-best-practices-for-data-science/)
- [Understanding Docker Layers](https://medium.com/@jessgreb01/digging-into-docker-layers-c22f948ed612)

#### Optimization
- [Reduce Docker Image Size](https://devopscube.com/reduce-docker-image-size/)
- [Multi-Stage Builds Explained](https://docs.docker.com/build/building/multi-stage/)
- [Docker Layer Caching in CI/CD](https://testdriven.io/blog/faster-ci-builds-with-docker-cache/)

#### GPU Support
- [Running GPU-Accelerated Containers](https://developer.nvidia.com/blog/gpu-accelerated-sql-queries-rapids/)
- [Docker and CUDA: Getting Started](https://blog.paperspace.com/nvidia-docker/)

#### Security
- [Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [10 Docker Security Best Practices](https://snyk.io/blog/10-docker-image-security-best-practices/)

---

## Tools and Software

### Docker Desktop Alternatives (Free)
- **[Rancher Desktop](https://rancherdesktop.io/)** - Free, open-source Docker Desktop alternative
- **[Podman](https://podman.io/)** - Daemonless container engine, Docker-compatible
- **[containerd](https://containerd.io/)** - Industry-standard container runtime

### Container Registries
- **[Docker Hub](https://hub.docker.com/)** - Official registry, free public repos
- **[GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)** - Free with GitHub
- **[Quay.io](https://quay.io/)** - Red Hat container registry
- **[AWS ECR](https://aws.amazon.com/ecr/)** - Amazon Elastic Container Registry
- **[GCP Artifact Registry](https://cloud.google.com/artifact-registry)** - Google Cloud registry
- **[Azure ACR](https://azure.microsoft.com/en-us/services/container-registry/)** - Azure Container Registry

### Image Scanning and Security
- **[Trivy](https://github.com/aquasecurity/trivy)** - Vulnerability scanner for containers
- **[Snyk](https://snyk.io/product/container-vulnerability-management/)** - Container security platform
- **[Clair](https://github.com/quay/clair)** - Open-source vulnerability scanner
- **[Anchore](https://anchore.com/)** - Container security and compliance

### Build Tools
- **[Docker Buildx](https://docs.docker.com/buildx/working-with-buildx/)** - Extended build capabilities
- **[BuildKit](https://docs.docker.com/build/buildkit/)** - Improved build backend
- **[Kaniko](https://github.com/GoogleContainerTools/kaniko)** - Build images in Kubernetes

### Development Tools
- **[VS Code Docker Extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)** - Docker support in VS Code
- **[Portainer](https://www.portainer.io/)** - Container management UI
- **[Dive](https://github.com/wagoodman/dive)** - Tool for exploring Docker image layers
- **[Hadolint](https://github.com/hadolint/hadolint)** - Dockerfile linter

---

## Pre-Built ML Container Images

### PyTorch
- **[pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch)** - Official PyTorch images
- **[nvidia/pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)** - NVIDIA-optimized PyTorch

### TensorFlow
- **[tensorflow/tensorflow](https://hub.docker.com/r/tensorflow/tensorflow)** - Official TensorFlow images
- **[nvidia/tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)** - NVIDIA-optimized TensorFlow

### Hugging Face
- **[huggingface/transformers-pytorch-gpu](https://hub.docker.com/r/huggingface/transformers-pytorch-gpu)** - Transformers with PyTorch
- **[huggingface/transformers-tensorflow-gpu](https://hub.docker.com/r/huggingface/transformers-tensorflow-gpu)** - Transformers with TensorFlow

### Jupyter
- **[jupyter/datascience-notebook](https://hub.docker.com/r/jupyter/datascience-notebook)** - Jupyter with data science stack
- **[jupyter/tensorflow-notebook](https://hub.docker.com/r/jupyter/tensorflow-notebook)** - Jupyter with TensorFlow
- **[jupyter/pytorch-notebook](https://hub.docker.com/r/jupyter/pytorch-notebook)** - Jupyter with PyTorch

### NVIDIA NGC
- [NVIDIA GPU Cloud Catalog](https://catalog.ngc.nvidia.com/) - Curated GPU-optimized containers
- RAPIDS, CUDA, cuDNN, TensorRT pre-built images

---

## Community and Forums

### Discussion Forums
- **[Docker Community Forums](https://forums.docker.com/)** - Official Docker discussions
- **[r/docker](https://reddit.com/r/docker)** - Docker Reddit community
- **[Stack Overflow - docker](https://stackoverflow.com/questions/tagged/docker)** - Q&A
- **[Docker Community Slack](https://dockercommunity.slack.com/)** - Real-time chat

### Social Media
- **[@Docker](https://twitter.com/docker)** - Official Docker Twitter
- **[@solomonstre](https://twitter.com/solomonstre)** - Solomon Hykes, Docker co-founder
- **[@jpetazzo](https://twitter.com/jpetazzo)** - Jérôme Petazzoni, Docker expert
- **[@bfirsh](https://twitter.com/bfirsh)** - Ben Firshman, Docker Compose creator

---

## Cheat Sheets and Reference

### Command Cheat Sheets
- [Docker CLI Cheat Sheet (PDF)](https://docs.docker.com/get-started/docker_cheatsheet.pdf)
- [Dockerfile Cheat Sheet](https://kapeli.com/cheat_sheets/Dockerfile.docset/Contents/Resources/Documents/index)
- [Docker Compose Cheat Sheet](https://devhints.io/docker-compose)

### Dockerfile Examples
- [Dockerfile Examples GitHub](https://github.com/jessfraz/dockerfiles) - Jess Frazelle's collection
- [Docker Official Images](https://github.com/docker-library/official-images) - How official images are built
- [Awesome Docker](https://github.com/veggiemonk/awesome-docker) - Curated Docker resources

---

## YouTube Channels

- **[Docker](https://www.youtube.com/c/DockerInc)** - Official Docker channel
- **[TechWorld with Nana](https://www.youtube.com/c/TechWorldwithNana)** - Docker and DevOps tutorials
- **[NetworkChuck](https://www.youtube.com/c/NetworkChuck)** - Docker crash courses
- **[freeCodeCamp.org](https://www.youtube.com/c/Freecodecamp)** - Full Docker courses

---

## Podcasts

- **[The Podlets](https://thepodlets.io/)** - Cloud-native and container discussions
- **[Software Engineering Daily - Docker Episodes](https://softwareengineeringdaily.com/?s=docker)** - Docker interviews

---

## GitHub Repositories

### Example Projects
- **[Docker Samples](https://github.com/dockersamples)** - Official Docker sample applications
- **[Awesome Compose](https://github.com/docker/awesome-compose)** - Sample Docker Compose applications
- **[ML Docker Examples](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/deploy-to-cloud)** - Azure ML Docker examples

### Tools
- **[Dive](https://github.com/wagoodman/dive)** - Explore Docker image layers
- **[Dockerize](https://github.com/jwilder/dockerize)** - Simplify running applications in containers
- **[Docker Slim](https://github.com/docker-slim/docker-slim)** - Minify Docker images
- **[Watchtower](https://github.com/containrrr/watchtower)** - Auto-update running containers

---

## Certifications

- **[Docker Certified Associate (DCA)](https://training.mirantis.com/dca-certification-exam/)**
  - Official Docker certification
  - Validates Docker skills
  - Good for career advancement

---

## Best Practices Guides

### Production Best Practices
- [Docker Production Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [12-Factor App Methodology](https://12factor.net/) - Principles for containerized apps
- [CNCF Cloud Native Security](https://www.cncf.io/blog/2022/06/07/introduction-to-cloud-native-security/)

### Image Optimization
- [Optimizing Docker Images](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#minimize-the-number-of-layers)
- [Multi-Stage Build Best Practices](https://www.docker.com/blog/faster-multi-platform-builds-dockerfile-cross-compilation-guide/)

### Security
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker) - Security configuration benchmark
- [Docker Security Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

---

## Troubleshooting Resources

### Common Issues
- [Docker Troubleshooting Guide](https://docs.docker.com/config/daemon/troubleshoot/)
- [Stack Overflow - docker](https://stackoverflow.com/questions/tagged/docker) - Search existing solutions
- [Docker Community Forums](https://forums.docker.com/) - Ask questions

### Performance
- [Docker Performance Tuning](https://docs.docker.com/config/containers/resource_constraints/)
- [Monitoring Docker Containers](https://docs.docker.com/config/containers/runmetrics/)

---

## Additional Learning Paths

### After Module 03

**Next Module:**
- **Module 04: Kubernetes Fundamentals** - Orchestrate containers at scale

**Deep Dive Options:**
- **Advanced Docker Networking** - Custom networks, plugins, service mesh
- **Docker Swarm** - Docker's native orchestration (alternative to Kubernetes)
- **Container Security** - Advanced security, scanning, runtime protection
- **Docker in CI/CD** - Automate builds, tests, deployments

---

## Practical Exercises Repository

Create a GitHub repository with:
- Example Dockerfiles for different ML frameworks
- Docker Compose stacks for common ML architectures
- Optimization examples (before/after image sizes)
- GPU-enabled examples
- CI/CD workflows with Docker

---

## Keep Learning

- **Build containers daily** - Practice is essential
- **Optimize every image** - Always aim for smaller, faster builds
- **Study layer caching** - This is the #1 performance optimization
- **Learn from official images** - Read Dockerfiles of popular images
- **Experiment with GPU containers** - Even without GPU hardware
- **Join Docker communities** - Learn from practitioners
- **Stay updated** - Docker evolves rapidly, follow release notes

---

**Questions or suggestions for this resource list?** Open an issue on GitHub!

**Want to contribute?** Submit a PR with additional high-quality resources!
