# Module 03 Quiz: Containerization with Docker

**Total Questions:** 20
**Passing Score:** 70% (14/20 correct)
**Time Limit:** 40 minutes
**Type:** Mixed (Multiple Choice, True/False)

---

## Section 1: Docker Fundamentals (Questions 1-5)

### Question 1
**What is the main difference between a Docker image and a Docker container?**

A) Images are running instances; containers are templates
B) Images are templates; containers are running instances of images
C) They are the same thing
D) Images are smaller than containers

<details>
<summary>Answer</summary>
B) Images are templates (read-only); containers are running instances of images (read-write layer on top of image)
</details>

---

### Question 2
**True or False: Docker containers share the host OS kernel.**

A) True
B) False

<details>
<summary>Answer</summary>
A) True - Containers use the host OS kernel, unlike VMs which run their own kernel. This makes containers lighter and faster than VMs.
</details>

---

### Question 3
**Which Dockerfile instruction is used to specify the base image?**

A) BASE
B) IMAGE
C) FROM
D) SOURCE

<details>
<summary>Answer</summary>
C) FROM - Every Dockerfile must start with FROM to specify the base image (e.g., FROM python:3.11)
</details>

---

### Question 4
**What is the purpose of the `-p` flag in `docker run -p 8080:80`?**

A) Set container priority
B) Map host port 8080 to container port 80
C) Set container to private mode
D) Enable port forwarding to external network

<details>
<summary>Answer</summary>
B) Map host port 8080 to container port 80 - This allows you to access the container's service on port 80 via the host's port 8080
</details>

---

### Question 5
**Which command would you use to see logs from a running container?**

A) docker log <container_id>
B) docker logs <container_id>
C) docker print <container_id>
D) docker output <container_id>

<details>
<summary>Answer</summary>
B) docker logs <container_id> - Shows STDOUT/STDERR output from the container
</details>

---

## Section 2: Dockerfiles and Image Building (Questions 6-10)

### Question 6
**What is the recommended strategy to minimize Docker image rebuild time?**

A) Put frequently changing files at the beginning of Dockerfile
B) Order instructions from least frequently changed to most frequently changed
C) Use only one RUN instruction for everything
D) Always use `--no-cache` flag

<details>
<summary>Answer</summary>
B) Order instructions from least frequently changed to most frequently changed - This maximizes layer cache utilization. Install dependencies first, copy code last.
</details>

---

### Question 7
**Which Dockerfile instruction creates a new layer?**

A) Only FROM and RUN
B) FROM, RUN, COPY, and ADD
C) All instructions create layers
D) Only RUN instructions

<details>
<summary>Answer</summary>
B) FROM, RUN, COPY, and ADD - These instructions modify the filesystem and create new layers. Instructions like ENV, LABEL, CMD don't create layers.
</details>

---

### Question 8
**What is the primary benefit of multi-stage builds?**

A) Faster build times
B) Smaller final image size by excluding build tools
C) Better security through encryption
D) Easier to write Dockerfiles

<details>
<summary>Answer</summary>
B) Smaller final image size by excluding build tools - Multi-stage builds allow you to build in one stage (with compilers, build tools) and copy only artifacts to final stage, resulting in images 50-80% smaller.
</details>

---

### Question 9
**True or False: COPY and ADD instructions are identical in functionality.**

A) True
B) False

<details>
<summary>Answer</summary>
B) False - While both copy files, ADD has additional features (extract tar files, download from URLs). COPY is preferred for simple file copying as it's more explicit and predictable.
</details>

---

### Question 10
**Which file would you create to exclude files (like `__pycache__/`, `.git/`, or large datasets) from the Docker build context?**

A) `.gitignore`
B) `.dockerignore`
C) `Dockerfile.ignore`
D) `.buildignore`

<details>
<summary>Answer</summary>
B) `.dockerignore` - Excludes matching files/directories from the build context, which speeds up builds, shrinks images, and keeps secrets or large datasets from being copied in by accident. Common ML exclusions: `__pycache__/`, `.git/`, `data/`/`datasets/`, `venv/`, `*.ipynb`.
</details>

---

## Section 3: Docker Volumes and Networking (Questions 11-15)

### Question 11
**Why should you use volumes instead of storing data inside containers?**

A) Volumes are faster
B) Data in containers is lost when container is removed
C) Volumes are more secure
D) Containers don't support data storage

<details>
<summary>Answer</summary>
B) Data in containers is lost when container is removed - Containers are ephemeral. Volumes persist data beyond container lifecycle and allow data sharing between containers.
</details>

---

### Question 12
**Which Docker networking mode gives containers direct access to host network interfaces?**

A) bridge
B) none
C) host
D) overlay

<details>
<summary>Answer</summary>
C) host - Host networking mode removes network isolation; container uses host's network stack directly. Useful for performance but reduces isolation.
</details>

---

### Question 13
**What is the correct syntax to mount a local directory to a container?**

A) `docker run -v /host/path:/container/path image`
B) `docker run -mount /host/path:/container/path image`
C) `docker run -d /host/path:/container/path image`
D) `docker run --volume=/container/path image`

<details>
<summary>Answer</summary>
A) `docker run -v /host/path:/container/path image` - The -v flag maps host directory to container directory
</details>

---

### Question 14
**True or False: Containers on the default bridge network can communicate using container names.**

A) True
B) False

<details>
<summary>Answer</summary>
B) False - Containers on the default bridge network must use IP addresses. Custom bridge networks support automatic DNS resolution, allowing containers to communicate via names.
</details>

---

### Question 15
**What is the primary use case for Docker named volumes vs bind mounts?**

A) Named volumes are managed by Docker and portable; bind mounts depend on host filesystem structure
B) Bind mounts are faster than named volumes
C) Named volumes work only on Linux
D) There is no difference

<details>
<summary>Answer</summary>
A) Named volumes are managed by Docker and portable; bind mounts depend on host filesystem structure - Named volumes are better for production; bind mounts are convenient for development.
</details>

---

## Section 4: GPU Support and Optimization (Questions 16-20)

### Question 16
**Which component must be installed to enable GPU access in Docker containers?**

A) CUDA Toolkit in the container
B) NVIDIA Container Toolkit on the host
C) cuDNN in the container
D) Docker GPU Plugin

<details>
<summary>Answer</summary>
B) NVIDIA Container Toolkit on the host - This allows Docker to pass through GPU devices to containers. CUDA toolkit would be in the container image.
</details>

---

### Question 17
**What flag is used to allocate all GPUs to a container?**

A) `--gpu all`
B) `--gpus all`
C) `--nvidia all`
D) `--cuda all`

<details>
<summary>Answer</summary>
B) `--gpus all` - Example: `docker run --gpus all nvidia/cuda:12.0-base nvidia-smi`
</details>

---

### Question 18
**What is the primary strategy for reducing Docker image size for ML applications?**

A) Install all dependencies with conda instead of pip
B) Use multi-stage builds — build with full tooling in one stage, copy only the needed artifacts to a minimal runtime stage
C) Combine every instruction into a single `RUN` command
D) Always use the full (non-slim) base image so nothing is missing at runtime

<details>
<summary>Answer</summary>
B) Multi-stage builds - build artifacts in a stage with compilers and build tools, then copy only the compiled artifacts into a minimal runtime stage:

```dockerfile
FROM python:3.11 AS builder
RUN pip install --user torch torchvision

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
```

Other contributing strategies: slim/alpine base images, minimizing layers, removing build dependencies after use, `.dockerignore`, and never copying large datasets into the image — but multi-stage builds are the single biggest lever, typically 50-80% smaller.
</details>

---

### Question 19
**True or False: Using `nvidia/cuda` base images automatically gives you GPU support without NVIDIA Container Toolkit.**

A) True
B) False

<details>
<summary>Answer</summary>
B) False - The NVIDIA Container Toolkit must be installed on the host to pass through GPU devices. The nvidia/cuda image contains CUDA libraries needed inside the container, but won't work without the toolkit on the host.
</details>

---

### Question 20
**What Docker Compose instruction allows you to specify that one service depends on another?**

A) requires
B) depends_on
C) needs
D) after

<details>
<summary>Answer</summary>
B) depends_on - Example:
```yaml
services:
  web:
    depends_on:
      - db
  db:
    image: postgres
```
Note: depends_on only controls start order, not readiness. Use health checks for true dependency management.
</details>

---