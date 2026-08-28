# Module 03 Quiz: Containerization with Docker

**Total Questions:** 20
**Passing Score:** 70% (14/20 correct)
**Time Limit:** 40 minutes
**Type:** Mixed (Multiple Choice, True/False, Short Answer)

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

<details>
<summary>Answer</summary>
True - Containers use the host OS kernel, unlike VMs which run their own kernel. This makes containers lighter and faster than VMs.
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

<details>
<summary>Answer</summary>
False - While both copy files, ADD has additional features (extract tar files, download from URLs). COPY is preferred for simple file copying as it's more explicit and predictable.
</details>

---

### Question 10
**Short Answer: What file would you create to exclude files from Docker build context, and give two examples of what to exclude for ML projects?**

<details>
<summary>Sample Answer</summary>

**File:** `.dockerignore`

**Examples to exclude:**
1. `__pycache__/` and `*.pyc` - Python bytecode files (not needed in container)
2. `.git/` - Git repository history (large and unnecessary)
3. `data/` or `datasets/` - Large training datasets (should be mounted as volumes)
4. `venv/` or `.venv/` - Virtual environments (install fresh in container)
5. `*.ipynb` - Jupyter notebooks (development files)
6. `README.md`, `docs/` - Documentation (optional)

(Any 2 valid examples acceptable)
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

<details>
<summary>Answer</summary>
False - Containers on the default bridge network must use IP addresses. Custom bridge networks support automatic DNS resolution, allowing containers to communicate via names.
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
**Short Answer: Explain the primary strategy for reducing Docker image size for ML applications.**

<details>
<summary>Sample Answer</summary>

**Primary Strategy: Multi-Stage Builds**

Build artifacts in one stage with all build tools (compilers, build-essential, etc.), then copy only necessary artifacts to a minimal runtime stage.

**Example:**
```dockerfile
# Stage 1: Build
FROM python:3.11 as builder
RUN pip install --user pytorch torchvision

# Stage 2: Runtime (much smaller)
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
```

**Additional strategies:**
- Use slim/alpine base images
- Minimize layers (combine RUN commands)
- Remove build dependencies after installation
- Use .dockerignore to exclude unnecessary files
- Avoid copying large datasets into images

(Core concept + example acceptable)
</details>

---

### Question 19
**True or False: Using `nvidia/cuda` base images automatically gives you GPU support without NVIDIA Container Toolkit.**

<details>
<summary>Answer</summary>
False - The NVIDIA Container Toolkit must be installed on the host to pass through GPU devices. The nvidia/cuda image contains CUDA libraries needed inside the container, but won't work without the toolkit on the host.
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

## Scoring Guide

### Grading Rubric

- **18-20 correct (90-100%):** Excellent - You have mastered Docker containerization
- **16-17 correct (80-89%):** Very Good - Strong understanding with minor gaps
- **14-15 correct (70-79%):** Passing - Adequate knowledge, review areas where you struggled
- **Below 14 (< 70%):** Not Passing - Review module material and retake quiz

### What to Do Next

**If you passed (≥ 70%):**
1. Review any questions you got wrong
2. Complete the Module 03 practical exercises
3. Work on the practical assessment (build production-ready container)
4. Proceed to Module 04: Kubernetes Fundamentals

**If you didn't pass (< 70%):**
1. Review lessons corresponding to questions you missed
2. Practice writing Dockerfiles and building images
3. Work through hands-on exercises again
4. Retake quiz after additional study

---

## Answer Key Summary

1. B
2. True
3. C
4. B
5. B
6. B
7. B
8. B
9. False
10. Short Answer (see details)
11. B
12. C
13. A
14. False
15. A
16. B
17. B
18. Short Answer (see details)
19. False
20. B

---

## Key Concepts to Review

### If you struggled with Section 1 (Fundamentals):
- Review Lesson 01: Docker Introduction
- Practice: Run containers, explore docker ps, docker logs, docker exec
- Understand: Image vs container, layers, Docker architecture

### If you struggled with Section 2 (Dockerfiles):
- Review Lesson 02: Dockerfiles for ML Apps
- Review Lesson 03: Image Optimization
- Practice: Write Dockerfiles, build images, use multi-stage builds
- Understand: Layer caching, instruction ordering, .dockerignore

### If you struggled with Section 3 (Volumes/Networking):
- Review Lesson 04: Docker Networking and Volumes
- Practice: Mount volumes, connect containers, test networking
- Understand: Volume persistence, network modes, DNS resolution

### If you struggled with Section 4 (GPU/Optimization):
- Review Lesson 07: GPU Support in Docker
- Review Lesson 03: Image Optimization
- Practice: Build optimized images, test GPU access
- Understand: NVIDIA Container Toolkit, multi-stage builds, size optimization

---

## Practical Application

After passing the quiz, demonstrate skills by:

1. **Build a production Dockerfile** for an ML model (< 500MB, multi-stage)
2. **Use Docker Compose** to run model + database + cache
3. **Enable GPU** in a container and run GPU-accelerated inference
4. **Push image** to Docker Hub or cloud registry
5. **Document** your work in a README with build instructions

---

**Time to Review:** 20-30 minutes to review answers and understand mistakes

**Ready for Module 04?** If you scored ≥ 70% and completed exercises, move forward to Kubernetes!
