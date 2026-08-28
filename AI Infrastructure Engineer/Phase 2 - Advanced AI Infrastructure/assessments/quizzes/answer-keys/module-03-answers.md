# Module 03: Containerization — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-03-containerization/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** What is the main difference between a Docker image and a Docker container?

**Answer:** B) Images are templates; containers are running instances of images

**Explanation:**
A Docker image is an immutable, read-only template composed of stacked filesystem layers, build metadata, and a default command. When you `docker run` an image, Docker creates a container by adding a thin read-write layer on top of those read-only layers and starting the entrypoint process. Many containers can be launched from the same image, but each gets its own writable layer, process namespace, and runtime state.

**Common Mistakes:**
- Choosing A inverts the relationship — images are not running instances and containers are not templates.
- Choosing C ignores that an image has no running process or writable layer, while a container does.
- Choosing D conflates artifact type with size; a container reuses image layers and can actually be smaller on disk than the image itself.

**Related Material:** `lessons/mod-03-containerization/01-docker-introduction.md`

---

## Question 2
**Q:** True or False: Docker containers share the host OS kernel.

**Answer:** True

**Explanation:**
Containers are isolated user-space processes that rely on Linux kernel features such as namespaces (PID, mount, network, UTS, IPC, user) for isolation and cgroups for resource limits, but they all execute against the host's running kernel. This is why containers start in milliseconds and use far less memory than VMs, which boot a full guest kernel inside a hypervisor. It also explains why a Linux container image cannot run natively on a Windows kernel without a Linux VM underneath.

**Common Mistakes:**
- Marking False often comes from confusing containers with VMs, which do bundle their own kernel.
- Forgetting that Docker Desktop on macOS/Windows runs a hidden Linux VM, which can make it look like the kernel is "separate" when it is actually shared inside that VM.

**Related Material:** `lessons/mod-03-containerization/01-docker-introduction.md`

---

## Question 3
**Q:** Which Dockerfile instruction is used to specify the base image?

**Answer:** C) FROM

**Explanation:**
Every Dockerfile must begin with a `FROM` instruction (optionally preceded only by `ARG`s used in `FROM`), which sets the base image for the build stage. The base image determines the starting filesystem, default user, and any preinstalled tooling — for ML work this is often something like `FROM python:3.11-slim` or `FROM nvidia/cuda:12.0-base`. Multi-stage builds use multiple `FROM` lines to define independent stages that can be selectively copied from.

**Common Mistakes:**
- Picking A (`BASE`) or B (`IMAGE`) — these are not Dockerfile instructions at all.
- Picking D (`SOURCE`) — sometimes guessed because of "source image," but no such instruction exists.

**Related Material:** `lessons/mod-03-containerization/02-dockerfile-ml-apps.md`

---

## Question 4
**Q:** What is the purpose of the `-p` flag in `docker run -p 8080:80`?

**Answer:** B) Map host port 8080 to container port 80

**Explanation:**
The `-p HOST:CONTAINER` flag publishes a container port to the host by installing an iptables/NAT rule, so traffic arriving on the host's TCP port 8080 is forwarded to the container's port 80. Without publishing, the container's port is only reachable on the container's internal network. The order matters: host port is always on the left, container port on the right.

**Common Mistakes:**
- Choosing A confuses `-p` (publish) with priority/CPU flags like `--cpu-shares`.
- Choosing C invents a "private mode" that does not exist for `-p`.
- Choosing D is partially right in spirit but misleading — publishing exposes the port to the host, not directly to an external network (firewall rules still apply).

**Related Material:** `lessons/mod-03-containerization/04-docker-networking-volumes.md`

---

## Question 5
**Q:** Which command would you use to see logs from a running container?

**Answer:** B) `docker logs <container_id>`

**Explanation:**
`docker logs` reads whatever the container's PID 1 wrote to STDOUT and STDERR via the configured logging driver (json-file by default). Useful flags include `-f` to follow output in real time, `--tail N` to limit history, and `--since`/`--until` for time-bounded windows. It is the first tool to reach for when debugging why a container exited or is misbehaving.

**Common Mistakes:**
- Choosing A (`docker log`, singular) — the command is pluralized; this typo is a frequent source of "command not found."
- Choosing C or D — `print` and `output` are not Docker subcommands.

**Related Material:** `lessons/mod-03-containerization/01-docker-introduction.md`

---

## Question 6
**Q:** What is the recommended strategy to minimize Docker image rebuild time?

**Answer:** B) Order instructions from least frequently changed to most frequently changed

**Explanation:**
Docker caches each instruction as a layer; a cache hit requires that the instruction and all inputs match a previous build. By placing stable steps (base image, system packages, `pip install -r requirements.txt`) before volatile ones (`COPY . /app`), you preserve cache hits across most rebuilds and only re-run the cheap final layers. Reordering so that source code is copied last is the single biggest win for ML iteration speed.

**Common Mistakes:**
- Choosing A is the exact opposite — putting volatile files first invalidates the cache on every change.
- Choosing C ("one giant RUN") trades cacheability for slightly smaller images and makes every rebuild a full reinstall.
- Choosing D (`--no-cache`) guarantees worst-case rebuild time on every build.

**Related Material:** `lessons/mod-03-containerization/03-image-optimization.md`

---

## Question 7
**Q:** Which Dockerfile instruction creates a new layer?

**Answer:** B) FROM, RUN, COPY, and ADD

**Explanation:**
Layers are created by instructions that change the filesystem: `FROM` pulls in the base layers, while `RUN`, `COPY`, and `ADD` each commit filesystem deltas as a new read-only layer. Metadata-only instructions such as `ENV`, `LABEL`, `EXPOSE`, `WORKDIR`, `CMD`, and `ENTRYPOINT` update the image manifest but do not add a filesystem layer (modern BuildKit treats these as essentially free).

**Common Mistakes:**
- Choosing A omits `COPY` and `ADD`, which clearly write files to the image.
- Choosing C overcounts — metadata-only instructions do not produce filesystem layers.
- Choosing D forgets that `COPY`/`ADD`/`FROM` also create layers.

**Related Material:** `lessons/mod-03-containerization/03-image-optimization.md`

---

## Question 8
**Q:** What is the primary benefit of multi-stage builds?

**Answer:** B) Smaller final image size by excluding build tools

**Explanation:**
A multi-stage Dockerfile uses one stage to compile/install with heavyweight tooling (compilers, build-essential, CUDA dev libraries) and a final, minimal stage that only `COPY --from=builder` pulls the runtime artifacts. The build-only stage is discarded from the final image, commonly cutting size by 50–80% and shrinking the attack surface by leaving compilers and source out of production. Build time is typically similar or slightly slower; the win is image footprint and security.

**Common Mistakes:**
- Choosing A confuses build speed (which is not the primary benefit) with image size.
- Choosing C invents an encryption story — multi-stage builds do not encrypt anything.
- Choosing D is the opposite; multi-stage Dockerfiles are usually more complex to author.

**Related Material:** `lessons/mod-03-containerization/03-image-optimization.md`

---

## Question 9
**Q:** True or False: COPY and ADD instructions are identical in functionality.

**Answer:** False

**Explanation:**
Both copy local files into the image, but `ADD` has two extra behaviors: it can fetch from remote URLs and it automatically extracts recognized local tar archives into the destination. Those side effects make `ADD` surprising and harder to audit, which is why the official guidance is to default to `COPY` and only reach for `ADD` when you specifically need tar extraction (URL downloads are better handled by `RUN curl` so you can verify checksums and clean up).

**Common Mistakes:**
- Marking True overlooks the auto-extraction of tarballs, which has caused real-world bugs when a tar file was meant to be copied as-is.
- Forgetting that linters like Hadolint flag `ADD` for non-tar use precisely because of this difference.

**Related Material:** `lessons/mod-03-containerization/02-dockerfile-ml-apps.md`

---

## Question 10
**Q:** Short Answer: What file would you create to exclude files from Docker build context, and give two examples of what to exclude for ML projects?

**Answer:** `.dockerignore` — two valid examples include `__pycache__/` and `*.pyc` (Python bytecode), `.git/` (repo history), `data/` or `datasets/` (large training data that should be a volume), `.venv/` (host virtualenvs), `*.ipynb` (notebooks), and `*.ckpt`/`*.pt`/`*.bin` (large model weights that belong in object storage).

**Explanation:**
The Docker client tars the entire build context and sends it to the daemon before any instruction runs, so anything not excluded inflates upload time, build cache invalidation, and final image size if accidentally `COPY`d. `.dockerignore` lives at the root of the build context and uses the same glob syntax as `.gitignore`. For ML projects, excluding datasets, checkpoints, virtualenvs, and VCS metadata can shrink the context from gigabytes to megabytes and prevent secrets in `.env` files from leaking into images.

**Common Mistakes:**
- Naming the file `.gitignore` or `dockerignore` (missing leading dot) — Docker will silently ignore it and send the entire directory.
- Forgetting to exclude `node_modules/`, `.venv/`, or large `data/` directories, then wondering why the build context is several GB.
- Excluding source code or `requirements.txt` by accident, which causes `COPY` steps to fail.

**Related Material:** `lessons/mod-03-containerization/03-image-optimization.md`

---

## Question 11
**Q:** Why should you use volumes instead of storing data inside containers?

**Answer:** B) Data in containers is lost when container is removed

**Explanation:**
A container's writable layer lives and dies with the container — `docker rm` deletes everything written above the image's read-only layers. Volumes (named or bind) live outside the container lifecycle, persist across restarts and re-creations, can be shared between containers, and are managed independently by Docker or the host filesystem. They are also significantly faster than the copy-on-write storage driver for write-heavy workloads such as databases or training checkpoints.

**Common Mistakes:**
- Choosing A is true as a secondary benefit (volumes bypass the union filesystem) but is not the primary reason to reach for them.
- Choosing C overstates security; volumes are not inherently more secure than the writable layer.
- Choosing D is factually wrong — containers can store data, it just doesn't persist.

**Related Material:** `lessons/mod-03-containerization/04-docker-networking-volumes.md`

---

## Question 12
**Q:** Which Docker networking mode gives containers direct access to host network interfaces?

**Answer:** C) host

**Explanation:**
`--network host` disables the network namespace for the container, so its processes bind directly to the host's NICs and ports — no NAT, no `-p` publishing required. This eliminates the small bridge/NAT performance overhead (useful for high-throughput inference servers) but removes port isolation: the container's services compete with host services for ports and there is no firewall boundary between them. It is also Linux-only; Docker Desktop simulates it via a VM with caveats.

**Common Mistakes:**
- Choosing A (`bridge`) — bridge is the default isolated network with NAT, not direct host access.
- Choosing B (`none`) — `none` disables networking entirely.
- Choosing D (`overlay`) — overlay is for multi-host Swarm/cluster networking, not host-interface access.

**Related Material:** `lessons/mod-03-containerization/04-docker-networking-volumes.md`

---

## Question 13
**Q:** What is the correct syntax to mount a local directory to a container?

**Answer:** A) `docker run -v /host/path:/container/path image`

**Explanation:**
The `-v` (or `--volume`) flag uses the form `SRC:DST[:OPTIONS]`, where an absolute host path on the left creates a bind mount and a bare name creates/uses a named volume. The newer, more verbose `--mount type=bind,source=/host/path,target=/container/path` syntax is also valid and preferred in production scripts for its explicitness, but `--mount` (option B) was spelled with a single dash, which is invalid.

**Common Mistakes:**
- Choosing B uses `-mount` (single dash) which is not a real flag — the correct long form is `--mount`.
- Choosing C confuses `-d` (detach) with volume mounting.
- Choosing D supplies only a container path with no source, which mounts an anonymous volume rather than a host directory.

**Related Material:** `lessons/mod-03-containerization/04-docker-networking-volumes.md`

---

## Question 14
**Q:** True or False: Containers on the default bridge network can communicate using container names.

**Answer:** False

**Explanation:**
The default `bridge` network is a legacy compatibility network that does not provide embedded DNS for container names; peers must talk to each other by IP address (or use the deprecated `--link` flag). User-defined bridge networks created with `docker network create` do include an internal DNS resolver that maps container names (and `--network-alias` values) to IPs automatically, which is why the documentation recommends always creating a custom network for multi-container apps and Compose does so by default.

**Common Mistakes:**
- Marking True is the most common error because Compose-based experience implies DNS "just works" everywhere; Compose actually creates a user-defined network behind the scenes.
- Forgetting that the `default` bridge and a user-defined bridge behave very differently in this respect.

**Related Material:** `lessons/mod-03-containerization/04-docker-networking-volumes.md`

---

## Question 15
**Q:** What is the primary use case for Docker named volumes vs bind mounts?

**Answer:** A) Named volumes are managed by Docker and portable; bind mounts depend on host filesystem structure

**Explanation:**
Named volumes live under `/var/lib/docker/volumes/<name>` (or a volume driver such as NFS/EBS) and are created, backed up, and removed via `docker volume` commands, making them portable across hosts and the preferred choice for production state. Bind mounts attach an arbitrary host path directly into the container, which is convenient for development (live source reloading) but couples the container to the host's directory layout, file ownership, and SELinux/AppArmor labels.

**Common Mistakes:**
- Choosing B overgeneralizes — bind mounts are not categorically faster; performance depends on the storage driver and OS.
- Choosing C is false — named volumes work on every platform Docker runs on.
- Choosing D ignores meaningful operational differences.

**Related Material:** `lessons/mod-03-containerization/04-docker-networking-volumes.md`

---

## Question 16
**Q:** Which component must be installed to enable GPU access in Docker containers?

**Answer:** B) NVIDIA Container Toolkit on the host

**Explanation:**
The NVIDIA Container Toolkit (formerly `nvidia-docker2`) installs a runtime hook on the host that injects the host's NVIDIA driver libraries and device files into containers when `--gpus` is requested. The host still needs the matching NVIDIA driver kernel module, and the container image must include user-space CUDA/cuDNN libraries compatible with that driver — but without the toolkit, Docker has no way to expose `/dev/nvidia*` and the CUDA runtime to the container.

**Common Mistakes:**
- Choosing A or C confuses what goes in the container (CUDA Toolkit, cuDNN) with what enables device passthrough on the host.
- Choosing D ("Docker GPU Plugin") invents a component that doesn't exist; the toolkit registers as an OCI runtime hook, not a Docker plugin.

**Related Material:** `lessons/mod-03-containerization/07-gpu-docker.md`

---

## Question 17
**Q:** What flag is used to allocate all GPUs to a container?

**Answer:** B) `--gpus all`

**Explanation:**
`--gpus` (plural) is the modern Docker CLI flag introduced alongside the NVIDIA Container Toolkit; `all` exposes every GPU on the host. You can also request specific devices (`--gpus '"device=0,1"'`), a count (`--gpus 2`), or capabilities (`--gpus 'all,capabilities=compute,utility'`). The older `--runtime=nvidia` plus `NVIDIA_VISIBLE_DEVICES=all` environment variable still works but is considered legacy.

**Common Mistakes:**
- Choosing A (`--gpu`, singular) — the flag is pluralized; the singular form is silently ignored or errors out.
- Choosing C/D — `--nvidia` and `--cuda` are not Docker flags.

**Related Material:** `lessons/mod-03-containerization/07-gpu-docker.md`

---

## Question 18
**Q:** Short Answer: Explain the primary strategy for reducing Docker image size for ML applications.

**Answer:** Use multi-stage builds: install/compile heavy dependencies in a `builder` stage, then `COPY --from=builder` only the runtime artifacts into a minimal final stage (e.g., `python:3.11-slim`, `nvidia/cuda:*-runtime` instead of `*-devel`). Complement this with `.dockerignore`, combining `RUN` commands and cleaning apt/pip caches in the same layer, choosing the smallest viable base image, and keeping large datasets/model weights out of the image (mount as volumes or pull from object storage at runtime).

**Explanation:**
ML images bloat for predictable reasons: full Python/CUDA dev images, pip build caches, apt lists, accidentally copied datasets, and compiler toolchains that are never needed at inference time. Multi-stage builds attack the biggest contributor — build tooling — by physically excluding it from the final image, and pair naturally with `runtime`-flavored base images. Layer hygiene (`apt-get update && install ... && rm -rf /var/lib/apt/lists/*` in a single `RUN`) and a strict `.dockerignore` close the remaining gaps.

**Common Mistakes:**
- Only mentioning "use alpine" — Alpine often breaks Python wheels (musl vs glibc) and is rarely appropriate for ML workloads; `*-slim` is usually the right call.
- Forgetting that `RUN apt-get clean` in a separate `RUN` does not shrink the image because the previous layer already committed the cache.
- Copying datasets or model weights into the image instead of mounting them, undoing all other size wins.

**Related Material:** `lessons/mod-03-containerization/03-image-optimization.md`

---

## Question 19
**Q:** True or False: Using `nvidia/cuda` base images automatically gives you GPU support without NVIDIA Container Toolkit.

**Answer:** False

**Explanation:**
The `nvidia/cuda` images ship CUDA user-space libraries (libcudart, cuDNN in `cudnn` variants, etc.) so applications inside the container can link against CUDA — but they cannot see a GPU unless the host runtime injects the device files and the matching host driver. That injection is exactly what the NVIDIA Container Toolkit performs when you run with `--gpus`. Without the toolkit, `nvidia-smi` inside the container will fail with "command not found" or "no devices."

**Common Mistakes:**
- Marking True assumes the base image is self-sufficient and conflates user-space CUDA with kernel-driver access, which Docker cannot bridge by itself.
- Forgetting that the host's NVIDIA driver version must also be compatible with the CUDA version baked into the image.

**Related Material:** `lessons/mod-03-containerization/07-gpu-docker.md`

---

## Question 20
**Q:** What Docker Compose instruction allows you to specify that one service depends on another?

**Answer:** B) `depends_on`

**Explanation:**
`depends_on` controls the start and stop order of services and, in v2+ Compose, can also wait for a referenced service's healthcheck to report healthy by using the long-form `condition: service_healthy`. The short-form list only guarantees that the dependency's container has started, not that the application inside is ready to accept connections — for databases and message queues, always pair `depends_on` with a healthcheck or in-app retry logic.

**Common Mistakes:**
- Choosing A (`requires`) or C (`needs`) — these are not Compose keys; `needs` is a GitHub Actions keyword that often leaks into guesses.
- Choosing D (`after`) — sometimes guessed from systemd unit syntax, but Compose uses `depends_on`.
- Even when `depends_on` is chosen, learners forget that it does not wait for readiness without `condition: service_healthy`.

**Related Material:** `lessons/mod-03-containerization/05-docker-compose.md`

---
