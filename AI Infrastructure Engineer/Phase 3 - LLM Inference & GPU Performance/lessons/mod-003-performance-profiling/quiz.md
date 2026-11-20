# Module 03: Performance Profiling — Quiz

10 questions. 70% pass.

### 1. nsys vs ncu — what's the right split of work?
- a) They are synonyms; pick one
- b) nsys for memory; ncu for compute
- c) nsys is the older deprecated tool
- d) nsys for system-level timeline + workflow; ncu for per-kernel deep dive

### 2. NVTX ranges in your training code:
- a) Annotate code regions so nsys shows them as named bands on the timeline
- b) Replace nsys; they emit their own trace
- c) Only track memory allocations
- d) Are required for the CUDA compiler

### 3. The roofline model classifies kernels into:
- a) Fast vs slow
- b) PyTorch ops vs raw CUDA
- c) Single-threaded vs parallel
- d) Memory-bound (under the slope) vs compute-bound (under the ceiling)

### 4. If GPU Speed-of-Light shows both compute % and memory % below 30%, the diagnosis is usually:
- a) The kernel is latency-bound — too few blocks/warps to fill the GPU, dependency stalls, or launch overhead
- b) Add more memory to the GPU
- c) Use a bigger batch size automatically
- d) Switch the workload to CPU

### 5. PyTorch memory snapshots can be uploaded to:
- a) Console-only output; no visualization
- b) PagerDuty for incident routing
- c) The Apple App Store
- d) pytorch.org/memory_viz for interactive exploration

### 6. Gradient checkpointing trades:
- a) Speed for accuracy
- b) Accuracy for speed
- c) Nothing — it's a free optimization
- d) ~30% more recompute time for ~50% less activation memory

### 7. AdamW8bit (from bitsandbytes) reduces:
- a) Optimizer state memory by roughly 4× (fp32 → int8)
- b) Wall-clock training time
- c) Final model accuracy
- d) Gradient magnitude

### 8. FlashAttention's primary benefit:
- a) Always slower but more numerically accurate
- b) Required for GPT-style architectures
- c) Replaces the softmax function
- d) Memory: O(N²) → O(N) in sequence length; often faster too

### 9. A kernel that's small but called thousands of times per second is bottlenecked by:
- a) Per-launch overhead — batch the work or capture as a CUDA graph
- b) Nothing; small kernels are always good
- c) Cannot be optimized
- d) Required by autograd machinery

### 10. To benchmark GPU code accurately:
- a) A single call is sufficient
- b) Time wall-clock from process start to end
- c) Trust the IDE / debugger timer
- d) Warm up the kernel, use `torch.cuda.synchronize()` between runs, report the median

Answers: [quiz-answers.md](quiz-answers.md)
