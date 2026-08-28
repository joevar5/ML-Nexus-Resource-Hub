# Module 07: GPU Computing & Distributed Training - Resources

## Official Documentation

### NVIDIA CUDA
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Comprehensive CUDA programming reference
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - Performance optimization guide
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) - Complete CUDA documentation
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/) - Deep learning primitives

### PyTorch Distributed
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html) - Introduction to distributed training
- [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) - DDP API reference
- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) - Step-by-step DDP guide
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) - How PyTorch uses CUDA

### NVIDIA Deep Learning
- [Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/) - Optimization best practices
- [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/) - AMP guide
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/) - Collective communications library

## Books

### GPU Computing
1. **"Programming Massively Parallel Processors"** by David Kirk and Wen-mei Hwu
   - Comprehensive introduction to GPU programming
   - CUDA fundamentals and advanced techniques
   - [Available on Amazon](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923)

2. **"Professional CUDA C Programming"** by John Cheng, Max Grossman, Ty McKercher
   - Practical CUDA programming guide
   - Real-world optimization techniques
   - [Available on O'Reilly](https://www.oreilly.com/library/view/professional-cuda-c/9781118739310/)

### Distributed Training
3. **"Distributed Machine Learning Patterns"** by Yuan Tang
   - Modern distributed ML patterns
   - Practical implementations
   - [Available on Manning](https://www.manning.com/books/distributed-machine-learning-patterns)

4. **"Deep Learning at Scale"** by Suneeta Mall
   - Scaling deep learning systems
   - Infrastructure considerations
   - [Available on Manning](https://www.manning.com/books/deep-learning-at-scale)

## Research Papers

### Foundational
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) - AlexNet (started GPU revolution in DL)
- [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) - Training with large batches

### Distributed Training
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704) - DDP implementation details
- [Horovod: Fast and Easy Distributed Deep Learning](https://arxiv.org/abs/1802.05799) - Uber's distributed training framework
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) - DeepSpeed ZeRO

### Model Parallelism
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) - Tensor parallelism for transformers
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) - Pipeline parallelism
- [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377) - Advanced pipeline strategies

### Memory Optimization
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) - Gradient checkpointing
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) - Improved Flash Attention

### Mixed Precision
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) - AMP foundations
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) - Next-generation precision

## Online Courses

### NVIDIA Deep Learning Institute
- [Fundamentals of Accelerated Computing with CUDA C/C++](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1/)
- [Fundamentals of Deep Learning](https://courses.nvidia.com/courses/course-v1:DLI+C-FX-01+V1/)
- [Scaling Workloads Across Multiple GPUs](https://courses.nvidia.com/courses/course-v1:DLI+C-MG-01+V1/)

### Coursera
- [Parallel Programming in CUDA](https://www.coursera.org/specializations/gpu-programming) - Johns Hopkins University
- [High Performance Computing](https://www.coursera.org/specializations/computational-thinking) - University of Colorado Boulder

### Fast.ai
- [Practical Deep Learning for Coders](https://course.fast.ai/) - Includes GPU optimization techniques
- [From Deep Learning Foundations to Stable Diffusion](https://course.fast.ai/Lessons/part2.html) - Advanced topics

## Tools & Libraries

### PyTorch Ecosystem
- **PyTorch** - https://pytorch.org/
- **PyTorch Lightning** - https://lightning.ai/ - High-level training framework
- **Accelerate** - https://huggingface.co/docs/accelerate/ - Simplified distributed training
- **torchrun** - Built-in launcher for distributed training

### Distributed Training Frameworks
- **DeepSpeed** - https://www.deepspeed.ai/ - Microsoft's training optimization library
- **Megatron-LM** - https://github.com/NVIDIA/Megatron-LM - NVIDIA's large model training
- **Horovod** - https://horovod.ai/ - Uber's distributed DL framework
- **Alpa** - https://alpa.ai/ - Automated model parallelism

### Memory Optimization
- **Flash Attention** - https://github.com/Dao-AILab/flash-attention
- **xFormers** - https://github.com/facebookresearch/xformers - Memory-efficient operations
- **bitsandbytes** - https://github.com/TimDettmers/bitsandbytes - 8-bit optimizers
- **Apex** - https://github.com/NVIDIA/apex - NVIDIA's mixed precision tools

### Profiling & Monitoring
- **NVIDIA Nsight Systems** - https://developer.nvidia.com/nsight-systems
- **NVIDIA Nsight Compute** - https://developer.nvidia.com/nsight-compute
- **PyTorch Profiler** - Built into PyTorch
- **TensorBoard** - https://www.tensorflow.org/tensorboard - Visualization
- **Weights & Biases** - https://wandb.ai/ - Experiment tracking
- **nvidia-ml-py** - https://pypi.org/project/nvidia-ml-py3/ - Python bindings for nvidia-smi

### Infrastructure
- **Ray** - https://ray.io/ - Distributed computing framework
- **Kubernetes** - https://kubernetes.io/ - Container orchestration
- **Kubeflow** - https://kubeflow.org/ - ML on Kubernetes
- **MLflow** - https://mlflow.org/ - ML lifecycle management

## Blog Posts & Tutorials

### NVIDIA
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/) - Latest GPU computing insights
- [Mixed Precision Training Tutorial](https://developer.nvidia.com/automatic-mixed-precision)
- [Multi-GPU Programming Models](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)

### PyTorch
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Distributed Training Tutorial Series](https://pytorch.org/tutorials/beginner/ddp_series_intro.html)
- [PyTorch Blog](https://pytorch.org/blog/) - Official PyTorch updates

### Hugging Face
- [Training Large Language Models](https://huggingface.co/blog/bloom-megatron-deepspeed)
- [Efficient Training on Single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [Multi-GPU Training](https://huggingface.co/docs/transformers/perf_train_gpu_many)

### DeepSpeed
- [Getting Started with DeepSpeed](https://www.deepspeed.ai/getting-started/)
- [ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

## Community & Forums

### Discussion Forums
- [PyTorch Forums - Distributed](https://discuss.pytorch.org/c/distributed/) - Official PyTorch discussion
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/) - CUDA and GPU computing
- [r/MachineLearning](https://reddit.com/r/MachineLearning) - ML community
- [r/CUDA](https://reddit.com/r/CUDA) - CUDA programming

### GitHub Organizations
- [PyTorch](https://github.com/pytorch/pytorch)
- [NVIDIA](https://github.com/NVIDIA)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Hugging Face](https://github.com/huggingface)

## Video Content

### Conference Talks
- [GTC (GPU Technology Conference)](https://www.nvidia.com/en-us/on-demand/session-library/) - NVIDIA's annual conference
- [PyTorch Conference](https://pytorch.org/ecosystem/conferences/) - PyTorch developer summit
- [ICML/NeurIPS/ICLR](https://slideslive.com/) - ML conference recordings

### YouTube Channels
- [NVIDIA Developer](https://www.youtube.com/c/NVIDIADeveloper) - GPU computing tutorials
- [PyTorch](https://www.youtube.com/c/PyTorch) - Official PyTorch channel
- [Weights & Biases](https://www.youtube.com/c/WeightsBiases) - ML engineering talks
- [Two Minute Papers](https://www.youtube.com/c/K%C3%A1rolyZsolnai) - AI research summaries

## Hardware References

### GPU Specifications
- [NVIDIA GPU Specifications](https://www.nvidia.com/en-us/data-center/resources/ai-inferencing-technical-overview/) - Data center GPUs
- [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) - Compute capability lookup
- [GPU Benchmarks](https://lambdalabs.com/gpu-benchmarks) - Performance comparisons

### Architecture Whitepapers
- [NVIDIA Hopper Architecture](https://resources.nvidia.com/en-us-tensor-core) - H100
- [NVIDIA Ampere Architecture](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) - A100
- [NVIDIA Volta Architecture](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) - V100

## Benchmarks & Datasets

### ML Benchmarks
- [MLPerf Training](https://mlcommons.org/en/training-normal-20/) - Industry standard ML benchmarks
- [DAWNBench](https://dawn.cs.stanford.edu/benchmark/) - Training speed and cost
- [Papers With Code](https://paperswithcode.com/sota) - SOTA benchmarks

### Profiling Datasets
- [ImageNet](https://www.image-net.org/) - Standard vision benchmark
- [GLUE](https://gluebenchmark.com/) - NLP benchmark suite
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) - Question answering

## Cheat Sheets

### Quick References
- [CUDA C++ Syntax](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface) - Language extensions
- [PyTorch Distributed Cheat Sheet](https://pytorch.org/tutorials/beginner/dist_overview.html#comparison-table) - API comparison
- [GPU Memory Optimization](https://huggingface.co/docs/transformers/performance) - Quick wins

### Command Line Tools
```bash
# Essential GPU commands
nvidia-smi                          # GPU status
nvidia-smi -l 1                     # Monitor in real-time
nvidia-smi --query-gpu=...          # Query specific metrics
nvtop                               # Interactive monitoring
watch -n 1 nvidia-smi               # Refresh every second

# CUDA info
nvcc --version                      # CUDA compiler version
nvidia-smi --query-gpu=compute_cap  # Compute capability

# PyTorch distributed
torchrun --nproc_per_node=4         # Launch 4 processes
python -m torch.distributed.launch  # Legacy launcher
```

## Practice Environments

### Cloud Platforms
- **Google Colab** - https://colab.research.google.com/ - Free GPU notebooks
- **Kaggle Kernels** - https://www.kaggle.com/code - Free P100 GPUs
- **AWS EC2** - https://aws.amazon.com/ec2/instance-types/p4/ - P4d instances
- **Google Cloud** - https://cloud.google.com/compute/docs/gpus - A100 instances
- **Azure** - https://azure.microsoft.com/en-us/products/virtual-machines/gpu - ND-series

### Academic Resources
- [NCSA GPU Cluster](https://www.ncsa.illinois.edu/) - University resources
- [XSEDE](https://www.xsede.org/) - NSF supercomputing resources
- [Google TRC](https://sites.research.google/trc/) - TPU Research Cloud

## Advanced Topics

### Emerging Technologies
- **FP8 Training** - Next-generation precision format
- **Sparse Training** - Train with sparse gradients
- **Mixture of Experts** - Conditionally activated networks
- **Federated Learning** - Distributed training across devices

### Research Directions
- [Papers With Code - Distributed Training](https://paperswithcode.com/task/distributed-training)
- [Efficient Deep Learning](https://efficientdlsystems.github.io/) - MIT course
- [Stanford CS149](https://gfxcourses.stanford.edu/cs149/fall21) - Parallel computing

## Staying Updated

### Newsletters
- [NVIDIA Developer Newsletter](https://developer.nvidia.com/newsletter)
- [PyTorch Newsletter](https://pytorch.org/get-started/locally/)
- [ImportAI](https://jack-clark.net/) - AI news by Jack Clark
- [The Batch](https://www.deeplearning.ai/the-batch/) - DeepLearning.AI weekly

### Podcasts
- [TWIML AI Podcast](https://twimlai.com/) - ML/AI interviews
- [Gradient Dissent](https://www.gradient-dissent.com/) - Weights & Biases podcast
- [The NVIDIA AI Podcast](https://blogs.nvidia.com/ai-podcast/) - AI innovations

---

## Recommended Learning Path

### Beginner (Weeks 1-2)
1. Start with NVIDIA DLI "Fundamentals of Accelerated Computing"
2. Read PyTorch CUDA Semantics documentation
3. Complete lessons 01-03 of this module
4. Practice: Convert single-GPU training to multi-GPU

### Intermediate (Weeks 3-4)
1. Deep dive into DDP tutorial series
2. Study Megatron-LM paper
3. Complete lessons 04-06 of this module
4. Practice: Implement data parallelism with DDP

### Advanced (Weeks 5-6)
1. Study ZeRO and Flash Attention papers
2. Experiment with DeepSpeed
3. Complete lessons 07-08 of this module
4. Practice: Optimize large model training

### Expert (Ongoing)
1. Read latest papers on distributed training
2. Contribute to open source (PyTorch, DeepSpeed)
3. Attend GTC and PyTorch conferences
4. Build production ML infrastructure

---

**Keep learning and stay updated with the rapidly evolving GPU computing landscape!**
