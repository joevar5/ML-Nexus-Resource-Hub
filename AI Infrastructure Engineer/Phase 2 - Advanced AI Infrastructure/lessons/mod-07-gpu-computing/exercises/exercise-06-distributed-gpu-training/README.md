# Exercise 06: Distributed GPU Training at Scale

**Estimated Time**: 36-44 hours

## Business Context

Your company trains large language models (LLMs) and computer vision models that don't fit on a single GPU. The current situation:

- **Model size**: 13B parameter LLM (52GB in FP32, 26GB in FP16)
- **Available hardware**: 8×A100 (80GB each) per node, 4 nodes total (32 GPUs)
- **Current approach**: Single-GPU training with severe limitations
  - Can only train 7B models (13B doesn't fit)
  - Training time: 3 weeks for 7B model
  - GPU utilization: 60% (communication overhead, load imbalance)

The CTO has allocated a $2.4M GPU cluster and wants you to:
1. **Enable 13B model training** across multiple GPUs/nodes
2. **Reduce training time by 80%** (3 weeks → 4 days)
3. **Achieve >90% scaling efficiency** for 2-32 GPUs
4. **Minimize development complexity** (don't reinvent PyTorch)

You need to implement a production-grade distributed training framework supporting multiple parallelism strategies.

## Learning Objectives

After completing this exercise, you will be able to:

1. Implement data parallelism (DDP) with gradient synchronization
2. Apply model parallelism (tensor/pipeline) for models that don't fit on one GPU
3. Use ZeRO optimizer states sharding (DeepSpeed/FSDP) to reduce memory
4. Optimize inter-GPU communication with NCCL and gradient compression
5. Handle fault tolerance with checkpointing and automatic recovery
6. Monitor and debug distributed training at scale

## Prerequisites

- Module 07 Exercise 05 (GPU Performance Optimization)
- Understanding of deep learning training loops
- Familiarity with PyTorch or TensorFlow
- Basic knowledge of MPI and distributed systems
- Linux and networking fundamentals

## Problem Statement

Build a **Distributed Training Framework** that:

1. **Supports multiple parallelism strategies**:
   - Data Parallelism (DDP) - replicate model, split data
   - Model Parallelism (Tensor/Pipeline) - split model across GPUs
   - ZeRO (Sharded DDP) - shard optimizer states and gradients

2. **Scales efficiently** to 32+ GPUs across multiple nodes

3. **Handles failures gracefully** with checkpointing and recovery

4. **Provides observability** into training progress and performance

### Success Metrics

- 13B parameter model trains successfully on 32 GPUs
- Training time: <5 days (vs 3 weeks baseline)
- Scaling efficiency: >90% for 2-8 GPUs, >80% for 32 GPUs
- Recovery from node failure in <5 minutes
- Memory efficiency: train 2× larger models with ZeRO

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                 Distributed Training Framework                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐      ┌─────────────────┐                 │
│  │ Parallelism     │      │ Communication   │                 │
│  │ Strategies      │◀────▶│ Backend         │                 │
│  │ - DDP           │      │ - NCCL          │                 │
│  │ - Tensor        │      │ - Gloo          │                 │
│  │ - Pipeline      │      │ - MPI           │                 │
│  │ - ZeRO/FSDP     │      │ - Gradient comp │                 │
│  └─────────────────┘      └─────────────────┘                 │
│           │                         │                          │
│           │                         │                          │
│           ▼                         ▼                          │
│  ┌─────────────────┐      ┌─────────────────┐                 │
│  │ Fault Tolerance │      │ Observability   │                 │
│  │ - Checkpointing │      │ - Metrics       │                 │
│  │ - Auto recovery │      │ - Logging       │                 │
│  │ - Elastic train │      │ - Profiling     │                 │
│  └─────────────────┘      └─────────────────┘                 │
│                                                                 │
│  ┌────────────────────────────────────────────────┐            │
│  │           Multi-Node GPU Cluster               │            │
│  │  Node 0         Node 1         Node 2          │            │
│  │  GPU 0-7        GPU 8-15       GPU 16-23       │            │
│  │  [Rank 0-7]     [Rank 8-15]    [Rank 16-23]    │            │
│  │       └───────────NCCL/InfiniBand──────────┘   │            │
│  └────────────────────────────────────────────────┘            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Part 1: Data Parallel Training (DDP) (8-10 hours)

Implement PyTorch Distributed Data Parallel for efficient multi-GPU training.

#### 1.1 DDP Trainer

Create `src/parallelism/ddp_trainer.py`:

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class DDPConfig:
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"  # or "tcp://..."
    world_size: int = 1  # Total number of processes
    rank: int = 0  # Current process rank
    local_rank: int = 0  # GPU ID on current node
    gradient_as_bucket_view: bool = True  # Memory optimization
    find_unused_parameters: bool = False  # Set True if not all params used
    broadcast_buffers: bool = True  # Sync buffers like BatchNorm running stats

class DDPTrainer:
    """
    Distributed Data Parallel trainer.

    How DDP works:
    1. Each GPU has a full copy of the model
    2. Each GPU processes different data batches
    3. Gradients are synchronized via allreduce after backward()
    4. All GPUs have identical model weights after each step
    """

    def __init__(self, config: DDPConfig):
        self.config = config
        self.setup_distributed()

    def setup_distributed(self):
        """
        TODO: Initialize distributed process group

        Environment variables (set by torchrun or SLURM):
        - MASTER_ADDR: IP of rank 0 node
        - MASTER_PORT: Port for communication
        - WORLD_SIZE: Total number of processes
        - RANK: Global rank (0 to world_size-1)
        - LOCAL_RANK: Rank on current node (0 to 7 for 8-GPU node)

        Initialize:
        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method
        )

        Set device:
        torch.cuda.set_device(self.config.local_rank)

        Verify setup:
        assert dist.is_initialized()
        assert dist.get_world_size() == self.config.world_size
        """
        pass

    def wrap_model(self, model: nn.Module) -> DDP:
        """
        TODO: Wrap model with DDP

        Steps:
        1. Move model to GPU: model = model.to(self.config.local_rank)
        2. Wrap with DDP:
           model = DDP(
               model,
               device_ids=[self.config.local_rank],
               output_device=self.config.local_rank,
               gradient_as_bucket_view=True,  # Memory optimization
               broadcast_buffers=True,  # Sync BatchNorm stats
               find_unused_parameters=False  # Faster if all params used
           )

        DDP optimizations:
        - Gradient bucketing: Groups gradients into buckets for efficient allreduce
        - Overlapping: Overlaps allreduce with backward pass computation
        - gradient_as_bucket_view: Avoids gradient copy, saves memory

        Return DDP-wrapped model
        """
        pass

    def create_distributed_sampler(
        self,
        dataset,
        shuffle: bool = True,
        seed: int = 0
    ) -> DistributedSampler:
        """
        TODO: Create DistributedSampler for data loading

        DistributedSampler ensures:
        - Each GPU gets different data samples (no overlap)
        - All GPUs process same number of batches per epoch
        - Shuffling is consistent across all GPUs (same seed)

        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle,
            seed=seed
        )

        Usage with DataLoader:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # Don't use shuffle=True with sampler
            num_workers=4,
            pin_memory=True
        )

        Important: Call sampler.set_epoch(epoch) before each epoch
        to ensure different shuffling each epoch
        """
        pass

    def train_step(
        self,
        model: DDP,
        batch: Dict,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        TODO: Single training step with DDP

        optimizer.zero_grad()

        outputs = model(batch['input'])  # Forward pass
        loss = criterion(outputs, batch['target'])

        loss.backward()  # Backward pass
        # DDP automatically synchronizes gradients via allreduce here

        optimizer.step()  # Update weights (same on all GPUs)

        return loss.item()
        """
        pass

    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        TODO: Average metrics across all GPUs

        For each metric (loss, accuracy, etc.):
        1. Convert to tensor
        2. AllReduce with SUM operation
        3. Divide by world_size to get average

        tensor = torch.tensor(value).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        avg_value = tensor.item() / self.config.world_size

        This ensures all GPUs report consistent metrics
        """
        pass

    def is_main_process(self) -> bool:
        """Return True if this is rank 0 (main process)."""
        return self.config.rank == 0

    def barrier(self):
        """
        TODO: Synchronize all processes

        dist.barrier()

        Blocks until all processes reach this point.
        Useful for coordinating checkpointing, logging, etc.
        """
        pass

    def cleanup(self):
        """
        TODO: Clean up distributed training

        dist.destroy_process_group()

        Call this at end of training
        """
        pass

class DDPCheckpointer:
    """Handle checkpointing in distributed training."""

    def save_checkpoint(
        self,
        model: DDP,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        path: str,
        rank: int
    ):
        """
        TODO: Save checkpoint (only on rank 0)

        if rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # .module to unwrap DDP
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, path)

        Important: Use model.module.state_dict(), not model.state_dict()
        DDP wraps the model, .module accesses the underlying model
        """
        pass

    def load_checkpoint(
        self,
        model: DDP,
        optimizer: torch.optim.Optimizer,
        path: str
    ) -> int:
        """
        TODO: Load checkpoint (all ranks load same checkpoint)

        checkpoint = torch.load(path, map_location=f'cuda:{local_rank}')

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch']
        """
        pass
```

#### 1.2 Launch Script

Create `scripts/launch_ddp.py`:

```python
#!/usr/bin/env python3
"""
Launch distributed training with torchrun.

Single-node (8 GPUs):
    torchrun --nproc_per_node=8 scripts/train_ddp.py --config config.yaml

Multi-node (4 nodes × 8 GPUs = 32 GPUs):
    # On each node:
    torchrun \
        --nproc_per_node=8 \
        --nnodes=4 \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=29500 \
        scripts/train_ddp.py --config config.yaml
"""

import argparse
import torch
import torch.distributed as dist
from src.parallelism.ddp_trainer import DDPTrainer, DDPConfig

def main():
    # TODO: Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # TODO: Get distributed config from environment
    # torchrun sets these automatically
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    config = DDPConfig(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        local_rank=local_rank
    )

    # TODO: Initialize trainer and run training
    trainer = DDPTrainer(config)
    # ... training loop ...
    trainer.cleanup()

if __name__ == '__main__':
    main()
```

### Part 2: Model Parallelism (Tensor + Pipeline) (10-12 hours)

Implement model parallelism for models too large to fit on one GPU.

#### 2.1 Tensor Parallelism

Create `src/parallelism/tensor_parallel.py`:

```python
import torch
import torch.nn as nn
from typing import List

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-wise parallelism.

    Input: [batch, in_features]
    Weight split: [in_features, out_features] split along columns

    GPU 0: W[:, 0:out_features//2]
    GPU 1: W[:, out_features//2:out_features]

    Each GPU computes partial output, concatenate results.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        # TODO: Calculate output features for this GPU
        assert out_features % world_size == 0
        self.out_features_per_partition = out_features // world_size

        # TODO: Create partial weight matrix
        self.weight = nn.Parameter(
            torch.empty(in_features, self.out_features_per_partition)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition)
            )
        else:
            self.bias = None

        # TODO: Initialize weights
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Forward pass with column parallelism

        1. Each GPU computes partial output:
           output_partial = x @ self.weight + self.bias
           Shape: [batch, out_features_per_partition]

        2. AllGather to combine results from all GPUs:
           output_list = [torch.zeros_like(output_partial) for _ in range(world_size)]
           dist.all_gather(output_list, output_partial)
           output = torch.cat(output_list, dim=-1)
           Shape: [batch, out_features]

        Return combined output
        """
        pass

class RowParallelLinear(nn.Module):
    """
    Linear layer with row-wise parallelism.

    Input split: [batch, in_features] split along features
    Weight split: [in_features, out_features] split along rows

    GPU 0: W[0:in_features//2, :]
    GPU 1: W[in_features//2:in_features, :]

    Each GPU computes partial output, sum results via AllReduce.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True
    ):
        super().__init__()
        # TODO: Similar to ColumnParallelLinear but split rows
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Forward pass with row parallelism

        1. Input already split across GPUs (from previous layer)
           x shape: [batch, in_features_per_partition]

        2. Each GPU computes partial output:
           output_partial = x @ self.weight
           Shape: [batch, out_features]

        3. AllReduce to sum results from all GPUs:
           dist.all_reduce(output_partial, op=dist.ReduceOp.SUM)

        4. Add bias (only on one GPU to avoid double counting):
           if self.rank == 0 and self.bias is not None:
               output_partial += self.bias

        Return combined output
        """
        pass

class TensorParallelTransformer(nn.Module):
    """
    Transformer block with tensor parallelism.

    Split attention and FFN across GPUs.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        world_size: int,
        rank: int
    ):
        super().__init__()

        # TODO: Parallelize attention
        # Q, K, V projections: ColumnParallel (split heads across GPUs)
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, world_size, rank
        )

        # Attention output: RowParallel (combine heads)
        self.out_proj = RowParallelLinear(
            hidden_size, hidden_size, world_size, rank
        )

        # TODO: Parallelize FFN
        # FFN up: ColumnParallel (split intermediate dimension)
        self.ffn_up = ColumnParallelLinear(
            hidden_size, 4 * hidden_size, world_size, rank
        )

        # FFN down: RowParallel (combine results)
        self.ffn_down = RowParallelLinear(
            4 * hidden_size, hidden_size, world_size, rank
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Forward pass through transformer block

        1. Attention:
           qkv = self.qkv_proj(x)  # Column parallel
           # ... attention computation (each GPU has subset of heads)
           attn_out = self.out_proj(attn)  # Row parallel

        2. FFN:
           ffn = self.ffn_up(x + attn_out)  # Column parallel
           ffn = gelu(ffn)
           ffn_out = self.ffn_down(ffn)  # Row parallel

        Return x + attn_out + ffn_out
        """
        pass
```

#### 2.2 Pipeline Parallelism

Create `src/parallelism/pipeline_parallel.py`:

```python
import torch
import torch.nn as nn
from typing import List
from collections import deque

class PipelineStage(nn.Module):
    """One stage of the pipeline (runs on one GPU)."""

    def __init__(self, layers: nn.ModuleList, stage_id: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class PipelineParallel:
    """
    Pipeline parallelism with GPipe schedule.

    Split model into stages, each stage on different GPU.
    Split batch into micro-batches for pipeline efficiency.

    Example with 4 stages, 4 micro-batches:

    Time   GPU0    GPU1    GPU2    GPU3
    1      F0      -       -       -
    2      F1      F0      -       -
    3      F2      F1      F0      -
    4      F3      F2      F1      F0
    5      B3      F3      F2      F1
    6      -       B3      F3      F2
    7      -       -       B3      F3
    8      -       -       -       B3

    F = Forward, B = Backward
    Number = micro-batch ID
    """

    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        num_microbatches: int
    ):
        self.model = model
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches

        # TODO: Split model into stages
        self.stages = self._split_model_into_stages(model, num_stages)

    def _split_model_into_stages(
        self,
        model: nn.Module,
        num_stages: int
    ) -> List[PipelineStage]:
        """
        TODO: Split model layers evenly across stages

        Example for 12-layer transformer:
        - Stage 0 (GPU 0): Layers 0-2
        - Stage 1 (GPU 1): Layers 3-5
        - Stage 2 (GPU 2): Layers 6-8
        - Stage 3 (GPU 3): Layers 9-11

        Return list of PipelineStage objects
        """
        pass

    def forward_backward(
        self,
        batch: torch.Tensor,
        target: torch.Tensor,
        criterion: nn.Module
    ) -> torch.Tensor:
        """
        TODO: GPipe forward-backward pass

        1. Split batch into micro-batches:
           micro_batches = batch.chunk(self.num_microbatches, dim=0)

        2. Forward pass (fill pipeline):
           For each micro-batch:
               - Send to first stage
               - Each stage processes and sends to next stage
               - Store activations for backward pass

        3. Backward pass (drain pipeline):
           For each micro-batch (reverse order):
               - Compute gradients at last stage
               - Backward through each stage
               - Accumulate gradients

        4. Synchronize gradients across all stages

        Challenge: Manage activation memory
        - Can't store all activations (OOM for large models)
        - Solution: Recompute activations during backward (activation checkpointing)

        Return total loss
        """
        pass

    def get_pipeline_schedule(self) -> List[tuple]:
        """
        TODO: Generate GPipe schedule

        Return list of (time_step, stage_id, operation, microbatch_id)

        Example for 4 stages, 4 micro-batches:
        [
            (0, 0, 'F', 0),  # Stage 0 forward micro-batch 0
            (1, 0, 'F', 1),  # Stage 0 forward micro-batch 1
            (1, 1, 'F', 0),  # Stage 1 forward micro-batch 0
            ...
        ]

        This schedule maximizes GPU utilization (minimizes bubbles)
        """
        pass
```

### Part 3: ZeRO Optimizer (DeepSpeed/FSDP) (8-10 hours)

Implement Zero Redundancy Optimizer to reduce memory usage.

#### 3.1 ZeRO Implementation

Create `src/parallelism/zero_optimizer.py`:

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

class ZeROConfig:
    """
    ZeRO optimization stages:

    ZeRO-1: Shard optimizer states only (4× memory reduction)
    ZeRO-2: Shard optimizer states + gradients (8× memory reduction)
    ZeRO-3: Shard optimizer states + gradients + parameters (N× memory reduction, N = num_GPUs)

    Example for 13B parameter model (52GB in FP32):
    - Single GPU: 52GB parameters + 52GB gradients + 104GB optimizer states = 208GB ❌ Doesn't fit
    - ZeRO-1: 52GB + 52GB + (104GB / 8 GPUs) = 117GB ❌ Still doesn't fit
    - ZeRO-2: 52GB + (52GB / 8) + (104GB / 8) = 71.5GB ❌ Still doesn't fit
    - ZeRO-3: (52GB / 8) + (52GB / 8) + (104GB / 8) = 26GB ✅ Fits on A100-80GB
    """

    def __init__(
        self,
        stage: int = 3,  # ZeRO stage (1, 2, or 3)
        offload_optimizer: bool = False,  # Offload optimizer to CPU
        offload_params: bool = False,  # Offload params to CPU (ZeRO-Infinity)
        overlap_comm: bool = True,  # Overlap communication with computation
    ):
        self.stage = stage
        self.offload_optimizer = offload_optimizer
        self.offload_params = offload_params
        self.overlap_comm = overlap_comm

class FSDPTrainer:
    """
    Fully Sharded Data Parallel (PyTorch implementation of ZeRO-3).

    How FSDP works:
    1. Shard model parameters across all GPUs
    2. Before forward: AllGather parameters for needed layers
    3. After forward: Discard parameters (free memory)
    4. Before backward: AllGather parameters again
    5. After backward: Reduce-scatter gradients, discard parameters
    """

    def __init__(self, zero_config: ZeROConfig):
        self.config = zero_config

    def wrap_model_fsdp(self, model: nn.Module) -> FSDP:
        """
        TODO: Wrap model with FSDP

        # Auto-wrap policy: Wrap layers >100M parameters
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=100_000_000  # 100M parameters
        )

        # Mixed precision config
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,  # Store params in FP16
            reduce_dtype=torch.float16,  # Reduce gradients in FP16
            buffer_dtype=torch.float32,  # Keep buffers (BatchNorm) in FP32
        )

        # CPU offload config (optional, for very large models)
        cpu_offload = CPUOffload(offload_params=self.config.offload_params)

        # Wrap model
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=cpu_offload if self.config.offload_params else None,
            sharding_strategy=...,  # FULL_SHARD for ZeRO-3
            backward_prefetch=...,  # BACKWARD_PRE for overlapping
            device_id=torch.cuda.current_device(),
        )

        Return FSDP-wrapped model
        """
        pass

    def train_step(
        self,
        model: FSDP,
        batch: Dict,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        TODO: Training step with FSDP

        Same as DDP, but FSDP handles parameter sharding automatically:

        optimizer.zero_grad()
        outputs = model(batch['input'])  # FSDP allgathers params as needed
        loss = criterion(outputs, batch['target'])
        loss.backward()  # FSDP reduce-scatters gradients
        optimizer.step()  # FSDP updates sharded params

        return loss.item()
        """
        pass

    def estimate_memory_savings(
        self,
        num_parameters: int,
        num_gpus: int,
        stage: int
    ) -> Dict[str, float]:
        """
        TODO: Estimate memory savings from ZeRO

        Assume FP16 training (2 bytes per parameter):
        - Parameters: num_parameters × 2 bytes
        - Gradients: num_parameters × 2 bytes
        - Optimizer states (Adam): num_parameters × 12 bytes
          (FP32 copy: 4 bytes, momentum: 4 bytes, variance: 4 bytes)

        Memory per GPU:
        - Baseline (DDP): params + grads + optimizer = 16 bytes/param
        - ZeRO-1: params + grads + (optimizer / num_gpus)
        - ZeRO-2: params + (grads + optimizer) / num_gpus
        - ZeRO-3: (params + grads + optimizer) / num_gpus

        Return:
        {
            'baseline_memory_gb': ...,
            'zero_memory_gb': ...,
            'memory_reduction_factor': ...,
            'max_model_size_with_zero': ...
        }
        """
        pass

class DeepSpeedIntegration:
    """
    Integration with Microsoft DeepSpeed (alternative to FSDP).

    DeepSpeed provides additional optimizations:
    - ZeRO-Offload: Offload optimizer to CPU
    - ZeRO-Infinity: Offload parameters to NVMe SSD
    - 1-bit Adam: Compressed communication for gradients
    """

    def create_deepspeed_config(
        self,
        zero_stage: int = 3,
        offload_optimizer: bool = False,
        gradient_clipping: float = 1.0,
        train_micro_batch_size: int = 16,
    ) -> Dict:
        """
        TODO: Generate DeepSpeed configuration

        config = {
            "train_micro_batch_size_per_gpu": train_micro_batch_size,
            "gradient_clipping": gradient_clipping,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,  # Dynamic loss scaling
                "initial_scale_power": 16,
            },
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if offload_optimizer else "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
            }
        }

        return config
        """
        pass

    def initialize_deepspeed(
        self,
        model: nn.Module,
        config: Dict
    ):
        """
        TODO: Initialize model with DeepSpeed

        import deepspeed

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=config,
            model_parameters=model.parameters()
        )

        return model_engine, optimizer
        """
        pass
```

### Part 4: Fault Tolerance & Checkpointing (6-8 hours)

Implement robust checkpointing and automatic recovery from failures.

#### 4.1 Fault Tolerant Checkpointing

Create `src/checkpointing/fault_tolerant.py`:

```python
import torch
import torch.distributed as dist
from pathlib import Path
import time
from typing import Optional
import os

class DistributedCheckpointer:
    """
    Fault-tolerant checkpointing for distributed training.

    Requirements:
    1. Atomic writes (no corrupted checkpoints)
    2. Fast checkpointing (<2 minutes for 13B model)
    3. Automatic cleanup (keep last N checkpoints)
    4. Resume from latest checkpoint automatically
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        checkpoint_interval_minutes: int = 30
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.checkpoint_interval_seconds = checkpoint_interval_minutes * 60
        self.last_checkpoint_time = time.time()

    def should_checkpoint(self) -> bool:
        """
        TODO: Determine if it's time to checkpoint

        Check if checkpoint_interval has elapsed since last checkpoint

        return (time.time() - self.last_checkpoint_time) >= self.checkpoint_interval_seconds
        """
        pass

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        step: int,
        rank: int,
        is_fsdp: bool = False
    ):
        """
        TODO: Save checkpoint atomically

        Strategy for atomic writes:
        1. Write to temporary file: checkpoint_epoch{epoch}_step{step}.tmp
        2. Once complete, rename to: checkpoint_epoch{epoch}_step{step}.pt
        3. Atomic rename ensures no partial checkpoints

        For FSDP models:
        - Use FSDP.state_dict() with state_dict_type=FULL_STATE_DICT (only rank 0)
        - Or use SHARDED_STATE_DICT (all ranks save their shard)

        Only rank 0 writes (unless using sharded checkpoints):
        if rank == 0 or is_fsdp:
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'timestamp': time.time(),
            }

            temp_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.tmp"
            final_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"

            torch.save(checkpoint, temp_path)
            temp_path.rename(final_path)  # Atomic

        Cleanup old checkpoints after saving new one
        """
        pass

    def load_latest_checkpoint(
        self,
        model,
        optimizer,
        rank: int
    ) -> Optional[Dict]:
        """
        TODO: Load most recent checkpoint

        1. List all checkpoint files in directory
        2. Sort by modification time (most recent first)
        3. Try loading latest:
           try:
               checkpoint = torch.load(latest_path)
               model.load_state_dict(checkpoint['model_state_dict'])
               optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
               return checkpoint
           except Exception:
               # Corrupted checkpoint, try next one
               continue

        4. If all checkpoints fail, return None (start from scratch)
        """
        pass

    def cleanup_old_checkpoints(self):
        """
        TODO: Remove old checkpoints, keep last N

        1. List all checkpoint files
        2. Sort by modification time
        3. Delete all except last keep_last_n checkpoints
        """
        pass

class ElasticTraining:
    """
    Elastic training: Automatically recover from node failures.

    Uses torch.distributed.elastic (torchelastic) for:
    - Automatic restart on failure
    - Dynamic scaling (add/remove nodes during training)
    """

    def __init__(self, checkpointer: DistributedCheckpointer):
        self.checkpointer = checkpointer

    def train_with_fault_tolerance(
        self,
        model,
        train_loader,
        num_epochs: int,
        rank: int
    ):
        """
        TODO: Training loop with automatic recovery

        # Try to resume from checkpoint
        checkpoint = self.checkpointer.load_latest_checkpoint(model, optimizer, rank)
        start_epoch = checkpoint['epoch'] + 1 if checkpoint else 0

        for epoch in range(start_epoch, num_epochs):
            try:
                for step, batch in enumerate(train_loader):
                    # Training step
                    loss = train_step(model, batch, optimizer, criterion)

                    # Periodic checkpointing
                    if self.checkpointer.should_checkpoint():
                        self.checkpointer.save_checkpoint(
                            model, optimizer, epoch, step, rank
                        )

            except Exception as e:
                # Log error, checkpoint will be restored on restart
                print(f"Training failed: {e}")
                raise  # torchelastic will restart

        """
        pass

    def launch_elastic_training(self, config: Dict):
        """
        TODO: Launch training with torchrun (torchelastic)

        torchrun provides:
        - Automatic restart on failure (--max_restarts=3)
        - Failure detection via heartbeats
        - Rendezvous for nodes to find each other

        Command:
        torchrun \
            --nnodes=4 \  # 4 nodes
            --nproc_per_node=8 \  # 8 GPUs per node
            --max_restarts=3 \  # Restart up to 3 times
            --rdzv_backend=c10d \  # Rendezvous backend
            --rdzv_endpoint=$MASTER_ADDR:29500 \
            train_elastic.py --config config.yaml

        If node fails:
        1. Other nodes detect failure via timeout
        2. torchrun restarts training on remaining nodes
        3. Load latest checkpoint and continue
        """
        pass
```

### Part 5: Observability & Monitoring (4-6 hours)

Build monitoring for distributed training.

#### 5.1 Distributed Training Monitor

Create `src/monitoring/distributed_monitor.py`:

```python
import torch
import torch.distributed as dist
from prometheus_client import Gauge, Counter, Histogram
import time
from typing import Dict

class DistributedTrainingMonitor:
    """Monitor distributed training performance and health."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

        # Prometheus metrics
        self.samples_per_second = Gauge(
            'training_samples_per_second',
            'Training throughput',
            ['rank']
        )

        self.step_time = Histogram(
            'training_step_time_seconds',
            'Time per training step',
            ['rank']
        )

        self.gpu_memory = Gauge(
            'gpu_memory_allocated_bytes',
            'GPU memory allocated',
            ['rank', 'device']
        )

        self.communication_time = Histogram(
            'communication_time_seconds',
            'Time spent in collective communication',
            ['rank', 'operation']
        )

    def log_step_metrics(
        self,
        step: int,
        loss: float,
        step_time: float,
        batch_size: int
    ):
        """
        TODO: Log metrics for current step

        1. Calculate throughput:
           samples_per_sec = batch_size / step_time

        2. Log to Prometheus:
           self.samples_per_second.labels(rank=self.rank).set(samples_per_sec)
           self.step_time.labels(rank=self.rank).observe(step_time)

        3. Log GPU memory:
           memory_allocated = torch.cuda.memory_allocated()
           self.gpu_memory.labels(rank=self.rank, device=0).set(memory_allocated)

        4. All-reduce metrics to rank 0 for logging:
           if self.rank == 0:
               avg_loss = self._all_reduce_metric(loss)
               print(f"Step {step}: loss={avg_loss:.4f}, throughput={samples_per_sec:.0f} samples/sec")
        """
        pass

    def measure_communication_time(
        self,
        operation: str,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        TODO: Measure time for collective communication

        start_time = time.time()

        if operation == 'allreduce':
            dist.all_reduce(tensor)
        elif operation == 'allgather':
            dist.all_gather(...)
        elif operation == 'broadcast':
            dist.broadcast(tensor, src=0)

        comm_time = time.time() - start_time
        self.communication_time.labels(rank=self.rank, operation=operation).observe(comm_time)

        return tensor
        """
        pass

    def check_gradient_sync(self, model) -> Dict:
        """
        TODO: Verify all GPUs have identical gradients after sync

        For each parameter:
        1. Compute hash of gradient on each GPU
        2. AllGather hashes from all GPUs
        3. Verify all hashes match

        If mismatch detected:
        - Log warning
        - Identify which parameters differ
        - This indicates a bug in gradient synchronization

        return {'all_synced': True/False, 'mismatched_params': [...]}
        """
        pass

    def measure_scaling_efficiency(
        self,
        single_gpu_throughput: float,
        current_throughput: float
    ) -> Dict:
        """
        TODO: Calculate scaling efficiency

        speedup = current_throughput / single_gpu_throughput
        ideal_speedup = self.world_size
        efficiency = (speedup / ideal_speedup) * 100

        return {
            'speedup': speedup,
            'ideal_speedup': ideal_speedput,
            'efficiency_percent': efficiency
        }
        """
        pass
```

## Acceptance Criteria

### Functional Requirements

- [ ] DDP trainer supports multi-GPU and multi-node training
- [ ] Tensor parallelism splits transformer across GPUs
- [ ] Pipeline parallelism implements GPipe schedule
- [ ] FSDP/ZeRO enables training 13B model on 32 GPUs
- [ ] Checkpointing saves/loads state correctly
- [ ] Elastic training recovers from node failures automatically
- [ ] Monitoring tracks throughput, memory, scaling efficiency

### Performance Requirements

- [ ] 13B parameter model trains successfully (doesn't OOM)
- [ ] Training time <5 days (vs 3 weeks baseline for 7B model)
- [ ] Scaling efficiency >90% for 2-8 GPUs
- [ ] Scaling efficiency >80% for 32 GPUs
- [ ] Checkpoint save time <2 minutes for 13B model
- [ ] Recovery from failure <5 minutes

### Code Quality

- [ ] All distributed code has error handling for common failures
- [ ] Comprehensive logging (rank, step, loss, throughput)
- [ ] Unit tests for each parallelism strategy
- [ ] Integration test for full distributed training
- [ ] Documentation with architecture diagrams

## Testing Strategy

### Unit Tests

```python
# tests/test_ddp.py
def test_ddp_gradient_sync():
    """Verify gradients are synchronized across all GPUs."""
    # Requires multi-GPU environment
    pass

# tests/test_fsdp.py
def test_fsdp_memory_reduction():
    """Verify FSDP reduces memory usage."""
    # Compare memory with DDP vs FSDP
    pass
```

### Integration Tests

```python
# tests/test_distributed_training.py
def test_full_training_pipeline():
    """Test complete distributed training workflow."""
    # 1. Initialize distributed
    # 2. Create model with FSDP
    # 3. Train for 10 steps
    # 4. Checkpoint
    # 5. Resume from checkpoint
    # 6. Verify loss continues from checkpoint
    pass
```

## Deliverables

1. **Source Code** (`src/`):
   - `parallelism/ddp_trainer.py` - DDP implementation
   - `parallelism/tensor_parallel.py` - Tensor parallelism
   - `parallelism/pipeline_parallel.py` - Pipeline parallelism
   - `parallelism/zero_optimizer.py` - FSDP/DeepSpeed integration
   - `checkpointing/fault_tolerant.py` - Fault-tolerant checkpointing
   - `monitoring/distributed_monitor.py` - Monitoring and observability

2. **Scripts** (`scripts/`):
   - `launch_ddp.py` - Launch DDP training
   - `launch_fsdp.py` - Launch FSDP training
   - `benchmark_scaling.py` - Measure scaling efficiency

3. **Documentation** (`docs/`):
   - `DISTRIBUTED_TRAINING_GUIDE.md` - Complete guide
   - `PARALLELISM_STRATEGIES.md` - When to use each strategy
   - `TROUBLESHOOTING.md` - Common issues and solutions

4. **Configuration Examples** (`configs/`):
   - `ddp_config.yaml` - DDP configuration
   - `fsdp_config.yaml` - FSDP configuration
   - `deepspeed_config.json` - DeepSpeed configuration

## Bonus Challenges

1. **3D Parallelism (Data + Tensor + Pipeline)** (+10 hours):
   - Combine all three parallelism strategies
   - Train 175B parameter model (GPT-3 scale)
   - Optimize communication patterns

2. **Gradient Compression** (+6 hours):
   - Implement PowerSGD or 1-bit Adam
   - Reduce communication volume by 10×
   - Measure impact on convergence

3. **Heterogeneous Training** (+8 hours):
   - Mix GPU types (A100 + V100)
   - Dynamic load balancing based on GPU speed
   - Handle stragglers efficiently

## Resources

### Official Documentation

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [Microsoft DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM (NVIDIA)](https://github.com/NVIDIA/Megatron-LM)

### Research Papers

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

## Submission

Submit your implementation via Git:

```bash
git add .
git commit -m "Complete Exercise 06: Distributed GPU Training"
git push origin exercise-06-distributed-gpu-training
```

Ensure your submission includes:
- Complete implementation of all parallelism strategies
- Checkpointing and fault tolerance
- Monitoring and observability
- Tests and documentation
- Benchmark results showing scaling efficiency

---

**Estimated Time Breakdown**:
- Part 1 (DDP): 8-10 hours
- Part 2 (Model Parallelism): 10-12 hours
- Part 3 (ZeRO/FSDP): 8-10 hours
- Part 4 (Fault Tolerance): 6-8 hours
- Part 5 (Monitoring): 4-6 hours
- **Total**: 36-44 hours
