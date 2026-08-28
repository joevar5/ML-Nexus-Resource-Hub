# Lesson 05: LLM Fine-Tuning Infrastructure

## Table of Contents
1. [Introduction](#introduction)
2. [Fine-Tuning vs RAG: Making the Right Choice](#fine-tuning-vs-rag-making-the-right-choice)
3. [Fine-Tuning Fundamentals](#fine-tuning-fundamentals)
4. [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
5. [LoRA: Low-Rank Adaptation](#lora-low-rank-adaptation)
6. [QLoRA: Quantized LoRA](#qlora-quantized-lora)
7. [Infrastructure Requirements](#infrastructure-requirements)
8. [Distributed Training with DeepSpeed](#distributed-training-with-deepspeed)
9. [Distributed Training with FSDP](#distributed-training-with-fsdp)
10. [Hugging Face Ecosystem](#hugging-face-ecosystem)
11. [Complete Fine-Tuning Workflows](#complete-fine-tuning-workflows)
12. [Training Data Management](#training-data-management)
13. [Experiment Tracking](#experiment-tracking)
14. [Hyperparameter Tuning](#hyperparameter-tuning)
15. [Cost Optimization](#cost-optimization)
16. [Production Fine-Tuning Pipeline](#production-fine-tuning-pipeline)
17. [Summary](#summary)

## Introduction

Fine-tuning Large Language Models allows you to adapt pre-trained models to specific domains, tasks, or use cases. This lesson covers the complete infrastructure required to fine-tune LLMs at scale, from understanding when to fine-tune versus using RAG, to implementing production-ready fine-tuning pipelines.

### Learning Objectives

After completing this lesson, you will be able to:
- Decide when to use fine-tuning vs RAG for your use case
- Understand and implement Parameter-Efficient Fine-Tuning (PEFT) techniques
- Implement LoRA and QLoRA for memory-efficient fine-tuning
- Set up GPU clusters for distributed training
- Use DeepSpeed and FSDP for large-scale training
- Manage training data at scale
- Track experiments with MLflow and Weights & Biases
- Build production-ready fine-tuning pipelines
- Optimize costs for fine-tuning workloads

## Fine-Tuning vs RAG: Making the Right Choice

### Decision Framework

The choice between fine-tuning and RAG depends on your specific requirements:

```python
# Decision framework implementation
class FineTuningVsRAGDecision:
    """
    Framework to help decide between fine-tuning and RAG
    """
    def __init__(self, use_case_params: dict):
        self.params = use_case_params

    def analyze(self):
        """Analyze use case and recommend approach"""
        score_ft = 0  # Fine-tuning score
        score_rag = 0  # RAG score

        # Factor 1: Data characteristics
        if self.params.get('data_volume') == 'large':  # >10k examples
            score_ft += 2
        else:
            score_rag += 2

        # Factor 2: Knowledge type
        if self.params.get('knowledge_type') == 'behavioral':
            score_ft += 3  # Style, tone, format
        else:
            score_rag += 3  # Factual knowledge

        # Factor 3: Update frequency
        if self.params.get('update_frequency') == 'frequent':
            score_rag += 3  # Easy to update vector DB
        else:
            score_ft += 1

        # Factor 4: Budget
        if self.params.get('budget') == 'limited':
            score_rag += 2  # Lower upfront costs
        else:
            score_ft += 1

        # Factor 5: Latency requirements
        if self.params.get('latency_sensitive'):
            score_ft += 1  # No retrieval overhead
        else:
            score_rag += 0

        # Factor 6: Accuracy requirements
        if self.params.get('accuracy_critical'):
            score_rag += 2  # Grounded in documents

        recommendation = "Fine-Tuning" if score_ft > score_rag else "RAG"

        return {
            'recommendation': recommendation,
            'fine_tuning_score': score_ft,
            'rag_score': score_rag,
            'confidence': abs(score_ft - score_rag) / max(score_ft, score_rag),
            'hybrid_feasible': abs(score_ft - score_rag) <= 2
        }

# Example usage
decision = FineTuningVsRAGDecision({
    'data_volume': 'large',
    'knowledge_type': 'behavioral',
    'update_frequency': 'rare',
    'budget': 'medium',
    'latency_sensitive': True,
    'accuracy_critical': False
})

print(decision.analyze())
# Output: {'recommendation': 'Fine-Tuning', 'fine_tuning_score': 7, 'rag_score': 2, ...}
```

### When to Choose Fine-Tuning

**Use fine-tuning when you need to:**

1. **Adapt Model Behavior**
   - Change writing style or tone
   - Teach specific output formats
   - Improve instruction following
   - Adapt to domain-specific language

2. **Improve Task Performance**
   - Increase accuracy on specific tasks
   - Reduce hallucinations for your domain
   - Learn complex reasoning patterns
   - Optimize for specific metrics

3. **Reduce Inference Costs**
   - Use smaller fine-tuned model instead of larger base model
   - Eliminate retrieval overhead
   - Reduce prompt length requirements

**Examples:**
- Medical diagnosis chatbot (domain adaptation)
- Code generation for specific framework
- Legal document drafting with firm's style
- Customer service with brand voice

### When to Choose RAG

**Use RAG when you need to:**

1. **Access Dynamic Knowledge**
   - Frequently updated information
   - Real-time data integration
   - Document-based Q&A
   - Citation requirements

2. **Work with Limited Training Data**
   - Few domain examples (<1000)
   - Rapidly changing requirements
   - Multiple knowledge sources

3. **Maintain Factual Accuracy**
   - Grounded responses required
   - Auditable information sources
   - Compliance requirements

**Examples:**
- Company knowledge base Q&A
- Product documentation assistant
- Legal research tool
- News analysis system

### Hybrid Approach

Often, the best solution combines both:

```python
# Hybrid approach example
class HybridLLMSystem:
    """
    Combines fine-tuned model with RAG for optimal results
    """
    def __init__(self, fine_tuned_model_path: str, vector_db_config: dict):
        from vllm import LLM
        from qdrant_client import QdrantClient

        # Fine-tuned model for behavior/style
        self.llm = LLM(model=fine_tuned_model_path)

        # Vector DB for factual knowledge
        self.vector_db = QdrantClient(**vector_db_config)

    def generate_response(self, query: str):
        """
        Generate response using hybrid approach
        """
        # 1. Retrieve relevant context (RAG)
        context_docs = self.retrieve_context(query, top_k=3)

        # 2. Build prompt with context
        prompt = self.build_prompt(query, context_docs)

        # 3. Generate with fine-tuned model
        response = self.llm.generate([prompt])

        return {
            'response': response[0].outputs[0].text,
            'sources': context_docs
        }

    def retrieve_context(self, query: str, top_k: int = 3):
        """Retrieve relevant documents"""
        # Implementation details...
        pass

    def build_prompt(self, query: str, context: list):
        """Build prompt with retrieved context"""
        # Fine-tuned model knows how to use context effectively
        return f"""Context: {context}

Query: {query}

Answer:"""
```

## Fine-Tuning Fundamentals

### What is Fine-Tuning?

Fine-tuning is the process of further training a pre-trained model on a specific dataset to adapt it to a particular task or domain.

```python
# Conceptual flow of fine-tuning
"""
Base Model (Pre-trained on broad data)
         ↓
    Fine-Tuning (Train on specific data)
         ↓
Fine-Tuned Model (Optimized for your task)
"""
```

### Types of Fine-Tuning

#### 1. Full Fine-Tuning

Update all parameters of the model.

**Pros:**
- Maximum adaptation capability
- Best performance potential

**Cons:**
- Requires massive GPU memory
- Expensive and time-consuming
- Risk of catastrophic forgetting

```python
# Full fine-tuning memory requirements
def full_finetuning_memory(num_parameters_billions: float):
    """
    Calculate GPU memory needed for full fine-tuning
    """
    # Model weights (FP16)
    model_memory_gb = num_parameters_billions * 2

    # Gradients (same size as model)
    gradient_memory_gb = num_parameters_billions * 2

    # Optimizer states (AdamW needs 2x model size)
    optimizer_memory_gb = num_parameters_billions * 4

    # Activations (rough estimate)
    activation_memory_gb = num_parameters_billions * 2

    total = model_memory_gb + gradient_memory_gb + optimizer_memory_gb + activation_memory_gb

    return {
        'model_gb': model_memory_gb,
        'gradients_gb': gradient_memory_gb,
        'optimizer_gb': optimizer_memory_gb,
        'activations_gb': activation_memory_gb,
        'total_gb': total,
        'recommended_gpu': 'A100-80GB' if total <= 80 else 'Multi-GPU Required'
    }

# Example: Llama 2 7B
print("7B Model Full Fine-Tuning:")
print(full_finetuning_memory(7))
# Requires ~70GB GPU memory!

# Example: Llama 2 70B
print("\n70B Model Full Fine-Tuning:")
print(full_finetuning_memory(70))
# Requires ~700GB - needs distributed training!
```

#### 2. Parameter-Efficient Fine-Tuning (PEFT)

Update only a small subset of parameters.

**Pros:**
- Much lower memory requirements
- Faster training
- Can train on single GPU
- Less risk of overfitting

**Cons:**
- Slightly lower performance ceiling
- May not adapt as deeply to domain

## Parameter-Efficient Fine-Tuning (PEFT)

### PEFT Methods Overview

```python
# Comparison of PEFT methods
peft_methods_comparison = {
    'LoRA': {
        'trainable_params': '0.1-1%',
        'memory_reduction': '~3x',
        'performance': '95-99% of full FT',
        'best_for': 'General purpose, production use',
        'complexity': 'Low'
    },
    'QLoRA': {
        'trainable_params': '0.1-1%',
        'memory_reduction': '~4x',
        'performance': '90-98% of full FT',
        'best_for': 'Limited GPU memory',
        'complexity': 'Low'
    },
    'Prefix Tuning': {
        'trainable_params': '0.1-0.5%',
        'memory_reduction': '~2x',
        'performance': '85-95% of full FT',
        'best_for': 'Task-specific adaptation',
        'complexity': 'Medium'
    },
    'Adapter Layers': {
        'trainable_params': '1-3%',
        'memory_reduction': '~2.5x',
        'performance': '90-97% of full FT',
        'best_for': 'Multi-task learning',
        'complexity': 'Medium'
    },
    'IA3': {
        'trainable_params': '0.01-0.1%',
        'memory_reduction': '~4x',
        'performance': '85-93% of full FT',
        'best_for': 'Minimal adaptation',
        'complexity': 'Low'
    }
}
```

### Why PEFT Works

The key insight: **most of the knowledge is already in the pre-trained model**. We only need to adapt a small part to our specific task.

```python
# Mathematical intuition for PEFT
"""
Full fine-tuning updates:
W_new = W_old + ΔW

Where ΔW is often low-rank (most changes are in a low-dimensional subspace)

PEFT explicitly models this:
W_new = W_old + A × B (where A and B are low-rank matrices)

This reduces parameters from d×d to d×r + r×d (where r << d)
"""

def parameter_reduction_example():
    """
    Show parameter reduction from LoRA
    """
    d = 4096  # Hidden dimension
    r = 16    # Rank

    # Full fine-tuning
    full_params = d * d

    # LoRA
    lora_params = d * r + r * d

    reduction_factor = full_params / lora_params
    percentage = (lora_params / full_params) * 100

    print(f"Full parameters: {full_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Reduction: {reduction_factor:.1f}x")
    print(f"Trainable: {percentage:.2f}%")

parameter_reduction_example()
# Output:
# Full parameters: 16,777,216
# LoRA parameters: 131,072
# Reduction: 128.0x
# Trainable: 0.78%
```

## LoRA: Low-Rank Adaptation

### Mathematical Foundation

LoRA decomposes weight updates into low-rank matrices:

```
W' = W + ΔW = W + BA

Where:
- W ∈ ℝ^(d×k): Original weight matrix (frozen)
- B ∈ ℝ^(d×r): Low-rank matrix (trainable)
- A ∈ ℝ^(r×k): Low-rank matrix (trainable)
- r << min(d, k): Rank (typically 8, 16, 32)
```

### LoRA Implementation

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    LoRA layer implementation
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Scaling factor
        self.scaling = alpha / rank

        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, original_weight: torch.Tensor):
        """
        Forward pass combining original weight with LoRA adaptation

        Args:
            x: Input tensor
            original_weight: Original frozen weight matrix

        Returns:
            Output tensor
        """
        # Original forward pass (frozen)
        result = F.linear(x, original_weight)

        # LoRA adaptation
        lora_result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

        # Combine with scaling
        return result + lora_result * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation
    """
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16
    ):
        super().__init__()

        # Freeze original layer
        self.linear = linear_layer
        for param in self.linear.parameters():
            param.requires_grad = False

        # Add LoRA
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x: torch.Tensor):
        return self.lora(x, self.linear.weight)


# Example: Add LoRA to a model
def add_lora_to_model(model, target_modules, rank=16):
    """
    Add LoRA layers to specified modules in a model

    Args:
        model: Base model
        target_modules: List of module names to apply LoRA
        rank: LoRA rank
    """
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)

                # Replace with LoRA version
                lora_linear = LoRALinear(module, rank=rank)
                setattr(parent, child_name, lora_linear)

    return model


# Calculate trainable parameters
def count_trainable_parameters(model):
    """Count trainable vs total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {
        'trainable': trainable,
        'total': total,
        'percentage': 100 * trainable / total
    }
```

### Hugging Face PEFT Library

The production-ready way to use LoRA:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=16,                           # Rank
    lora_alpha=32,                  # Alpha (scaling factor)
    target_modules=[                # Which layers to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,              # Dropout rate
    bias="none",                    # Don't train biases
    task_type=TaskType.CAUSAL_LM    # Task type
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%

# The model is now ready for fine-tuning!
```

### LoRA Hyperparameters

```python
# Comprehensive guide to LoRA hyperparameters
lora_hyperparameters_guide = {
    'rank (r)': {
        'typical_values': [4, 8, 16, 32, 64],
        'effect': 'Higher rank = more capacity but more parameters',
        'recommendation': {
            'simple_tasks': 8,
            'general_purpose': 16,
            'complex_adaptation': 32,
            'domain_shift': 64
        },
        'memory_impact': 'Linear with rank'
    },
    'alpha': {
        'typical_values': [8, 16, 32, 64],
        'effect': 'Scaling factor for LoRA updates',
        'recommendation': 'Usually 2x the rank',
        'formula': 'scaling = alpha / rank'
    },
    'dropout': {
        'typical_values': [0.0, 0.05, 0.1],
        'effect': 'Regularization to prevent overfitting',
        'recommendation': {
            'large_dataset': 0.0,
            'medium_dataset': 0.05,
            'small_dataset': 0.1
        }
    },
    'target_modules': {
        'minimal': ['q_proj', 'v_proj'],
        'recommended': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'aggressive': ['all_linear_layers'],
        'effect': 'More modules = more adaptation but more parameters'
    }
}


# Example configuration function
def get_lora_config(
    task_complexity: str,
    dataset_size: str,
    available_memory_gb: int
):
    """
    Get recommended LoRA configuration
    """
    configs = {
        ('simple', 'small'): {'r': 8, 'alpha': 16, 'dropout': 0.1},
        ('simple', 'medium'): {'r': 8, 'alpha': 16, 'dropout': 0.05},
        ('simple', 'large'): {'r': 16, 'alpha': 32, 'dropout': 0.0},
        ('medium', 'small'): {'r': 16, 'alpha': 32, 'dropout': 0.1},
        ('medium', 'medium'): {'r': 16, 'alpha': 32, 'dropout': 0.05},
        ('medium', 'large'): {'r': 32, 'alpha': 64, 'dropout': 0.0},
        ('complex', 'small'): {'r': 32, 'alpha': 64, 'dropout': 0.1},
        ('complex', 'medium'): {'r': 32, 'alpha': 64, 'dropout': 0.05},
        ('complex', 'large'): {'r': 64, 'alpha': 128, 'dropout': 0.0},
    }

    config = configs.get((task_complexity, dataset_size), {'r': 16, 'alpha': 32, 'dropout': 0.05})

    # Adjust for memory constraints
    if available_memory_gb < 24 and config['r'] > 16:
        config['r'] = 16
        config['alpha'] = 32

    return LoraConfig(
        r=config['r'],
        lora_alpha=config['alpha'],
        lora_dropout=config['dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
```

## QLoRA: Quantized LoRA

### What is QLoRA?

QLoRA combines quantization with LoRA to enable fine-tuning on even more limited hardware.

**Key innovations:**
1. **4-bit Quantization**: Base model stored in 4-bit precision
2. **Double Quantization**: Quantize the quantization constants
3. **Paged Optimizers**: Handle memory spikes with CPU offloading
4. **NormalFloat4 (NF4)**: Better 4-bit data type for normally distributed weights

```python
# Memory comparison: Full vs LoRA vs QLoRA
def memory_comparison(model_size_b: float):
    """
    Compare memory requirements across methods
    """
    # Full fine-tuning (FP16)
    full_ft = model_size_b * 10  # Model + gradients + optimizer states + activations

    # LoRA (FP16 model, FP16 adapters)
    lora = model_size_b * 2 + (model_size_b * 0.01) * 4  # Model + adapter training overhead

    # QLoRA (4-bit model, FP16 adapters)
    qlora = model_size_b * 0.5 + (model_size_b * 0.01) * 4

    return {
        'full_ft_gb': round(full_ft, 2),
        'lora_gb': round(lora, 2),
        'qlora_gb': round(qlora, 2),
        'lora_reduction': f"{full_ft/lora:.1f}x",
        'qlora_reduction': f"{full_ft/qlora:.1f}x"
    }

# Example: 7B model
print("7B Model Memory Requirements:")
print(memory_comparison(7))
# Output:
# {
#   'full_ft_gb': 70.0,
#   'lora_gb': 14.28,
#   'qlora_gb': 3.78,
#   'lora_reduction': '4.9x',
#   'qlora_reduction': '18.5x'
# }

# 70B model can now fit on a single A100!
print("\n70B Model Memory Requirements:")
print(memory_comparison(70))
```

### QLoRA Implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit loading
    bnb_4bit_use_double_quant=True,         # Double quantization
    bnb_4bit_quant_type="nf4",              # NormalFloat4 data type
    bnb_4bit_compute_dtype=torch.bfloat16   # Compute dtype for stability
)

# Load model with quantization
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA (same as before)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to quantized model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# Now ready for fine-tuning with minimal memory!
```

### QLoRA Best Practices

```python
# QLoRA training best practices
qlora_best_practices = {
    'quantization_type': {
        'nf4': 'Best for normally distributed weights (recommended for LLMs)',
        'fp4': 'Alternative, slightly worse performance',
        'recommendation': 'Use nf4 for LLMs'
    },
    'compute_dtype': {
        'bfloat16': 'Best stability and performance (requires Ampere or newer)',
        'float16': 'Compatible with older GPUs, slightly less stable',
        'recommendation': 'bfloat16 if available, else float16'
    },
    'double_quantization': {
        'enabled': 'Saves ~0.4GB per 1B parameters',
        'disabled': 'Slightly faster but uses more memory',
        'recommendation': 'Enable for memory-constrained setups'
    },
    'gradient_checkpointing': {
        'purpose': 'Reduce activation memory at cost of 20% slower training',
        'recommendation': 'Enable for very large models or small GPUs'
    },
    'batch_size': {
        'consideration': 'Start small (1-4) and increase if memory allows',
        'use_gradient_accumulation': 'To simulate larger batch sizes'
    }
}


# Complete QLoRA training setup
def setup_qlora_training(
    model_name: str,
    output_dir: str,
    dataset,
    max_memory_gb: int = 24
):
    """
    Complete QLoRA training setup with memory optimization
    """
    from transformers import TrainingArguments, Trainer

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    # Training arguments optimized for memory
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1 if max_memory_gb < 24 else 4,
        gradient_accumulation_steps=16 if max_memory_gb < 24 else 4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True if max_memory_gb < 24 else False,
        optim="paged_adamw_8bit",  # QLoRA's paged optimizer
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine"
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    return trainer, model
```

## Infrastructure Requirements

### GPU Cluster for Fine-Tuning

```python
# GPU requirements calculator for fine-tuning
class FineTuningGPUCalculator:
    """
    Calculate GPU requirements for fine-tuning scenarios
    """
    def __init__(self):
        self.gpu_specs = {
            'T4': {'memory_gb': 16, 'cost_per_hour': 0.40},
            'A10G': {'memory_gb': 24, 'cost_per_hour': 1.20},
            'A100-40GB': {'memory_gb': 40, 'cost_per_hour': 3.50},
            'A100-80GB': {'memory_gb': 80, 'cost_per_hour': 5.00},
            'H100': {'memory_gb': 80, 'cost_per_hour': 8.50}
        }

    def calculate_requirements(
        self,
        model_size_b: float,
        method: str = 'qlora',
        batch_size: int = 4,
        gradient_accumulation: int = 1
    ):
        """
        Calculate GPU requirements for fine-tuning
        """
        # Memory calculations based on method
        if method == 'full':
            memory_needed = model_size_b * 10  # Conservative estimate
        elif method == 'lora':
            memory_needed = model_size_b * 2.5
        elif method == 'qlora':
            memory_needed = model_size_b * 0.6
        else:
            raise ValueError(f"Unknown method: {method}")

        # Adjust for batch size
        effective_batch = batch_size * gradient_accumulation
        memory_needed *= (1 + 0.1 * (effective_batch - 1))

        # Find suitable GPUs
        suitable_gpus = []
        for gpu_name, specs in self.gpu_specs.items():
            if specs['memory_gb'] >= memory_needed:
                suitable_gpus.append({
                    'gpu': gpu_name,
                    'memory_gb': specs['memory_gb'],
                    'cost_per_hour': specs['cost_per_hour'],
                    'utilization': memory_needed / specs['memory_gb']
                })

        return {
            'memory_needed_gb': round(memory_needed, 2),
            'suitable_gpus': suitable_gpus,
            'recommended_gpu': suitable_gpus[0]['gpu'] if suitable_gpus else 'Multi-GPU Required'
        }

# Example usage
calc = FineTuningGPUCalculator()

print("7B Model QLoRA:")
print(calc.calculate_requirements(7, 'qlora', batch_size=4))

print("\n13B Model LoRA:")
print(calc.calculate_requirements(13, 'lora', batch_size=4))

print("\n70B Model Full Fine-Tuning:")
print(calc.calculate_requirements(70, 'full', batch_size=1))
```

### Kubernetes Deployment for Fine-Tuning

```yaml
# Fine-tuning job on Kubernetes
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-finetuning-job
  namespace: ml-training
spec:
  template:
    metadata:
      labels:
        app: llm-finetuning
    spec:
      restartPolicy: OnFailure
      containers:
      - name: trainer
        image: your-registry/llm-trainer:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "64Gi"
            cpu: "16"
          limits:
            nvidia.com/gpu: 1
            memory: "64Gi"
            cpu: "16"
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-7b-hf"
        - name: TRAINING_METHOD
          value: "qlora"
        - name: OUTPUT_DIR
          value: "/models/output"
        - name: DATASET_PATH
          value: "/data/training_data.jsonl"
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-secret
              key: api-key
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: data-storage
          mountPath: /data
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: data-storage
        persistentVolumeClaim:
          claimName: data-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 16Gi
      nodeSelector:
        gpu-type: a100
        nvidia.com/gpu.memory: 80GB
```

### Multi-GPU Training Configuration

```yaml
# Multi-GPU distributed training job
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llm-distributed-training
  namespace: ml-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: your-registry/llm-trainer:latest
            resources:
              requests:
                nvidia.com/gpu: 4
                memory: "256Gi"
                cpu: "64"
              limits:
                nvidia.com/gpu: 4
                memory: "256Gi"
                cpu: "64"
            env:
            - name: WORLD_SIZE
              value: "4"
            - name: MASTER_PORT
              value: "29500"
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: your-registry/llm-trainer:latest
            resources:
              requests:
                nvidia.com/gpu: 4
                memory: "256Gi"
                cpu: "64"
              limits:
                nvidia.com/gpu: 4
                memory: "256Gi"
                cpu: "64"
```

## Distributed Training with DeepSpeed

### DeepSpeed Overview

DeepSpeed is Microsoft's library for distributed deep learning, enabling training of massive models.

**Key features:**
- ZeRO (Zero Redundancy Optimizer): Eliminate memory redundancies
- 3D Parallelism: Data + Pipeline + Tensor parallelism
- Optimized kernels for faster training

### ZeRO Stages

```python
# ZeRO stages explanation
zero_stages = {
    'Stage 0': {
        'description': 'Disabled (no optimization)',
        'memory_reduction': '1x',
        'use_case': 'Baseline'
    },
    'Stage 1': {
        'description': 'Optimizer state partitioning',
        'memory_reduction': '4x',
        'what_is_partitioned': 'Optimizer states',
        'use_case': 'Moderate memory savings'
    },
    'Stage 2': {
        'description': 'Optimizer + Gradient partitioning',
        'memory_reduction': '8x',
        'what_is_partitioned': 'Optimizer states + Gradients',
        'use_case': 'Significant memory savings'
    },
    'Stage 3': {
        'description': 'Optimizer + Gradient + Parameter partitioning',
        'memory_reduction': '64x (with large #GPUs)',
        'what_is_partitioned': 'Everything',
        'use_case': 'Maximum memory savings, largest models'
    }
}
```

### DeepSpeed Configuration

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": false
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 1,
  "wall_clock_breakdown": false
}
```

### DeepSpeed Training Script

```python
# Training with DeepSpeed
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model

def train_with_deepspeed(
    model_name: str,
    dataset,
    output_dir: str,
    deepspeed_config: str = "ds_config.json"
):
    """
    Train model with DeepSpeed
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )

    # Add LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Training arguments with DeepSpeed
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        deepspeed=deepspeed_config,  # Enable DeepSpeed
        report_to="wandb"
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(output_dir)

    return trainer


# Launch with DeepSpeed
if __name__ == "__main__":
    # Prepare dataset
    from datasets import load_dataset
    dataset = load_dataset("your_dataset")

    # Train
    train_with_deepspeed(
        model_name="meta-llama/Llama-2-7b-hf",
        dataset=dataset["train"],
        output_dir="./output"
    )
```

### Launch DeepSpeed Training

```bash
# Single node, multiple GPUs
deepspeed --num_gpus=4 train_script.py \
    --deepspeed ds_config.json \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir ./output

# Multiple nodes
deepspeed --num_nodes=4 \
    --num_gpus=8 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train_script.py \
    --deepspeed ds_config.json \
    --model_name meta-llama/Llama-2-70b-hf \
    --output_dir ./output
```

## Distributed Training with FSDP

### FSDP Overview

Fully Sharded Data Parallel (FSDP) is PyTorch's native solution for distributed training.

**Key features:**
- Built into PyTorch (no external dependencies)
- Shards model parameters, gradients, and optimizer states
- Supports mixed precision training
- Good integration with PyTorch ecosystem

### FSDP Configuration

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# FSDP configuration
fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    "cpu_offload": CPUOffload(offload_params=True),     # Offload to CPU
    "mixed_precision": MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    ),
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
    "auto_wrap_policy": transformer_auto_wrap_policy(
        LlamaDecoderLayer,
        lambda_fn=lambda module: True
    ),
}


# Training with FSDP
def train_with_fsdp(
    model_name: str,
    dataset,
    output_dir: str
):
    """
    Train model with FSDP
    """
    import torch.distributed as dist

    # Initialize distributed training
    dist.init_process_group(backend="nccl")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )

    # Add LoRA (optional)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Wrap with FSDP
    model = FSDP(
        model,
        **fsdp_config
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        fsdp="full_shard auto_wrap",  # Enable FSDP
        fsdp_config=fsdp_config,
        report_to="wandb"
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # Train
    trainer.train()

    return trainer
```

### DeepSpeed vs FSDP Comparison

```python
comparison = {
    'DeepSpeed': {
        'pros': [
            'More mature and battle-tested',
            'ZeRO-Offload for CPU offloading',
            'Extensive optimizations',
            'Better documentation and examples',
            'MoE (Mixture of Experts) support'
        ],
        'cons': [
            'External dependency',
            'More complex configuration',
            'Some NVIDIA-specific optimizations'
        ],
        'best_for': 'Maximum performance, largest models, production deployments'
    },
    'FSDP': {
        'pros': [
            'Native PyTorch (no extra dependencies)',
            'Simpler configuration',
            'Better integration with PyTorch ecosystem',
            'Active development'
        ],
        'cons': [
            'Less mature than DeepSpeed',
            'Fewer optimization features',
            'Less community support/examples'
        ],
        'best_for': 'PyTorch-native workflows, simpler setups, experimentation'
    }
}
```

## Hugging Face Ecosystem

### Complete Training Pipeline

```python
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

class LLMFineTuner:
    """
    Complete fine-tuning pipeline using Hugging Face ecosystem
    """
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        use_qlora: bool = True
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_qlora = use_qlora

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self):
        """Load model with optional quantization"""
        if self.use_qlora:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def add_lora(self, lora_config: LoraConfig = None):
        """Add LoRA adapters to model"""
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, dataset_name: str, split: str = "train"):
        """Load and prepare dataset"""
        dataset = load_dataset(dataset_name, split=split)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ):
        """Train the model"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=not self.use_qlora,
            bf16=self.use_qlora,
            logging_steps=10,
            logging_dir=f"{self.output_dir}/logs",
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit" if self.use_qlora else "adamw_torch",
            gradient_checkpointing=True,
            report_to="wandb"
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        self.trainer.train()

    def save_model(self):
        """Save the fine-tuned model"""
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def run_complete_pipeline(
        self,
        dataset_name: str,
        lora_config: LoraConfig = None
    ):
        """Run complete fine-tuning pipeline"""
        print("Loading model...")
        self.load_model()

        print("Adding LoRA adapters...")
        self.add_lora(lora_config)

        print("Preparing dataset...")
        dataset = self.prepare_dataset(dataset_name)

        # Split dataset
        split_dataset = dataset.train_test_split(test_size=0.1)

        print("Starting training...")
        self.train(
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"]
        )

        print("Saving model...")
        self.save_model()

        print("Fine-tuning complete!")


# Usage
if __name__ == "__main__":
    finetuner = LLMFineTuner(
        model_name="meta-llama/Llama-2-7b-hf",
        output_dir="./llama-2-7b-finetuned",
        use_qlora=True
    )

    finetuner.run_complete_pipeline(
        dataset_name="your_dataset_name"
    )
```

## Complete Fine-Tuning Workflows

### Custom Dataset Preparation

```python
import json
from datasets import Dataset

class DatasetPreparator:
    """
    Prepare custom datasets for fine-tuning
    """
    @staticmethod
    def from_jsonl(file_path: str):
        """Load dataset from JSONL file"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return Dataset.from_list(data)

    @staticmethod
    def format_instruction_dataset(examples):
        """
        Format instruction-following dataset
        Expected format: {"instruction": "...", "input": "...", "output": "..."}
        """
        prompts = []
        for instruction, input_text, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        ):
            if input_text:
                prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
            else:
                prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""
            prompts.append(prompt)

        return {"text": prompts}

    @staticmethod
    def format_chat_dataset(examples):
        """
        Format chat dataset
        Expected format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        """
        formatted = []
        for messages in examples["messages"]:
            conversation = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    conversation += f"User: {content}\n"
                elif role == "assistant":
                    conversation += f"Assistant: {content}\n"
            formatted.append(conversation)

        return {"text": formatted}

    @staticmethod
    def validate_dataset(dataset, required_columns=["text"]):
        """Validate dataset has required columns"""
        for col in required_columns:
            if col not in dataset.column_names:
                raise ValueError(f"Dataset missing required column: {col}")

        # Check for empty examples
        for i, example in enumerate(dataset):
            if not example["text"] or len(example["text"].strip()) == 0:
                print(f"Warning: Empty text at index {i}")

        print(f"Dataset validation passed. {len(dataset)} examples.")
        return True


# Example usage
preparator = DatasetPreparator()

# Load from JSONL
dataset = preparator.from_jsonl("training_data.jsonl")

# Format for instruction following
dataset = dataset.map(
    preparator.format_instruction_dataset,
    batched=True,
    remove_columns=dataset.column_names
)

# Validate
preparator.validate_dataset(dataset)
```

## Training Data Management

### Data Version Control with DVC

```bash
# Initialize DVC
dvc init

# Add large training dataset to DVC
dvc add data/training_data.jsonl

# Commit DVC files to git
git add data/training_data.jsonl.dvc data/.gitignore
git commit -m "Add training dataset"

# Configure remote storage (S3, GCS, etc.)
dvc remote add -d storage s3://my-bucket/dvc-storage

# Push data to remote
dvc push

# Pull data on another machine
dvc pull
```

### Data Quality Checks

```python
class DataQualityChecker:
    """
    Perform quality checks on training data
    """
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def check_length_distribution(self, max_length=2048):
        """Check token length distribution"""
        lengths = []
        for example in self.dataset:
            tokens = self.tokenizer.encode(example["text"])
            lengths.append(len(tokens))

        import numpy as np
        return {
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'p95_length': np.percentile(lengths, 95),
            'max_length': np.max(lengths),
            'num_truncated': sum(1 for l in lengths if l > max_length),
            'truncation_rate': sum(1 for l in lengths if l > max_length) / len(lengths)
        }

    def check_for_duplicates(self):
        """Check for duplicate examples"""
        texts = [example["text"] for example in self.dataset]
        unique_texts = set(texts)

        duplicate_rate = 1 - (len(unique_texts) / len(texts))

        return {
            'total_examples': len(texts),
            'unique_examples': len(unique_texts),
            'duplicates': len(texts) - len(unique_texts),
            'duplicate_rate': duplicate_rate
        }

    def check_for_common_issues(self):
        """Check for common data quality issues"""
        issues = {
            'empty_examples': 0,
            'too_short': 0,
            'too_long': 0,
            'encoding_errors': 0
        }

        for example in self.dataset:
            text = example["text"]

            # Empty
            if not text or len(text.strip()) == 0:
                issues['empty_examples'] += 1
                continue

            # Too short
            if len(text) < 10:
                issues['too_short'] += 1

            # Too long
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 4096:
                issues['too_long'] += 1

            # Encoding errors
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                issues['encoding_errors'] += 1

        return issues

    def generate_report(self):
        """Generate complete data quality report"""
        print("=== Data Quality Report ===\n")

        print("Length Distribution:")
        length_stats = self.check_length_distribution()
        for key, value in length_stats.items():
            print(f"  {key}: {value}")

        print("\nDuplicate Check:")
        dup_stats = self.check_for_duplicates()
        for key, value in dup_stats.items():
            print(f"  {key}: {value}")

        print("\nCommon Issues:")
        issues = self.check_for_common_issues()
        for key, value in issues.items():
            print(f"  {key}: {value}")

        print("\n=== End Report ===")


# Usage
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
checker = DataQualityChecker(dataset, tokenizer)
checker.generate_report()
```

## Experiment Tracking

### Weights & Biases Integration

```python
import wandb
from transformers import TrainingArguments, Trainer

# Initialize wandb
wandb.init(
    project="llm-finetuning",
    name="llama-2-7b-custom-dataset",
    config={
        "model": "meta-llama/Llama-2-7b-hf",
        "method": "qlora",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size": 4
    }
)

# Training arguments with wandb
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    logging_steps=10,
    report_to="wandb",  # Enable wandb logging
    run_name="llama-2-7b-custom-dataset"
)

# Trainer will automatically log to wandb
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# Log custom metrics
wandb.log({
    "final_train_loss": trainer.state.log_history[-1]["train_loss"],
    "final_eval_loss": trainer.state.log_history[-1]["eval_loss"]
})

wandb.finish()
```

### MLflow Integration

```python
import mlflow
from transformers import TrainingArguments, Trainer

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("llm-finetuning")

# Start MLflow run
with mlflow.start_run(run_name="llama-2-7b-qlora"):
    # Log parameters
    mlflow.log_params({
        "model": "meta-llama/Llama-2-7b-hf",
        "method": "qlora",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size": 4
    })

    # Train model
    trainer = Trainer(...)
    trainer.train()

    # Log metrics
    mlflow.log_metrics({
        "final_train_loss": trainer.state.log_history[-1]["train_loss"],
        "final_eval_loss": trainer.state.log_history[-1]["eval_loss"]
    })

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifacts("./output", artifact_path="checkpoints")
```

## Hyperparameter Tuning

### Grid Search for LoRA Parameters

```python
import itertools
from transformers import TrainingArguments, Trainer

class LoRAHyperparameterTuner:
    """
    Hyperparameter tuning for LoRA fine-tuning
    """
    def __init__(self, model_name: str, dataset, eval_dataset):
        self.model_name = model_name
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.results = []

    def grid_search(
        self,
        param_grid: dict,
        output_base_dir: str = "./tuning_runs"
    ):
        """
        Perform grid search over hyperparameters
        """
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))

        print(f"Testing {len(combinations)} combinations...")

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")

            # Train with these parameters
            result = self.train_with_params(params, f"{output_base_dir}/run_{i}")

            self.results.append({
                'params': params,
                'eval_loss': result['eval_loss'],
                'train_loss': result['train_loss']
            })

        # Find best configuration
        best_result = min(self.results, key=lambda x: x['eval_loss'])
        print(f"\nBest configuration: {best_result['params']}")
        print(f"Best eval loss: {best_result['eval_loss']}")

        return best_result

    def train_with_params(self, params: dict, output_dir: str):
        """Train model with specific hyperparameters"""
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Add LoRA
        lora_config = LoraConfig(
            r=params.get('lora_r', 16),
            lora_alpha=params.get('lora_alpha', 32),
            lora_dropout=params.get('lora_dropout', 0.05),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=params.get('epochs', 3),
            per_device_train_batch_size=params.get('batch_size', 4),
            learning_rate=params.get('learning_rate', 2e-4),
            fp16=True,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset
        )

        trainer.train()

        # Get final metrics
        eval_results = trainer.evaluate()

        return {
            'eval_loss': eval_results['eval_loss'],
            'train_loss': trainer.state.log_history[-1]['loss']
        }


# Example usage
tuner = LoRAHyperparameterTuner(
    model_name="meta-llama/Llama-2-7b-hf",
    dataset=train_dataset,
    eval_dataset=eval_dataset
)

param_grid = {
    'lora_r': [8, 16, 32],
    'lora_alpha': [16, 32, 64],
    'learning_rate': [1e-4, 2e-4, 5e-4],
    'batch_size': [4, 8]
}

best_config = tuner.grid_search(param_grid)
```

## Cost Optimization

### Fine-Tuning Cost Calculator

```python
class FineTuningCostCalculator:
    """
    Calculate and optimize fine-tuning costs
    """
    def __init__(self):
        self.gpu_costs = {
            'T4': 0.40,
            'A10G': 1.20,
            'A100-40GB': 3.50,
            'A100-80GB': 5.00,
            'H100': 8.50
        }

    def estimate_training_time(
        self,
        num_examples: int,
        avg_tokens_per_example: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        num_epochs: int,
        gpu_type: str,
        model_size_b: float
    ):
        """
        Estimate training time
        """
        # Tokens per second estimates (rough)
        throughput = {
            ('T4', 7): 50,
            ('A10G', 7): 150,
            ('A100-40GB', 7): 350,
            ('A100-80GB', 7): 400,
            ('T4', 13): 25,
            ('A10G', 13): 75,
            ('A100-40GB', 13): 175,
            ('A100-80GB', 13): 200,
        }

        tokens_per_sec = throughput.get((gpu_type, int(model_size_b)), 100)

        # Calculate total training tokens
        total_tokens = num_examples * avg_tokens_per_example * num_epochs

        # Effective batch size
        effective_batch = batch_size * gradient_accumulation_steps

        # Adjust throughput for batch size
        adjusted_throughput = tokens_per_sec * effective_batch

        # Training time in hours
        training_hours = (total_tokens / adjusted_throughput) / 3600

        return training_hours

    def calculate_cost(
        self,
        num_examples: int,
        avg_tokens_per_example: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        num_epochs: int,
        gpu_type: str,
        model_size_b: float,
        spot_instance: bool = False
    ):
        """
        Calculate total fine-tuning cost
        """
        # Estimate training time
        training_hours = self.estimate_training_time(
            num_examples,
            avg_tokens_per_example,
            batch_size,
            gradient_accumulation_steps,
            num_epochs,
            gpu_type,
            model_size_b
        )

        # GPU cost
        gpu_cost_per_hour = self.gpu_costs[gpu_type]

        # Spot discount
        if spot_instance:
            gpu_cost_per_hour *= 0.3  # ~70% discount

        total_cost = training_hours * gpu_cost_per_hour

        return {
            'training_hours': round(training_hours, 2),
            'gpu_cost_per_hour': gpu_cost_per_hour,
            'total_cost': round(total_cost, 2),
            'gpu_type': gpu_type,
            'spot_instance': spot_instance,
            'cost_per_example': round(total_cost / num_examples, 4)
        }

    def optimize_configuration(
        self,
        num_examples: int,
        avg_tokens_per_example: int,
        budget: float
    ):
        """
        Find optimal configuration within budget
        """
        configurations = []

        for gpu_type in ['T4', 'A10G', 'A100-40GB']:
            for spot in [False, True]:
                for batch_size in [1, 2, 4, 8]:
                    for grad_accum in [1, 2, 4, 8]:
                        cost_est = self.calculate_cost(
                            num_examples=num_examples,
                            avg_tokens_per_example=avg_tokens_per_example,
                            batch_size=batch_size,
                            gradient_accumulation_steps=grad_accum,
                            num_epochs=3,
                            gpu_type=gpu_type,
                            model_size_b=7,
                            spot_instance=spot
                        )

                        if cost_est['total_cost'] <= budget:
                            configurations.append({
                                **cost_est,
                                'batch_size': batch_size,
                                'grad_accum': grad_accum,
                                'effective_batch': batch_size * grad_accum
                            })

        # Sort by training time (faster is better within budget)
        configurations.sort(key=lambda x: x['training_hours'])

        return configurations[:5]  # Top 5 configurations


# Example usage
calculator = FineTuningCostCalculator()

# Estimate cost for specific configuration
cost = calculator.calculate_cost(
    num_examples=10000,
    avg_tokens_per_example=512,
    batch_size=4,
    gradient_accumulation_steps=4,
    num_epochs=3,
    gpu_type='A10G',
    model_size_b=7,
    spot_instance=True
)
print("Cost estimate:", cost)

# Find optimal configuration within budget
budget = 50  # $50 budget
optimal_configs = calculator.optimize_configuration(
    num_examples=10000,
    avg_tokens_per_example=512,
    budget=budget
)
print("\nOptimal configurations within budget:")
for config in optimal_configs:
    print(config)
```

## Production Fine-Tuning Pipeline

### Complete Production Pipeline

```python
# production_finetuning_pipeline.py

import os
import json
from datetime import datetime
from pathlib import Path
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

class ProductionFineTuningPipeline:
    """
    Production-ready fine-tuning pipeline with complete lifecycle management
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(self.config['output_dir']) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config['wandb_project'],
                name=self.run_id,
                config=self.config
            )

    def load_and_prepare_data(self):
        """Load and prepare training data"""
        print("Loading dataset...")

        if self.config['dataset']['type'] == 'huggingface':
            dataset = load_dataset(
                self.config['dataset']['name'],
                split=self.config['dataset']['split']
            )
        elif self.config['dataset']['type'] == 'jsonl':
            dataset = Dataset.from_json(self.config['dataset']['path'])
        else:
            raise ValueError(f"Unknown dataset type: {self.config['dataset']['type']}")

        # Data quality checks
        print("Running data quality checks...")
        self.run_data_quality_checks(dataset)

        # Tokenize
        print("Tokenizing dataset...")
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config['max_length'],
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Train/eval split
        split_dataset = tokenized_dataset.train_test_split(
            test_size=self.config.get('eval_split', 0.1)
        )

        return split_dataset, tokenizer

    def run_data_quality_checks(self, dataset):
        """Run comprehensive data quality checks"""
        # Implement checks from DataQualityChecker class
        pass

    def load_model(self, tokenizer):
        """Load model with optional quantization"""
        print("Loading model...")

        if self.config['training']['method'] == 'qlora':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                torch_dtype=torch.float16,
                device_map="auto"
            )

        # Add LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['rank'],
            lora_alpha=self.config['lora']['alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def train(self, model, tokenizer, dataset):
        """Train the model"""
        print("Starting training...")

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config['training']['epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            fp16=self.config['training']['method'] != 'qlora',
            bf16=self.config['training']['method'] == 'qlora',
            logging_steps=self.config['training']['logging_steps'],
            logging_dir=str(self.output_dir / "logs"),
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit" if self.config['training']['method'] == 'qlora' else "adamw_torch",
            gradient_checkpointing=True,
            report_to="wandb" if self.config.get('use_wandb') else "none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"]
        )

        trainer.train()

        return trainer

    def evaluate_model(self, trainer):
        """Evaluate trained model"""
        print("Evaluating model...")

        eval_results = trainer.evaluate()

        # Save evaluation results
        with open(self.output_dir / "eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)

        return eval_results

    def save_model(self, trainer, tokenizer):
        """Save model and artifacts"""
        print("Saving model...")

        # Save model
        trainer.save_model(str(self.output_dir / "final_model"))
        tokenizer.save_pretrained(str(self.output_dir / "final_model"))

        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Model saved to {self.output_dir}")

    def run(self):
        """Run complete pipeline"""
        try:
            # Load data
            dataset, tokenizer = self.load_and_prepare_data()

            # Load model
            model = self.load_model(tokenizer)

            # Train
            trainer = self.train(model, tokenizer, dataset)

            # Evaluate
            eval_results = self.evaluate_model(trainer)

            # Save
            self.save_model(trainer, tokenizer)

            print("\nPipeline completed successfully!")
            print(f"Final eval loss: {eval_results['eval_loss']}")

            if self.config.get('use_wandb'):
                wandb.finish()

        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            if self.config.get('use_wandb'):
                wandb.finish(exit_code=1)
            raise


# Example config.json
example_config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "output_dir": "./finetuned_models",
    "wandb_project": "llm-finetuning",
    "use_wandb": True,
    "dataset": {
        "type": "huggingface",
        "name": "tatsu-lab/alpaca",
        "split": "train"
    },
    "max_length": 512,
    "eval_split": 0.1,
    "training": {
        "method": "qlora",
        "epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "logging_steps": 10
    },
    "lora": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
}

# Save example config
with open("finetune_config.json", 'w') as f:
    json.dump(example_config, f, indent=2)

# Run pipeline
if __name__ == "__main__":
    pipeline = ProductionFineTuningPipeline("finetune_config.json")
    pipeline.run()
```

## Summary

This lesson covered the complete infrastructure and process for fine-tuning Large Language Models:

### Key Takeaways

1. **Fine-Tuning vs RAG**: Choose based on your use case:
   - Fine-tuning for behavior/style adaptation
   - RAG for dynamic, factual knowledge
   - Hybrid for best of both worlds

2. **PEFT Methods**: Dramatically reduce resource requirements:
   - LoRA: 128x parameter reduction
   - QLoRA: Train 70B models on single GPU
   - Production-ready with Hugging Face PEFT

3. **Infrastructure**: Plan GPU resources carefully:
   - QLoRA: 7B model on 24GB GPU
   - Distributed training for larger models
   - DeepSpeed and FSDP for scale

4. **Data Management**: Quality matters:
   - Validate and clean datasets
   - Version control with DVC
   - Monitor data quality metrics

5. **Experiment Tracking**: Track everything:
   - Weights & Biases for visualization
   - MLflow for model registry
   - Comprehensive logging

6. **Cost Optimization**: Control expenses:
   - Use spot instances
   - Optimize batch sizes
   - Choose appropriate GPU tier
   - Monitor training efficiency

7. **Production Pipelines**: Automate and monitor:
   - End-to-end automation
   - Quality gates
   - Artifact management
   - Reproducibility

### Next Steps

In the next lesson, we'll cover LLM serving optimization techniques including quantization, flash attention, continuous batching, and more.

---

**Next Lesson**: [06-llm-serving-optimization.md](./06-llm-serving-optimization.md)
