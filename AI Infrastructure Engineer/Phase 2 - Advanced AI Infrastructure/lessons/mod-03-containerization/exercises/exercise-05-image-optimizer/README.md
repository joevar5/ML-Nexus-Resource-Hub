# Exercise 05: Multi-Stage Build Optimizer

**Estimated Time**: 28-35 hours
**Difficulty**: Advanced
**Prerequisites**: Docker, Python 3.9+, Docker BuildKit

## Overview

Build a comprehensive tool that analyzes Dockerfiles and container images to identify optimization opportunities, automatically applies multi-stage build patterns, and reduces image sizes by 50%+ while maintaining functionality. This exercise teaches Docker layer caching, build optimization strategies, and automated image analysis.

In production ML infrastructure, container image sizes directly impact:
- **Deployment Speed**: Smaller images = faster pulls across clusters
- **Storage Costs**: Registry storage and bandwidth costs
- **Attack Surface**: Fewer packages = reduced security vulnerabilities
- **Cold Start Latency**: Critical for serverless ML inference

## Learning Objectives

By completing this exercise, you will:

1. **Analyze Docker build efficiency** using BuildKit traces and layer inspection
2. **Implement multi-stage builds** to separate build-time and runtime dependencies
3. **Optimize layer caching** for faster CI/CD builds (order dependencies correctly)
4. **Reduce image sizes** by 50-80% using distroless/slim base images
5. **Automate Dockerfile optimization** with AST parsing and rewriting
6. **Measure build performance** with detailed metrics and recommendations

## Business Context

**Real-World Scenario**: Your ML engineering team deploys models to Kubernetes clusters across 5 AWS regions. Current Docker images are 2.5 GB, causing:

- **15-minute pod startup times** due to slow image pulls
- **$3,000/month** in ECR storage and data transfer costs
- **Failed autoscaling** during traffic spikes (can't scale fast enough)
- **Security alerts** for 87 vulnerabilities in unnecessary build tools

Your task: Build an optimizer that reduces images to <500 MB, cuts startup time to <3 minutes, and eliminates build-time dependencies from production images.

## Project Structure

```
exercise-05-image-optimizer/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Poetry/pip-tools config
├── src/
│   └── image_optimizer/
│       ├── __init__.py
│       ├── dockerfile_parser.py       # Parse and analyze Dockerfiles
│       ├── layer_analyzer.py          # Inspect image layers and sizes
│       ├── optimizer.py               # Apply optimization strategies
│       ├── multistage_builder.py      # Generate multi-stage Dockerfiles
│       ├── cache_analyzer.py          # Analyze and improve build caching
│       ├── reporter.py                # Generate optimization reports
│       └── cli.py                     # Command-line interface
├── tests/
│   ├── test_dockerfile_parser.py
│   ├── test_layer_analyzer.py
│   ├── test_optimizer.py
│   ├── test_multistage_builder.py
│   ├── test_cache_analyzer.py
│   └── fixtures/
│       ├── dockerfiles/               # Sample Dockerfiles to optimize
│       │   ├── python_ml.Dockerfile   # Typical ML image (2.5 GB)
│       │   ├── nodejs_api.Dockerfile  # Node.js API (1.8 GB)
│       │   └── java_service.Dockerfile # Java microservice (3.2 GB)
│       └── expected/                  # Expected optimized outputs
├── examples/
│   ├── before_after_comparison.md     # Real optimization examples
│   └── optimization_strategies.md     # Common patterns
├── benchmarks/
│   ├── measure_build_time.sh          # Benchmark build performance
│   └── measure_image_size.sh          # Compare image sizes
└── docs/
    ├── DESIGN.md                      # Architecture decisions
    └── OPTIMIZATION_GUIDE.md          # Best practices reference
```

## Requirements

### Functional Requirements

Your image optimizer must:

1. **Parse Dockerfiles** into an abstract syntax tree (AST)
2. **Analyze image layers** using `docker history` and manifest inspection
3. **Detect optimization opportunities**:
   - Single-stage builds that should be multi-stage
   - Build tools in final image (gcc, npm, maven, cargo)
   - Inefficient layer ordering (dependencies change frequently)
   - Large files in intermediate layers (not properly removed)
   - Suboptimal base images (use slim/alpine/distroless variants)
4. **Generate optimized Dockerfiles** with multi-stage patterns
5. **Measure improvements** with before/after size comparisons
6. **Provide actionable recommendations** with estimated savings

### Non-Functional Requirements

- **Performance**: Analyze Dockerfile in <2 seconds, build comparison in <5 minutes
- **Accuracy**: Size reduction predictions within ±10% of actual
- **Safety**: Never break image functionality (validate with test commands)
- **Compatibility**: Support Dockerfile syntax versions 1.0-1.6

## Implementation Tasks

### Task 1: Dockerfile Parser (5-6 hours)

Build a parser that converts Dockerfiles into structured data for analysis.

```python
# src/image_optimizer/dockerfile_parser.py

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum

class InstructionType(Enum):
    FROM = "FROM"
    RUN = "RUN"
    COPY = "COPY"
    ADD = "ADD"
    WORKDIR = "WORKDIR"
    ENV = "ENV"
    ARG = "ARG"
    EXPOSE = "EXPOSE"
    CMD = "CMD"
    ENTRYPOINT = "ENTRYPOINT"
    LABEL = "LABEL"
    USER = "USER"
    VOLUME = "VOLUME"

@dataclass
class DockerfileInstruction:
    """Represents a single Dockerfile instruction"""
    type: InstructionType
    arguments: str
    line_number: int
    raw_line: str

    def is_layer_creating(self) -> bool:
        """RUN, COPY, ADD create layers; others are metadata"""
        # TODO: Return True if instruction creates a new layer
        raise NotImplementedError

@dataclass
class Stage:
    """Represents a build stage in multi-stage Dockerfile"""
    name: Optional[str]  # FROM foo AS builder -> name="builder"
    base_image: str
    instructions: List[DockerfileInstruction]
    stage_number: int

    def get_dependencies(self) -> List[str]:
        """Extract packages installed in this stage"""
        # TODO: Parse RUN apt-get install, pip install, npm install
        # Return list of packages
        raise NotImplementedError

    def is_build_stage(self) -> bool:
        """Detect if stage contains build tools"""
        # TODO: Check for gcc, make, npm, maven, cargo, rustc, etc.
        raise NotImplementedError

class DockerfileParser:
    """Parse Dockerfile into structured format"""

    def __init__(self, dockerfile_path: Path):
        self.dockerfile_path = dockerfile_path
        self.stages: List[Stage] = []

    def parse(self) -> List[Stage]:
        """
        Parse Dockerfile into stages

        Handle:
        - Multi-line instructions with backslash continuation
        - Comments (lines starting with #)
        - ARG before FROM (global args)
        - Multi-stage builds (multiple FROM instructions)
        """
        # TODO: Read file, split into lines
        # TODO: Handle line continuations (backslash)
        # TODO: Parse each instruction type
        # TODO: Group instructions into stages (split on FROM)
        # TODO: Extract stage names (FROM ... AS name)
        raise NotImplementedError

    def get_base_images(self) -> List[str]:
        """Extract all base images used"""
        # TODO: Return list of images from FROM instructions
        raise NotImplementedError

    def count_layers(self) -> int:
        """Count layer-creating instructions"""
        # TODO: Count RUN, COPY, ADD across all stages
        raise NotImplementedError

    def extract_package_managers(self) -> Dict[str, List[str]]:
        """
        Identify package managers used

        Returns:
            {
                "apt": ["python3", "gcc", "git"],
                "pip": ["torch==2.0.0", "transformers"],
                "npm": ["typescript", "@types/node"]
            }
        """
        # TODO: Parse RUN commands for apt-get, yum, apk, pip, npm, etc.
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Parse single-stage Dockerfiles into Stage objects
- ✅ Parse multi-stage Dockerfiles with named stages
- ✅ Handle line continuations and comments correctly
- ✅ Extract base images, packages, and layer counts
- ✅ Tests pass for 10+ real-world Dockerfiles

**Test Example**:
```python
def test_parse_multistage_dockerfile():
    parser = DockerfileParser(Path("fixtures/dockerfiles/python_ml.Dockerfile"))
    stages = parser.parse()

    assert len(stages) == 2
    assert stages[0].name == "builder"
    assert stages[0].is_build_stage() == True
    assert stages[1].name is None  # Final stage

    packages = parser.extract_package_managers()
    assert "pip" in packages
    assert "torch" in packages["pip"][0]
```

---

### Task 2: Layer Analyzer (5-6 hours)

Analyze built images to identify large layers and inefficiencies.

```python
# src/image_optimizer/layer_analyzer.py

from dataclasses import dataclass
from typing import List, Dict, Optional
import subprocess
import json

@dataclass
class Layer:
    """Represents a single image layer"""
    id: str
    size_bytes: int
    created_by: str  # Dockerfile instruction that created it
    created_at: str
    comment: Optional[str] = None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def is_large(self, threshold_mb: float = 100.0) -> bool:
        """Check if layer exceeds size threshold"""
        return self.size_mb > threshold_mb

@dataclass
class ImageAnalysis:
    """Complete analysis of a Docker image"""
    image_name: str
    total_size_bytes: int
    layers: List[Layer]
    base_image_size_bytes: int

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def added_size_mb(self) -> float:
        """Size added on top of base image"""
        return (self.total_size_bytes - self.base_image_size_bytes) / (1024 * 1024)

    def get_largest_layers(self, n: int = 10) -> List[Layer]:
        """Get the N largest layers"""
        # TODO: Sort layers by size, return top N
        raise NotImplementedError

class LayerAnalyzer:
    """Analyze Docker image layers for optimization opportunities"""

    def analyze_image(self, image_name: str) -> ImageAnalysis:
        """
        Analyze image layers using docker history and docker inspect

        Steps:
        1. Run: docker history --no-trunc --format json <image>
        2. Run: docker inspect <image> to get total size
        3. Parse layer information
        4. Identify base image size
        """
        # TODO: Execute docker commands
        # TODO: Parse JSON output
        # TODO: Create Layer objects
        # TODO: Calculate base image size (first layer)
        raise NotImplementedError

    def find_removable_files(self, image_name: str) -> List[Dict]:
        """
        Find files that could be removed to reduce size

        Common culprits:
        - /var/cache/apt/*
        - /root/.cache/pip/*
        - *.pyc, __pycache__
        - /tmp/*
        - npm cache
        - Build artifacts (*.o, *.a)

        Returns:
            [
                {"path": "/var/cache/apt", "size_mb": 45.2},
                {"path": "/root/.cache/pip", "size_mb": 123.5}
            ]
        """
        # TODO: Run container with: docker run --rm <image> du -sh /var/cache /root/.cache /tmp
        # TODO: Parse output to find large directories
        raise NotImplementedError

    def detect_unnecessary_packages(self, image_name: str, runtime_deps: List[str]) -> List[str]:
        """
        Detect build-time packages that aren't needed at runtime

        Args:
            image_name: Image to analyze
            runtime_deps: List of packages actually needed at runtime

        Returns:
            List of package names that can be removed
        """
        # TODO: Run: docker run --rm <image> dpkg -l (for Debian/Ubuntu)
        # TODO: Compare installed packages vs runtime_deps
        # TODO: Flag build tools: gcc, g++, make, cmake, git, etc.
        raise NotImplementedError

    def analyze_layer_efficiency(self, layers: List[Layer]) -> Dict:
        """
        Analyze layer ordering for cache efficiency

        Returns:
            {
                "cache_friendly": False,
                "issues": [
                    "COPY . . comes before pip install (dependencies not cached)",
                    "apt-get update and install in separate layers"
                ],
                "recommendations": [
                    "Move COPY requirements.txt before COPY . .",
                    "Combine apt-get update && install in single RUN"
                ]
            }
        """
        # TODO: Analyze layer creation order
        # TODO: Detect anti-patterns
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Analyze image layers with accurate size calculations
- ✅ Identify layers >100 MB
- ✅ Detect removable cache directories
- ✅ Flag unnecessary build-time packages
- ✅ Provide cache efficiency recommendations

**Test Example**:
```python
def test_analyze_pytorch_image():
    analyzer = LayerAnalyzer()
    analysis = analyzer.analyze_image("pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime")

    assert analysis.total_size_mb > 1000  # PyTorch images are large
    largest = analysis.get_largest_layers(5)
    assert len(largest) == 5
    assert all(layer.size_mb > 50 for layer in largest)
```

---

### Task 3: Multi-Stage Builder (6-7 hours)

Generate optimized multi-stage Dockerfiles automatically.

```python
# src/image_optimizer/multistage_builder.py

from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
from .dockerfile_parser import Stage, DockerfileInstruction, InstructionType

@dataclass
class OptimizationStrategy:
    """Configuration for optimization approach"""
    use_multistage: bool = True
    use_slim_base: bool = True  # python:3.11 -> python:3.11-slim
    use_distroless: bool = False  # Google distroless for max security
    combine_run_commands: bool = True
    remove_cache: bool = True
    optimize_layer_order: bool = True

    def get_base_image_variant(self, base_image: str) -> str:
        """Convert base image to slim/alpine/distroless variant"""
        # TODO: python:3.11 -> python:3.11-slim
        # TODO: node:18 -> node:18-alpine
        # TODO: ubuntu:22.04 -> gcr.io/distroless/python3-debian11
        raise NotImplementedError

class MultiStageBuilder:
    """Generate optimized multi-stage Dockerfiles"""

    def __init__(self, strategy: OptimizationStrategy):
        self.strategy = strategy

    def optimize_dockerfile(
        self,
        stages: List[Stage],
        runtime_dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Convert single or multi-stage Dockerfile into optimized version

        Optimization steps:
        1. Separate build and runtime stages
        2. Use slim/alpine base images
        3. Combine RUN commands to reduce layers
        4. Order layers by change frequency (least to most)
        5. Remove caches in same layer they're created
        6. Copy only necessary artifacts from builder stage

        Args:
            stages: Parsed Dockerfile stages
            runtime_dependencies: List of packages needed at runtime

        Returns:
            Optimized Dockerfile as string
        """
        # TODO: Implement optimization logic
        raise NotImplementedError

    def _create_builder_stage(self, original_stage: Stage) -> str:
        """
        Create builder stage with all build dependencies

        Example output:
        ```
        FROM python:3.11 AS builder
        WORKDIR /build
        RUN apt-get update && apt-get install -y --no-install-recommends \
            gcc g++ make \
            && rm -rf /var/lib/apt/lists/*
        COPY requirements.txt .
        RUN pip install --user --no-cache-dir -r requirements.txt
        COPY . .
        ```
        """
        # TODO: Generate builder stage Dockerfile content
        raise NotImplementedError

    def _create_runtime_stage(
        self,
        original_stage: Stage,
        runtime_dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Create minimal runtime stage

        Example output:
        ```
        FROM python:3.11-slim
        WORKDIR /app

        # Copy Python packages from builder
        COPY --from=builder /root/.local /root/.local
        ENV PATH=/root/.local/bin:$PATH

        # Copy application code
        COPY --from=builder /build/src ./src
        COPY --from=builder /build/config ./config

        # Install only runtime dependencies
        RUN apt-get update && apt-get install -y --no-install-recommends \
            libgomp1 \
            && rm -rf /var/lib/apt/lists/*

        CMD ["python", "src/main.py"]
        ```
        """
        # TODO: Generate runtime stage with minimal dependencies
        # TODO: Copy artifacts from builder (use COPY --from=builder)
        # TODO: Install only runtime_dependencies
        raise NotImplementedError

    def _combine_run_commands(self, instructions: List[DockerfileInstruction]) -> List[DockerfileInstruction]:
        """
        Combine consecutive RUN commands into single layer

        Before:
        RUN apt-get update
        RUN apt-get install -y python3
        RUN pip install torch

        After:
        RUN apt-get update && \
            apt-get install -y python3 && \
            pip install torch && \
            rm -rf /var/lib/apt/lists/*
        """
        # TODO: Find consecutive RUN instructions
        # TODO: Combine with && operator
        # TODO: Add cleanup commands (rm -rf /var/lib/apt/lists/*)
        raise NotImplementedError

    def _optimize_layer_order(self, instructions: List[DockerfileInstruction]) -> List[DockerfileInstruction]:
        """
        Reorder instructions for better caching

        Optimal order (least to most frequently changed):
        1. System package installation (apt-get, yum)
        2. Application dependency files (requirements.txt, package.json)
        3. Dependency installation (pip install, npm install)
        4. Application code (COPY . .)
        5. Build commands (npm run build, make)

        This ensures changing app code doesn't invalidate dependency layers
        """
        # TODO: Categorize instructions by change frequency
        # TODO: Reorder while maintaining dependencies
        raise NotImplementedError

    def generate_dockerfile(self, optimized_content: str, output_path: Path) -> None:
        """Write optimized Dockerfile to disk"""
        # TODO: Write content to file
        # TODO: Add header comment explaining optimizations applied
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Convert single-stage to multi-stage Dockerfile
- ✅ Separate build and runtime dependencies
- ✅ Use slim/alpine base images appropriately
- ✅ Combine RUN commands to reduce layers
- ✅ Optimize layer ordering for caching
- ✅ Generated Dockerfile builds successfully

**Test Example**:
```python
def test_optimize_python_ml_dockerfile():
    strategy = OptimizationStrategy(
        use_multistage=True,
        use_slim_base=True,
        combine_run_commands=True
    )
    builder = MultiStageBuilder(strategy)

    # Original single-stage Dockerfile
    parser = DockerfileParser(Path("fixtures/dockerfiles/python_ml.Dockerfile"))
    stages = parser.parse()

    optimized = builder.optimize_dockerfile(
        stages,
        runtime_dependencies=["libgomp1"]  # Required for PyTorch
    )

    assert "FROM python:3.11 AS builder" in optimized
    assert "FROM python:3.11-slim" in optimized
    assert "COPY --from=builder" in optimized
    assert "rm -rf /var/lib/apt/lists/*" in optimized
```

---

### Task 4: Cache Analyzer (4-5 hours)

Analyze build cache effectiveness and recommend improvements.

```python
# src/image_optimizer/cache_analyzer.py

from dataclasses import dataclass
from typing import List, Dict
import subprocess
import re
from datetime import datetime

@dataclass
class BuildStep:
    """Represents a single build step from BuildKit output"""
    step_number: int
    instruction: str
    cached: bool
    duration_seconds: float
    size_bytes: int

@dataclass
class BuildAnalysis:
    """Analysis of Docker build performance"""
    total_duration_seconds: float
    steps: List[BuildStep]
    cache_hit_rate: float  # Percentage of steps using cache
    estimated_speedup: float  # How much faster with perfect caching

    @property
    def cache_misses(self) -> List[BuildStep]:
        return [step for step in self.steps if not step.cached]

class CacheAnalyzer:
    """Analyze Docker build caching effectiveness"""

    def analyze_build(self, dockerfile_path: Path, build_context: Path) -> BuildAnalysis:
        """
        Build image and analyze cache performance

        Uses BuildKit's --progress=plain output to track:
        - Which steps hit cache vs rebuilt
        - Time spent on each step
        - First cache miss (indicates where to optimize)

        Example BuildKit output:
        #5 [2/8] RUN apt-get update
        #5 CACHED

        #6 [3/8] COPY requirements.txt .
        #6 sha256:abc123...
        #6 DONE 0.5s
        """
        # TODO: Run: docker buildx build --progress=plain --no-cache <context>
        # TODO: Parse output to identify cached vs rebuilt steps
        # TODO: Calculate cache hit rate
        # TODO: Estimate speedup with better caching
        raise NotImplementedError

    def find_cache_breakers(self, analysis: BuildAnalysis) -> List[Dict]:
        """
        Identify instructions that frequently break cache

        Returns:
            [
                {
                    "step": 5,
                    "instruction": "COPY . .",
                    "reason": "Copies entire context, changes frequently",
                    "recommendation": "COPY only necessary files, use .dockerignore"
                }
            ]
        """
        # TODO: Find first cache miss
        # TODO: Identify common cache-breaking patterns:
        #   - COPY . . before dependency installation
        #   - apt-get update without version pinning
        #   - Timestamp-based commands (date, git commit hash)
        raise NotImplementedError

    def recommend_dockerignore(self, build_context: Path) -> List[str]:
        """
        Recommend .dockerignore patterns based on context

        Common patterns to ignore:
        - .git/
        - __pycache__/
        - *.pyc
        - .pytest_cache/
        - node_modules/ (if not needed in image)
        - .env
        - README.md, docs/
        """
        # TODO: Scan build context
        # TODO: Identify large directories not needed in image
        # TODO: Generate .dockerignore recommendations
        raise NotImplementedError

    def measure_layer_rebuild_time(self, dockerfile_path: Path) -> Dict[int, float]:
        """
        Measure how long each layer takes to rebuild

        Returns:
            {
                1: 0.3,  # Step 1 takes 0.3 seconds
                2: 45.2, # Step 2 takes 45.2 seconds (optimize this!)
                3: 12.1
            }
        """
        # TODO: Build image, parse BuildKit timings
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Analyze build with accurate cache hit rate
- ✅ Identify cache-breaking instructions
- ✅ Recommend .dockerignore patterns
- ✅ Measure per-layer build times
- ✅ Provide actionable optimization suggestions

---

### Task 5: Optimizer (4-5 hours)

Main optimizer that orchestrates all analysis and optimization.

```python
# src/image_optimizer/optimizer.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from .dockerfile_parser import DockerfileParser
from .layer_analyzer import LayerAnalyzer, ImageAnalysis
from .multistage_builder import MultiStageBuilder, OptimizationStrategy
from .cache_analyzer import CacheAnalyzer

@dataclass
class OptimizationResult:
    """Results of optimization process"""
    original_size_mb: float
    optimized_size_mb: float
    size_reduction_percent: float
    original_layers: int
    optimized_layers: int
    estimated_build_speedup: float
    optimizations_applied: List[str]
    warnings: List[str]

    def __str__(self) -> str:
        return f"""
Optimization Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original Size:    {self.original_size_mb:.1f} MB
Optimized Size:   {self.optimized_size_mb:.1f} MB
Size Reduction:   {self.size_reduction_percent:.1f}% 🎉
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original Layers:  {self.original_layers}
Optimized Layers: {self.optimized_layers}
Build Speedup:    {self.estimated_build_speedup:.1f}x faster
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Optimizations Applied:
{chr(10).join(f"  ✓ {opt}" for opt in self.optimizations_applied)}

{f"Warnings:{chr(10)}" + chr(10).join(f"  ⚠ {warn}" for warn in self.warnings) if self.warnings else ""}
"""

class ImageOptimizer:
    """Main optimizer orchestrating all optimization strategies"""

    def __init__(
        self,
        dockerfile_path: Path,
        build_context: Path,
        strategy: Optional[OptimizationStrategy] = None
    ):
        self.dockerfile_path = dockerfile_path
        self.build_context = build_context
        self.strategy = strategy or OptimizationStrategy()

        self.parser = DockerfileParser(dockerfile_path)
        self.layer_analyzer = LayerAnalyzer()
        self.builder = MultiStageBuilder(self.strategy)
        self.cache_analyzer = CacheAnalyzer()

    def analyze(self, build_original: bool = True) -> Dict:
        """
        Analyze Dockerfile without optimizing

        Returns detailed report of optimization opportunities
        """
        # TODO: Parse Dockerfile
        # TODO: If build_original, build image and analyze layers
        # TODO: Run cache analysis
        # TODO: Return comprehensive report
        raise NotImplementedError

    def optimize(
        self,
        output_path: Path,
        runtime_dependencies: Optional[List[str]] = None,
        verify: bool = True
    ) -> OptimizationResult:
        """
        Optimize Dockerfile and measure results

        Steps:
        1. Build original image, measure size
        2. Parse and analyze Dockerfile
        3. Generate optimized Dockerfile
        4. Build optimized image
        5. Compare sizes and performance
        6. Optionally verify functionality

        Args:
            output_path: Where to write optimized Dockerfile
            runtime_dependencies: Packages needed at runtime
            verify: Run test command to verify functionality

        Returns:
            OptimizationResult with before/after metrics
        """
        # TODO: Build original image
        original_analysis = self.layer_analyzer.analyze_image("original:latest")

        # TODO: Parse Dockerfile
        stages = self.parser.parse()

        # TODO: Generate optimized Dockerfile
        optimized_content = self.builder.optimize_dockerfile(stages, runtime_dependencies)
        self.builder.generate_dockerfile(optimized_content, output_path)

        # TODO: Build optimized image
        # docker build -f <output_path> -t optimized:latest <context>

        # TODO: Analyze optimized image
        optimized_analysis = self.layer_analyzer.analyze_image("optimized:latest")

        # TODO: Compare results
        result = OptimizationResult(
            original_size_mb=original_analysis.total_size_mb,
            optimized_size_mb=optimized_analysis.total_size_mb,
            size_reduction_percent=(
                (original_analysis.total_size_mb - optimized_analysis.total_size_mb)
                / original_analysis.total_size_mb * 100
            ),
            original_layers=len(original_analysis.layers),
            optimized_layers=len(optimized_analysis.layers),
            estimated_build_speedup=1.5,  # TODO: Calculate from cache analysis
            optimizations_applied=[],  # TODO: Track applied optimizations
            warnings=[]
        )

        # TODO: Optionally verify functionality
        if verify:
            self._verify_functionality("original:latest", "optimized:latest")

        return result

    def _verify_functionality(self, original_image: str, optimized_image: str) -> None:
        """
        Verify optimized image works the same as original

        Strategies:
        1. Run smoke test command (e.g., python -c "import torch")
        2. Compare package versions (pip list, npm list)
        3. Run unit tests inside both containers
        4. Compare file checksums for critical files
        """
        # TODO: Run test commands in both images
        # TODO: Compare outputs
        # TODO: Raise exception if verification fails
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Orchestrate full optimization pipeline
- ✅ Build both original and optimized images
- ✅ Calculate accurate size reductions
- ✅ Verify optimized image functionality
- ✅ Generate comprehensive optimization report

---

### Task 6: CLI and Reporting (4-5 hours)

Build command-line interface and generate reports.

```python
# src/image_optimizer/cli.py

import click
from pathlib import Path
from typing import Optional
from .optimizer import ImageOptimizer
from .multistage_builder import OptimizationStrategy
from .reporter import Reporter

@click.group()
def cli():
    """Docker Image Optimizer - Reduce image sizes by 50%+"""
    pass

@cli.command()
@click.argument('dockerfile', type=click.Path(exists=True))
@click.option('--context', type=click.Path(exists=True), default='.',
              help='Build context directory')
@click.option('--build/--no-build', default=True,
              help='Build image for analysis')
def analyze(dockerfile: str, context: str, build: bool):
    """Analyze Dockerfile for optimization opportunities"""
    # TODO: Create ImageOptimizer
    # TODO: Run analysis
    # TODO: Print report
    raise NotImplementedError

@cli.command()
@click.argument('dockerfile', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output path for optimized Dockerfile')
@click.option('--context', type=click.Path(exists=True), default='.',
              help='Build context directory')
@click.option('--runtime-deps', '-r', multiple=True,
              help='Runtime dependencies (e.g., -r libgomp1 -r libpng16)')
@click.option('--multistage/--no-multistage', default=True,
              help='Use multi-stage builds')
@click.option('--slim-base/--no-slim-base', default=True,
              help='Use slim/alpine base images')
@click.option('--verify/--no-verify', default=True,
              help='Verify optimized image works')
@click.option('--report', type=click.Path(),
              help='Generate HTML report at path')
def optimize(
    dockerfile: str,
    output: str,
    context: str,
    runtime_deps: tuple,
    multistage: bool,
    slim_base: bool,
    verify: bool,
    report: Optional[str]
):
    """Optimize Dockerfile and generate smaller image"""
    strategy = OptimizationStrategy(
        use_multistage=multistage,
        use_slim_base=slim_base
    )

    optimizer = ImageOptimizer(
        dockerfile_path=Path(dockerfile),
        build_context=Path(context),
        strategy=strategy
    )

    result = optimizer.optimize(
        output_path=Path(output),
        runtime_dependencies=list(runtime_deps) if runtime_deps else None,
        verify=verify
    )

    click.echo(result)

    if report:
        reporter = Reporter()
        reporter.generate_html_report(result, Path(report))
        click.echo(f"\n📊 Report saved to: {report}")

@cli.command()
@click.argument('image')
@click.option('--output', '-o', type=click.Path(), default='layer_report.html',
              help='Output path for report')
def inspect(image: str, output: str):
    """Inspect layers of existing image"""
    # TODO: Analyze image layers
    # TODO: Generate visualization
    raise NotImplementedError

if __name__ == '__main__':
    cli()
```

```python
# src/image_optimizer/reporter.py

from pathlib import Path
from typing import Dict, List
from .optimizer import OptimizationResult
from .layer_analyzer import ImageAnalysis
import json

class Reporter:
    """Generate optimization reports"""

    def generate_html_report(self, result: OptimizationResult, output_path: Path) -> None:
        """
        Generate interactive HTML report with:
        - Size comparison chart
        - Layer breakdown
        - Optimization recommendations
        - Before/after Dockerfile diff
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Docker Image Optimization Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; }
        .metric { display: inline-block; margin: 20px; padding: 20px; background: #f0f0f0; border-radius: 8px; }
        .metric h3 { margin: 0; color: #666; font-size: 14px; }
        .metric .value { font-size: 32px; font-weight: bold; color: #333; }
        .success { color: #22c55e; }
        canvas { max-width: 600px; }
    </style>
</head>
<body>
    <h1>Docker Image Optimization Report</h1>

    <div class="metrics">
        <div class="metric">
            <h3>Size Reduction</h3>
            <div class="value success">{{ size_reduction }}%</div>
        </div>
        <div class="metric">
            <h3>Original Size</h3>
            <div class="value">{{ original_size }} MB</div>
        </div>
        <div class="metric">
            <h3>Optimized Size</h3>
            <div class="value">{{ optimized_size }} MB</div>
        </div>
    </div>

    <h2>Size Comparison</h2>
    <canvas id="sizeChart"></canvas>

    <h2>Optimizations Applied</h2>
    <ul>
        {% for opt in optimizations %}
        <li>{{ opt }}</li>
        {% endfor %}
    </ul>

    <script>
        // TODO: Render Chart.js visualization
    </script>
</body>
</html>
"""
        # TODO: Render template with result data
        # TODO: Write HTML to output_path
        raise NotImplementedError

    def generate_json_report(self, result: OptimizationResult, output_path: Path) -> None:
        """Generate machine-readable JSON report"""
        # TODO: Serialize result to JSON
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ CLI commands work: `analyze`, `optimize`, `inspect`
- ✅ Support all optimization flags
- ✅ Generate HTML reports with charts
- ✅ Generate JSON reports for CI/CD integration
- ✅ Clear error messages for invalid inputs

---

## Testing Requirements

### Unit Tests

```python
# tests/test_optimizer.py

def test_optimize_python_ml_image():
    """Test optimizing a typical ML Docker image"""
    optimizer = ImageOptimizer(
        dockerfile_path=Path("fixtures/dockerfiles/python_ml.Dockerfile"),
        build_context=Path("fixtures/context"),
        strategy=OptimizationStrategy(use_multistage=True, use_slim_base=True)
    )

    result = optimizer.optimize(
        output_path=Path("/tmp/Dockerfile.optimized"),
        runtime_dependencies=["libgomp1"],
        verify=False  # Skip verification in tests
    )

    assert result.size_reduction_percent > 50.0, "Should reduce size by >50%"
    assert result.optimized_layers < result.original_layers
    assert "Multi-stage build" in result.optimizations_applied

def test_cache_analysis():
    """Test build cache analysis"""
    analyzer = CacheAnalyzer()
    analysis = analyzer.analyze_build(
        dockerfile_path=Path("fixtures/dockerfiles/python_ml.Dockerfile"),
        build_context=Path("fixtures/context")
    )

    assert analysis.cache_hit_rate >= 0.0
    assert analysis.cache_hit_rate <= 100.0
    assert len(analysis.steps) > 0

def test_dockerfile_parsing():
    """Test Dockerfile parsing accuracy"""
    parser = DockerfileParser(Path("fixtures/dockerfiles/multistage.Dockerfile"))
    stages = parser.parse()

    assert len(stages) == 2
    assert stages[0].name == "builder"
    assert "python:3.11" in stages[0].base_image
```

### Integration Tests

```bash
# tests/integration/test_end_to_end.sh

#!/bin/bash
set -e

echo "Building test image..."
docker build -t test-original:latest -f fixtures/dockerfiles/python_ml.Dockerfile fixtures/context

echo "Running optimizer..."
python -m image_optimizer.cli optimize \
    fixtures/dockerfiles/python_ml.Dockerfile \
    --output /tmp/Dockerfile.optimized \
    --context fixtures/context \
    --runtime-deps libgomp1

echo "Building optimized image..."
docker build -t test-optimized:latest -f /tmp/Dockerfile.optimized fixtures/context

echo "Comparing sizes..."
ORIGINAL_SIZE=$(docker images test-original:latest --format "{{.Size}}")
OPTIMIZED_SIZE=$(docker images test-optimized:latest --format "{{.Size}}")

echo "Original:  $ORIGINAL_SIZE"
echo "Optimized: $OPTIMIZED_SIZE"

echo "Verifying functionality..."
docker run --rm test-optimized:latest python -c "import torch; print('PyTorch works!')"

echo "✅ Integration test passed!"
```

### Benchmark Tests

```python
# benchmarks/measure_optimization_time.py

import time
from pathlib import Path
from image_optimizer import ImageOptimizer, OptimizationStrategy

def benchmark_optimization():
    dockerfiles = [
        "fixtures/dockerfiles/python_ml.Dockerfile",
        "fixtures/dockerfiles/nodejs_api.Dockerfile",
        "fixtures/dockerfiles/java_service.Dockerfile"
    ]

    for dockerfile_path in dockerfiles:
        print(f"\nBenchmarking: {dockerfile_path}")

        optimizer = ImageOptimizer(
            dockerfile_path=Path(dockerfile_path),
            build_context=Path("fixtures/context"),
            strategy=OptimizationStrategy()
        )

        start = time.time()
        result = optimizer.optimize(
            output_path=Path(f"/tmp/{Path(dockerfile_path).stem}.optimized"),
            verify=False
        )
        duration = time.time() - start

        print(f"  Time: {duration:.1f}s")
        print(f"  Reduction: {result.size_reduction_percent:.1f}%")
        print(f"  Original: {result.original_size_mb:.1f} MB")
        print(f"  Optimized: {result.optimized_size_mb:.1f} MB")

if __name__ == "__main__":
    benchmark_optimization()
```

## Expected Results

After completing this exercise, you should achieve:

| Metric | Target | Measured |
|--------|--------|----------|
| **Size Reduction** | >50% | ________% |
| **Layer Count Reduction** | >30% | ________% |
| **Build Time (with cache)** | <2 min | ________ |
| **Analysis Time** | <5s | ________ |

**Example Optimization Results**:

```
Original Dockerfile (PyTorch ML Service):
- Base: python:3.11 (1.2 GB)
- + System packages (gcc, g++, make): +450 MB
- + PyTorch + deps: +2.1 GB
- + Application code: +15 MB
- + Cache (/var/cache, /root/.cache): +180 MB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 3.9 GB, 18 layers

Optimized Dockerfile (Multi-Stage):
- Builder stage: python:3.11 (builds wheels)
- Runtime stage: python:3.11-slim (320 MB)
- + PyTorch wheels (no build deps): +1.8 GB
- + Runtime lib (libgomp1): +2 MB
- + Application code: +15 MB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 2.1 GB, 8 layers

Size Reduction: 46% (1.8 GB saved)
Build Time: 8m → 45s (with cache)
```

## Validation

Submit the following for review:

1. **Complete implementation** of all 6 tasks
2. **Test suite** with >80% coverage
3. **Optimization report** for 3 different Dockerfile types:
   - Python ML service (PyTorch/TensorFlow)
   - Node.js API
   - Java microservice
4. **Benchmark results** showing:
   - Size reductions >50%
   - Build time improvements
   - Successful functionality verification
5. **Documentation**:
   - `DESIGN.md`: Architecture and design decisions
   - `OPTIMIZATION_GUIDE.md`: Best practices reference
   - `README.md`: Usage examples

## Bonus Challenges

1. **Distroless Support** (2-3h): Add Google distroless base image support for maximum security
2. **Layer Visualization** (3-4h): Generate interactive layer size diagrams (using D3.js or Plotly)
3. **GitHub Actions Integration** (2-3h): Create GitHub Action that comments PR with optimization suggestions
4. **Multi-Architecture Builds** (3-4h): Support ARM64 optimization for AWS Graviton instances
5. **Cost Calculator** (2-3h): Estimate ECR storage and data transfer cost savings

## Resources

- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker BuildKit](https://docs.docker.com/build/buildkit/)
- [Google Distroless Images](https://github.com/GoogleContainerTools/distroless)
- [Dive - Image Layer Analysis Tool](https://github.com/wagoodman/dive)
- [Docker Slim](https://github.com/slimtoolkit/slim)

## Deliverables Checklist

- [ ] `dockerfile_parser.py` - Parse Dockerfiles into AST
- [ ] `layer_analyzer.py` - Analyze image layers and sizes
- [ ] `multistage_builder.py` - Generate optimized multi-stage Dockerfiles
- [ ] `cache_analyzer.py` - Analyze build cache effectiveness
- [ ] `optimizer.py` - Main optimization orchestrator
- [ ] `reporter.py` - Generate HTML/JSON reports
- [ ] `cli.py` - Command-line interface
- [ ] Comprehensive test suite (>80% coverage)
- [ ] `DESIGN.md` - Architecture documentation
- [ ] `OPTIMIZATION_GUIDE.md` - Best practices
- [ ] 3 optimization reports (Python ML, Node.js, Java)
- [ ] Benchmark results showing >50% size reduction

---

**Estimated Completion Time**: 28-35 hours

**Skills Practiced**:
- Docker multi-stage builds
- Container image optimization
- Build cache analysis
- Dockerfile parsing and generation
- Performance benchmarking
- CLI tool development
