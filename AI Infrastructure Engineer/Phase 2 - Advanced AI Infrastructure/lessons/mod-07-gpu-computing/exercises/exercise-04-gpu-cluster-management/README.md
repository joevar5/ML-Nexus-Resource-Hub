# Exercise 04: GPU Cluster Management and Scheduling

**Estimated Time**: 34-42 hours
**Difficulty**: Advanced
**Prerequisites**: Python 3.9+, Kubernetes, NVIDIA DCGM, Slurm (optional), Ray

## Overview

Build a production-grade GPU cluster management system for ML workloads. Implement GPU scheduling, resource allocation, multi-tenancy, job queuing, and cost optimization. Handle GPU sharing, fractional GPU allocation, and heterogeneous GPU types (V100, A100, T4). This exercise teaches critical skills for operating large-scale ML infrastructure with expensive GPU resources.

In production ML platforms, GPU cluster management is essential for:
- **Resource Efficiency**: $30K/month GPU cluster, maximize utilization
- **Fair Sharing**: Multiple teams sharing limited GPU resources
- **Cost Attribution**: Track GPU usage per team/project
- **Job Scheduling**: Queue jobs, prioritize urgent workloads
- **Multi-Tenancy**: Isolate workloads, prevent interference

## Learning Objectives

By completing this exercise, you will:

1. **Build GPU resource scheduler** with priority queues
2. **Implement GPU sharing** (time-slicing, MIG, MPS)
3. **Track GPU utilization** with NVIDIA DCGM
4. **Build cost allocation** system for chargeback
5. **Implement job preemption** for urgent workloads
6. **Handle heterogeneous GPUs** (different types/generations)
7. **Monitor GPU health** and detect failures

## Business Context

**Real-World Scenario**: Your ML platform has 50 GPUs ($1.5M infrastructure) serving 5 teams:

- **Low utilization**: GPUs idle 60% of time (wasting $18K/month)
- **Resource conflicts**: Training jobs fight for GPUs, no priorities
- **No sharing**: Users reserve whole GPUs, use only 30%
- **No accountability**: Can't track which team uses how much
- **Queue management**: Manual coordination in Slack for GPU access
- **Heterogeneous cluster**: Mix of V100, A100, T4 (need smart scheduling)

Your task: Build GPU cluster manager that:
- Schedules jobs based on priority and resource needs
- Enables GPU sharing (multiple jobs per GPU)
- Tracks usage per team for cost allocation
- Achieves >80% GPU utilization
- Handles preemption for urgent jobs
- Monitors GPU health and failures

## Project Structure

```
exercise-04-gpu-cluster-management/
├── README.md
├── requirements.txt
├── docker-compose.yaml
├── config/
│   ├── scheduler_config.yaml
│   ├── gpu_types.yaml
│   └── team_quotas.yaml
├── src/
│   └── gpu_manager/
│       ├── __init__.py
│       ├── scheduler/
│       │   ├── __init__.py
│       │   ├── job_scheduler.py         # Main scheduler
│       │   ├── priority_queue.py        # Priority-based queuing
│       │   ├── gpu_allocator.py         # GPU allocation logic
│       │   └── preemption.py            # Job preemption
│       ├── sharing/
│       │   ├── __init__.py
│       │   ├── time_slicing.py          # Time-slicing scheduler
│       │   ├── mig_manager.py           # NVIDIA MIG support
│       │   └── mps_manager.py           # Multi-Process Service
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── dcgm_collector.py        # DCGM metrics
│       │   ├── utilization_tracker.py   # Track GPU usage
│       │   └── health_checker.py        # GPU health monitoring
│       ├── accounting/
│       │   ├── __init__.py
│       │   ├── usage_tracker.py         # Track usage per team
│       │   ├── cost_allocator.py        # Cost allocation
│       │   └── billing_report.py        # Generate reports
│       └── api/
│           ├── __init__.py
│           ├── job_api.py               # Job submission API
│           └── metrics_api.py           # Metrics API
├── kubernetes/
│   ├── gpu-operator.yaml               # NVIDIA GPU Operator
│   ├── device-plugin.yaml              # GPU device plugin
│   └── scheduler-deployment.yaml
├── tests/
│   ├── test_scheduler.py
│   ├── test_allocator.py
│   └── test_sharing.py
├── examples/
│   ├── submit_job.py
│   ├── query_usage.py
│   └── generate_report.py
└── docs/
    ├── DESIGN.md
    ├── SCHEDULING.md
    └── GPU_SHARING.md
```

## Requirements

### Functional Requirements

1. **Job Scheduling**:
   - Priority-based queue (high/medium/low)
   - Resource requests (GPU count, memory, CPU)
   - Gang scheduling (all-or-nothing allocation)
   - Backfill scheduling (fill gaps with small jobs)

2. **GPU Sharing**:
   - Time-slicing (multiple jobs share GPU over time)
   - MIG (Multi-Instance GPU for A100)
   - MPS (Multi-Process Service for training)
   - Fractional GPU allocation (0.5 GPU)

3. **Monitoring**:
   - GPU utilization (compute, memory)
   - Power consumption
   - Temperature
   - Error rates (ECC, XID errors)

4. **Cost Allocation**:
   - Track GPU-hours per team/project
   - Calculate costs based on GPU type
   - Generate monthly billing reports
   - Enforce quotas

5. **Preemption**:
   - Preempt low-priority jobs for high-priority
   - Checkpoint and resume preempted jobs
   - Fair sharing over time

### Non-Functional Requirements

- **Scheduling Latency**: <5 seconds from submission to allocation
- **Utilization Target**: >80% average GPU utilization
- **Fairness**: No team starved for >1 hour
- **Reliability**: Handle GPU failures gracefully

## Implementation Tasks

### Task 1: GPU Resource Scheduler (8-10 hours)

Build priority-based GPU scheduler.

```python
# src/gpu_manager/scheduler/job_scheduler.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import heapq

class Priority(Enum):
    """Job priority levels"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class GPURequirement:
    """GPU resource requirements"""
    count: int = 1
    memory_gb: int = 16
    gpu_type: Optional[str] = None  # "V100", "A100", "T4", or None (any)
    fractional: bool = False  # Allow fractional GPU

@dataclass
class Job:
    """GPU job definition"""
    job_id: str
    user: str
    team: str
    priority: Priority
    gpu_requirements: GPURequirement
    command: str
    created_at: datetime
    estimated_duration_hours: float

    def __lt__(self, other):
        """For priority queue ordering"""
        # Lower priority value = higher priority
        return self.priority.value < other.priority.value

@dataclass
class GPUNode:
    """Represents a GPU node in cluster"""
    node_id: str
    gpu_type: str
    gpu_count: int
    available_gpus: int
    memory_per_gpu_gb: int
    jobs: List[str]  # Job IDs running on this node

class GPUScheduler:
    """
    GPU job scheduler with priority queuing

    Scheduling algorithm:
    1. Sort jobs by priority
    2. For each job:
       a. Find nodes with required GPU type
       b. Check if sufficient GPUs available
       c. Allocate GPUs to job
       d. If no resources, queue job
    3. Backfill: Try to fit queued jobs in gaps
    """

    def __init__(self):
        self.job_queue: List[Job] = []  # Priority queue
        self.running_jobs: Dict[str, Job] = {}
        self.nodes: List[GPUNode] = []
        self.allocations: Dict[str, List[str]] = {}  # job_id -> [node_ids]

    def add_node(self, node: GPUNode):
        """Register GPU node with scheduler"""
        self.nodes.append(node)

    def submit_job(self, job: Job) -> str:
        """
        Submit job to scheduler

        Returns:
            Job status: "queued" or "scheduled"
        """
        # TODO: Try to schedule immediately
        allocated = self._try_allocate(job)

        if allocated:
            self.running_jobs[job.job_id] = job
            return "scheduled"
        else:
            # Add to priority queue
            heapq.heappush(self.job_queue, job)
            return "queued"

    def _try_allocate(self, job: Job) -> bool:
        """
        Try to allocate GPUs for job

        Returns:
            True if allocated, False otherwise
        """
        req = job.gpu_requirements

        # TODO: Find suitable nodes
        suitable_nodes = self._find_suitable_nodes(req)

        if not suitable_nodes:
            return False

        # TODO: Gang scheduling - allocate all GPUs or none
        total_available = sum(node.available_gpus for node in suitable_nodes)

        if total_available < req.count:
            return False

        # TODO: Allocate GPUs across nodes
        gpus_needed = req.count
        allocated_nodes = []

        for node in suitable_nodes:
            if gpus_needed == 0:
                break

            gpus_from_node = min(node.available_gpus, gpus_needed)
            node.available_gpus -= gpus_from_node
            node.jobs.append(job.job_id)
            allocated_nodes.append(node.node_id)
            gpus_needed -= gpus_from_node

        # TODO: Record allocation
        self.allocations[job.job_id] = allocated_nodes

        return True

    def _find_suitable_nodes(self, req: GPURequirement) -> List[GPUNode]:
        """
        Find nodes matching GPU requirements

        Filters:
        - GPU type (if specified)
        - Available GPUs
        - GPU memory
        """
        suitable = []

        for node in self.nodes:
            # Check GPU type
            if req.gpu_type and node.gpu_type != req.gpu_type:
                continue

            # Check GPU memory
            if node.memory_per_gpu_gb < req.memory_gb:
                continue

            # Check availability
            if node.available_gpus > 0:
                suitable.append(node)

        # Sort by available GPUs (descending)
        suitable.sort(key=lambda n: n.available_gpus, reverse=True)

        return suitable

    def complete_job(self, job_id: str):
        """
        Mark job as complete and free resources

        Triggers scheduling of queued jobs
        """
        if job_id not in self.running_jobs:
            return

        job = self.running_jobs.pop(job_id)
        allocated_nodes = self.allocations.pop(job_id)

        # TODO: Free GPUs
        for node_id in allocated_nodes:
            node = next(n for n in self.nodes if n.node_id == node_id)
            node.jobs.remove(job_id)
            node.available_gpus += 1  # Simplified - assumes 1 GPU per job

        # TODO: Try to schedule queued jobs
        self._schedule_queued_jobs()

    def _schedule_queued_jobs(self):
        """
        Try to schedule jobs from queue

        Uses backfilling to maximize utilization
        """
        # Sort queue by priority
        heapq.heapify(self.job_queue)

        scheduled = []

        # Try to schedule each queued job
        for job in self.job_queue:
            if self._try_allocate(job):
                self.running_jobs[job.job_id] = job
                scheduled.append(job)

        # Remove scheduled jobs from queue
        self.job_queue = [j for j in self.job_queue if j not in scheduled]

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get position of job in queue"""
        for i, job in enumerate(sorted(self.job_queue)):
            if job.job_id == job_id:
                return i
        return None

    def get_cluster_status(self) -> Dict:
        """
        Get cluster status

        Returns:
            {
                "total_gpus": 50,
                "available_gpus": 15,
                "utilization": 0.70,
                "queued_jobs": 5,
                "running_jobs": 12
            }
        """
        total_gpus = sum(node.gpu_count for node in self.nodes)
        available_gpus = sum(node.available_gpus for node in self.nodes)

        return {
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "utilization": (total_gpus - available_gpus) / total_gpus if total_gpus > 0 else 0,
            "queued_jobs": len(self.job_queue),
            "running_jobs": len(self.running_jobs)
        }
```

**Acceptance Criteria**:
- ✅ Priority-based scheduling
- ✅ Gang scheduling (all-or-nothing)
- ✅ Backfilling for utilization
- ✅ Queue position tracking
- ✅ Cluster status reporting

---

### Task 2: GPU Sharing Implementation (7-9 hours)

Implement GPU sharing mechanisms.

```python
# src/gpu_manager/sharing/time_slicing.py

from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta

@dataclass
class TimeSlice:
    """Time slice allocation for GPU sharing"""
    job_id: str
    gpu_id: str
    start_time: datetime
    duration_seconds: int
    percentage: float  # 0.0-1.0

class TimeSlicingScheduler:
    """
    GPU time-slicing scheduler

    Allows multiple jobs to share GPU by time-multiplexing.
    Each job gets exclusive GPU access during its time slice.

    Example:
    - Job A: 50% GPU (runs for 5s, waits 5s)
    - Job B: 50% GPU (runs for 5s, waits 5s)
    """

    def __init__(self, slice_duration_seconds: int = 10):
        self.slice_duration = slice_duration_seconds
        self.gpu_schedules: Dict[str, List[TimeSlice]] = {}

    def allocate_shared_gpu(
        self,
        job_id: str,
        gpu_id: str,
        percentage: float
    ) -> bool:
        """
        Allocate fractional GPU using time-slicing

        Args:
            percentage: Fraction of GPU time (0.0-1.0)

        Returns:
            True if allocated successfully
        """
        if gpu_id not in self.gpu_schedules:
            self.gpu_schedules[gpu_id] = []

        # TODO: Check if GPU has capacity
        current_allocation = sum(
            slice.percentage for slice in self.gpu_schedules[gpu_id]
        )

        if current_allocation + percentage > 1.0:
            return False

        # TODO: Create time slice
        slice = TimeSlice(
            job_id=job_id,
            gpu_id=gpu_id,
            start_time=datetime.utcnow(),
            duration_seconds=int(self.slice_duration * percentage),
            percentage=percentage
        )

        self.gpu_schedules[gpu_id].append(slice)
        return True

    def get_gpu_utilization(self, gpu_id: str) -> float:
        """Get current GPU allocation percentage"""
        if gpu_id not in self.gpu_schedules:
            return 0.0

        return sum(slice.percentage for slice in self.gpu_schedules[gpu_id])

    def release_gpu(self, job_id: str):
        """Release GPU allocation for job"""
        for gpu_id, slices in self.gpu_schedules.items():
            self.gpu_schedules[gpu_id] = [
                s for s in slices if s.job_id != job_id
            ]
```

```python
# src/gpu_manager/sharing/mig_manager.py

from typing import List, Dict
import subprocess

class MIGManager:
    """
    NVIDIA Multi-Instance GPU (MIG) Manager

    MIG allows partitioning A100 GPUs into up to 7 instances.
    Each instance has isolated memory and compute resources.

    MIG Profiles (A100 40GB):
    - 1g.5gb: 1/7 GPU, 5GB memory
    - 2g.10gb: 2/7 GPU, 10GB memory
    - 3g.20gb: 3/7 GPU, 20GB memory
    - 7g.40gb: Full GPU, 40GB memory
    """

    def __init__(self):
        self.gpu_instances: Dict[str, List[str]] = {}  # gpu_id -> [instance_ids]

    def enable_mig(self, gpu_id: str) -> bool:
        """
        Enable MIG mode on GPU

        Requires GPU reset, so should be done during cluster setup
        """
        try:
            # TODO: Enable MIG mode
            subprocess.run(
                ["nvidia-smi", "-i", gpu_id, "-mig", "1"],
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to enable MIG: {e}")
            return False

    def create_instance(
        self,
        gpu_id: str,
        profile: str = "1g.5gb"
    ) -> Optional[str]:
        """
        Create MIG instance

        Args:
            profile: MIG profile (1g.5gb, 2g.10gb, 3g.20gb, 7g.40gb)

        Returns:
            Instance ID if successful
        """
        try:
            # TODO: Create compute instance
            result = subprocess.run(
                ["nvidia-smi", "mig", "-cgi", profile, "-C"],
                capture_output=True,
                text=True,
                check=True
            )

            # TODO: Parse instance ID from output
            # Output format: "Successfully created GPU instance ID X ..."
            instance_id = self._parse_instance_id(result.stdout)

            # TODO: Track instance
            if gpu_id not in self.gpu_instances:
                self.gpu_instances[gpu_id] = []
            self.gpu_instances[gpu_id].append(instance_id)

            return instance_id

        except subprocess.CalledProcessError as e:
            print(f"Failed to create MIG instance: {e}")
            return None

    def destroy_instance(self, gpu_id: str, instance_id: str):
        """Destroy MIG instance"""
        try:
            subprocess.run(
                ["nvidia-smi", "mig", "-dci", "-ci", instance_id],
                check=True
            )

            self.gpu_instances[gpu_id].remove(instance_id)

        except subprocess.CalledProcessError as e:
            print(f"Failed to destroy instance: {e}")

    def list_instances(self, gpu_id: str) -> List[str]:
        """List all MIG instances on GPU"""
        return self.gpu_instances.get(gpu_id, [])

    def _parse_instance_id(self, output: str) -> str:
        """Parse instance ID from nvidia-smi output"""
        # TODO: Parse output
        import re
        match = re.search(r"ID (\d+)", output)
        return match.group(1) if match else None
```

**Acceptance Criteria**:
- ✅ Time-slicing implementation
- ✅ MIG support for A100
- ✅ Fractional GPU allocation
- ✅ GPU utilization tracking
- ✅ Instance lifecycle management

---

### Task 3: GPU Monitoring with DCGM (6-7 hours)

Monitor GPU health and utilization.

```python
# src/gpu_manager/monitoring/dcgm_collector.py

from typing import Dict, List
import subprocess
import json
from datetime import datetime
from dataclasses import dataclass

@dataclass
class GPUMetrics:
    """GPU metrics from DCGM"""
    gpu_id: str
    timestamp: datetime
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_watts: float
    sm_clock_mhz: int
    pcie_tx_mb: float
    pcie_rx_mb: float

class DCGMCollector:
    """
    Collect GPU metrics using NVIDIA DCGM

    DCGM (Data Center GPU Manager) provides low-overhead GPU monitoring.
    Metrics include: utilization, memory, temperature, power, errors.
    """

    def __init__(self):
        self.dcgm_initialized = self._initialize_dcgm()

    def _initialize_dcgm(self) -> bool:
        """Initialize DCGM"""
        try:
            # Check if DCGM is running
            subprocess.run(["dcgmi", "discovery", "-l"], check=True, capture_output=True)
            return True
        except:
            print("DCGM not available, falling back to nvidia-smi")
            return False

    def collect_metrics(self, gpu_id: str) -> GPUMetrics:
        """
        Collect current metrics for GPU

        Uses DCGM if available, otherwise nvidia-smi
        """
        if self.dcgm_initialized:
            return self._collect_with_dcgm(gpu_id)
        else:
            return self._collect_with_nvidia_smi(gpu_id)

    def _collect_with_dcgm(self, gpu_id: str) -> GPUMetrics:
        """
        Collect metrics using DCGM

        Command: dcgmi dmon -e <metric_ids> -i <gpu_id>
        """
        try:
            # TODO: Run dcgmi dmon
            result = subprocess.run(
                [
                    "dcgmi", "dmon",
                    "-e", "150,155,203,204,230,233",  # GPU util, mem, temp, power
                    "-i", gpu_id,
                    "-c", "1"  # Single sample
                ],
                capture_output=True,
                text=True,
                check=True
            )

            # TODO: Parse output
            metrics = self._parse_dcgm_output(result.stdout, gpu_id)
            return metrics

        except subprocess.CalledProcessError as e:
            print(f"DCGM collection failed: {e}")
            return None

    def _collect_with_nvidia_smi(self, gpu_id: str) -> GPUMetrics:
        """
        Fallback collection using nvidia-smi

        Command: nvidia-smi --query-gpu=... --format=csv
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    "-i", gpu_id
                ],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse CSV output
            values = result.stdout.strip().split(", ")

            return GPUMetrics(
                gpu_id=gpu_id,
                timestamp=datetime.utcnow(),
                utilization_percent=float(values[0]),
                memory_used_mb=float(values[1]),
                memory_total_mb=float(values[2]),
                temperature_c=float(values[3]),
                power_watts=float(values[4]),
                sm_clock_mhz=0,  # Not available via nvidia-smi query
                pcie_tx_mb=0,
                pcie_rx_mb=0
            )

        except Exception as e:
            print(f"nvidia-smi collection failed: {e}")
            return None

    def _parse_dcgm_output(self, output: str, gpu_id: str) -> GPUMetrics:
        """Parse DCGM dmon output"""
        # TODO: Parse DCGM output format
        # Format: GPU util mem temp power ...
        lines = output.strip().split('\n')
        if len(lines) < 2:
            return None

        data_line = lines[-1]  # Last line has data
        values = data_line.split()

        return GPUMetrics(
            gpu_id=gpu_id,
            timestamp=datetime.utcnow(),
            utilization_percent=float(values[1]),
            memory_used_mb=float(values[2]),
            memory_total_mb=16384,  # TODO: Get from GPU info
            temperature_c=float(values[3]),
            power_watts=float(values[4]),
            sm_clock_mhz=0,
            pcie_tx_mb=0,
            pcie_rx_mb=0
        )

    def detect_gpu_errors(self, gpu_id: str) -> List[str]:
        """
        Detect GPU errors (ECC, XID errors)

        Returns:
            List of error descriptions
        """
        errors = []

        try:
            # TODO: Query ECC errors
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=ecc.errors.corrected.volatile.total", "--format=csv,noheader", "-i", gpu_id],
                capture_output=True,
                text=True,
                check=True
            )

            ecc_errors = int(result.stdout.strip())
            if ecc_errors > 0:
                errors.append(f"ECC errors: {ecc_errors}")

            # TODO: Check XID errors from dmesg
            result = subprocess.run(
                ["dmesg", "|", "grep", "NVRM"],
                capture_output=True,
                text=True,
                shell=True
            )

            if "Xid" in result.stdout:
                errors.append("XID errors detected")

        except Exception as e:
            print(f"Error detection failed: {e}")

        return errors
```

**Acceptance Criteria**:
- ✅ Collect GPU metrics with DCGM
- ✅ Fallback to nvidia-smi
- ✅ Track utilization, memory, temp, power
- ✅ Detect ECC and XID errors
- ✅ Efficient polling (<1% overhead)

---

### Task 4: Cost Allocation System (5-6 hours)

Track GPU usage and allocate costs.

```python
# src/gpu_manager/accounting/usage_tracker.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List
import sqlite3

@dataclass
class UsageRecord:
    """Record of GPU usage"""
    job_id: str
    user: str
    team: str
    gpu_type: str
    gpu_count: float  # Fractional GPUs allowed
    start_time: datetime
    end_time: Optional[datetime]
    duration_hours: float

class UsageTracker:
    """
    Track GPU usage per team/user

    Stores usage records in SQLite database
    """

    def __init__(self, db_path: str = "gpu_usage.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create database tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                job_id TEXT PRIMARY KEY,
                user TEXT NOT NULL,
                team TEXT NOT NULL,
                gpu_type TEXT NOT NULL,
                gpu_count REAL NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_hours REAL
            )
        """)
        self.conn.commit()

    def start_usage(
        self,
        job_id: str,
        user: str,
        team: str,
        gpu_type: str,
        gpu_count: float
    ):
        """Record start of GPU usage"""
        self.conn.execute("""
            INSERT INTO usage (job_id, user, team, gpu_type, gpu_count, start_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (job_id, user, team, gpu_type, gpu_count, datetime.utcnow()))
        self.conn.commit()

    def end_usage(self, job_id: str):
        """Record end of GPU usage"""
        # TODO: Get start time
        cursor = self.conn.execute(
            "SELECT start_time FROM usage WHERE job_id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        if not row:
            return

        start_time = datetime.fromisoformat(row[0])
        end_time = datetime.utcnow()
        duration_hours = (end_time - start_time).total_seconds() / 3600

        # TODO: Update record
        self.conn.execute("""
            UPDATE usage
            SET end_time = ?, duration_hours = ?
            WHERE job_id = ?
        """, (end_time, duration_hours, job_id))
        self.conn.commit()

    def get_team_usage(
        self,
        team: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """
        Get GPU usage for team in time period

        Returns:
            {"V100": 123.5, "A100": 45.2}  # GPU-hours by type
        """
        cursor = self.conn.execute("""
            SELECT gpu_type, SUM(gpu_count * duration_hours)
            FROM usage
            WHERE team = ?
              AND start_time >= ?
              AND start_time < ?
              AND end_time IS NOT NULL
            GROUP BY gpu_type
        """, (team, start_date, end_date))

        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_user_usage(
        self,
        user: str,
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """Get total GPU-hours for user"""
        cursor = self.conn.execute("""
            SELECT SUM(gpu_count * duration_hours)
            FROM usage
            WHERE user = ?
              AND start_time >= ?
              AND start_time < ?
              AND end_time IS NOT NULL
        """, (user, start_date, end_date))

        result = cursor.fetchone()
        return result[0] if result[0] else 0.0
```

```python
# src/gpu_manager/accounting/cost_allocator.py

from typing import Dict
from datetime import datetime, timedelta
from .usage_tracker import UsageTracker

class CostAllocator:
    """
    Allocate GPU costs to teams

    Cost model:
    - V100: $3.00/hour
    - A100: $4.50/hour
    - T4: $1.50/hour
    """

    GPU_HOURLY_COST = {
        "V100": 3.00,
        "A100": 4.50,
        "T4": 1.50
    }

    def __init__(self, usage_tracker: UsageTracker):
        self.usage_tracker = usage_tracker

    def calculate_team_cost(
        self,
        team: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Calculate cost for team

        Returns:
            {
                "total_cost": 1234.56,
                "by_gpu_type": {
                    "V100": {"hours": 123.5, "cost": 370.50},
                    "A100": {"hours": 45.2, "cost": 203.40}
                }
            }
        """
        usage_by_type = self.usage_tracker.get_team_usage(team, start_date, end_date)

        by_gpu_type = {}
        total_cost = 0.0

        for gpu_type, hours in usage_by_type.items():
            cost = hours * self.GPU_HOURLY_COST.get(gpu_type, 0)
            by_gpu_type[gpu_type] = {
                "hours": hours,
                "cost": cost
            }
            total_cost += cost

        return {
            "total_cost": total_cost,
            "by_gpu_type": by_gpu_type
        }

    def generate_monthly_report(self, month: int, year: int) -> Dict:
        """
        Generate monthly cost report for all teams

        Returns:
            {
                "month": "2024-01",
                "teams": {
                    "ml-team": {"total_cost": 5000, ...},
                    "research-team": {"total_cost": 3000, ...}
                },
                "total": 8000
            }
        """
        # TODO: Get all teams
        # TODO: Calculate cost per team
        # TODO: Generate report
        pass
```

**Acceptance Criteria**:
- ✅ Track GPU usage per job
- ✅ Calculate costs per team
- ✅ Generate monthly reports
- ✅ Support fractional GPUs
- ✅ Historical usage queries

---

### Task 5: Job Preemption (4-5 hours)

Implement preemption for urgent jobs.

```python
# src/gpu_manager/scheduler/preemption.py

from typing import List, Optional
from dataclasses import dataclass
from .job_scheduler import Job, Priority

@dataclass
class PreemptionPolicy:
    """Policy for job preemption"""
    allow_preemption: bool = True
    max_preemptions_per_job: int = 3
    preempt_priority_threshold: Priority = Priority.MEDIUM
    checkpoint_before_preempt: bool = True

class PreemptionManager:
    """
    Manage job preemption

    Preemption scenarios:
    1. High-priority job arrives, no GPUs available
    2. Preempt lowest-priority running job
    3. Checkpoint preempted job (if supported)
    4. Resume preempted job later
    """

    def __init__(self, policy: PreemptionPolicy):
        self.policy = policy
        self.preemption_count: Dict[str, int] = {}

    def can_preempt(self, job: Job) -> bool:
        """Check if job can be preempted"""
        # Don't preempt high-priority jobs
        if job.priority.value <= self.policy.preempt_priority_threshold.value:
            return False

        # Don't preempt if already preempted too many times
        if self.preemption_count.get(job.job_id, 0) >= self.policy.max_preemptions_per_job:
            return False

        return self.policy.allow_preemption

    def find_preemption_candidate(
        self,
        running_jobs: Dict[str, Job],
        required_gpus: int
    ) -> Optional[Job]:
        """
        Find job to preempt

        Selects lowest-priority job with enough GPUs
        """
        # Sort by priority (lowest first)
        candidates = sorted(
            running_jobs.values(),
            key=lambda j: j.priority.value,
            reverse=True
        )

        for job in candidates:
            if not self.can_preempt(job):
                continue

            if job.gpu_requirements.count >= required_gpus:
                return job

        return None

    def preempt_job(self, job: Job) -> bool:
        """
        Preempt running job

        Steps:
        1. Checkpoint job (if supported)
        2. Kill job
        3. Free GPUs
        4. Re-queue job
        """
        # TODO: Checkpoint job
        if self.policy.checkpoint_before_preempt:
            checkpoint_success = self._checkpoint_job(job)
            if not checkpoint_success:
                print(f"Warning: Failed to checkpoint job {job.job_id}")

        # TODO: Kill job
        self._kill_job(job)

        # TODO: Track preemption count
        self.preemption_count[job.job_id] = self.preemption_count.get(job.job_id, 0) + 1

        return True

    def _checkpoint_job(self, job: Job) -> bool:
        """
        Create checkpoint of job state

        For ML training jobs, save model checkpoint
        """
        # TODO: Implement checkpointing
        # This is job-specific and requires integration with training framework
        return False

    def _kill_job(self, job: Job):
        """Kill running job"""
        # TODO: Send termination signal to job
        import signal
        import os
        # os.kill(job.pid, signal.SIGTERM)
        pass
```

**Acceptance Criteria**:
- ✅ Find preemption candidates
- ✅ Preempt low-priority jobs
- ✅ Checkpoint before preemption
- ✅ Track preemption count
- ✅ Prevent excessive preemptions

---

## Testing Requirements

```python
def test_gpu_scheduling():
    """Test GPU scheduler"""
    scheduler = GPUScheduler()

    # Add nodes
    scheduler.add_node(GPUNode(
        node_id="node-1",
        gpu_type="V100",
        gpu_count=4,
        available_gpus=4,
        memory_per_gpu_gb=16
    ))

    # Submit job
    job = Job(
        job_id="job-1",
        user="alice",
        team="ml-team",
        priority=Priority.HIGH,
        gpu_requirements=GPURequirement(count=2),
        command="python train.py",
        created_at=datetime.utcnow(),
        estimated_duration_hours=2.0
    )

    status = scheduler.submit_job(job)
    assert status == "scheduled"

    # Check allocation
    cluster_status = scheduler.get_cluster_status()
    assert cluster_status['available_gpus'] == 2
```

## Expected Results

| Metric | Target | Measured |
|--------|--------|----------|
| **GPU Utilization** | >80% | ________% |
| **Scheduling Latency** | <5s | ________s |
| **Cost Tracking Accuracy** | 100% | ________% |

## Validation

Submit:
1. GPU scheduler implementation
2. GPU sharing (time-slicing, MIG)
3. DCGM monitoring integration
4. Cost allocation system
5. Preemption manager
6. Test suite
7. Documentation

## Resources

- [NVIDIA DCGM](https://developer.nvidia.com/dcgm)
- [NVIDIA MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Ray for GPU Scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)

---

**Estimated Completion Time**: 34-42 hours

**Skills Practiced**:
- GPU cluster management
- Resource scheduling
- GPU sharing (MIG, time-slicing)
- NVIDIA DCGM monitoring
- Cost allocation
- Job preemption
- Multi-tenancy
