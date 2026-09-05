# Exercise 06: Container Registry Manager

**Estimated Time**: 32-40 hours
**Difficulty**: Advanced
**Prerequisites**: Docker, Python 3.9+, AWS/GCP/Azure CLI, Docker Registry HTTP API

## Overview

Build a production-grade container registry management system that synchronizes images across multiple registries (ECR, GCR, ACR, Docker Hub), implements automated image promotion pipelines (dev → staging → prod), enforces retention policies, and provides audit trails. This exercise teaches multi-cloud registry operations, image lifecycle management, and automated deployment workflows.

In production ML infrastructure, registry management is critical for:
- **Multi-Region Availability**: Replicate images to reduce cross-region pull latency
- **Disaster Recovery**: Backup images across cloud providers
- **Compliance**: Enforce retention policies and maintain audit logs
- **Security**: Scan images before promotion, prevent unauthorized access
- **Cost Optimization**: Delete unused images, implement lifecycle policies

## Learning Objectives

By completing this exercise, you will:

1. **Interact with Docker Registry HTTP API v2** for image metadata and manifests
2. **Sync images across registries** (ECR ↔ GCR ↔ ACR ↔ Docker Hub)
3. **Implement promotion pipelines** with approval workflows
4. **Enforce retention policies** based on age, tags, and usage
5. **Track image lineage** with tagging and metadata
6. **Generate audit reports** for compliance and security
7. **Optimize registry costs** by identifying and removing unused images

## Business Context

**Real-World Scenario**: Your ML platform runs across AWS (us-east-1, eu-west-1), GCP (us-central1), and Azure (eastus). Current challenges:

- **Slow deployments**: Pulling 2.5 GB ML images from us-east-1 ECR to eu-west-1 EKS takes 8 minutes
- **No disaster recovery**: Single registry failure blocks all deployments
- **Uncontrolled costs**: 4,500 images in ECR ($450/month), 70% unused
- **Manual promotions**: Engineers manually tag and push images through environments (error-prone)
- **No audit trail**: Can't track which images were deployed when and by whom

Your task: Build a registry manager that:
- Syncs images to regional registries (reduce pull time to <1 min)
- Implements automated dev → staging → prod promotion with approvals
- Deletes images older than 90 days with <5 pulls
- Maintains complete audit log of all image operations

## Project Structure

```
exercise-06-registry-manager/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Poetry/pip-tools config
├── config/
│   ├── registries.yaml                # Registry configurations
│   ├── promotion_policy.yaml          # Promotion rules
│   └── retention_policy.yaml          # Retention rules
├── src/
│   └── registry_manager/
│       ├── __init__.py
│       ├── registry/
│       │   ├── __init__.py
│       │   ├── base.py                # Abstract registry interface
│       │   ├── ecr.py                 # AWS ECR implementation
│       │   ├── gcr.py                 # Google GCR implementation
│       │   ├── acr.py                 # Azure ACR implementation
│       │   └── dockerhub.py           # Docker Hub implementation
│       ├── sync.py                    # Image synchronization
│       ├── promotion.py               # Promotion pipeline
│       ├── retention.py               # Retention policy enforcement
│       ├── audit.py                   # Audit logging
│       ├── metadata.py                # Image metadata tracking
│       ├── cost_analyzer.py           # Cost optimization
│       └── cli.py                     # Command-line interface
├── tests/
│   ├── test_registry_ecr.py
│   ├── test_registry_gcr.py
│   ├── test_registry_acr.py
│   ├── test_sync.py
│   ├── test_promotion.py
│   ├── test_retention.py
│   └── fixtures/
│       ├── sample_manifests/          # Docker image manifests
│       └── mock_responses/            # API response mocks
├── examples/
│   ├── sync_to_all_regions.py
│   ├── promote_to_production.py
│   └── cleanup_old_images.py
├── benchmarks/
│   ├── measure_sync_performance.sh
│   └── compare_pull_latency.sh
└── docs/
    ├── DESIGN.md                      # Architecture decisions
    ├── REGISTRY_API.md                # Registry API reference
    └── PROMOTION_WORKFLOW.md          # Promotion best practices
```

## Requirements

### Functional Requirements

Your registry manager must:

1. **Registry Operations**:
   - List repositories and images
   - Get image manifests and metadata
   - Tag images (create aliases)
   - Delete images and tags
   - Copy images between registries

2. **Synchronization**:
   - Sync specific images to target registries
   - Sync all images with specific tags (e.g., `latest`, `prod-*`)
   - Verify image integrity (checksum validation)
   - Handle concurrent syncs (rate limiting)

3. **Promotion Pipeline**:
   - Promote images from dev → staging → prod
   - Require approvals for production promotions
   - Run security scans before promotion
   - Tag images with environment labels
   - Rollback to previous versions

4. **Retention Policies**:
   - Delete images older than N days
   - Keep last N versions of each image
   - Preserve images with specific tags (e.g., `prod-*`, `stable`)
   - Delete images with <N pulls in last M days

5. **Audit and Reporting**:
   - Log all image operations (who, what, when)
   - Generate compliance reports
   - Track image lineage (which commit/build produced image)
   - Cost analysis (storage costs per registry)

### Non-Functional Requirements

- **Performance**: Sync 1 GB image in <3 minutes (within same cloud region)
- **Reliability**: Retry failed operations with exponential backoff
- **Idempotency**: Safe to re-run sync/promotion operations
- **Security**: Use IAM roles, never hardcode credentials
- **Observability**: Structured logging with correlation IDs

## Implementation Tasks

### Task 1: Registry Abstraction (6-7 hours)

Build abstract interface and implementations for each registry type.

```python
# src/registry_manager/registry/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class ImageManifest:
    """Docker image manifest (OCI format)"""
    digest: str  # sha256:abc123...
    media_type: str  # application/vnd.docker.distribution.manifest.v2+json
    size_bytes: int
    config: Dict  # Image configuration
    layers: List[Dict]  # Layer digests and sizes

@dataclass
class ImageMetadata:
    """Image metadata from registry"""
    repository: str  # my-ml-model
    tag: str  # v1.2.3
    digest: str  # sha256:abc123...
    created_at: datetime
    size_bytes: int
    pushed_at: Optional[datetime] = None
    pull_count: int = 0
    labels: Dict[str, str] = None

    @property
    def full_name(self) -> str:
        return f"{self.repository}:{self.tag}"

class RegistryInterface(ABC):
    """Abstract interface for container registries"""

    @abstractmethod
    def list_repositories(self) -> List[str]:
        """List all repositories in registry"""
        raise NotImplementedError

    @abstractmethod
    def list_tags(self, repository: str) -> List[str]:
        """List all tags for a repository"""
        raise NotImplementedError

    @abstractmethod
    def get_manifest(self, repository: str, tag: str) -> ImageManifest:
        """Get image manifest"""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self, repository: str, tag: str) -> ImageMetadata:
        """Get image metadata (size, created date, etc.)"""
        raise NotImplementedError

    @abstractmethod
    def tag_image(self, repository: str, source_tag: str, target_tag: str) -> None:
        """Create a new tag pointing to existing image"""
        raise NotImplementedError

    @abstractmethod
    def delete_image(self, repository: str, tag: str) -> None:
        """Delete image tag"""
        raise NotImplementedError

    @abstractmethod
    def delete_by_digest(self, repository: str, digest: str) -> None:
        """Delete image by digest (deletes all tags pointing to it)"""
        raise NotImplementedError

    @abstractmethod
    def image_exists(self, repository: str, tag: str) -> bool:
        """Check if image exists"""
        raise NotImplementedError
```

```python
# src/registry_manager/registry/ecr.py

import boto3
from typing import List, Dict, Optional
from datetime import datetime
from .base import RegistryInterface, ImageManifest, ImageMetadata

class ECRRegistry(RegistryInterface):
    """AWS Elastic Container Registry implementation"""

    def __init__(self, region: str, registry_id: Optional[str] = None):
        """
        Initialize ECR client

        Args:
            region: AWS region (e.g., us-east-1)
            registry_id: AWS account ID (defaults to caller's account)
        """
        self.region = region
        self.registry_id = registry_id
        self.client = boto3.client('ecr', region_name=region)

    def list_repositories(self) -> List[str]:
        """
        List all ECR repositories

        Uses: ecr.describe_repositories()
        """
        # TODO: Paginate through all repositories
        # TODO: Extract repository names
        raise NotImplementedError

    def list_tags(self, repository: str) -> List[str]:
        """
        List all image tags in repository

        Uses: ecr.list_images(repositoryName=...)
        Filter imageIds by imageTag (not digest-only references)
        """
        # TODO: Call list_images API
        # TODO: Extract tags (filter out None/digest-only)
        # TODO: Sort tags by push date (newest first)
        raise NotImplementedError

    def get_manifest(self, repository: str, tag: str) -> ImageManifest:
        """
        Get image manifest

        Uses: ecr.batch_get_image(imageIds=[{'imageTag': tag}])
        Parse imageManifest JSON
        """
        # TODO: Call batch_get_image
        # TODO: Parse manifest JSON
        # TODO: Extract digest, layers, config
        raise NotImplementedError

    def get_metadata(self, repository: str, tag: str) -> ImageMetadata:
        """
        Get image metadata

        Uses: ecr.describe_images(imageIds=[{'imageTag': tag}])
        Returns imageSizeInBytes, imagePushedAt, etc.
        """
        # TODO: Call describe_images
        # TODO: Extract metadata
        # TODO: Get pull count from CloudWatch metrics (if available)
        raise NotImplementedError

    def tag_image(self, repository: str, source_tag: str, target_tag: str) -> None:
        """
        Create new tag pointing to existing image

        Uses: ecr.put_image(repositoryName, imageManifest, imageTag)
        Get manifest from source_tag, put with target_tag
        """
        # TODO: Get manifest of source_tag
        # TODO: Put manifest with target_tag
        raise NotImplementedError

    def delete_image(self, repository: str, tag: str) -> None:
        """
        Delete image tag

        Uses: ecr.batch_delete_image(imageIds=[{'imageTag': tag}])
        """
        # TODO: Call batch_delete_image
        raise NotImplementedError

    def delete_by_digest(self, repository: str, digest: str) -> None:
        """Delete image by digest"""
        # TODO: Call batch_delete_image with imageDigest
        raise NotImplementedError

    def image_exists(self, repository: str, tag: str) -> bool:
        """Check if image exists"""
        try:
            self.get_metadata(repository, tag)
            return True
        except:
            return False
```

**Similar implementations needed for**:

```python
# src/registry_manager/registry/gcr.py
class GCRRegistry(RegistryInterface):
    """
    Google Container Registry implementation

    Uses Google Artifact Registry API or Docker Registry HTTP API v2
    Authentication via: gcloud auth print-access-token
    """
    # TODO: Implement using google-cloud-artifact-registry SDK

# src/registry_manager/registry/acr.py
class ACRRegistry(RegistryInterface):
    """
    Azure Container Registry implementation

    Uses Azure SDK or Docker Registry HTTP API v2
    Authentication via: az acr login
    """
    # TODO: Implement using azure-mgmt-containerregistry SDK

# src/registry_manager/registry/dockerhub.py
class DockerHubRegistry(RegistryInterface):
    """
    Docker Hub implementation

    Uses Docker Hub API v2
    Authentication via: username/password or access token
    """
    # TODO: Implement using requests + Docker Hub API
```

**Acceptance Criteria**:
- ✅ ECR implementation with all methods working
- ✅ GCR implementation with all methods working
- ✅ ACR implementation with all methods working
- ✅ Docker Hub implementation (bonus)
- ✅ All registries pass common interface tests
- ✅ Proper error handling and retries

**Test Example**:
```python
def test_ecr_list_repositories():
    ecr = ECRRegistry(region="us-east-1")
    repos = ecr.list_repositories()
    assert isinstance(repos, list)
    assert "my-ml-model" in repos

def test_ecr_get_metadata():
    ecr = ECRRegistry(region="us-east-1")
    metadata = ecr.get_metadata("my-ml-model", "v1.2.3")
    assert metadata.tag == "v1.2.3"
    assert metadata.size_bytes > 0
    assert metadata.digest.startswith("sha256:")
```

---

### Task 2: Image Synchronization (6-7 hours)

Implement cross-registry image synchronization.

```python
# src/registry_manager/sync.py

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import subprocess
import hashlib
from .registry.base import RegistryInterface, ImageMetadata

@dataclass
class SyncJob:
    """Represents an image sync operation"""
    source_registry: str  # us-east-1-ecr
    target_registry: str  # eu-west-1-ecr
    repository: str  # my-ml-model
    tag: str  # v1.2.3
    verify_checksum: bool = True

@dataclass
class SyncResult:
    """Result of sync operation"""
    job: SyncJob
    success: bool
    duration_seconds: float
    size_bytes: int
    error: Optional[str] = None

    def __str__(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        return f"{status} | {self.job.repository}:{self.job.tag} | {self.size_bytes / 1e9:.2f} GB in {self.duration_seconds:.1f}s"

class ImageSynchronizer:
    """Synchronize images between registries"""

    def __init__(self, registries: Dict[str, RegistryInterface]):
        """
        Args:
            registries: {
                "us-east-1-ecr": ECRRegistry(region="us-east-1"),
                "eu-west-1-ecr": ECRRegistry(region="eu-west-1"),
                "us-central1-gcr": GCRRegistry(project="my-project", location="us-central1")
            }
        """
        self.registries = registries

    def sync_image(self, job: SyncJob) -> SyncResult:
        """
        Sync single image from source to target registry

        Strategy:
        1. Check if image already exists in target (skip if digest matches)
        2. Pull image from source using `docker pull`
        3. Re-tag for target registry
        4. Push to target using `docker push`
        5. Verify digest matches
        6. Clean up local image

        Alternative for cloud-native sync (faster):
        - ECR to ECR: Use ECR replication rules or `skopeo copy`
        - GCR to GCR: Use `gcloud container images add-tag`
        """
        import time
        start = time.time()

        try:
            source_reg = self.registries[job.source_registry]
            target_reg = self.registries[job.target_registry]

            # TODO: Get source image metadata
            source_metadata = source_reg.get_metadata(job.repository, job.tag)

            # TODO: Check if already exists in target
            if target_reg.image_exists(job.repository, job.tag):
                target_metadata = target_reg.get_metadata(job.repository, job.tag)
                if target_metadata.digest == source_metadata.digest:
                    return SyncResult(
                        job=job,
                        success=True,
                        duration_seconds=time.time() - start,
                        size_bytes=0,  # No transfer needed
                        error="Image already exists with same digest"
                    )

            # TODO: Pull from source
            source_image = f"{self._get_registry_url(job.source_registry)}/{job.repository}:{job.tag}"
            self._docker_pull(source_image)

            # TODO: Re-tag for target
            target_image = f"{self._get_registry_url(job.target_registry)}/{job.repository}:{job.tag}"
            self._docker_tag(source_image, target_image)

            # TODO: Push to target
            self._docker_push(target_image)

            # TODO: Verify digest
            if job.verify_checksum:
                target_metadata = target_reg.get_metadata(job.repository, job.tag)
                if target_metadata.digest != source_metadata.digest:
                    raise ValueError(f"Digest mismatch: {source_metadata.digest} != {target_metadata.digest}")

            # TODO: Clean up local images
            self._docker_rmi(source_image)
            self._docker_rmi(target_image)

            return SyncResult(
                job=job,
                success=True,
                duration_seconds=time.time() - start,
                size_bytes=source_metadata.size_bytes
            )

        except Exception as e:
            return SyncResult(
                job=job,
                success=False,
                duration_seconds=time.time() - start,
                size_bytes=0,
                error=str(e)
            )

    def sync_repository(
        self,
        repository: str,
        source_registry: str,
        target_registries: List[str],
        tag_filter: Optional[str] = None
    ) -> List[SyncResult]:
        """
        Sync all tags of a repository to multiple target registries

        Args:
            repository: Repository name (e.g., "my-ml-model")
            source_registry: Source registry key
            target_registries: List of target registry keys
            tag_filter: Regex pattern (e.g., "v\\d+\\.\\d+\\.\\d+" for semantic versions)

        Returns:
            List of SyncResults for each tag
        """
        # TODO: List all tags in source repository
        # TODO: Filter tags by pattern if specified
        # TODO: Create SyncJob for each (tag, target_registry) combination
        # TODO: Execute syncs (optionally in parallel using ThreadPoolExecutor)
        # TODO: Return results
        raise NotImplementedError

    def _get_registry_url(self, registry_key: str) -> str:
        """Get registry URL for docker commands"""
        # TODO: Extract registry URL from registry object
        # ECR: 123456789012.dkr.ecr.us-east-1.amazonaws.com
        # GCR: gcr.io/my-project
        # ACR: myregistry.azurecr.io
        raise NotImplementedError

    def _docker_pull(self, image: str) -> None:
        """Pull image using docker CLI"""
        # TODO: Run: docker pull <image>
        # TODO: Handle authentication (assume already logged in via aws ecr get-login-password, etc.)
        raise NotImplementedError

    def _docker_tag(self, source: str, target: str) -> None:
        """Tag image"""
        # TODO: Run: docker tag <source> <target>
        raise NotImplementedError

    def _docker_push(self, image: str) -> None:
        """Push image"""
        # TODO: Run: docker push <image>
        raise NotImplementedError

    def _docker_rmi(self, image: str) -> None:
        """Remove local image"""
        # TODO: Run: docker rmi <image>
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Sync single image between registries
- ✅ Sync all tags in repository
- ✅ Skip sync if image already exists with same digest
- ✅ Verify image integrity with checksums
- ✅ Parallel sync for better performance
- ✅ Proper cleanup of local images

---

### Task 3: Promotion Pipeline (7-8 hours)

Implement automated image promotion workflow.

```python
# src/registry_manager/promotion.py

from dataclasses import dataclass
from typing import List, Optional, Callable
from enum import Enum
from datetime import datetime
from .registry.base import RegistryInterface
from .sync import ImageSynchronizer, SyncJob

class Environment(Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

@dataclass
class PromotionPolicy:
    """Rules for promoting images"""
    from_env: Environment
    to_env: Environment
    require_approval: bool
    require_security_scan: bool
    require_tests: bool
    auto_rollback_on_failure: bool

@dataclass
class PromotionRequest:
    """Request to promote an image"""
    repository: str
    tag: str
    from_env: Environment
    to_env: Environment
    approved_by: Optional[str] = None
    reason: Optional[str] = None

@dataclass
class PromotionResult:
    """Result of promotion"""
    request: PromotionRequest
    success: bool
    new_tag: str  # e.g., prod-v1.2.3-20240125
    timestamp: datetime
    error: Optional[str] = None

class PromotionPipeline:
    """Manage image promotion across environments"""

    def __init__(
        self,
        registries: Dict[str, RegistryInterface],
        synchronizer: ImageSynchronizer,
        policies: List[PromotionPolicy]
    ):
        self.registries = registries
        self.synchronizer = synchronizer
        self.policies = {(p.from_env, p.to_env): p for p in policies}

    def promote(
        self,
        request: PromotionRequest,
        security_scan_fn: Optional[Callable] = None,
        test_fn: Optional[Callable] = None
    ) -> PromotionResult:
        """
        Promote image from one environment to another

        Workflow:
        1. Validate promotion is allowed (check policy)
        2. Check approval if required
        3. Run security scan if required
        4. Run tests if required
        5. Tag image with environment prefix
        6. Sync to target environment registries
        7. Record promotion in audit log
        8. Optionally rollback on failure

        Args:
            request: Promotion request
            security_scan_fn: Function to run security scan
            test_fn: Function to run tests

        Returns:
            PromotionResult
        """
        try:
            # TODO: Get promotion policy
            policy = self.policies.get((request.from_env, request.to_env))
            if not policy:
                raise ValueError(f"No policy for {request.from_env} → {request.to_env}")

            # TODO: Check approval
            if policy.require_approval and not request.approved_by:
                raise ValueError("Approval required for this promotion")

            # TODO: Run security scan
            if policy.require_security_scan:
                if not security_scan_fn:
                    raise ValueError("Security scan required but no scan function provided")
                scan_result = security_scan_fn(request.repository, request.tag)
                if not scan_result.passed:
                    raise ValueError(f"Security scan failed: {scan_result.message}")

            # TODO: Run tests
            if policy.require_tests:
                if not test_fn:
                    raise ValueError("Tests required but no test function provided")
                test_result = test_fn(request.repository, request.tag)
                if not test_result.passed:
                    raise ValueError(f"Tests failed: {test_result.message}")

            # TODO: Generate new tag with environment prefix
            new_tag = self._generate_env_tag(request.to_env, request.tag)

            # TODO: Tag image in source registry
            source_registry = self._get_registry_for_env(request.from_env)
            source_registry.tag_image(request.repository, request.tag, new_tag)

            # TODO: Sync to target environment registries
            target_registries = self._get_registries_for_env(request.to_env)
            for target_reg_key in target_registries:
                sync_job = SyncJob(
                    source_registry=self._get_registry_key_for_env(request.from_env),
                    target_registry=target_reg_key,
                    repository=request.repository,
                    tag=new_tag
                )
                sync_result = self.synchronizer.sync_image(sync_job)
                if not sync_result.success:
                    raise Exception(f"Sync failed: {sync_result.error}")

            # TODO: Record in audit log
            self._log_promotion(request, new_tag)

            return PromotionResult(
                request=request,
                success=True,
                new_tag=new_tag,
                timestamp=datetime.now()
            )

        except Exception as e:
            # TODO: Rollback if configured
            if policy.auto_rollback_on_failure:
                self._rollback_promotion(request)

            return PromotionResult(
                request=request,
                success=False,
                new_tag="",
                timestamp=datetime.now(),
                error=str(e)
            )

    def rollback(self, repository: str, environment: Environment, to_tag: str) -> PromotionResult:
        """
        Rollback to previous image version

        Steps:
        1. Find current production tag
        2. Verify rollback target exists
        3. Re-tag rollback target as current
        4. Sync to all environment registries
        """
        # TODO: Implement rollback logic
        raise NotImplementedError

    def get_promotion_history(self, repository: str, limit: int = 100) -> List[Dict]:
        """Get promotion history for repository"""
        # TODO: Read from audit log
        raise NotImplementedError

    def _generate_env_tag(self, env: Environment, original_tag: str) -> str:
        """
        Generate environment-specific tag

        Examples:
        - dev-v1.2.3-20240125-abc123
        - staging-v1.2.3-20240126-def456
        - prod-v1.2.3-20240127-ghi789
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{env.value}-{original_tag}-{timestamp}"

    def _get_registry_for_env(self, env: Environment) -> RegistryInterface:
        """Get primary registry for environment"""
        # TODO: Map environment to registry
        raise NotImplementedError

    def _get_registries_for_env(self, env: Environment) -> List[str]:
        """Get all registries for environment"""
        # TODO: Return list of registry keys for this environment
        raise NotImplementedError

    def _log_promotion(self, request: PromotionRequest, new_tag: str) -> None:
        """Log promotion to audit trail"""
        # TODO: Write to audit log (database or file)
        raise NotImplementedError

    def _rollback_promotion(self, request: PromotionRequest) -> None:
        """Rollback failed promotion"""
        # TODO: Implement rollback
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Promote images between environments
- ✅ Enforce approval requirements
- ✅ Integrate security scans before promotion
- ✅ Run tests before promotion
- ✅ Generate environment-specific tags
- ✅ Rollback on failure
- ✅ Maintain promotion audit log

---

### Task 4: Retention Policy Engine (5-6 hours)

Implement automated image cleanup based on retention policies.

```python
# src/registry_manager/retention.py

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
from .registry.base import RegistryInterface, ImageMetadata

@dataclass
class RetentionRule:
    """Rule for image retention"""
    name: str
    max_age_days: Optional[int] = None  # Delete if older than N days
    keep_last_n: Optional[int] = None  # Keep last N versions
    min_pull_count: Optional[int] = None  # Delete if <N pulls in last 30 days
    preserve_tags: List[str] = None  # Never delete these tags (e.g., ["prod-*", "latest"])

    def should_delete(self, metadata: ImageMetadata, current_time: datetime) -> bool:
        """
        Determine if image should be deleted based on this rule

        Returns True if image violates ANY condition
        """
        # TODO: Check if tag is preserved
        if self.preserve_tags:
            for pattern in self.preserve_tags:
                if self._tag_matches_pattern(metadata.tag, pattern):
                    return False

        # TODO: Check age
        if self.max_age_days:
            age_days = (current_time - metadata.created_at).days
            if age_days > self.max_age_days:
                return True

        # TODO: Check pull count
        if self.min_pull_count:
            if metadata.pull_count < self.min_pull_count:
                return True

        return False

    def _tag_matches_pattern(self, tag: str, pattern: str) -> bool:
        """Check if tag matches pattern (supports wildcards)"""
        import re
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(regex_pattern, tag))

class RetentionPolicyEngine:
    """Enforce image retention policies"""

    def __init__(self, registries: Dict[str, RegistryInterface]):
        self.registries = registries

    def apply_policy(
        self,
        registry_key: str,
        repository: str,
        rule: RetentionRule,
        dry_run: bool = True
    ) -> Dict:
        """
        Apply retention policy to repository

        Args:
            registry_key: Registry to clean up
            repository: Repository name
            rule: Retention rule to apply
            dry_run: If True, only report what would be deleted

        Returns:
            {
                "total_images": 100,
                "to_delete": 45,
                "to_keep": 55,
                "deleted_tags": ["v1.0.0", "v1.0.1", ...],
                "space_freed_mb": 2500.0
            }
        """
        registry = self.registries[registry_key]

        # TODO: List all image tags
        tags = registry.list_tags(repository)

        # TODO: Get metadata for each tag
        all_metadata = []
        for tag in tags:
            metadata = registry.get_metadata(repository, tag)
            all_metadata.append(metadata)

        # TODO: Apply retention rules
        current_time = datetime.now()
        to_delete = []
        to_keep = []

        # Handle keep_last_n rule separately
        if rule.keep_last_n:
            # Sort by created_at, keep newest N
            sorted_metadata = sorted(all_metadata, key=lambda m: m.created_at, reverse=True)
            to_keep = sorted_metadata[:rule.keep_last_n]
            to_delete = sorted_metadata[rule.keep_last_n:]
        else:
            # Apply age/pull_count rules
            for metadata in all_metadata:
                if rule.should_delete(metadata, current_time):
                    to_delete.append(metadata)
                else:
                    to_keep.append(metadata)

        # TODO: Delete images if not dry_run
        deleted_tags = []
        space_freed = 0
        if not dry_run:
            for metadata in to_delete:
                registry.delete_image(repository, metadata.tag)
                deleted_tags.append(metadata.tag)
                space_freed += metadata.size_bytes

        return {
            "total_images": len(all_metadata),
            "to_delete": len(to_delete),
            "to_keep": len(to_keep),
            "deleted_tags": [m.tag for m in to_delete] if dry_run else deleted_tags,
            "space_freed_mb": sum(m.size_bytes for m in to_delete) / (1024 * 1024)
        }

    def apply_to_all_repositories(
        self,
        registry_key: str,
        rule: RetentionRule,
        dry_run: bool = True
    ) -> List[Dict]:
        """Apply retention policy to all repositories in registry"""
        registry = self.registries[registry_key]

        # TODO: List all repositories
        repositories = registry.list_repositories()

        # TODO: Apply policy to each repository
        results = []
        for repo in repositories:
            result = self.apply_policy(registry_key, repo, rule, dry_run)
            result["repository"] = repo
            results.append(result)

        return results
```

**Acceptance Criteria**:
- ✅ Delete images older than N days
- ✅ Keep last N versions of each repository
- ✅ Delete images with low pull counts
- ✅ Preserve specific tags (prod-*, latest)
- ✅ Dry-run mode to preview deletions
- ✅ Report space savings

---

### Task 5: Audit and Cost Analysis (4-5 hours)

Implement audit logging and cost optimization tools.

```python
# src/registry_manager/audit.py

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import json
from pathlib import Path

@dataclass
class AuditEvent:
    """Represents an auditable event"""
    timestamp: datetime
    event_type: str  # sync, promote, delete, tag
    user: str  # Who performed action
    registry: str
    repository: str
    tag: str
    details: dict
    correlation_id: str  # For tracing related events

class AuditLogger:
    """Log all registry operations for compliance"""

    def __init__(self, log_file: Path):
        self.log_file = log_file

    def log_event(self, event: AuditEvent) -> None:
        """Write event to audit log"""
        # TODO: Append to JSONL file (one JSON object per line)
        # TODO: Include structured fields for querying
        raise NotImplementedError

    def query_events(
        self,
        repository: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Query audit log"""
        # TODO: Read log file
        # TODO: Filter by criteria
        # TODO: Return matching events
        raise NotImplementedError

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Generate compliance report

        Returns:
            {
                "period": "2024-01-01 to 2024-01-31",
                "total_events": 1523,
                "by_type": {
                    "sync": 800,
                    "promote": 45,
                    "delete": 678
                },
                "by_user": {
                    "alice@example.com": 234,
                    "bob@example.com": 189
                },
                "security_events": [
                    "Production promotion without approval attempt blocked"
                ]
            }
        """
        # TODO: Aggregate events by type, user, repository
        # TODO: Identify security-relevant events
        raise NotImplementedError
```

```python
# src/registry_manager/cost_analyzer.py

from dataclasses import dataclass
from typing import Dict, List
from .registry.base import RegistryInterface

@dataclass
class RegistryCosts:
    """Cost breakdown for a registry"""
    registry_name: str
    storage_gb: float
    storage_cost_usd: float  # Based on pricing tier
    data_transfer_gb: float
    data_transfer_cost_usd: float
    total_cost_usd: float

class CostAnalyzer:
    """Analyze registry storage costs"""

    # Pricing (approximate, update with actual rates)
    PRICING = {
        "ecr": {"storage_per_gb": 0.10, "data_transfer_per_gb": 0.09},
        "gcr": {"storage_per_gb": 0.10, "data_transfer_per_gb": 0.12},
        "acr": {"storage_per_gb": 0.10, "data_transfer_per_gb": 0.09}
    }

    def __init__(self, registries: Dict[str, RegistryInterface]):
        self.registries = registries

    def analyze_costs(self, registry_key: str) -> RegistryCosts:
        """
        Calculate storage costs for registry

        Steps:
        1. List all repositories
        2. Get size of each image
        3. Sum total storage
        4. Calculate cost based on pricing
        """
        registry = self.registries[registry_key]

        # TODO: List all repositories
        repositories = registry.list_repositories()

        # TODO: Calculate total storage
        total_size_bytes = 0
        for repo in repositories:
            tags = registry.list_tags(repo)
            for tag in tags:
                metadata = registry.get_metadata(repo, tag)
                total_size_bytes += metadata.size_bytes

        total_size_gb = total_size_bytes / (1024 ** 3)

        # TODO: Calculate costs
        registry_type = self._get_registry_type(registry_key)
        pricing = self.PRICING.get(registry_type, self.PRICING["ecr"])

        storage_cost = total_size_gb * pricing["storage_per_gb"]

        # TODO: Estimate data transfer (would need actual metrics)
        data_transfer_gb = 0  # Placeholder
        data_transfer_cost = 0

        return RegistryCosts(
            registry_name=registry_key,
            storage_gb=total_size_gb,
            storage_cost_usd=storage_cost,
            data_transfer_gb=data_transfer_gb,
            data_transfer_cost_usd=data_transfer_cost,
            total_cost_usd=storage_cost + data_transfer_cost
        )

    def find_cost_savings(self, registry_key: str) -> List[Dict]:
        """
        Identify cost optimization opportunities

        Returns:
            [
                {
                    "opportunity": "Delete 45 images older than 90 days",
                    "savings_usd_monthly": 125.40,
                    "repositories_affected": ["my-ml-model", "legacy-api"]
                },
                {
                    "opportunity": "Delete duplicate images (same digest, different tags)",
                    "savings_usd_monthly": 45.20
                }
            ]
        """
        # TODO: Analyze old images
        # TODO: Find duplicate images (same digest)
        # TODO: Find unused images (zero pulls)
        # TODO: Calculate potential savings
        raise NotImplementedError

    def _get_registry_type(self, registry_key: str) -> str:
        """Determine registry type from key"""
        if "ecr" in registry_key.lower():
            return "ecr"
        elif "gcr" in registry_key.lower():
            return "gcr"
        elif "acr" in registry_key.lower():
            return "acr"
        return "ecr"
```

**Acceptance Criteria**:
- ✅ Log all registry operations to audit trail
- ✅ Query audit log by criteria
- ✅ Generate compliance reports
- ✅ Calculate registry storage costs
- ✅ Identify cost savings opportunities

---

### Task 6: CLI Interface (4-5 hours)

Build comprehensive command-line interface.

```python
# src/registry_manager/cli.py

import click
from pathlib import Path
from typing import Optional
import yaml
from .registry import ECRRegistry, GCRRegistry, ACRRegistry
from .sync import ImageSynchronizer, SyncJob
from .promotion import PromotionPipeline, PromotionRequest, Environment
from .retention import RetentionPolicyEngine, RetentionRule
from .cost_analyzer import CostAnalyzer

@click.group()
def cli():
    """Container Registry Manager - Multi-cloud registry operations"""
    pass

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def list_repos(config_file: str):
    """List all repositories in registries"""
    # TODO: Load config
    # TODO: Initialize registries
    # TODO: List repositories in each registry
    # TODO: Print results
    pass

@cli.command()
@click.option('--source', required=True, help='Source registry (e.g., us-east-1-ecr)')
@click.option('--target', required=True, multiple=True, help='Target registries')
@click.option('--repository', required=True, help='Repository name')
@click.option('--tag', required=True, help='Image tag')
@click.option('--config', type=click.Path(exists=True), default='config/registries.yaml')
def sync(source: str, target: tuple, repository: str, tag: str, config: str):
    """Sync image to target registries"""
    # TODO: Load registries from config
    # TODO: Create ImageSynchronizer
    # TODO: Sync to each target
    # TODO: Print results
    pass

@cli.command()
@click.option('--repository', required=True, help='Repository name')
@click.option('--tag', required=True, help='Image tag')
@click.option('--from-env', type=click.Choice(['dev', 'staging', 'prod']), required=True)
@click.option('--to-env', type=click.Choice(['dev', 'staging', 'prod']), required=True)
@click.option('--approved-by', help='Approver email (required for prod promotions)')
@click.option('--reason', help='Promotion reason')
@click.option('--config', type=click.Path(exists=True), default='config/registries.yaml')
def promote(repository: str, tag: str, from_env: str, to_env: str,
            approved_by: Optional[str], reason: Optional[str], config: str):
    """Promote image to next environment"""
    # TODO: Load config
    # TODO: Create PromotionPipeline
    # TODO: Execute promotion
    # TODO: Print result
    pass

@cli.command()
@click.option('--registry', required=True, help='Registry to clean up')
@click.option('--repository', help='Specific repository (or all if not specified)')
@click.option('--max-age-days', type=int, help='Delete images older than N days')
@click.option('--keep-last-n', type=int, help='Keep last N versions')
@click.option('--dry-run/--no-dry-run', default=True, help='Preview deletions without executing')
@click.option('--config', type=click.Path(exists=True), default='config/registries.yaml')
def cleanup(registry: str, repository: Optional[str], max_age_days: Optional[int],
            keep_last_n: Optional[int], dry_run: bool, config: str):
    """Clean up old images based on retention policy"""
    # TODO: Load config
    # TODO: Create RetentionRule
    # TODO: Apply policy
    # TODO: Print results (what was/would be deleted)
    pass

@cli.command()
@click.option('--registry', required=True, help='Registry to analyze')
@click.option('--config', type=click.Path(exists=True), default='config/registries.yaml')
def costs(registry: str, config: str):
    """Analyze registry storage costs"""
    # TODO: Load config
    # TODO: Create CostAnalyzer
    # TODO: Calculate costs
    # TODO: Find savings opportunities
    # TODO: Print report
    pass

@cli.command()
@click.option('--repository', help='Filter by repository')
@click.option('--event-type', help='Filter by event type')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--output', type=click.Path(), help='Output file for report')
def audit(repository: Optional[str], event_type: Optional[str],
          start_date: Optional[str], end_date: Optional[str], output: Optional[str]):
    """Query audit log"""
    # TODO: Create AuditLogger
    # TODO: Query events
    # TODO: Print or export results
    pass

if __name__ == '__main__':
    cli()
```

**Configuration File Example**:

```yaml
# config/registries.yaml

registries:
  us-east-1-ecr:
    type: ecr
    region: us-east-1
    environment: prod

  eu-west-1-ecr:
    type: ecr
    region: eu-west-1
    environment: prod

  us-central1-gcr:
    type: gcr
    project: my-ml-project
    location: us-central1
    environment: dev

  eastus-acr:
    type: acr
    registry_name: mymlregistry
    environment: staging

promotion_policies:
  - from_env: dev
    to_env: staging
    require_approval: false
    require_security_scan: true
    require_tests: true

  - from_env: staging
    to_env: prod
    require_approval: true
    require_security_scan: true
    require_tests: true
    auto_rollback_on_failure: true

retention_policies:
  dev:
    max_age_days: 30
    preserve_tags: ["latest"]

  staging:
    keep_last_n: 10
    preserve_tags: ["latest", "staging-*"]

  prod:
    keep_last_n: 20
    preserve_tags: ["prod-*", "latest", "stable"]
```

**Acceptance Criteria**:
- ✅ CLI commands for all major operations
- ✅ Load configuration from YAML
- ✅ Clear progress indicators
- ✅ Detailed error messages
- ✅ Support for dry-run mode

---

## Testing Requirements

### Unit Tests

```python
# tests/test_sync.py

def test_sync_image_success():
    """Test successful image sync"""
    source_ecr = ECRRegistry(region="us-east-1")
    target_ecr = ECRRegistry(region="eu-west-1")

    synchronizer = ImageSynchronizer({
        "source": source_ecr,
        "target": target_ecr
    })

    job = SyncJob(
        source_registry="source",
        target_registry="target",
        repository="my-ml-model",
        tag="v1.2.3"
    )

    result = synchronizer.sync_image(job)
    assert result.success
    assert result.size_bytes > 0

def test_sync_skips_existing():
    """Test sync skips if image already exists with same digest"""
    # TODO: Mock registries
    # TODO: Set up scenario where target already has image
    # TODO: Verify sync is skipped
    pass
```

### Integration Tests

```bash
# tests/integration/test_promotion_workflow.sh

#!/bin/bash
set -e

echo "Testing full promotion workflow..."

# Build test image
docker build -t test-app:v1.0.0 test-fixtures/app

# Tag for dev
docker tag test-app:v1.0.0 123456789012.dkr.ecr.us-east-1.amazonaws.com/test-app:dev-v1.0.0

# Push to dev
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/test-app:dev-v1.0.0

# Promote dev → staging
python -m registry_manager.cli promote \
    --repository test-app \
    --tag dev-v1.0.0 \
    --from-env dev \
    --to-env staging

# Verify image exists in staging
aws ecr describe-images \
    --repository-name test-app \
    --image-ids imageTag=staging-v1.0.0-* \
    --region us-east-1

echo "✅ Promotion workflow test passed!"
```

## Expected Results

After completing this exercise, you should achieve:

| Metric | Target | Measured |
|--------|--------|----------|
| **Sync Time (1 GB, same region)** | <3 min | ________ |
| **Sync Time (1 GB, cross-region)** | <5 min | ________ |
| **Retention Policy Execution** | <10s for 100 images | ________ |
| **Cost Savings** | >30% reduction | ________% |

## Validation

Submit the following for review:

1. **Complete implementation** of all 6 tasks
2. **Test suite** with >75% coverage
3. **Example workflows**:
   - Sync repository to 3 target registries
   - Promote image from dev → staging → prod
   - Apply retention policy and measure savings
4. **Performance benchmarks**:
   - Sync performance (different image sizes)
   - Retention policy execution time
5. **Cost analysis report** for your registries
6. **Documentation**:
   - `DESIGN.md`: Architecture
   - `REGISTRY_API.md`: API reference
   - `PROMOTION_WORKFLOW.md`: Best practices

## Bonus Challenges

1. **Image Signing** (4-5h): Integrate Sigstore/Cosign for image signing and verification
2. **Multi-Arch Support** (3-4h): Handle AMD64/ARM64 multi-architecture images
3. **Kubernetes Operator** (6-8h): Deploy as K8s operator that watches ImagePolicy CRDs
4. **Webhook Integration** (2-3h): Trigger promotions via GitHub/GitLab webhooks
5. **Grafana Dashboard** (3-4h): Build dashboard showing registry metrics and costs

## Resources

- [Docker Registry HTTP API V2](https://docs.docker.com/registry/spec/api/)
- [AWS ECR API Reference](https://docs.aws.amazon.com/AmazonECR/latest/APIReference/)
- [GCR/Artifact Registry Docs](https://cloud.google.com/artifact-registry/docs)
- [Azure ACR Docs](https://learn.microsoft.com/en-us/azure/container-registry/)
- [Skopeo - Image Copy Tool](https://github.com/containers/skopeo)
- [Crane - Container Registry Tool](https://github.com/google/go-containerregistry/tree/main/cmd/crane)

## Deliverables Checklist

- [ ] Registry abstraction (base.py, ecr.py, gcr.py, acr.py)
- [ ] Image synchronization (sync.py)
- [ ] Promotion pipeline (promotion.py)
- [ ] Retention policy engine (retention.py)
- [ ] Audit logging (audit.py)
- [ ] Cost analyzer (cost_analyzer.py)
- [ ] CLI interface (cli.py)
- [ ] Comprehensive test suite (>75% coverage)
- [ ] Configuration files (registries.yaml, policies)
- [ ] `DESIGN.md` - Architecture documentation
- [ ] `REGISTRY_API.md` - API reference
- [ ] `PROMOTION_WORKFLOW.md` - Best practices
- [ ] Performance benchmarks
- [ ] Cost analysis report

---

**Estimated Completion Time**: 32-40 hours

**Skills Practiced**:
- Multi-cloud registry management
- Docker Registry HTTP API
- Image synchronization and replication
- Promotion pipelines and workflows
- Retention policies and lifecycle management
- Audit logging and compliance
- Cost optimization
