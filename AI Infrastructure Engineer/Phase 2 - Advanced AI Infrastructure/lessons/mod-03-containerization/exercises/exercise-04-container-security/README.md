# Exercise 04: Container Security Scanner

## Learning Objectives

By completing this exercise, you will:
- Implement container vulnerability scanning
- Integrate security tools (Trivy, Snyk, Grype)
- Generate Software Bill of Materials (SBOM)
- Enforce security policies
- Automate security in CI/CD pipelines

## Overview

Container security is critical for production ML systems. This exercise builds a comprehensive security scanning tool that detects vulnerabilities, generates SBOMs, and enforces security policies before deployment.

## Prerequisites

- Docker and container knowledge
- Understanding of CVEs and security vulnerabilities
- Python 3.11+
- Familiarity with CI/CD concepts

## Problem Statement

Build `containersec`, a security scanner that:

1. **Scans container images** for vulnerabilities
2. **Generates SBOM** (Software Bill of Materials)
3. **Enforces security policies** (block critical vulnerabilities)
4. **Integrates with CI/CD** pipelines
5. **Reports and tracks** vulnerabilities over time
6. **Recommends fixes** and safer base images

## Requirements

### Functional Requirements

#### FR1: Vulnerability Scanning
- Scan images with multiple tools:
  - Trivy (comprehensive, fast)
  - Grype (Anchore)
  - Snyk (commercial)
- Detect:
  - OS package vulnerabilities
  - Application dependency vulnerabilities
  - Secrets in images
  - Misconfigurations
- Support multiple image formats (Docker, OCI)

#### FR2: SBOM Generation
- Generate SBOM in standard formats:
  - CycloneDX
  - SPDX
  - Syft format
- Include:
  - Package inventory
  - License information
  - Dependency tree
  - File checksums

#### FR3: Policy Enforcement
- Define security policies:
  - Maximum severity (no CRITICAL)
  - CVE blocklist
  - License restrictions
  - Base image requirements
- Pass/fail decisions for CI/CD
- Policy-as-code (YAML/Rego)

#### FR4: Reporting
- Generate security reports:
  - HTML reports with details
  - JSON for automation
  - SARIF for GitHub Security
- Track vulnerabilities over time
- Compare scans (diff)

#### FR5: CI/CD Integration
- GitHub Actions workflow
- GitLab CI integration
- Jenkins pipeline
- Pre-commit hooks
- Registry webhook integration

### Non-Functional Requirements

#### NFR1: Performance
- Scan 1GB image in < 2 minutes
- Cache scan results
- Parallel scanning

#### NFR2: Accuracy
- Low false positive rate
- Up-to-date vulnerability database
- Cross-reference multiple sources

#### NFR3: Usability
- Clear, actionable reports
- Remediation suggestions
- Developer-friendly output

## Implementation Tasks

### Task 1: Scanner Framework (4-5 hours)

Build the core scanning framework:

```python
# src/scanner/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class Severity(Enum):
    """Vulnerability severity"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NEGLIGIBLE = "NEGLIGIBLE"
    UNKNOWN = "UNKNOWN"

@dataclass
class Vulnerability:
    """Vulnerability information"""
    cve_id: str
    package_name: str
    installed_version: str
    fixed_version: Optional[str]
    severity: Severity
    description: str
    cvss_score: Optional[float]
    references: List[str]

@dataclass
class ScanResult:
    """Container scan result"""
    image_name: str
    image_digest: str
    scan_time: datetime
    scanner_name: str
    scanner_version: str

    vulnerabilities: List[Vulnerability]
    total_vulns: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int

    # SBOM
    sbom: Optional[Dict] = None

    # Secrets found
    secrets: List[Dict] = None

    # Misconfigurations
    misconfigs: List[Dict] = None

    def passes_policy(self, policy: 'SecurityPolicy') -> bool:
        """
        TODO: Check if scan passes security policy

        ```python
        # Check severity thresholds
        if self.critical_count > policy.max_critical:
            return False
        if self.high_count > policy.max_high:
            return False

        # Check specific CVEs
        for vuln in self.vulnerabilities:
            if vuln.cve_id in policy.blocked_cves:
                return False

        # Check secrets
        if self.secrets and not policy.allow_secrets:
            return False

        return True
        ```
        """
        pass

@dataclass
class SecurityPolicy:
    """Security policy"""
    name: str
    max_critical: int = 0
    max_high: int = 5
    max_medium: int = 20
    allow_secrets: bool = False
    blocked_cves: List[str] = None
    required_base_images: List[str] = None
    license_allowlist: List[str] = None

    def __post_init__(self):
        if self.blocked_cves is None:
            self.blocked_cves = []
        if self.license_allowlist is None:
            self.license_allowlist = []

class SecurityScanner(ABC):
    """Abstract security scanner"""

    def __init__(self):
        self.name = self.__class__.__name__
        self.version = "1.0.0"

    @abstractmethod
    def scan_image(
        self,
        image: str,
        include_sbom: bool = True
    ) -> ScanResult:
        """
        Scan container image

        TODO: Implement in subclasses
        - Pull image if needed
        - Run scanner
        - Parse results
        - Generate SBOM if requested
        """
        pass

    @abstractmethod
    def scan_filesystem(
        self,
        path: str
    ) -> ScanResult:
        """Scan filesystem/directory"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if scanner is installed and available"""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get scanner version"""
        pass
```

**Acceptance Criteria**:
- [ ] Abstract scanner interface defined
- [ ] Vulnerability data model complete
- [ ] Security policy model defined
- [ ] Policy enforcement logic implemented

---

### Task 2: Trivy Integration (4-5 hours)

```python
# src/scanner/trivy.py

import subprocess
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

class TrivyScanner(SecurityScanner):
    """Trivy vulnerability scanner"""

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__()
        self.cache_dir = cache_dir or Path.home() / ".cache" / "trivy"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """
        TODO: Check if Trivy is installed

        ```python
        try:
            result = subprocess.run(
                ["trivy", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        ```
        """
        pass

    def get_version(self) -> str:
        """
        TODO: Get Trivy version

        ```python
        result = subprocess.run(
            ["trivy", "--version"],
            capture_output=True,
            text=True
        )
        # Parse version from output
        return result.stdout.split()[1]
        ```
        """
        pass

    def scan_image(
        self,
        image: str,
        include_sbom: bool = True
    ) -> ScanResult:
        """
        TODO: Scan image with Trivy

        ```python
        # Run Trivy scan
        cmd = [
            "trivy",
            "image",
            "--format", "json",
            "--quiet",
            "--cache-dir", str(self.cache_dir),
            image
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Trivy scan failed: {result.stderr}")

        # Parse JSON output
        trivy_results = json.loads(result.stdout)

        # Extract vulnerabilities
        vulnerabilities = []
        for target in trivy_results.get("Results", []):
            for vuln in target.get("Vulnerabilities", []):
                vulnerabilities.append(
                    Vulnerability(
                        cve_id=vuln.get("VulnerabilityID"),
                        package_name=vuln.get("PkgName"),
                        installed_version=vuln.get("InstalledVersion"),
                        fixed_version=vuln.get("FixedVersion"),
                        severity=Severity[vuln.get("Severity", "UNKNOWN")],
                        description=vuln.get("Description", ""),
                        cvss_score=self._parse_cvss(vuln),
                        references=vuln.get("References", [])
                    )
                )

        # Count by severity
        severity_counts = self._count_by_severity(vulnerabilities)

        # Generate SBOM if requested
        sbom = self._generate_sbom(image) if include_sbom else None

        # Scan for secrets
        secrets = self._scan_secrets(image)

        # Scan for misconfigurations
        misconfigs = self._scan_misconfigs(image)

        return ScanResult(
            image_name=image,
            image_digest=self._get_image_digest(image),
            scan_time=datetime.now(),
            scanner_name=self.name,
            scanner_version=self.get_version(),
            vulnerabilities=vulnerabilities,
            total_vulns=len(vulnerabilities),
            critical_count=severity_counts.get(Severity.CRITICAL, 0),
            high_count=severity_counts.get(Severity.HIGH, 0),
            medium_count=severity_counts.get(Severity.MEDIUM, 0),
            low_count=severity_counts.get(Severity.LOW, 0),
            sbom=sbom,
            secrets=secrets,
            misconfigs=misconfigs
        )
        ```
        """
        pass

    def scan_filesystem(self, path: str) -> ScanResult:
        """
        TODO: Scan filesystem with Trivy

        Similar to scan_image but uses 'trivy fs' command
        """
        pass

    def _generate_sbom(self, image: str) -> Dict:
        """
        TODO: Generate SBOM using Trivy

        ```python
        cmd = [
            "trivy",
            "image",
            "--format", "cyclonedx",
            "--quiet",
            image
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        return json.loads(result.stdout)
        ```
        """
        pass

    def _scan_secrets(self, image: str) -> List[Dict]:
        """
        TODO: Scan for secrets in image

        ```python
        cmd = [
            "trivy",
            "image",
            "--scanners", "secret",
            "--format", "json",
            "--quiet",
            image
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        secrets = []
        for target in data.get("Results", []):
            for secret in target.get("Secrets", []):
                secrets.append({
                    "category": secret.get("Category"),
                    "severity": secret.get("Severity"),
                    "file": target.get("Target"),
                    "line": secret.get("StartLine")
                })

        return secrets
        ```
        """
        pass

    def _scan_misconfigs(self, image: str) -> List[Dict]:
        """
        TODO: Scan for misconfigurations

        Check for:
        - Running as root
        - Exposed sensitive ports
        - Insecure configurations
        """
        pass

    def _parse_cvss(self, vuln: Dict) -> Optional[float]:
        """TODO: Parse CVSS score from vulnerability"""
        pass

    def _count_by_severity(
        self,
        vulnerabilities: List[Vulnerability]
    ) -> Dict[Severity, int]:
        """TODO: Count vulnerabilities by severity"""
        pass

    def _get_image_digest(self, image: str) -> str:
        """
        TODO: Get image digest

        ```python
        result = subprocess.run(
            ["docker", "inspect", "--format={{.RepoDigests}}", image],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
        ```
        """
        pass
```

**Acceptance Criteria**:
- [ ] Trivy scanner fully integrated
- [ ] Scans images and filesystems
- [ ] Generates SBOM
- [ ] Detects secrets
- [ ] Identifies misconfigurations
- [ ] Handles errors gracefully

---

### Task 3: Multi-Scanner Aggregator (4-5 hours)

```python
# src/scanner/aggregator.py

from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .trivy import TrivyScanner
from .grype import GrypeScan ner  # Similar implementation
from .snyk import SnykScanner  # Similar implementation

class ScannerAggregator:
    """Aggregate results from multiple scanners"""

    def __init__(self):
        self.scanners = {
            'trivy': TrivyScanner(),
            'grype': GrypeScanner(),
            'snyk': SnykScanner()
        }

    def scan_with_all(
        self,
        image: str,
        parallel: bool = True
    ) -> Dict[str, ScanResult]:
        """
        TODO: Scan with all available scanners

        ```python
        results = {}
        available_scanners = {
            name: scanner
            for name, scanner in self.scanners.items()
            if scanner.is_available()
        }

        if not available_scanners:
            raise RuntimeError("No scanners available")

        if parallel:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(scanner.scan_image, image): name
                    for name, scanner in available_scanners.items()
                }

                for future in as_completed(futures):
                    scanner_name = futures[future]
                    try:
                        results[scanner_name] = future.result()
                    except Exception as e:
                        print(f"Scanner {scanner_name} failed: {e}")
        else:
            for name, scanner in available_scanners.items():
                try:
                    results[name] = scanner.scan_image(image)
                except Exception as e:
                    print(f"Scanner {name} failed: {e}")

        return results
        ```
        """
        pass

    def aggregate_results(
        self,
        results: Dict[str, ScanResult]
    ) -> ScanResult:
        """
        TODO: Aggregate results from multiple scanners

        Strategy:
        - Union of all vulnerabilities
        - Deduplicate by CVE ID
        - Take highest severity if conflict
        - Combine SBOMs
        """
        pass

    def compare_scanners(
        self,
        results: Dict[str, ScanResult]
    ) -> Dict:
        """
        TODO: Compare scanner results

        Generate report showing:
        - Unique vulnerabilities found by each scanner
        - Common vulnerabilities
        - Disagreements on severity
        - Performance (scan time)
        """
        pass

    def recommend_fixes(
        self,
        scan_result: ScanResult
    ) -> List[Dict]:
        """
        TODO: Recommend vulnerability fixes

        ```python
        recommendations = []

        # Group vulnerabilities by package
        by_package = {}
        for vuln in scan_result.vulnerabilities:
            by_package.setdefault(vuln.package_name, []).append(vuln)

        # Recommend upgrades
        for package, vulns in by_package.items():
            # Find minimum version that fixes all vulnerabilities
            fixed_versions = [
                v.fixed_version for v in vulns
                if v.fixed_version
            ]

            if fixed_versions:
                recommended_version = max(fixed_versions)
                recommendations.append({
                    "type": "upgrade",
                    "package": package,
                    "current_version": vulns[0].installed_version,
                    "recommended_version": recommended_version,
                    "fixes": len(vulns),
                    "critical_fixes": sum(
                        1 for v in vulns
                        if v.severity == Severity.CRITICAL
                    )
                })

        # Recommend base image changes
        base_image_rec = self._recommend_base_image(scan_result)
        if base_image_rec:
            recommendations.append(base_image_rec)

        return sorted(
            recommendations,
            key=lambda r: r.get("critical_fixes", 0),
            reverse=True
        )
        ```
        """
        pass

    def _recommend_base_image(self, scan_result: ScanResult) -> Optional[Dict]:
        """
        TODO: Recommend safer base image

        Suggest:
        - Distroless images
        - Slim/alpine variants
        - Newer versions
        """
        pass
```

**Acceptance Criteria**:
- [ ] Runs multiple scanners in parallel
- [ ] Aggregates results intelligently
- [ ] Compares scanner effectiveness
- [ ] Provides fix recommendations
- [ ] Suggests safer base images

---

### Task 4: Policy Engine (3-4 hours)

```python
# src/policy/engine.py

import yaml
from pathlib import Path
from typing import Dict, List
from dataclasses import asdict

class PolicyEngine:
    """Enforce security policies"""

    def __init__(self, policy_file: Optional[Path] = None):
        self.policies = {}
        if policy_file:
            self.load_policies(policy_file)

    def load_policies(self, policy_file: Path) -> None:
        """
        TODO: Load policies from YAML

        policies.yaml format:
        ```yaml
        policies:
          - name: production
            max_critical: 0
            max_high: 0
            max_medium: 10
            allow_secrets: false
            blocked_cves:
              - CVE-2021-44228  # Log4Shell
              - CVE-2021-45046
            required_base_images:
              - "gcr.io/distroless/.*"
              - "python:.*-slim"

          - name: development
            max_critical: 5
            max_high: 20
            allow_secrets: false
        ```
        """
        pass

    def evaluate(
        self,
        scan_result: ScanResult,
        policy_name: str = "production"
    ) -> Dict[str, any]:
        """
        TODO: Evaluate scan against policy

        ```python
        policy = self.policies.get(policy_name)
        if not policy:
            raise ValueError(f"Policy '{policy_name}' not found")

        violations = []

        # Check severity thresholds
        if scan_result.critical_count > policy.max_critical:
            violations.append({
                "rule": "max_critical",
                "limit": policy.max_critical,
                "actual": scan_result.critical_count,
                "severity": "critical"
            })

        if scan_result.high_count > policy.max_high:
            violations.append({
                "rule": "max_high",
                "limit": policy.max_high,
                "actual": scan_result.high_count,
                "severity": "high"
            })

        # Check blocked CVEs
        for vuln in scan_result.vulnerabilities:
            if vuln.cve_id in policy.blocked_cves:
                violations.append({
                    "rule": "blocked_cve",
                    "cve": vuln.cve_id,
                    "package": vuln.package_name,
                    "severity": "critical"
                })

        # Check secrets
        if scan_result.secrets and not policy.allow_secrets:
            violations.append({
                "rule": "no_secrets",
                "count": len(scan_result.secrets),
                "severity": "critical"
            })

        passed = len(violations) == 0

        return {
            "policy": policy_name,
            "passed": passed,
            "violations": violations,
            "scan_summary": {
                "critical": scan_result.critical_count,
                "high": scan_result.high_count,
                "medium": scan_result.medium_count,
                "low": scan_result.low_count
            }
        }
        ```
        """
        pass

    def generate_policy_report(
        self,
        evaluation: Dict
    ) -> str:
        """
        TODO: Generate human-readable policy report

        ```
        Security Policy Evaluation: production
        Status: FAILED ❌

        Violations:
        1. [CRITICAL] max_critical: 3 critical vulnerabilities found (limit: 0)
        2. [CRITICAL] blocked_cve: CVE-2021-44228 found in log4j-core:2.14.1
        3. [HIGH] max_high: 8 high vulnerabilities found (limit: 5)

        Scan Summary:
        - Critical: 3
        - High: 8
        - Medium: 15
        - Low: 42
        ```
        """
        pass

    def create_opa_policy(self, policy: SecurityPolicy) -> str:
        """
        TODO: Convert to OPA (Open Policy Agent) Rego policy

        For advanced use cases with Kubernetes admission controllers
        """
        pass
```

**Acceptance Criteria**:
- [ ] Loads policies from YAML
- [ ] Evaluates scans against policies
- [ ] Generates violation reports
- [ ] Supports multiple policy profiles
- [ ] OPA policy generation (bonus)

---

### Task 5: Reporting and Visualization (4-5 hours)

```python
# src/reporting/generator.py

from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json
from jinja2 import Template
import plotly.graph_objects as go

class SecurityReporter:
    """Generate security reports"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(
        self,
        scan_result: ScanResult,
        policy_evaluation: Dict,
        output_file: str = "security-report.html"
    ) -> Path:
        """
        TODO: Generate comprehensive HTML report

        Include:
        - Executive summary
        - Vulnerability table (sortable, filterable)
        - Charts (by severity, by package)
        - SBOM viewer
        - Remediation recommendations
        - Policy violations
        """
        pass

    def generate_sarif(
        self,
        scan_result: ScanResult,
        output_file: str = "results.sarif"
    ) -> Path:
        """
        TODO: Generate SARIF format for GitHub Security

        ```python
        sarif = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": scan_result.scanner_name,
                            "version": scan_result.scanner_version
                        }
                    },
                    "results": [
                        {
                            "ruleId": vuln.cve_id,
                            "level": self._severity_to_sarif_level(vuln.severity),
                            "message": {
                                "text": f"{vuln.package_name} has vulnerability {vuln.cve_id}"
                            },
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {
                                            "uri": scan_result.image_name
                                        }
                                    }
                                }
                            ]
                        }
                        for vuln in scan_result.vulnerabilities
                    ]
                }
            ]
        }

        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(sarif, f, indent=2)

        return output_path
        ```
        """
        pass

    def generate_json_report(
        self,
        scan_result: ScanResult,
        output_file: str = "scan-results.json"
    ) -> Path:
        """TODO: Generate JSON report for automation"""
        pass

    def generate_trend_report(
        self,
        scan_history: List[ScanResult],
        output_file: str = "trend-report.html"
    ) -> Path:
        """
        TODO: Generate trend report

        Show vulnerability trends over time:
        - Total vulnerabilities
        - By severity
        - New vs fixed
        - MTTR (Mean Time To Remediate)
        """
        pass

    def plot_vulnerabilities_by_severity(
        self,
        scan_result: ScanResult
    ) -> go.Figure:
        """
        TODO: Plot vulnerabilities by severity

        ```python
        severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        counts = [
            scan_result.critical_count,
            scan_result.high_count,
            scan_result.medium_count,
            scan_result.low_count
        ]
        colors = ['#DC3545', '#FD7E14', '#FFC107', '#28A745']

        fig = go.Figure(data=[
            go.Bar(
                x=severities,
                y=counts,
                marker_color=colors
            )
        ])

        fig.update_layout(
            title='Vulnerabilities by Severity',
            xaxis_title='Severity',
            yaxis_title='Count'
        )

        return fig
        ```
        """
        pass

    def compare_scans(
        self,
        old_scan: ScanResult,
        new_scan: ScanResult
    ) -> Dict:
        """
        TODO: Compare two scans

        Return:
        - New vulnerabilities
        - Fixed vulnerabilities
        - Changed severities
        """
        pass
```

**Acceptance Criteria**:
- [ ] HTML report with interactive elements
- [ ] SARIF format for GitHub
- [ ] JSON for automation
- [ ] Trend analysis
- [ ] Scan comparison

---

### Task 6: CI/CD Integration (3-4 hours)

```python
# src/cicd/github_action.py

from pathlib import Path
from typing import Optional

class GitHubActionsIntegration:
    """Generate GitHub Actions workflow"""

    @staticmethod
    def generate_workflow(
        policy_file: str = ".security-policy.yaml",
        fail_on_violation: bool = True,
        upload_sarif: bool = True
    ) -> str:
        """
        TODO: Generate GitHub Actions workflow

        ```yaml
        name: Container Security Scan

        on:
          push:
            branches: [ main, develop ]
          pull_request:
            branches: [ main ]

        jobs:
          security-scan:
            runs-on: ubuntu-latest

            steps:
            - uses: actions/checkout@v3

            - name: Build Docker image
              run: docker build -t ${{ github.repository }}:${{ github.sha }} .

            - name: Run Trivy scan
              uses: aquasecurity/trivy-action@master
              with:
                image-ref: ${{ github.repository }}:${{ github.sha }}
                format: 'sarif'
                output: 'trivy-results.sarif'

            - name: Upload SARIF to GitHub Security
              uses: github/codeql-action/upload-sarif@v2
              with:
                sarif_file: 'trivy-results.sarif'

            - name: Run policy check
              run: |
                python -m containersec check \\
                  --image ${{ github.repository }}:${{ github.sha }} \\
                  --policy {{ policy_file }} \\
                  --fail-on-violation

            - name: Generate report
              if: always()
              run: |
                python -m containersec report \\
                  --image ${{ github.repository }}:${{ github.sha }} \\
                  --output security-report.html

            - name: Upload report
              if: always()
              uses: actions/upload-artifact@v3
              with:
                name: security-report
                path: security-report.html
        ```
        """
        pass
```

**Acceptance Criteria**:
- [ ] GitHub Actions workflow generated
- [ ] GitLab CI template created
- [ ] Jenkins pipeline example
- [ ] Pre-commit hook script

---

## CLI Interface

```bash
# Scan image
containersec scan myapp:latest

# Scan with specific scanner
containersec scan --scanner trivy myapp:latest

# Scan with all scanners
containersec scan --all myapp:latest

# Check against policy
containersec check --policy production myapp:latest

# Generate SBOM
containersec sbom --format cyclonedx myapp:latest

# Generate report
containersec report --format html myapp:latest

# Compare scans
containersec diff old-scan.json new-scan.json

# Setup CI/CD
containersec setup github
```

---

## Deliverables

1. **Scanner implementations**
2. **Policy engine**
3. **Reporting tools**
4. **CI/CD integrations**
5. **Documentation**
6. **Tests**

---

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| **Accuracy** | 30% |
| **Completeness** | 25% |
| **Policy Enforcement** | 20% |
| **CI/CD Integration** | 15% |
| **Code Quality** | 10% |

**Passing**: 70%+
**Excellence**: 90%+

---

## Estimated Time

- Task 1: 4-5 hours
- Task 2: 4-5 hours
- Task 3: 4-5 hours
- Task 4: 3-4 hours
- Task 5: 4-5 hours
- Task 6: 3-4 hours
- Testing: 4-5 hours
- Documentation: 2-3 hours

**Total**: 28-36 hours

---

**This exercise teaches critical container security skills for production ML infrastructure.**
