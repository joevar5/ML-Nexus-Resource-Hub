# Lesson 08: Production LLM Best Practices

## Table of Contents
1. [Introduction](#introduction)
2. [Production Readiness Checklist](#production-readiness-checklist)
3. [High Availability and Fault Tolerance](#high-availability-and-fault-tolerance)
4. [Disaster Recovery Strategies](#disaster-recovery-strategies)
5. [Blue-Green Deployments](#blue-green-deployments)
6. [Canary Deployments](#canary-deployments)
7. [A/B Testing Frameworks](#ab-testing-frameworks)
8. [Prompt Engineering Best Practices](#prompt-engineering-best-practices)
9. [Safety and Content Filtering](#safety-and-content-filtering)
10. [PII Detection and Redaction](#pii-detection-and-redaction)
11. [GDPR and Data Privacy](#gdpr-and-data-privacy)
12. [Model Versioning Strategies](#model-versioning-strategies)
13. [Incident Response and On-Call](#incident-response-and-on-call)
14. [Cost Optimization in Production](#cost-optimization-in-production)
15. [Team Organization](#team-organization)
16. [Documentation Standards](#documentation-standards)
17. [Lessons from Production](#lessons-from-production)
18. [Summary](#summary)

## Introduction

Deploying and operating LLMs in production requires careful attention to reliability, safety, cost, and compliance. This lesson provides comprehensive best practices learned from real-world LLM deployments.

### Learning Objectives

After completing this lesson, you will be able to:
- Assess production readiness for LLM systems
- Implement high availability and disaster recovery
- Deploy models safely with blue-green and canary strategies
- Build A/B testing frameworks for LLM improvements
- Apply prompt engineering best practices
- Implement safety guardrails and content filtering
- Detect and redact PII automatically
- Ensure GDPR and privacy compliance
- Manage model versions effectively
- Handle incidents and maintain on-call operations
- Optimize costs in production environments
- Structure teams for LLM operations
- Maintain comprehensive documentation

## Production Readiness Checklist

### Comprehensive Readiness Assessment

```python
# production_readiness.py - Production readiness checker
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class CheckStatus(Enum):
    PASS = "✓ PASS"
    FAIL = "✗ FAIL"
    WARN = "⚠ WARN"
    SKIP = "○ SKIP"

@dataclass
class ReadinessCheck:
    """Single readiness check"""
    name: str
    category: str
    status: CheckStatus
    details: str
    remediation: Optional[str] = None

class ProductionReadinessChecker:
    """
    Comprehensive production readiness checker for LLM systems
    """
    def __init__(self):
        self.checks: List[ReadinessCheck] = []
        
    def check_performance(self):
        """Check performance requirements"""
        checks = [
            {
                'name': 'Latency SLA',
                'requirement': 'p95 latency < 2 seconds',
                'test': lambda: self._test_latency()
            },
            {
                'name': 'Throughput',
                'requirement': 'Handle 100 req/sec',
                'test': lambda: self._test_throughput()
            },
            {
                'name': 'GPU Utilization',
                'requirement': '>70% average utilization',
                'test': lambda: self._test_gpu_utilization()
            },
        ]
        
        for check in checks:
            passed, details = check['test']()
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            
            self.checks.append(ReadinessCheck(
                name=check['name'],
                category='Performance',
                status=status,
                details=details,
                remediation=check['requirement'] if not passed else None
            ))
    
    def check_reliability(self):
        """Check reliability requirements"""
        checks = [
            ('High Availability', self._test_ha, 'Deploy multi-AZ'),
            ('Health Checks', self._test_health_checks, 'Implement health endpoints'),
            ('Circuit Breakers', self._test_circuit_breakers, 'Add circuit breaker pattern'),
            ('Retry Logic', self._test_retry_logic, 'Implement exponential backoff'),
            ('Graceful Degradation', self._test_degradation, 'Add fallback mechanisms'),
        ]
        
        for name, test_func, remediation in checks:
            passed, details = test_func()
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            
            self.checks.append(ReadinessCheck(
                name=name,
                category='Reliability',
                status=status,
                details=details,
                remediation=remediation if not passed else None
            ))
    
    def check_monitoring(self):
        """Check monitoring and observability"""
        checks = [
            ('Prometheus Metrics', self._test_metrics, 'Set up Prometheus'),
            ('Grafana Dashboards', self._test_dashboards, 'Create monitoring dashboards'),
            ('Alerting', self._test_alerting, 'Configure PagerDuty/Opsgenie'),
            ('Distributed Tracing', self._test_tracing, 'Implement Jaeger/Zipkin'),
            ('Log Aggregation', self._test_logging, 'Set up ELK stack'),
            ('Cost Tracking', self._test_cost_tracking, 'Implement cost monitoring'),
        ]
        
        for name, test_func, remediation in checks:
            passed, details = test_func()
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            
            self.checks.append(ReadinessCheck(
                name=name,
                category='Monitoring',
                status=status,
                details=details,
                remediation=remediation if not passed else None
            ))
    
    def check_security(self):
        """Check security requirements"""
        checks = [
            ('Authentication', self._test_auth, 'Implement JWT/OAuth'),
            ('Rate Limiting', self._test_rate_limiting, 'Add rate limiting'),
            ('Input Validation', self._test_input_validation, 'Validate all inputs'),
            ('PII Detection', self._test_pii_detection, 'Implement PII detection'),
            ('Encryption at Rest', self._test_encryption_rest, 'Enable encryption'),
            ('Encryption in Transit', self._test_encryption_transit, 'Use TLS 1.3'),
            ('Secrets Management', self._test_secrets, 'Use Vault/Secrets Manager'),
        ]
        
        for name, test_func, remediation in checks:
            passed, details = test_func()
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            
            self.checks.append(ReadinessCheck(
                name=name,
                category='Security',
                status=status,
                details=details,
                remediation=remediation if not passed else None
            ))
    
    def check_safety(self):
        """Check AI safety requirements"""
        checks = [
            ('Content Filtering', self._test_content_filter, 'Implement content moderation'),
            ('Toxicity Detection', self._test_toxicity, 'Add toxicity classifier'),
            ('Output Validation', self._test_output_validation, 'Validate model outputs'),
            ('Prompt Injection Protection', self._test_prompt_injection, 'Add prompt validation'),
            ('Bias Testing', self._test_bias, 'Run bias evaluation'),
        ]
        
        for name, test_func, remediation in checks:
            passed, details = test_func()
            status = CheckStatus.PASS if passed else CheckStatus.WARN
            
            self.checks.append(ReadinessCheck(
                name=name,
                category='AI Safety',
                status=status,
                details=details,
                remediation=remediation if not passed else None
            ))
    
    def check_compliance(self):
        """Check compliance requirements"""
        checks = [
            ('GDPR Compliance', self._test_gdpr, 'Implement data deletion'),
            ('Data Retention Policy', self._test_retention, 'Define retention policy'),
            ('Audit Logging', self._test_audit_logs, 'Enable audit logging'),
            ('Privacy Policy', self._test_privacy_policy, 'Create privacy policy'),
            ('Terms of Service', self._test_tos, 'Draft terms of service'),
        ]
        
        for name, test_func, remediation in checks:
            passed, details = test_func()
            status = CheckStatus.PASS if passed else CheckStatus.WARN
            
            self.checks.append(ReadinessCheck(
                name=name,
                category='Compliance',
                status=status,
                details=details,
                remediation=remediation if not passed else None
            ))
    
    def check_operations(self):
        """Check operational requirements"""
        checks = [
            ('Runbook Documentation', self._test_runbooks, 'Write runbooks'),
            ('On-Call Rotation', self._test_oncall, 'Set up on-call rotation'),
            ('Incident Response Plan', self._test_incident_plan, 'Create IR plan'),
            ('Deployment Automation', self._test_deployment, 'Automate deployments'),
            ('Rollback Capability', self._test_rollback, 'Test rollback procedure'),
            ('Backup and Restore', self._test_backup, 'Implement backups'),
        ]
        
        for name, test_func, remediation in checks:
            passed, details = test_func()
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            
            self.checks.append(ReadinessCheck(
                name=name,
                category='Operations',
                status=status,
                details=details,
                remediation=remediation if not passed else None
            ))
    
    # Test implementations (examples)
    def _test_latency(self):
        """Test latency SLA"""
        # Implement actual latency testing
        # Return (passed, details)
        return (True, "p95 latency: 1.2s")
    
    def _test_throughput(self):
        """Test throughput capacity"""
        return (True, "Sustained 150 req/sec")
    
    def _test_gpu_utilization(self):
        """Test GPU utilization"""
        return (True, "Average utilization: 78%")
    
    def _test_ha(self):
        """Test high availability setup"""
        return (True, "Multi-AZ deployment configured")
    
    def _test_health_checks(self):
        """Test health check endpoints"""
        return (True, "Health endpoints responding")
    
    def _test_circuit_breakers(self):
        """Test circuit breaker implementation"""
        return (False, "Circuit breakers not implemented")
    
    def _test_retry_logic(self):
        """Test retry logic"""
        return (True, "Exponential backoff implemented")
    
    def _test_degradation(self):
        """Test graceful degradation"""
        return (True, "Fallback to cached responses")
    
    def _test_metrics(self):
        """Test metrics collection"""
        return (True, "Prometheus scraping active")
    
    def _test_dashboards(self):
        """Test monitoring dashboards"""
        return (True, "5 dashboards configured")
    
    def _test_alerting(self):
        """Test alerting setup"""
        return (True, "PagerDuty integration active")
    
    def _test_tracing(self):
        """Test distributed tracing"""
        return (False, "Tracing not configured")
    
    def _test_logging(self):
        """Test log aggregation"""
        return (True, "ELK stack configured")
    
    def _test_cost_tracking(self):
        """Test cost tracking"""
        return (True, "Cost tracking implemented")
    
    def _test_auth(self):
        """Test authentication"""
        return (True, "JWT authentication active")
    
    def _test_rate_limiting(self):
        """Test rate limiting"""
        return (True, "Redis-based rate limiting")
    
    def _test_input_validation(self):
        """Test input validation"""
        return (True, "Input sanitization active")
    
    def _test_pii_detection(self):
        """Test PII detection"""
        return (True, "PII detection implemented")
    
    def _test_encryption_rest(self):
        """Test encryption at rest"""
        return (True, "AES-256 encryption enabled")
    
    def _test_encryption_transit(self):
        """Test encryption in transit"""
        return (True, "TLS 1.3 enforced")
    
    def _test_secrets(self):
        """Test secrets management"""
        return (True, "AWS Secrets Manager in use")
    
    def _test_content_filter(self):
        """Test content filtering"""
        return (True, "Content moderation API integrated")
    
    def _test_toxicity(self):
        """Test toxicity detection"""
        return (False, "Toxicity classifier not deployed")
    
    def _test_output_validation(self):
        """Test output validation"""
        return (True, "Output validation rules applied")
    
    def _test_prompt_injection(self):
        """Test prompt injection protection"""
        return (True, "Prompt validation implemented")
    
    def _test_bias(self):
        """Test bias evaluation"""
        return (False, "Bias testing pending")
    
    def _test_gdpr(self):
        """Test GDPR compliance"""
        return (True, "Data deletion API implemented")
    
    def _test_retention(self):
        """Test data retention policy"""
        return (True, "30-day retention configured")
    
    def _test_audit_logs(self):
        """Test audit logging"""
        return (True, "Audit logs enabled")
    
    def _test_privacy_policy(self):
        """Test privacy policy"""
        return (True, "Privacy policy published")
    
    def _test_tos(self):
        """Test terms of service"""
        return (True, "ToS published")
    
    def _test_runbooks(self):
        """Test runbook documentation"""
        return (True, "Runbooks in wiki")
    
    def _test_oncall(self):
        """Test on-call rotation"""
        return (True, "24/7 on-call rotation")
    
    def _test_incident_plan(self):
        """Test incident response plan"""
        return (True, "IR plan documented")
    
    def _test_deployment(self):
        """Test deployment automation"""
        return (True, "GitOps with ArgoCD")
    
    def _test_rollback(self):
        """Test rollback capability"""
        return (True, "Rollback tested successfully")
    
    def _test_backup(self):
        """Test backup and restore"""
        return (True, "Daily backups configured")
    
    def run_all_checks(self):
        """Run all readiness checks"""
        self.check_performance()
        self.check_reliability()
        self.check_monitoring()
        self.check_security()
        self.check_safety()
        self.check_compliance()
        self.check_operations()
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate readiness report"""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.status == CheckStatus.PASS)
        failed = sum(1 for c in self.checks if c.status == CheckStatus.FAIL)
        warned = sum(1 for c in self.checks if c.status == CheckStatus.WARN)
        
        # Group by category
        by_category = {}
        for check in self.checks:
            if check.category not in by_category:
                by_category[check.category] = []
            by_category[check.category].append(check)
        
        # Calculate readiness score
        score = (passed / total) * 100 if total > 0 else 0
        
        # Determine readiness level
        if score >= 95 and failed == 0:
            readiness = "PRODUCTION READY"
        elif score >= 80 and failed <= 2:
            readiness = "READY WITH WARNINGS"
        elif score >= 60:
            readiness = "NOT READY - MAJOR ISSUES"
        else:
            readiness = "NOT READY - CRITICAL ISSUES"
        
        return {
            'readiness': readiness,
            'score': round(score, 1),
            'total_checks': total,
            'passed': passed,
            'failed': failed,
            'warned': warned,
            'by_category': by_category,
            'all_checks': self.checks
        }
    
    def print_report(self):
        """Print formatted report"""
        report = self.generate_report()
        
        print("=" * 70)
        print(f"PRODUCTION READINESS REPORT".center(70))
        print("=" * 70)
        print()
        print(f"Status: {report['readiness']}")
        print(f"Score: {report['score']}%")
        print(f"Checks: {report['passed']}/{report['total_checks']} passed, "
              f"{report['failed']} failed, {report['warned']} warnings")
        print()
        
        # Print by category
        for category, checks in report['by_category'].items():
            print(f"\n{category}:")
            print("-" * 70)
            
            for check in checks:
                print(f"{check.status.value} {check.name}")
                print(f"    {check.details}")
                if check.remediation:
                    print(f"    → Remediation: {check.remediation}")
        
        print("\n" + "=" * 70)
        
        # Action items
        if report['failed'] > 0 or report['warned'] > 0:
            print("\nACTION ITEMS:")
            print("-" * 70)
            
            for check in report['all_checks']:
                if check.status in [CheckStatus.FAIL, CheckStatus.WARN]:
                    print(f"• {check.name}: {check.remediation}")
        
        print()


# Usage
if __name__ == "__main__":
    checker = ProductionReadinessChecker()
    checker.run_all_checks()
    checker.print_report()
```

## High Availability and Fault Tolerance

### Multi-Region Deployment

```yaml
# multi-region-deployment.yaml - High availability setup
apiVersion: v1
kind: Namespace
metadata:
  name: llm-platform-ha

---
# Primary region deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-primary
  namespace: llm-platform-ha
  labels:
    region: us-east-1
spec:
  replicas: 5
  selector:
    matchLabels:
      app: llm-server
      region: us-east-1
  template:
    metadata:
      labels:
        app: llm-server
        region: us-east-1
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - llm-server
            topologyKey: "kubernetes.io/hostname"
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300
          periodSeconds: 30
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          failureThreshold: 2

---
# Secondary region deployment (DR)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-secondary
  namespace: llm-platform-ha
  labels:
    region: us-west-2
spec:
  replicas: 3  # Smaller capacity for DR
  selector:
    matchLabels:
      app: llm-server
      region: us-west-2
  template:
    metadata:
      labels:
        app: llm-server
        region: us-west-2
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1

---
# Global load balancer configuration (AWS Global Accelerator example)
apiVersion: v1
kind: Service
metadata:
  name: llm-global-lb
  namespace: llm-platform-ha
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  selector:
    app: llm-server
  ports:
  - port: 80
    targetPort: 8000
```

### Circuit Breaker Implementation

```python
# circuit_breaker.py - Circuit breaker for LLM calls
from enum import Enum
from datetime import datetime, timedelta
import asyncio
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failures detected, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit breaker for LLM service calls
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60,
        half_open_timeout: int = 30
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_timeout = half_open_timeout
        
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time = None
        self.last_state_change = datetime.utcnow()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        """
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. "
                    f"Will retry at {self.last_failure_time + timedelta(seconds=self.timeout)}"
                )
        
        try:
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) \
                else func(*args, **kwargs)
            
            # Record success
            self._on_success()
            return result
            
        except Exception as e:
            # Record failure
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failures = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            if self.successes >= self.success_threshold:
                self._transition_to_closed()
    
    def _on_failure(self):
        """Handle failed call"""
        self.failures += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failures >= self.failure_threshold:
            self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try resetting"""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.timeout
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_state_change = datetime.utcnow()
        print(f"Circuit breaker: CLOSED")
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.utcnow()
        print(f"Circuit breaker: OPEN (too many failures)")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.successes = 0
        self.last_state_change = datetime.utcnow()
        print(f"Circuit breaker: HALF_OPEN (testing recovery)")
    
    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            'state': self.state.value,
            'failures': self.failures,
            'successes': self.successes,
            'last_state_change': self.last_state_change.isoformat()
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Usage with LLM calls
import httpx

class ResilientLLMClient:
    """
    LLM client with circuit breaker
    """
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60
        )
    
    async def generate(self, prompt: str, **kwargs):
        """
        Generate completion with circuit breaker protection
        """
        async def llm_call():
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.endpoint}/v1/completions",
                    json={'prompt': prompt, **kwargs}
                )
                response.raise_for_status()
                return response.json()
        
        try:
            return await self.circuit_breaker.call(llm_call)
        except CircuitBreakerOpenError as e:
            # Return cached or fallback response
            return self._get_fallback_response(prompt)
    
    def _get_fallback_response(self, prompt: str):
        """Fallback response when circuit is open"""
        return {
            'text': "Service temporarily unavailable. Please try again later.",
            'fallback': True
        }
```

## Disaster Recovery Strategies

### Backup and Restore Procedures

```python
# disaster_recovery.py - DR procedures for LLM platform
import boto3
from datetime import datetime, timedelta
import subprocess
import json

class DisasterRecoveryManager:
    """
    Manage disaster recovery for LLM platform
    """
    def __init__(self, config: dict):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.backup_bucket = config['backup_bucket']
    
    def backup_model_artifacts(self, model_path: str):
        """
        Backup model files to S3
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_key = f"models/backup_{timestamp}/"
        
        print(f"Backing up model from {model_path}")
        
        # Upload to S3 with versioning
        for root, dirs, files in os.walk(model_path):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = backup_key + os.path.relpath(local_path, model_path)
                
                self.s3_client.upload_file(
                    local_path,
                    self.backup_bucket,
                    s3_path
                )
        
        # Store backup metadata
        metadata = {
            'timestamp': timestamp,
            'model_path': model_path,
            's3_location': f"s3://{self.backup_bucket}/{backup_key}",
            'backup_type': 'full'
        }
        
        self.s3_client.put_object(
            Bucket=self.backup_bucket,
            Key=f"{backup_key}metadata.json",
            Body=json.dumps(metadata)
        )
        
        return metadata
    
    def backup_configuration(self):
        """
        Backup Kubernetes configurations
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_key = f"configs/backup_{timestamp}/"
        
        # Export Kubernetes resources
        namespaces = ['llm-platform', 'llm-platform-ha']
        
        for namespace in namespaces:
            # Export deployments
            result = subprocess.run(
                f"kubectl get all -n {namespace} -o yaml".split(),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.s3_client.put_object(
                    Bucket=self.backup_bucket,
                    Key=f"{backup_key}{namespace}_resources.yaml",
                    Body=result.stdout
                )
        
        print(f"Configuration backed up to {backup_key}")
        return backup_key
    
    def backup_database(self):
        """
        Backup application databases
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # PostgreSQL backup
        db_config = self.config['database']
        dump_file = f"/tmp/db_backup_{timestamp}.sql"
        
        subprocess.run([
            'pg_dump',
            '-h', db_config['host'],
            '-U', db_config['user'],
            '-d', db_config['database'],
            '-f', dump_file
        ])
        
        # Upload to S3
        self.s3_client.upload_file(
            dump_file,
            self.backup_bucket,
            f"database/backup_{timestamp}.sql"
        )
        
        # Cleanup
        os.remove(dump_file)
        
        print(f"Database backed up")
    
    def restore_from_backup(self, backup_timestamp: str):
        """
        Restore from backup
        """
        print(f"Restoring from backup: {backup_timestamp}")
        
        # Download model
        model_key = f"models/backup_{backup_timestamp}/"
        # Implementation details...
        
        # Restore configuration
        config_key = f"configs/backup_{backup_timestamp}/"
        # Implementation details...
        
        # Restore database
        db_key = f"database/backup_{backup_timestamp}.sql"
        # Implementation details...
        
        print("Restore complete")
    
    def test_disaster_recovery(self):
        """
        Test DR procedures
        """
        print("Testing disaster recovery procedures...")
        
        # 1. Create backup
        backup_metadata = self.backup_model_artifacts('/models/llama-2-7b')
        
        # 2. Simulate failure
        # (In testing environment only)
        
        # 3. Restore from backup
        self.restore_from_backup(backup_metadata['timestamp'])
        
        # 4. Verify restoration
        # Check model loads correctly
        # Check service is operational
        
        print("DR test complete")
    
    def create_recovery_plan(self):
        """
        Generate recovery plan document
        """
        plan = {
            'rpo': '1 hour',  # Recovery Point Objective
            'rto': '30 minutes',  # Recovery Time Objective
            'backup_frequency': 'hourly',
            'backup_retention': '30 days',
            'procedures': [
                {
                    'step': 1,
                    'action': 'Identify failure scope',
                    'time_estimate': '5 minutes'
                },
                {
                    'step': 2,
                    'action': 'Activate DR site',
                    'time_estimate': '10 minutes'
                },
                {
                    'step': 3,
                    'action': 'Restore from backup',
                    'time_estimate': '15 minutes'
                },
                {
                    'step': 4,
                    'action': 'Validate services',
                    'time_estimate': '5 minutes'
                },
                {
                    'step': 5,
                    'action': 'Update DNS/routing',
                    'time_estimate': '5 minutes'
                }
            ]
        }
        
        return plan
```

## Blue-Green Deployments

### Blue-Green Deployment Strategy

```python
# blue_green_deployment.py - Blue-green deployment for LLM models
from typing import Dict
import time

class BlueGreenDeployer:
    """
    Manage blue-green deployments for LLM models
    """
    def __init__(self, k8s_client):
        self.k8s = k8s_client
        self.namespace = 'llm-platform'
    
    def deploy_new_version(
        self,
        model_name: str,
        new_version: str,
        model_path: str
    ):
        """
        Deploy new model version using blue-green strategy
        """
        # Get current active color
        current_color = self._get_active_color(model_name)
        new_color = 'green' if current_color == 'blue' else 'blue'
        
        print(f"Current active: {current_color}")
        print(f"Deploying to: {new_color}")
        
        # 1. Deploy to inactive environment
        self._deploy_to_environment(
            model_name=model_name,
            version=new_version,
            color=new_color,
            model_path=model_path
        )
        
        # 2. Wait for deployment to be ready
        print("Waiting for new deployment to be ready...")
        self._wait_for_ready(model_name, new_color)
        
        # 3. Run smoke tests on new deployment
        print("Running smoke tests...")
        if not self._run_smoke_tests(model_name, new_color):
            print("Smoke tests failed! Rolling back...")
            self._rollback(model_name, new_color)
            return False
        
        # 4. Switch traffic to new deployment
        print(f"Switching traffic to {new_color}...")
        self._switch_traffic(model_name, new_color)
        
        # 5. Monitor for issues
        print("Monitoring new deployment...")
        time.sleep(300)  # Monitor for 5 minutes
        
        if not self._check_health_metrics(model_name, new_color):
            print("Health check failed! Rolling back...")
            self._switch_traffic(model_name, current_color)
            return False
        
        # 6. Keep old deployment for quick rollback
        print(f"Deployment successful! {new_color} is now active.")
        print(f"Keeping {current_color} for 24 hours for quick rollback.")
        
        return True
    
    def _deploy_to_environment(
        self,
        model_name: str,
        version: str,
        color: str,
        model_path: str
    ):
        """Deploy to specific environment (blue or green)"""
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_name}-{color}
  namespace: {self.namespace}
  labels:
    app: {model_name}
    color: {color}
    version: {version}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {model_name}
      color: {color}
  template:
    metadata:
      labels:
        app: {model_name}
        color: {color}
        version: {version}
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model={model_path}
        - --tensor-parallel-size=1
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: {model_name}-{color}
  namespace: {self.namespace}
spec:
  selector:
    app: {model_name}
    color: {color}
  ports:
  - port: 8000
    targetPort: 8000
"""
        # Apply deployment
        self.k8s.apply(deployment_yaml)
    
    def _switch_traffic(self, model_name: str, target_color: str):
        """Switch traffic to target environment"""
        # Update main service to point to target color
        service_yaml = f"""
apiVersion: v1
kind: Service
metadata:
  name: {model_name}
  namespace: {self.namespace}
spec:
  selector:
    app: {model_name}
    color: {target_color}
  ports:
  - port: 8000
    targetPort: 8000
"""
        self.k8s.apply(service_yaml)
    
    def _wait_for_ready(self, model_name: str, color: str, timeout: int = 600):
        """Wait for deployment to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.k8s.get_deployment_status(
                f"{model_name}-{color}",
                self.namespace
            )
            
            if status['ready_replicas'] == status['desired_replicas']:
                return True
            
            time.sleep(10)
        
        return False
    
    def _run_smoke_tests(self, model_name: str, color: str) -> bool:
        """Run smoke tests on deployment"""
        endpoint = f"http://{model_name}-{color}.{self.namespace}:8000"
        
        test_prompts = [
            "Hello, how are you?",
            "What is 2+2?",
            "Write a haiku about clouds."
        ]
        
        for prompt in test_prompts:
            try:
                response = requests.post(
                    f"{endpoint}/v1/completions",
                    json={'prompt': prompt, 'max_tokens': 50},
                    timeout=30
                )
                
                if response.status_code != 200:
                    return False
                
                # Validate response structure
                data = response.json()
                if 'text' not in data or not data['text']:
                    return False
                    
            except Exception as e:
                print(f"Smoke test failed: {e}")
                return False
        
        return True
    
    def _check_health_metrics(self, model_name: str, color: str) -> bool:
        """Check health metrics for deployment"""
        # Query Prometheus for error rate, latency, etc.
        # Return True if metrics are healthy
        return True
    
    def _rollback(self, model_name: str, color: str):
        """Rollback failed deployment"""
        # Delete failed deployment
        self.k8s.delete_deployment(f"{model_name}-{color}", self.namespace)
    
    def _get_active_color(self, model_name: str) -> str:
        """Get currently active color"""
        service = self.k8s.get_service(model_name, self.namespace)
        return service['spec']['selector'].get('color', 'blue')
```

## Canary Deployments

### Gradual Rollout Strategy

```python
# canary_deployment.py - Canary deployment for LLM models
import time
from typing import Dict

class CanaryDeployer:
    """
    Manage canary deployments with gradual traffic shifting
    """
    def __init__(self, k8s_client, metrics_client):
        self.k8s = k8s_client
        self.metrics = metrics_client
        self.namespace = 'llm-platform'
    
    def deploy_canary(
        self,
        model_name: str,
        new_version: str,
        model_path: str,
        stages: list = [5, 10, 25, 50, 100]
    ):
        """
        Deploy new version with canary strategy
        
        Args:
            stages: Percentage of traffic to route to canary at each stage
        """
        print(f"Starting canary deployment for {model_name} v{new_version}")
        
        # 1. Deploy canary with 0% traffic
        self._deploy_canary_version(model_name, new_version, model_path)
        
        # 2. Wait for canary to be ready
        if not self._wait_for_ready(model_name, 'canary'):
            print("Canary deployment failed to start")
            return False
        
        # 3. Gradually increase traffic
        for stage_percent in stages:
            print(f"\n=== Stage: {stage_percent}% traffic to canary ===")
            
            # Update traffic split
            self._update_traffic_split(model_name, stage_percent)
            
            # Monitor for duration
            monitor_duration = 300 if stage_percent < 50 else 600  # 5-10 minutes
            print(f"Monitoring for {monitor_duration} seconds...")
            
            if not self._monitor_canary(model_name, monitor_duration):
                print(f"Canary failed at {stage_percent}% stage!")
                self._rollback_canary(model_name)
                return False
            
            print(f"Stage {stage_percent}% successful!")
        
        # 4. Promote canary to stable
        print("\nPromoting canary to stable...")
        self._promote_canary(model_name, new_version)
        
        print("Canary deployment successful!")
        return True
    
    def _deploy_canary_version(
        self,
        model_name: str,
        version: str,
        model_path: str
    ):
        """Deploy canary version"""
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_name}-canary
  namespace: {self.namespace}
  labels:
    app: {model_name}
    track: canary
    version: {version}
spec:
  replicas: 2  # Start with fewer replicas
  selector:
    matchLabels:
      app: {model_name}
      track: canary
  template:
    metadata:
      labels:
        app: {model_name}
        track: canary
        version: {version}
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model={model_path}
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
"""
        self.k8s.apply(deployment_yaml)
    
    def _update_traffic_split(self, model_name: str, canary_percent: int):
        """Update traffic split using Istio VirtualService"""
        stable_percent = 100 - canary_percent
        
        virtualservice_yaml = f"""
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: {model_name}
  namespace: {self.namespace}
spec:
  hosts:
  - {model_name}
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: {model_name}
        subset: canary
      weight: 100
  - route:
    - destination:
        host: {model_name}
        subset: stable
      weight: {stable_percent}
    - destination:
        host: {model_name}
        subset: canary
      weight: {canary_percent}
"""
        self.k8s.apply(virtualservice_yaml)
    
    def _monitor_canary(self, model_name: str, duration: int) -> bool:
        """
        Monitor canary deployment for issues
        
        Returns False if any critical metric fails
        """
        start_time = time.time()
        check_interval = 30  # Check every 30 seconds
        
        thresholds = {
            'error_rate': 0.01,  # 1% error rate
            'latency_p95': 2000,  # 2 seconds
            'latency_increase': 1.5  # 50% increase over stable
        }
        
        while time.time() - start_time < duration:
            # Get metrics for canary and stable
            canary_metrics = self.metrics.get_metrics(model_name, 'canary')
            stable_metrics = self.metrics.get_metrics(model_name, 'stable')
            
            # Check error rate
            if canary_metrics['error_rate'] > thresholds['error_rate']:
                print(f"ERROR: Canary error rate too high: {canary_metrics['error_rate']}")
                return False
            
            # Check latency
            if canary_metrics['latency_p95'] > thresholds['latency_p95']:
                print(f"ERROR: Canary latency too high: {canary_metrics['latency_p95']}ms")
                return False
            
            # Check latency increase vs stable
            if stable_metrics['latency_p95'] > 0:
                latency_ratio = canary_metrics['latency_p95'] / stable_metrics['latency_p95']
                if latency_ratio > thresholds['latency_increase']:
                    print(f"ERROR: Canary latency {latency_ratio:.2f}x higher than stable")
                    return False
            
            # Print status
            print(f"  Canary metrics: "
                  f"error_rate={canary_metrics['error_rate']:.4f}, "
                  f"p95_latency={canary_metrics['latency_p95']:.0f}ms")
            
            time.sleep(check_interval)
        
        return True
    
    def _rollback_canary(self, model_name: str):
        """Rollback canary deployment"""
        print("Rolling back canary deployment...")
        
        # Set traffic to 0% canary
        self._update_traffic_split(model_name, 0)
        
        # Delete canary deployment
        self.k8s.delete_deployment(f"{model_name}-canary", self.namespace)
    
    def _promote_canary(self, model_name: str, version: str):
        """Promote canary to stable"""
        # Update stable deployment to new version
        # Delete canary deployment
        # Set traffic to 100% stable
        pass
```

Due to character limits, I'll complete the remaining sections more concisely to finish the lesson.

## A/B Testing Frameworks

### A/B Test Implementation

```python
# ab_testing.py - A/B testing framework for LLMs
import hashlib
import random
from typing import Dict, Optional

class ABTestManager:
    """
    Manage A/B tests for LLM models and prompts
    """
    def __init__(self):
        self.experiments = {}
        self.results = {}
    
    def create_experiment(
        self,
        experiment_id: str,
        variant_a: dict,
        variant_b: dict,
        traffic_split: float = 0.5,
        metrics: list = None
    ):
        """
        Create new A/B test experiment
        
        Args:
            experiment_id: Unique identifier
            variant_a: Control variant configuration
            variant_b: Treatment variant configuration
            traffic_split: Percentage to variant B (0.0-1.0)
            metrics: List of metrics to track
        """
        self.experiments[experiment_id] = {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'traffic_split': traffic_split,
            'metrics': metrics or ['latency', 'quality', 'user_satisfaction'],
            'active': True
        }
        
        self.results[experiment_id] = {
            'variant_a': {'count': 0, 'metrics': {}},
            'variant_b': {'count': 0, 'metrics': {}}
        }
    
    def get_variant(self, experiment_id: str, user_id: str) -> str:
        """
        Determine which variant to show user
        
        Uses consistent hashing to ensure same user always gets same variant
        """
        if experiment_id not in self.experiments:
            return 'variant_a'  # Default to control
        
        experiment = self.experiments[experiment_id]
        
        # Hash user_id to determine variant
        hash_value = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest(), 16)
        threshold = experiment['traffic_split']
        
        return 'variant_b' if (hash_value % 100) / 100 < threshold else 'variant_a'
    
    def track_result(
        self,
        experiment_id: str,
        variant: str,
        metrics: Dict[str, float]
    ):
        """Track experiment results"""
        if experiment_id not in self.results:
            return
        
        variant_results = self.results[experiment_id][variant]
        variant_results['count'] += 1
        
        for metric_name, value in metrics.items():
            if metric_name not in variant_results['metrics']:
                variant_results['metrics'][metric_name] = []
            variant_results['metrics'][metric_name].append(value)
    
    def get_results(self, experiment_id: str) -> Dict:
        """Get experiment results with statistical analysis"""
        if experiment_id not in self.results:
            return {}
        
        import numpy as np
        from scipy import stats
        
        results = self.results[experiment_id]
        analysis = {}
        
        for variant in ['variant_a', 'variant_b']:
            variant_data = results[variant]
            analysis[variant] = {
                'sample_size': variant_data['count'],
                'metrics': {}
            }
            
            for metric_name, values in variant_data['metrics'].items():
                if values:
                    analysis[variant]['metrics'][metric_name] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'p95': np.percentile(values, 95)
                    }
        
        # Statistical significance testing
        for metric_name in analysis['variant_a']['metrics'].keys():
            a_values = results['variant_a']['metrics'][metric_name]
            b_values = results['variant_b']['metrics'][metric_name]
            
            if a_values and b_values:
                t_stat, p_value = stats.ttest_ind(a_values, b_values)
                
                analysis['significance'] = analysis.get('significance', {})
                analysis['significance'][metric_name] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'winner': 'variant_b' if np.mean(b_values) > np.mean(a_values) else 'variant_a'
                }
        
        return analysis
```

## Prompt Engineering Best Practices

### Systematic Prompt Optimization

```python
# prompt_engineering.py - Best practices for prompt engineering
from typing import List, Dict
import re

class PromptOptimizer:
    """
    Systematic prompt engineering and optimization
    """
    def __init__(self):
        self.templates = {}
        self.test_cases = {}
    
    def create_template(
        self,
        template_id: str,
        template: str,
        variables: List[str]
    ):
        """
        Create reusable prompt template
        
        Example:
            template = "You are a {role}. {instruction}\n\nContext: {context}\n\nQuestion: {question}"
            variables = ['role', 'instruction', 'context', 'question']
        """
        self.templates[template_id] = {
            'template': template,
            'variables': variables
        }
    
    def format_prompt(
        self,
        template_id: str,
        **kwargs
    ) -> str:
        """Format prompt from template"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]['template']
        return template.format(**kwargs)
    
    def test_prompt(
        self,
        template_id: str,
        test_cases: List[Dict],
        llm_client
    ) -> List[Dict]:
        """
        Test prompt template with multiple cases
        
        Args:
            test_cases: List of {inputs: dict, expected_output: str}
        """
        results = []
        
        for case in test_cases:
            # Format prompt
            prompt = self.format_prompt(template_id, **case['inputs'])
            
            # Generate output
            response = llm_client.generate(prompt)
            
            # Evaluate
            result = {
                'inputs': case['inputs'],
                'prompt': prompt,
                'output': response['text'],
                'expected': case['expected_output'],
                'score': self._evaluate_output(
                    response['text'],
                    case['expected_output']
                )
            }
            results.append(result)
        
        return results
    
    def _evaluate_output(self, actual: str, expected: str) -> float:
        """
        Evaluate output quality
        
        Returns score from 0.0 to 1.0
        """
        # Simple similarity check (replace with more sophisticated evaluation)
        actual_lower = actual.lower().strip()
        expected_lower = expected.lower().strip()
        
        # Check if key phrases are present
        key_phrases = re.findall(r'\w+', expected_lower)
        matches = sum(1 for phrase in key_phrases if phrase in actual_lower)
        
        return matches / len(key_phrases) if key_phrases else 0.0
    
    def optimize_prompt(
        self,
        base_prompt: str,
        test_cases: List[Dict],
        llm_client,
        variations: List[str]
    ) -> Dict:
        """
        Test multiple prompt variations and find best performer
        
        Args:
            variations: List of prompt variations to test
        """
        results = {}
        
        for i, variation in enumerate(variations):
            variation_id = f"variation_{i}"
            self.create_template(variation_id, variation, [])
            
            test_results = self.test_prompt(variation_id, test_cases, llm_client)
            avg_score = sum(r['score'] for r in test_results) / len(test_results)
            
            results[variation_id] = {
                'prompt': variation,
                'average_score': avg_score,
                'test_results': test_results
            }
        
        # Find best variation
        best = max(results.items(), key=lambda x: x[1]['average_score'])
        
        return {
            'best_variation': best[0],
            'best_score': best[1]['average_score'],
            'all_results': results
        }


# Best Practices Collection
PROMPT_BEST_PRACTICES = {
    'specificity': {
        'principle': 'Be specific and detailed in instructions',
        'bad_example': 'Summarize this',
        'good_example': 'Summarize this article in 3 bullet points, focusing on the main findings',
    },
    'context': {
        'principle': 'Provide relevant context',
        'bad_example': 'What is it?',
        'good_example': 'Based on the product description above, what are the key features?',
    },
    'format': {
        'principle': 'Specify desired output format',
        'bad_example': 'List the items',
        'good_example': 'List the items in JSON format with keys: name, price, quantity',
    },
    'examples': {
        'principle': 'Include examples (few-shot learning)',
        'example': '''Task: Extract person names from text

Example 1:
Text: "John and Mary went to the store"
Names: ["John", "Mary"]

Example 2:
Text: "The CEO Sarah announced new policies"
Names: ["Sarah"]

Now extract from:
Text: "{input_text}"
Names:'''
    },
    'role': {
        'principle': 'Define the AI role/persona',
        'example': 'You are an expert Python developer. Review this code for potential bugs and security issues.'
    },
    'constraints': {
        'principle': 'Set clear constraints and boundaries',
        'example': 'Answer in less than 100 words. Do not include personal opinions. Only use information from the provided context.'
    }
}
```

## Safety and Content Filtering

### Content Moderation System

```python
# content_moderation.py - Safety and content filtering
from typing import Dict, List, Optional
import re

class ContentModerator:
    """
    Content moderation and safety filtering for LLM inputs/outputs
    """
    def __init__(self):
        # Load toxicity classifier
        from transformers import pipeline
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert"
        )
        
        # Define content categories
        self.categories = {
            'hate_speech': [],
            'violence': [],
            'sexual': [],
            'self_harm': [],
            'illegal': []
        }
        
        # Load category-specific classifiers
        self._load_classifiers()
    
    def _load_classifiers(self):
        """Load specialized classifiers for each category"""
        # Implementation would load actual models
        pass
    
    def moderate_input(self, text: str) -> Dict:
        """
        Moderate user input before sending to LLM
        
        Returns:
            {
                'safe': bool,
                'categories': dict,
                'filtered_text': str,
                'reason': str
            }
        """
        # Check for banned patterns
        if self._contains_banned_patterns(text):
            return {
                'safe': False,
                'categories': {'banned_pattern': True},
                'filtered_text': '[BLOCKED]',
                'reason': 'Contains prohibited content'
            }
        
        # Run toxicity classification
        toxicity_result = self.toxicity_classifier(text)[0]
        
        if toxicity_result['label'] == 'toxic' and toxicity_result['score'] > 0.7:
            return {
                'safe': False,
                'categories': {'toxicity': toxicity_result['score']},
                'filtered_text': '[BLOCKED]',
                'reason': 'High toxicity detected'
            }
        
        # Run category-specific checks
        categories = self._check_categories(text)
        
        if any(score > 0.8 for score in categories.values()):
            return {
                'safe': False,
                'categories': categories,
                'filtered_text': '[BLOCKED]',
                'reason': 'Unsafe content category detected'
            }
        
        return {
            'safe': True,
            'categories': categories,
            'filtered_text': text,
            'reason': None
        }
    
    def moderate_output(self, text: str) -> Dict:
        """
        Moderate LLM output before returning to user
        
        Includes additional checks for:
        - Hallucinations
        - PII leakage
        - Factual errors
        """
        result = self.moderate_input(text)
        
        # Additional output-specific checks
        if result['safe']:
            # Check for PII
            pii_check = self._check_pii(text)
            if pii_check['found']:
                result['safe'] = False
                result['reason'] = 'PII detected in output'
                result['pii_detected'] = pii_check['types']
        
        return result
    
    def _contains_banned_patterns(self, text: str) -> bool:
        """Check for explicitly banned patterns"""
        banned_patterns = [
            r'kill yourself',
            r'how to make explosives',
            # Add more banned patterns
        ]
        
        text_lower = text.lower()
        for pattern in banned_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _check_categories(self, text: str) -> Dict[str, float]:
        """Check content against all categories"""
        # Simplified - real implementation would use ML models
        return {
            'hate_speech': 0.1,
            'violence': 0.05,
            'sexual': 0.02,
            'self_harm': 0.01,
            'illegal': 0.03
        }
    
    def _check_pii(self, text: str) -> Dict:
        """Check for PII in text"""
        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
        found_types = []
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, text):
                found_types.append(pii_type)
        
        return {
            'found': len(found_types) > 0,
            'types': found_types
        }
```

## PII Detection and Redaction

Already covered in Lesson 07, but here's production implementation:

```python
# Advanced PII redaction with named entity recognition
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class ProductionPIIHandler:
    """Production-grade PII detection and anonymization"""
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
    
    def detect_and_redact(self, text: str) -> Dict:
        """Detect and redact PII"""
        # Analyze
        results = self.analyzer.analyze(
            text=text,
            language='en',
            entities=['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD', 'SSN']
        )
        
        # Anonymize
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )
        
        return {
            'original': text,
            'redacted': anonymized.text,
            'pii_found': [
                {
                    'type': result.entity_type,
                    'start': result.start,
                    'end': result.end,
                    'score': result.score
                }
                for result in results
            ]
        }
```

## GDPR and Data Privacy

### GDPR Compliance Implementation

```python
# gdpr_compliance.py
from datetime import datetime, timedelta
from typing import Dict, List

class GDPRComplianceManager:
    """
    Manage GDPR compliance for LLM platform
    """
    def __init__(self, db_client):
        self.db = db_client
    
    def handle_data_subject_request(
        self,
        request_type: str,
        user_id: str,
        email: str
    ) -> Dict:
        """
        Handle GDPR data subject requests
        
        Types: access, rectification, erasure, portability, restriction
        """
        if request_type == 'access':
            return self._handle_access_request(user_id, email)
        elif request_type == 'erasure':
            return self._handle_erasure_request(user_id, email)
        elif request_type == 'portability':
            return self._handle_portability_request(user_id, email)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    def _handle_access_request(self, user_id: str, email: str) -> Dict:
        """Provide all data for user (right to access)"""
        # Collect all user data
        user_data = {
            'user_profile': self.db.get_user_profile(user_id),
            'usage_history': self.db.get_usage_history(user_id),
            'preferences': self.db.get_preferences(user_id),
            'stored_prompts': self.db.get_prompts(user_id),
            'api_keys': self.db.get_api_keys(user_id)
        }
        
        # Generate downloadable package
        package_id = self._generate_data_package(user_data)
        
        return {
            'request_type': 'access',
            'user_id': user_id,
            'package_id': package_id,
            'download_url': f'/api/gdpr/download/{package_id}',
            'expires_at': (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
    
    def _handle_erasure_request(self, user_id: str, email: str) -> Dict:
        """Delete all user data (right to be forgotten)"""
        # Verify no legal obligation to retain
        if self._must_retain_data(user_id):
            return {
                'status': 'rejected',
                'reason': 'Legal obligation to retain data'
            }
        
        # Delete from all systems
        deletion_log = []
        
        # 1. Delete user profile
        self.db.delete_user_profile(user_id)
        deletion_log.append('user_profile')
        
        # 2. Anonymize usage logs (keep aggregated stats)
        self.db.anonymize_usage_logs(user_id)
        deletion_log.append('usage_logs_anonymized')
        
        # 3. Delete API keys
        self.db.delete_api_keys(user_id)
        deletion_log.append('api_keys')
        
        # 4. Delete cached data
        self.db.delete_cached_data(user_id)
        deletion_log.append('cached_data')
        
        # 5. Remove from backups (mark for deletion in next cycle)
        self.db.mark_for_backup_deletion(user_id)
        deletion_log.append('marked_for_backup_deletion')
        
        return {
            'status': 'completed',
            'user_id': user_id,
            'deleted_at': datetime.utcnow().isoformat(),
            'deleted_items': deletion_log
        }
    
    def _must_retain_data(self, user_id: str) -> bool:
        """Check if legally required to retain data"""
        # Check for active legal holds, investigations, etc.
        return False
    
    def _generate_data_package(self, user_data: Dict) -> str:
        """Generate downloadable data package"""
        import json
        import hashlib
        
        package_id = hashlib.md5(
            f"{user_data['user_profile']['user_id']}{datetime.utcnow()}".encode()
        ).hexdigest()
        
        # Store package
        self.db.store_data_package(package_id, json.dumps(user_data))
        
        return package_id
```

## Model Versioning Strategies

### Semantic Versioning for Models

```python
# model_versioning.py
from typing import Dict, List
from datetime import datetime

class ModelVersionManager:
    """
    Manage model versions with semantic versioning
    
    Version format: MAJOR.MINOR.PATCH-LABEL
    - MAJOR: Breaking changes (API changes, different architecture)
    - MINOR: New features (better performance, new capabilities)
    - PATCH: Bug fixes, minor improvements
    - LABEL: dev, beta, rc, stable
    """
    def __init__(self, registry_client):
        self.registry = registry_client
        self.versions = {}
    
    def register_version(
        self,
        model_name: str,
        version: str,
        metadata: Dict
    ):
        """Register new model version"""
        self.versions[f"{model_name}:{version}"] = {
            'model_name': model_name,
            'version': version,
            'registered_at': datetime.utcnow(),
            'metadata': metadata,
            'status': 'dev'
        }
        
        # Store in registry
        self.registry.push_model(model_name, version, metadata)
    
    def promote_version(
        self,
        model_name: str,
        version: str,
        to_status: str
    ):
        """
        Promote version through stages: dev -> beta -> rc -> stable
        """
        key = f"{model_name}:{version}"
        
        if key not in self.versions:
            raise ValueError(f"Version not found: {key}")
        
        valid_statuses = ['dev', 'beta', 'rc', 'stable']
        if to_status not in valid_statuses:
            raise ValueError(f"Invalid status: {to_status}")
        
        self.versions[key]['status'] = to_status
        self.versions[key]['promoted_at'] = datetime.utcnow()
        
        # Update registry
        self.registry.update_version_status(model_name, version, to_status)
    
    def get_latest_stable(self, model_name: str) -> str:
        """Get latest stable version"""
        stable_versions = [
            v for k, v in self.versions.items()
            if v['model_name'] == model_name and v['status'] == 'stable'
        ]
        
        if not stable_versions:
            return None
        
        # Sort by version
        stable_versions.sort(
            key=lambda x: self._version_tuple(x['version']),
            reverse=True
        )
        
        return stable_versions[0]['version']
    
    def _version_tuple(self, version: str) -> tuple:
        """Convert version string to tuple for comparison"""
        # Remove label
        version_num = version.split('-')[0]
        return tuple(map(int, version_num.split('.')))
```

## Incident Response and On-Call

### Incident Response Runbook

```python
# incident_response.py - Incident management system
from enum import Enum
from datetime import datetime
from typing import List, Dict

class IncidentSeverity(Enum):
    SEV1 = "Critical - Service Down"
    SEV2 = "Major - Degraded Performance"
    SEV3 = "Minor - Limited Impact"
    SEV4 = "Low - No User Impact"

class IncidentManager:
    """
    Manage incidents for LLM platform
    """
    def __init__(self, pagerduty_client, slack_client):
        self.pagerduty = pagerduty_client
        self.slack = slack_client
        self.incidents = {}
    
    def create_incident(
        self,
        title: str,
        severity: IncidentSeverity,
        description: str
    ) -> str:
        """Create new incident"""
        incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        incident = {
            'id': incident_id,
            'title': title,
            'severity': severity,
            'description': description,
            'status': 'investigating',
            'created_at': datetime.utcnow(),
            'timeline': [],
            'responders': []
        }
        
        self.incidents[incident_id] = incident
        
        # Page on-call for SEV1/SEV2
        if severity in [IncidentSeverity.SEV1, IncidentSeverity.SEV2]:
            self.pagerduty.create_incident(incident)
        
        # Create Slack channel
        channel = self.slack.create_channel(f"incident-{incident_id.lower()}")
        incident['slack_channel'] = channel
        
        # Post runbook
        self._post_runbook(incident)
        
        return incident_id
    
    def _post_runbook(self, incident: Dict):
        """Post relevant runbook to incident channel"""
        runbooks = {
            'high_latency': self._get_latency_runbook(),
            'service_down': self._get_downtime_runbook(),
            'high_error_rate': self._get_error_runbook(),
            'oom': self._get_oom_runbook()
        }
        
        # Determine relevant runbook based on incident
        # Post to Slack
        pass
    
    def _get_latency_runbook(self) -> str:
        return """
## High Latency Runbook

### 1. Check Current Metrics
- Grafana Dashboard: https://grafana.company.com/llm-latency
- Query: `histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))`

### 2. Immediate Actions
- [ ] Check GPU utilization: `kubectl top nodes -l gpu=true`
- [ ] Check for resource saturation
- [ ] Review recent deployments (rollback if needed)
- [ ] Check batch sizes and queue lengths

### 3. Investigation Steps
- [ ] Review application logs for slow queries
- [ ] Check database performance
- [ ] Verify network latency to downstream services
- [ ] Check for memory pressure

### 4. Mitigation Options
- [ ] Scale up GPU nodes
- [ ] Reduce batch size
- [ ] Enable caching for common requests
- [ ] Implement request shedding for overload

### 5. Communication
- Update status page every 15 minutes
- Post updates to #incidents channel
"""
```

## Summary

This lesson covered production best practices for LLM systems:

### Key Takeaways

1. **Production Readiness**: Comprehensive checklist covering performance, reliability, security, safety, compliance, and operations
2. **High Availability**: Multi-region deployments, circuit breakers, graceful degradation
3. **Disaster Recovery**: Backup strategies, recovery procedures, RTO/RPO objectives
4. **Safe Deployments**: Blue-green and canary strategies for risk-free rollouts
5. **A/B Testing**: Data-driven improvements with statistical significance
6. **Prompt Engineering**: Systematic optimization and testing
7. **Safety**: Content moderation, toxicity detection, output validation
8. **Privacy**: PII detection, GDPR compliance, data subject rights
9. **Versioning**: Semantic versioning, promotion pipelines
10. **Incident Response**: Runbooks, on-call procedures, escalation paths

### Production Checklist

- ✓ Comprehensive monitoring and alerting
- ✓ High availability across regions
- ✓ Disaster recovery tested regularly
- ✓ Safe deployment strategies (blue-green/canary)
- ✓ Content moderation and safety filters
- ✓ PII detection and redaction
- ✓ GDPR compliance mechanisms
- ✓ Model versioning and promotion
- ✓ Incident response procedures
- ✓ Cost optimization strategies
- ✓ Complete documentation

### Lessons from Production

1. **Start Simple**: Begin with proven patterns, add complexity as needed
2. **Monitor Everything**: Can't improve what you don't measure
3. **Automate Safety**: Human review doesn't scale
4. **Plan for Failures**: Assume everything will fail eventually
5. **Test Continuously**: Testing in production is inevitable, do it safely
6. **Document Thoroughly**: Future you will thank present you
7. **Iterate Based on Data**: Use metrics to guide improvements

### Next Steps

- Deploy a pilot project using these practices
- Build your incident response playbooks
- Establish baseline metrics for your use case
- Create your specific compliance checklist
- Set up monitoring and alerting
- Train your team on procedures
- Conduct DR drills regularly

---

**Congratulations!** You've completed the LLM Infrastructure module. You now have the knowledge to build and operate production-grade LLM systems.

**Module Complete**: [10-llm-infrastructure](./README.md)
