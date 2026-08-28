"""
Project 02: End-to-End MLOps Pipeline
Deployment Pipeline DAG - Automates model deployment to Kubernetes

This DAG demonstrates:
- Pulling models from MLflow Model Registry
- Building Docker images with new models
- Deploying to Kubernetes with rolling updates
- Health checks and rollback logic
- Integration with Project 01 (Kubernetes infrastructure)
- Notifications on deployment status

Learning Objectives:
- Understand CI/CD for ML models
- Learn Kubernetes deployment strategies
- Implement health checks and rollback mechanisms
- Integrate MLflow Model Registry with deployment
- Build production-ready deployment pipelines

Author: AI Infrastructure Learning
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.task_group import TaskGroup
import os
import time

# TODO: Import your custom deployment modules
# from src.deployment.deploy import pull_model_from_registry
# from src.deployment.deploy import build_docker_image
# from src.deployment.deploy import deploy_to_kubernetes
# from src.deployment.deploy import verify_deployment_health

# ==================== DAG Configuration ====================

default_args = {
    'owner': 'ml-ops-team',
    'depends_on_past': False,
    'email': ['mlops-team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}

# ==================== Helper Functions ====================

def check_model_registry_for_promotion(**context):
    """
    TODO: Check MLflow Model Registry for newly promoted models

    Steps:
    1. Connect to MLflow Model Registry
    2. Check for models in "Production" stage
    3. Compare with currently deployed version
    4. Determine if deployment is needed

    This task runs as a sensor - checks periodically until a new model is available.

    Returns:
        bool: True if new model is available for deployment, False otherwise
    """
    import logging
    import mlflow
    from mlflow.tracking import MlflowClient

    logger = logging.getLogger(__name__)
    logger.info("Checking MLflow Model Registry for new Production models...")

    # TODO: Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    client = MlflowClient()

    model_name = os.getenv('MODEL_NAME', 'mlops-classifier')

    # TODO: Get latest Production model
    """
    production_models = client.get_latest_versions(model_name, stages=["Production"])

    if not production_models:
        logger.info("No models in Production stage yet")
        return False

    latest_production = production_models[0]
    latest_version = latest_production.version

    # Check if this version is already deployed
    # You could store the deployed version in a ConfigMap or database
    currently_deployed_version = get_currently_deployed_version()

    if latest_version > currently_deployed_version:
        logger.info(f"New model version {latest_version} ready for deployment")
        return True
    else:
        logger.info(f"Model version {latest_version} already deployed")
        return False
    """

    # Placeholder - simulate new model available
    logger.info("Simulating new model available for deployment")
    return True


def pull_model_from_mlflow_registry(**context):
    """
    TODO: Pull the latest Production model from MLflow Model Registry

    Steps:
    1. Connect to MLflow Model Registry
    2. Get latest Production model
    3. Download model artifacts
    4. Extract model metadata (version, metrics, tags)
    5. Save model locally for containerization

    Returns:
        dict: Model information (name, version, path, metrics, uri)
    """
    import logging
    import mlflow
    from mlflow.tracking import MlflowClient

    logger = logging.getLogger(__name__)
    logger.info("Pulling model from MLflow Registry...")

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    client = MlflowClient()

    model_name = os.getenv('MODEL_NAME', 'mlops-classifier')

    # TODO: Download Production model
    """
    # Get latest Production version
    production_models = client.get_latest_versions(model_name, stages=["Production"])
    model_version = production_models[0]

    # Download model
    model_uri = f"models:/{model_name}/Production"
    local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path="./models/production")

    # Get model metadata
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics
    params = run.data.params

    model_info = {
        'model_name': model_name,
        'version': model_version.version,
        'path': local_path,
        'metrics': metrics,
        'params': params,
        'run_id': model_version.run_id,
        'model_uri': model_uri
    }
    """

    # Placeholder
    model_info = {
        'model_name': model_name,
        'version': 1,
        'path': './models/production/model',
        'metrics': {'accuracy': 0.92, 'f1_score': 0.90},
        'params': {'n_estimators': 100},
        'run_id': 'abc123',
        'model_uri': f"models:/{model_name}/Production"
    }

    logger.info(f"Model pulled: {model_info}")
    return model_info


def build_docker_image_with_model(**context):
    """
    TODO: Build Docker image containing the new model

    Steps:
    1. Get model information from previous task
    2. Create Dockerfile (or use template) that includes:
       - Base image (Python with ML dependencies)
       - Model files
       - Serving code (FastAPI/Flask endpoint)
       - Health check endpoint
    3. Build Docker image with version tag
    4. Tag image appropriately (version, latest)
    5. Push image to container registry

    Dockerfile should extend Project 01's model server with new model.

    Best Practices:
    - Use multi-stage builds to reduce image size
    - Include only necessary dependencies
    - Add health check instructions
    - Tag with model version and timestamp
    - Use semantic versioning

    Returns:
        dict: Docker image information (image_name, tag, registry_url)
    """
    import logging
    import subprocess

    logger = logging.getLogger(__name__)
    logger.info("Building Docker image with new model...")

    # Get model info from previous task
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='pull_model')

    # Docker image configuration
    registry = os.getenv('DOCKER_REGISTRY', 'localhost:5000')
    image_name = os.getenv('MODEL_IMAGE_NAME', 'mlops-model-server')
    model_version = model_info['version']
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Create tags
    version_tag = f"{registry}/{image_name}:v{model_version}"
    timestamp_tag = f"{registry}/{image_name}:v{model_version}-{timestamp}"
    latest_tag = f"{registry}/{image_name}:latest"

    # TODO: Build Docker image
    """
    # Build command
    build_cmd = [
        'docker', 'build',
        '-t', version_tag,
        '-t', timestamp_tag,
        '-t', latest_tag,
        '--build-arg', f'MODEL_VERSION={model_version}',
        '--build-arg', f'MODEL_PATH={model_info["path"]}',
        '-f', 'Dockerfile.model-server',
        '.'
    ]

    logger.info(f"Building image: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Docker build failed: {result.stderr}")
        raise RuntimeError(f"Docker build failed: {result.stderr}")

    logger.info(f"Image built successfully: {version_tag}")

    # Push to registry
    for tag in [version_tag, timestamp_tag, latest_tag]:
        push_cmd = ['docker', 'push', tag]
        logger.info(f"Pushing image: {tag}")
        result = subprocess.run(push_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Docker push failed: {result.stderr}")
            raise RuntimeError(f"Docker push failed: {result.stderr}")

    logger.info("Image pushed to registry successfully")
    """

    # Placeholder
    logger.info(f"Simulating Docker build for model version {model_version}")

    image_info = {
        'image_name': image_name,
        'version_tag': version_tag,
        'timestamp_tag': timestamp_tag,
        'latest_tag': latest_tag,
        'registry': registry,
        'model_version': model_version
    }

    logger.info(f"Docker image built: {image_info}")
    return image_info


def deploy_to_kubernetes_rolling_update(**context):
    """
    TODO: Deploy new model to Kubernetes using rolling update strategy

    Steps:
    1. Get image information from previous task
    2. Update Kubernetes deployment manifest with new image
    3. Apply deployment using kubectl or Python Kubernetes client
    4. Use rolling update strategy for zero-downtime deployment
    5. Configure readiness/liveness probes
    6. Set resource limits appropriately

    This extends Project 01's Kubernetes deployment with automated updates.

    Deployment Strategy:
    - Rolling update with maxSurge=1, maxUnavailable=0
    - Gradual rollout to minimize risk
    - Health checks before routing traffic

    Kubernetes Resources to Update:
    - Deployment: New image tag
    - ConfigMap: Model version, configuration
    - Service: Ensure selector matches (usually unchanged)

    Returns:
        dict: Deployment status information
    """
    import logging
    import subprocess

    logger = logging.getLogger(__name__)
    logger.info("Deploying to Kubernetes...")

    # Get image info from previous task
    ti = context['ti']
    image_info = ti.xcom_pull(task_ids='build_deployment.build_docker_image')

    deployment_name = os.getenv('K8S_DEPLOYMENT_NAME', 'mlops-model-deployment')
    namespace = os.getenv('K8S_NAMESPACE', 'default')
    image_tag = image_info['version_tag']

    # TODO: Update Kubernetes deployment
    """
    # Option 1: Using kubectl
    kubectl_cmd = [
        'kubectl', 'set', 'image',
        f'deployment/{deployment_name}',
        f'model-server={image_tag}',
        '-n', namespace,
        '--record'
    ]

    logger.info(f"Updating deployment: {' '.join(kubectl_cmd)}")
    result = subprocess.run(kubectl_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Kubernetes deployment failed: {result.stderr}")
        raise RuntimeError(f"Kubernetes deployment failed: {result.stderr}")

    # Option 2: Using Python Kubernetes client
    from kubernetes import client, config

    config.load_kube_config()
    apps_v1 = client.AppsV1Api()

    # Get current deployment
    deployment = apps_v1.read_namespaced_deployment(deployment_name, namespace)

    # Update image
    deployment.spec.template.spec.containers[0].image = image_tag

    # Add labels for tracking
    deployment.spec.template.metadata.labels['model-version'] = str(image_info['model_version'])
    deployment.spec.template.metadata.labels['deployment-timestamp'] = datetime.now().isoformat()

    # Update deployment
    apps_v1.patch_namespaced_deployment(
        name=deployment_name,
        namespace=namespace,
        body=deployment
    )

    logger.info(f"Deployment updated with image {image_tag}")
    """

    # Placeholder
    logger.info(f"Simulating Kubernetes deployment update to {image_tag}")

    deployment_status = {
        'deployment_name': deployment_name,
        'namespace': namespace,
        'image': image_tag,
        'model_version': image_info['model_version'],
        'deployment_time': datetime.now().isoformat(),
        'status': 'deployed'
    }

    logger.info(f"Deployment status: {deployment_status}")
    return deployment_status


def wait_for_rollout_complete(**context):
    """
    TODO: Wait for Kubernetes rollout to complete

    Steps:
    1. Monitor deployment rollout status
    2. Check that all pods are ready
    3. Verify new ReplicaSet is fully scaled
    4. Timeout after specified duration

    Returns:
        dict: Rollout status
    """
    import logging
    import subprocess
    import time

    logger = logging.getLogger(__name__)
    logger.info("Waiting for rollout to complete...")

    ti = context['ti']
    deployment_status = ti.xcom_pull(task_ids='build_deployment.deploy_to_k8s')

    deployment_name = deployment_status['deployment_name']
    namespace = deployment_status['namespace']
    timeout = int(os.getenv('ROLLOUT_TIMEOUT_SECONDS', 300))

    # TODO: Wait for rollout
    """
    kubectl_cmd = [
        'kubectl', 'rollout', 'status',
        f'deployment/{deployment_name}',
        '-n', namespace,
        f'--timeout={timeout}s'
    ]

    logger.info(f"Waiting for rollout: {' '.join(kubectl_cmd)}")
    result = subprocess.run(kubectl_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Rollout failed or timed out: {result.stderr}")
        raise RuntimeError(f"Rollout failed: {result.stderr}")

    logger.info("Rollout completed successfully")
    """

    # Placeholder - simulate wait
    logger.info(f"Simulating rollout wait for {deployment_name}...")
    time.sleep(2)  # In real implementation, this would be actual wait time

    return {'rollout_status': 'complete', 'deployment_name': deployment_name}


def perform_health_checks(**context):
    """
    TODO: Perform comprehensive health checks on deployed model

    Steps:
    1. Check pod status (all running and ready)
    2. Verify service endpoints are accessible
    3. Test model prediction endpoint with sample data
    4. Check response times meet SLA
    5. Verify metrics are being exported
    6. Test health and readiness endpoints

    Health Check Types:
    - Infrastructure health: Pods running, resources available
    - Application health: Endpoints responding
    - Model health: Predictions working correctly
    - Performance: Response times acceptable

    Returns:
        dict: Health check results
    """
    import logging
    import requests

    logger = logging.getLogger(__name__)
    logger.info("Performing health checks...")

    ti = context['ti']
    deployment_status = ti.xcom_pull(task_ids='build_deployment.deploy_to_k8s')

    service_url = os.getenv('MODEL_SERVICE_URL', 'http://model-service.default.svc.cluster.local')
    max_response_time_ms = int(os.getenv('MAX_RESPONSE_TIME_MS', 1000))

    health_checks = {
        'infrastructure': True,
        'application': True,
        'model_inference': True,
        'performance': True,
        'all_passed': True
    }

    # TODO: Perform actual health checks
    """
    # 1. Check health endpoint
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        health_checks['application'] = response.status_code == 200
        logger.info(f"Health endpoint: {response.status_code}")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_checks['application'] = False

    # 2. Test prediction endpoint with sample data
    try:
        sample_data = {"features": [1, 2, 3, 4, 5]}
        start_time = time.time()
        response = requests.post(f"{service_url}/predict", json=sample_data, timeout=5)
        response_time_ms = (time.time() - start_time) * 1000

        health_checks['model_inference'] = response.status_code == 200
        health_checks['performance'] = response_time_ms < max_response_time_ms

        logger.info(f"Prediction test: {response.status_code}, time: {response_time_ms:.2f}ms")
    except Exception as e:
        logger.error(f"Prediction test failed: {e}")
        health_checks['model_inference'] = False
        health_checks['performance'] = False

    # 3. Check Kubernetes pod status
    from kubernetes import client, config
    config.load_kube_config()
    v1 = client.CoreV1Api()

    pods = v1.list_namespaced_pod(
        namespace=deployment_status['namespace'],
        label_selector=f"app={deployment_status['deployment_name']}"
    )

    all_pods_ready = all(
        pod.status.phase == "Running" and
        all(condition.status == "True" for condition in pod.status.conditions if condition.type == "Ready")
        for pod in pods.items
    )

    health_checks['infrastructure'] = all_pods_ready
    logger.info(f"Pod status: {len(pods.items)} pods, all ready: {all_pods_ready}")

    # Overall status
    health_checks['all_passed'] = all([
        health_checks['infrastructure'],
        health_checks['application'],
        health_checks['model_inference'],
        health_checks['performance']
    ])
    """

    # Placeholder
    logger.info("Simulating health checks - all passed")
    health_checks['all_passed'] = True

    logger.info(f"Health check results: {health_checks}")
    return health_checks


def decide_deployment_action(**context):
    """
    TODO: Decide whether deployment is successful or needs rollback

    Decision Logic:
    1. Check health check results
    2. If all passed → proceed to success notification
    3. If any failed → proceed to rollback

    Returns:
        str: Next task ID ('deployment_success_notification' or 'rollback_deployment')
    """
    import logging

    logger = logging.getLogger(__name__)

    ti = context['ti']
    health_checks = ti.xcom_pull(task_ids='health_checks.perform_checks')

    if health_checks['all_passed']:
        logger.info("All health checks passed - deployment successful")
        return 'deployment_success_notification'
    else:
        logger.error("Health checks failed - initiating rollback")
        return 'rollback_deployment'


def rollback_deployment(**context):
    """
    TODO: Rollback deployment to previous version

    Steps:
    1. Get previous deployment revision
    2. Execute rollback using kubectl or K8s API
    3. Wait for rollback to complete
    4. Verify previous version is working
    5. Send notification about rollback

    Returns:
        dict: Rollback status
    """
    import logging
    import subprocess

    logger = logging.getLogger(__name__)
    logger.error("Initiating deployment rollback...")

    ti = context['ti']
    deployment_status = ti.xcom_pull(task_ids='build_deployment.deploy_to_k8s')

    deployment_name = deployment_status['deployment_name']
    namespace = deployment_status['namespace']

    # TODO: Perform rollback
    """
    kubectl_cmd = [
        'kubectl', 'rollout', 'undo',
        f'deployment/{deployment_name}',
        '-n', namespace
    ]

    logger.info(f"Rolling back: {' '.join(kubectl_cmd)}")
    result = subprocess.run(kubectl_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Rollback failed: {result.stderr}")
        raise RuntimeError(f"Rollback failed: {result.stderr}")

    logger.info("Rollback initiated, waiting for completion...")

    # Wait for rollback to complete
    wait_cmd = [
        'kubectl', 'rollout', 'status',
        f'deployment/{deployment_name}',
        '-n', namespace
    ]
    subprocess.run(wait_cmd, check=True)

    logger.info("Rollback completed successfully")
    """

    # Placeholder
    logger.info(f"Simulating rollback for {deployment_name}")

    rollback_status = {
        'deployment_name': deployment_name,
        'namespace': namespace,
        'rollback_status': 'completed',
        'rollback_time': datetime.now().isoformat()
    }

    logger.error(f"Rollback completed: {rollback_status}")
    return rollback_status


def send_deployment_success_notification(**context):
    """
    TODO: Send success notification for deployment

    Information to include:
    - Model version deployed
    - Deployment timestamp
    - Health check results
    - Performance metrics
    - Link to monitoring dashboard

    Returns:
        bool: True if notification sent
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Sending deployment success notification...")

    ti = context['ti']
    deployment_status = ti.xcom_pull(task_ids='build_deployment.deploy_to_k8s')
    health_checks = ti.xcom_pull(task_ids='health_checks.perform_checks')
    model_info = ti.xcom_pull(task_ids='pull_model')

    # TODO: Send notification via Slack/Email
    """
    import requests

    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    message = {
        'text': f"✅ Model Deployment Successful",
        'blocks': [
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"*Model Deployment Successful*\\n"
                            f"Model: {model_info['model_name']} v{model_info['version']}\\n"
                            f"Accuracy: {model_info['metrics']['accuracy']:.2%}\\n"
                            f"Deployment: {deployment_status['deployment_name']}\\n"
                            f"Time: {deployment_status['deployment_time']}"
                }
            }
        ]
    }
    requests.post(webhook_url, json=message)
    """

    logger.info("Deployment success notification sent")
    return True


def send_rollback_notification(**context):
    """
    TODO: Send notification about deployment rollback

    Returns:
        bool: True if notification sent
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Sending rollback notification...")

    # TODO: Send critical alert about rollback

    logger.info("Rollback notification sent")
    return True


# ==================== DAG Definition ====================

with DAG(
    dag_id='deployment_pipeline',
    default_args=default_args,
    description='Deploy ML models from MLflow Registry to Kubernetes',
    schedule_interval=None,  # Triggered by training_pipeline
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'deployment', 'kubernetes', 'cd'],
    max_active_runs=1,
) as dag:

    # Task 1: Check for new model in registry
    task_check_registry = PythonSensor(
        task_id='check_registry',
        python_callable=check_model_registry_for_promotion,
        poke_interval=60,  # Check every 60 seconds
        timeout=3600,  # Timeout after 1 hour
        mode='reschedule',  # Don't block worker slot while waiting
    )

    # Task 2: Pull model from MLflow Registry
    task_pull_model = PythonOperator(
        task_id='pull_model',
        python_callable=pull_model_from_mlflow_registry,
        provide_context=True,
    )

    # Task Group: Build and Deploy
    with TaskGroup(group_id='build_deployment') as build_deployment:
        # Build Docker image
        task_build_image = PythonOperator(
            task_id='build_docker_image',
            python_callable=build_docker_image_with_model,
            provide_context=True,
        )

        # Deploy to Kubernetes
        task_deploy_k8s = PythonOperator(
            task_id='deploy_to_k8s',
            python_callable=deploy_to_kubernetes_rolling_update,
            provide_context=True,
        )

        # Wait for rollout
        task_wait_rollout = PythonOperator(
            task_id='wait_rollout',
            python_callable=wait_for_rollout_complete,
            provide_context=True,
        )

        task_build_image >> task_deploy_k8s >> task_wait_rollout

    # Task Group: Health Checks
    with TaskGroup(group_id='health_checks') as health_checks:
        task_perform_checks = PythonOperator(
            task_id='perform_checks',
            python_callable=perform_health_checks,
            provide_context=True,
        )

    # Task: Decide on deployment success or rollback
    task_decide = BranchPythonOperator(
        task_id='decide_action',
        python_callable=decide_deployment_action,
        provide_context=True,
    )

    # Task: Rollback deployment
    task_rollback = PythonOperator(
        task_id='rollback_deployment',
        python_callable=rollback_deployment,
        provide_context=True,
    )

    # Task: Success notification
    task_success_notify = PythonOperator(
        task_id='deployment_success_notification',
        python_callable=send_deployment_success_notification,
        provide_context=True,
    )

    # Task: Rollback notification
    task_rollback_notify = PythonOperator(
        task_id='rollback_notification',
        python_callable=send_rollback_notification,
        provide_context=True,
        trigger_rule='all_success',
    )

    # Define task dependencies
    task_check_registry >> task_pull_model >> build_deployment >> health_checks
    health_checks >> task_decide
    task_decide >> [task_success_notify, task_rollback]
    task_rollback >> task_rollback_notify

# TODO: Add canary deployment strategy
# TODO: Implement blue/green deployment option
# TODO: Add A/B testing configuration
# TODO: Implement progressive delivery with traffic splitting
# TODO: Add automated rollback based on metrics degradation
# TODO: Implement deployment approval gates for production
