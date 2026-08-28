# Lesson 06: Model Deployment Strategies

## Overview
Deploying machine learning models to production requires careful consideration of performance, scalability, reliability, and cost. This lesson explores various deployment patterns and strategies for serving ML models at scale.

**Duration:** 3-4 hours
**Prerequisites:** Understanding of ML serving, Docker, Kubernetes basics
**Learning Objectives:**
- Understand different model serving architectures
- Implement batch and real-time inference
- Design multi-model serving systems
- Optimize model performance and cost
- Handle model updates and versioning

---

## 1. Model Serving Architectures

### 1.1 Deployment Patterns Overview

```
┌────────────────────────────────────────────────────────────────┐
│            Model Deployment Patterns                            │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │   Batch      │  │  Real-time   │  │  Stream Processing │   │
│  │  Inference   │  │  Inference   │  │                    │   │
│  │              │  │              │  │                    │   │
│  │  - Scheduled │  │  - REST API  │  │  - Kafka/Kinesis   │   │
│  │  - Large     │  │  - gRPC      │  │  - Continuous      │   │
│  │    volumes   │  │  - Low       │  │  - Windowed        │   │
│  │  - Minutes/  │  │    latency   │  │  - Event-driven    │   │
│  │    hours     │  │  - <100ms    │  │                    │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │   Embedded   │  │  Serverless  │  │  Edge Deployment   │   │
│  │              │  │              │  │                    │   │
│  │  - Mobile    │  │  - Lambda/   │  │  - IoT devices     │   │
│  │    apps      │  │    Cloud     │  │  - Low latency     │   │
│  │  - On-device │  │    Functions │  │  - Offline-capable │   │
│  │  - TFLite/   │  │  - Auto-     │  │  - Resource-       │   │
│  │    ONNX      │  │    scaling   │  │    constrained     │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Pattern Selection Criteria

| Pattern | Latency | Throughput | Cost | Use Cases |
|---------|---------|------------|------|-----------|
| **Batch** | Minutes-Hours | Very High | Low | Recommendation engines, ETL, Analytics |
| **Real-time** | <100ms | Medium-High | Medium | Web apps, APIs, Interactive systems |
| **Streaming** | Seconds | High | Medium | Real-time analytics, Fraud detection |
| **Serverless** | 100-500ms | Variable | Low-Medium | Sporadic workloads, Microservices |
| **Edge** | <10ms | Low | High | Mobile apps, IoT, Autonomous systems |

---

## 2. Real-Time Model Serving

### 2.1 REST API with FastAPI

```python
# src/serve_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
from typing import Dict, List
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ML Model API", version="1.0.0")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    """Load model when server starts"""
    global model
    try:
        model_uri = "models:/recommendation_model/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✅ Model loaded: {model_uri}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

# Request/Response models
class PredictionRequest(BaseModel):
    user_id: str
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    user_id: str
    prediction: float
    probability: float
    model_version: str

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction for a single request
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        features = np.array([[
            request.features['age'],
            request.features['purchase_count'],
            request.features['avg_amount']
        ]])

        # Make prediction
        prediction = model.predict(features)[0]
        probability = prediction if 0 <= prediction <= 1 else sigmoid(prediction)

        return PredictionResponse(
            user_id=request.user_id,
            prediction=float(prediction),
            probability=float(probability),
            model_version="1.0.0"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """
    Make predictions for multiple requests
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features for batch
        features = np.array([
            [r.features['age'], r.features['purchase_count'], r.features['avg_amount']]
            for r in requests
        ])

        # Batch prediction
        predictions = model.predict(features)

        # Format responses
        return [
            PredictionResponse(
                user_id=req.user_id,
                prediction=float(pred),
                probability=float(sigmoid(pred)),
                model_version="1.0.0"
            )
            for req, pred in zip(requests, predictions)
        ]

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model metadata endpoint
@app.get("/model/info")
async def model_info():
    """Get model metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": "recommendation_model",
        "version": "1.0.0",
        "framework": "scikit-learn",
        "input_schema": {
            "age": "float",
            "purchase_count": "int",
            "avg_amount": "float"
        }
    }

def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-x))

# Run with: uvicorn src.serve_fastapi:app --host 0.0.0.0 --port 8000
```

### 2.2 gRPC Model Serving

```python
# protos/prediction.proto
syntax = "proto3";

service PredictionService {
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc PredictBatch(PredictBatchRequest) returns (PredictBatchResponse);
}

message PredictRequest {
  string user_id = 1;
  map<string, float> features = 2;
}

message PredictResponse {
  string user_id = 1;
  float prediction = 2;
  float probability = 3;
  string model_version = 4;
}

message PredictBatchRequest {
  repeated PredictRequest requests = 1;
}

message PredictBatchResponse {
  repeated PredictResponse responses = 1;
}

# src/serve_grpc.py
import grpc
from concurrent import futures
import mlflow.pyfunc
import numpy as np
import prediction_pb2
import prediction_pb2_grpc

class PredictionServicer(prediction_pb2_grpc.PredictionServiceServicer):
    """gRPC service for model predictions"""

    def __init__(self, model_uri: str):
        self.model = mlflow.pyfunc.load_model(model_uri)

    def Predict(self, request, context):
        """Single prediction"""
        try:
            # Extract features
            features = np.array([[
                request.features['age'],
                request.features['purchase_count'],
                request.features['avg_amount']
            ]])

            # Predict
            prediction = self.model.predict(features)[0]

            return prediction_pb2.PredictResponse(
                user_id=request.user_id,
                prediction=float(prediction),
                probability=self._sigmoid(prediction),
                model_version="1.0.0"
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return prediction_pb2.PredictResponse()

    def PredictBatch(self, request, context):
        """Batch prediction"""
        try:
            # Extract features for all requests
            features = np.array([
                [
                    req.features['age'],
                    req.features['purchase_count'],
                    req.features['avg_amount']
                ]
                for req in request.requests
            ])

            # Batch predict
            predictions = self.model.predict(features)

            # Create responses
            responses = [
                prediction_pb2.PredictResponse(
                    user_id=req.user_id,
                    prediction=float(pred),
                    probability=self._sigmoid(pred),
                    model_version="1.0.0"
                )
                for req, pred in zip(request.requests, predictions)
            ]

            return prediction_pb2.PredictBatchResponse(responses=responses)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return prediction_pb2.PredictBatchResponse()

    def _sigmoid(self, x):
        return float(1 / (1 + np.exp(-x)))

def serve():
    """Start gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    prediction_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionServicer(model_uri="models:/recommendation_model/Production"),
        server
    )

    server.add_insecure_port('[::]:50051')
    server.start()
    print("✅ gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

### 2.3 TorchServe for PyTorch Models

```python
# handler.py - Custom TorchServe handler
import torch
import json
from ts.torch_handler.base_handler import BaseHandler

class RecommendationHandler(BaseHandler):
    """
    Custom handler for recommendation model
    """

    def initialize(self, context):
        """Load model and initialize"""
        super().initialize(context)
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        self.model = torch.jit.load(f"{model_dir}/model.pt")
        self.model.eval()

        # Load feature names
        with open(f"{model_dir}/feature_names.json") as f:
            self.feature_names = json.load(f)

        self.initialized = True

    def preprocess(self, requests):
        """
        Transform raw input into model input data
        """
        inputs = []
        for request in requests:
            data = request.get("body")
            if isinstance(data, dict):
                # Extract features in correct order
                features = [data['features'][name] for name in self.feature_names]
                inputs.append(features)

        return torch.tensor(inputs, dtype=torch.float32)

    def inference(self, model_input):
        """
        Run inference
        """
        with torch.no_grad():
            predictions = self.model(model_input)
        return predictions

    def postprocess(self, inference_output):
        """
        Format output
        """
        predictions = inference_output.cpu().numpy()
        return [
            {
                "prediction": float(pred),
                "probability": float(torch.sigmoid(torch.tensor(pred)))
            }
            for pred in predictions
        ]

# Package model for TorchServe
# torch-model-archiver --model-name recommendation \
#   --version 1.0 \
#   --model-file model.py \
#   --serialized-file model.pt \
#   --handler handler.py \
#   --export-path model-store

# Start TorchServe
# torchserve --start --model-store model-store --models recommendation=recommendation.mar
```

---

## 3. Batch Inference

### 3.1 Spark-based Batch Scoring

```python
# src/batch_scoring.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, udf
from pyspark.sql.types import FloatType
import mlflow.pyfunc
import pandas as pd

class BatchScorer:
    """Score large datasets using Spark"""

    def __init__(self, model_uri: str):
        self.spark = SparkSession.builder \
            .appName("BatchModelScoring") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()

        # Load model
        self.model = mlflow.pyfunc.load_model(model_uri)

    def score_batch(
        self,
        input_path: str,
        output_path: str,
        partition_size: int = 10000
    ):
        """
        Score large dataset in batches
        """
        # Read input data
        df = self.spark.read.parquet(input_path)

        print(f"📊 Scoring {df.count()} records")

        # Define pandas UDF for vectorized scoring
        @udf(FloatType())
        def predict_udf(*features):
            """UDF for model prediction"""
            # Create feature array
            feature_array = pd.DataFrame([features]).values
            prediction = self.model.predict(feature_array)[0]
            return float(prediction)

        # Apply predictions
        scored_df = df.withColumn(
            "prediction",
            predict_udf(
                col("age"),
                col("purchase_count"),
                col("avg_amount")
            )
        )

        # Write results
        scored_df.write \
            .mode("overwrite") \
            .partitionBy("date") \
            .parquet(output_path)

        print(f"✅ Results written to {output_path}")

        return scored_df

    def score_with_pandas_udf(self, input_path: str, output_path: str):
        """
        Score using Pandas UDF (more efficient)
        """
        from pyspark.sql.functions import pandas_udf, PandasUDFType

        df = self.spark.read.parquet(input_path)

        @pandas_udf(FloatType(), PandasUDFType.SCALAR)
        def predict_batch_udf(age, purchase_count, avg_amount):
            """Vectorized prediction using Pandas UDF"""
            # Create DataFrame
            features_df = pd.DataFrame({
                'age': age,
                'purchase_count': purchase_count,
                'avg_amount': avg_amount
            })

            # Batch predict
            predictions = self.model.predict(features_df.values)
            return pd.Series(predictions)

        # Apply predictions
        scored_df = df.withColumn(
            "prediction",
            predict_batch_udf(
                col("age"),
                col("purchase_count"),
                col("avg_amount")
            )
        )

        scored_df.write.parquet(output_path)

# Usage
scorer = BatchScorer(model_uri="models:/recommendation_model/Production")
scorer.score_with_pandas_udf(
    input_path="s3://data/users/",
    output_path="s3://predictions/users/"
)
```

### 3.2 Scheduled Batch Jobs with Airflow

```python
# dags/batch_scoring_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.emr import EmrCreateJobFlowOperator
from airflow.providers.amazon.aws.sensors.emr import EmrJobFlowSensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['ml-team@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_batch_scoring',
    default_args=default_args,
    description='Daily batch model scoring',
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False
)

# Task 1: Validate model
def validate_model(**context):
    """Validate model before scoring"""
    import mlflow
    client = mlflow.tracking.MlflowClient()

    # Get production model
    model = client.get_latest_versions("recommendation_model", stages=["Production"])[0]

    # Check model age
    model_age_days = (datetime.now() - datetime.fromtimestamp(model.creation_timestamp / 1000)).days

    if model_age_days > 30:
        raise ValueError(f"Model is {model_age_days} days old - requires retraining")

    return model.version

validate_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

# Task 2: Create EMR cluster
emr_cluster = EmrCreateJobFlowOperator(
    task_id='create_emr_cluster',
    job_flow_overrides={
        'Name': 'batch-scoring-cluster',
        'ReleaseLabel': 'emr-6.10.0',
        'Applications': [{'Name': 'Spark'}],
        'Instances': {
            'InstanceGroups': [
                {
                    'Name': 'Master nodes',
                    'Market': 'ON_DEMAND',
                    'InstanceRole': 'MASTER',
                    'InstanceType': 'r5.xlarge',
                    'InstanceCount': 1,
                },
                {
                    'Name': 'Worker nodes',
                    'Market': 'SPOT',
                    'InstanceRole': 'CORE',
                    'InstanceType': 'r5.2xlarge',
                    'InstanceCount': 10,
                }
            ],
            'KeepJobFlowAliveWhenNoSteps': False,
        },
        'Steps': [
            {
                'Name': 'Run Batch Scoring',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': [
                        'spark-submit',
                        '--deploy-mode', 'cluster',
                        '--py-files', 's3://ml-code/dependencies.zip',
                        's3://ml-code/batch_scoring.py',
                        '--input-path', f's3://data/users/date={{{{ ds }}}}/',
                        '--output-path', f's3://predictions/users/date={{{{ ds }}}}/',
                    ]
                }
            }
        ],
    },
    dag=dag
)

# Task 3: Wait for cluster completion
wait_for_cluster = EmrJobFlowSensor(
    task_id='wait_for_cluster',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    dag=dag
)

# Task 4: Validate output
def validate_output(**context):
    """Validate scoring output"""
    import boto3
    import pandas as pd

    s3 = boto3.client('s3')
    date = context['ds']

    # Check output exists
    output_path = f"predictions/users/date={date}/"
    objects = s3.list_objects_v2(Bucket='ml-bucket', Prefix=output_path)

    if objects['KeyCount'] == 0:
        raise ValueError(f"No output files found at {output_path}")

    # Sample and validate predictions
    # ... validation logic ...

    print(f"✅ Output validated for {date}")

validate_output_task = PythonOperator(
    task_id='validate_output',
    python_callable=validate_output,
    provide_context=True,
    dag=dag
)

# Define task dependencies
validate_task >> emr_cluster >> wait_for_cluster >> validate_output_task
```

---

## 4. Multi-Model Serving

### 4.1 Model Router

```python
# src/model_router.py
from fastapi import FastAPI, HTTPException
from typing import Dict, Optional
import mlflow.pyfunc
from enum import Enum

class ModelType(str, Enum):
    RECOMMENDATION = "recommendation"
    RANKING = "ranking"
    PERSONALIZATION = "personalization"

class ModelRouter:
    """Route requests to appropriate models"""

    def __init__(self):
        self.models: Dict[str, any] = {}
        self._load_models()

    def _load_models(self):
        """Load all models at startup"""
        model_configs = {
            ModelType.RECOMMENDATION: "models:/recommendation_model/Production",
            ModelType.RANKING: "models:/ranking_model/Production",
            ModelType.PERSONALIZATION: "models:/personalization_model/Production",
        }

        for model_type, model_uri in model_configs.items():
            try:
                self.models[model_type] = mlflow.pyfunc.load_model(model_uri)
                print(f"✅ Loaded {model_type} model")
            except Exception as e:
                print(f"❌ Failed to load {model_type}: {e}")

    def predict(
        self,
        model_type: ModelType,
        features: np.ndarray,
        model_version: Optional[str] = None
    ):
        """
        Route prediction to appropriate model
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not available")

        model = self.models[model_type]
        return model.predict(features)

    def predict_ensemble(self, features: np.ndarray, weights: Dict[str, float]):
        """
        Ensemble prediction from multiple models
        """
        predictions = {}
        for model_type, weight in weights.items():
            if model_type in self.models:
                pred = self.models[model_type].predict(features)
                predictions[model_type] = pred * weight

        # Weighted average
        ensemble_pred = sum(predictions.values()) / sum(weights.values())
        return ensemble_pred

# FastAPI integration
app = FastAPI()
router = ModelRouter()

@app.post("/predict/{model_type}")
async def predict(model_type: ModelType, request: PredictionRequest):
    """Route to specific model"""
    features = prepare_features(request)
    prediction = router.predict(model_type, features)

    return {"prediction": float(prediction[0])}

@app.post("/predict/ensemble")
async def predict_ensemble(request: EnsemblePredictionRequest):
    """Ensemble prediction"""
    features = prepare_features(request)
    prediction = router.predict_ensemble(
        features,
        weights=request.model_weights
    )

    return {"prediction": float(prediction[0])}
```

### 4.2 A/B Testing Multiple Models

```python
# src/ab_testing.py
from typing import Dict, Optional
import random
import hashlib

class ABTestingRouter:
    """Route requests to models based on A/B test configuration"""

    def __init__(self, experiment_config: Dict):
        """
        experiment_config = {
            'experiment_id': 'model_v2_test',
            'variants': {
                'control': {'model_version': 'v1', 'traffic': 0.9},
                'treatment': {'model_version': 'v2', 'traffic': 0.1}
            }
        }
        """
        self.config = experiment_config
        self.models = self._load_variants()

    def _load_variants(self) -> Dict:
        """Load all variant models"""
        models = {}
        for variant_name, variant_config in self.config['variants'].items():
            model_uri = f"models:/recommendation_model/{variant_config['model_version']}"
            models[variant_name] = mlflow.pyfunc.load_model(model_uri)
        return models

    def route_request(self, user_id: str) -> str:
        """
        Determine which variant to use for this user
        Uses consistent hashing for stable assignments
        """
        # Consistent hashing based on user_id
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        assignment_value = (hash_value % 100) / 100.0

        cumulative_traffic = 0
        for variant_name, variant_config in self.config['variants'].items():
            cumulative_traffic += variant_config['traffic']
            if assignment_value < cumulative_traffic:
                return variant_name

        # Default to control
        return 'control'

    def predict(self, user_id: str, features):
        """Make prediction with assigned variant"""
        variant = self.route_request(user_id)
        model = self.models[variant]

        prediction = model.predict(features)

        return {
            'prediction': prediction,
            'variant': variant,
            'experiment_id': self.config['experiment_id']
        }

# Usage
ab_router = ABTestingRouter({
    'experiment_id': 'model_v2_test',
    'variants': {
        'control': {'model_version': 'v1', 'traffic': 0.9},
        'treatment': {'model_version': 'v2', 'traffic': 0.1}
    }
})

# Track results for analysis
@app.post("/predict/ab")
async def predict_ab(request: PredictionRequest):
    result = ab_router.predict(request.user_id, request.features)

    # Log to analytics
    log_ab_test_event(
        user_id=request.user_id,
        experiment_id=result['experiment_id'],
        variant=result['variant'],
        prediction=result['prediction']
    )

    return result
```

---

## 5. Model Optimization

### 5.1 Model Compression

```python
# src/model_optimization.py
import torch
import torch.quantization

def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply dynamic quantization to reduce model size
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Layers to quantize
        dtype=torch.qint8
    )

    print(f"Original model size: {get_model_size(model):.2f} MB")
    print(f"Quantized model size: {get_model_size(quantized_model):.2f} MB")

    return quantized_model

def prune_model(model: torch.nn.Module, amount: float = 0.3):
    """
    Prune model weights
    """
    import torch.nn.utils.prune as prune

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

    print(f"Pruned {amount*100}% of weights")
    return model

def convert_to_onnx(model: torch.nn.Module, input_shape: tuple, output_path: str):
    """
    Convert PyTorch model to ONNX for optimized inference
    """
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ Model exported to ONNX: {output_path}")

def optimize_onnx(onnx_path: str, optimized_path: str):
    """
    Optimize ONNX model
    """
    from onnxruntime.transformers import optimizer

    optimized_model = optimizer.optimize_model(
        onnx_path,
        model_type='bert',  # or 'gpt2', etc.
        num_heads=12,
        hidden_size=768
    )

    optimized_model.save_model_to_file(optimized_path)
    print(f"✅ Optimized ONNX model saved: {optimized_path}")

def get_model_size(model) -> float:
    """Get model size in MB"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2
```

### 5.2 Inference Optimization with TensorRT

```python
# src/tensorrt_optimization.py
import torch
import torch_tensorrt

def optimize_with_tensorrt(model: torch.nn.Module, input_shape: tuple):
    """
    Optimize PyTorch model with TensorRT
    """
    # Trace the model
    example_input = torch.randn(input_shape).cuda()
    traced_model = torch.jit.trace(model, example_input)

    # Compile with TensorRT
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input(shape=input_shape)],
        enabled_precisions={torch.float16},  # Use FP16
        workspace_size=1 << 30  # 1GB
    )

    print("✅ Model optimized with TensorRT")

    # Benchmark
    import time

    # Original model
    start = time.time()
    for _ in range(1000):
        _ = model(example_input)
    torch.cuda.synchronize()
    original_time = (time.time() - start) / 1000

    # TensorRT model
    start = time.time()
    for _ in range(1000):
        _ = trt_model(example_input)
    torch.cuda.synchronize()
    trt_time = (time.time() - start) / 1000

    print(f"Original inference: {original_time*1000:.2f}ms")
    print(f"TensorRT inference: {trt_time*1000:.2f}ms")
    print(f"Speedup: {original_time/trt_time:.2f}x")

    return trt_model
```

---

## 6. Scaling Strategies

### 6.1 Horizontal Pod Autoscaling (Kubernetes)

```yaml
# k8s/model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: model-server
        image: ml-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Percent
        value: 50  # Scale down max 50% at a time
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Percent
        value: 100  # Double pods if needed
        periodSeconds: 15
```

---

## Summary

In this lesson, you learned:

✅ **Model Serving Architectures:**
- Batch, real-time, streaming patterns
- REST API and gRPC implementations
- TorchServe and model-specific servers

✅ **Production Deployment:**
- Multi-model serving
- A/B testing infrastructure
- Model routing strategies

✅ **Optimization:**
- Model compression techniques
- Quantization and pruning
- TensorRT and ONNX optimization

✅ **Scaling:**
- Horizontal pod autoscaling
- Resource management
- Performance monitoring

---

## Additional Resources

- [TorchServe Documentation](https://pytorch.org/serve/)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
- [KServe (KubeFlow Serving)](https://kserve.github.io/website/)

---

## Next Lesson

**Lesson 07: A/B Testing & Experimentation** - Learn how to run experiments and measure model performance in production.
