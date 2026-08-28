"""
FastAPI Application for ML Model Serving

This module implements a REST API for serving an image classification model.
Complete the TODOs to implement the full functionality.

Expected Endpoints:
- GET / - Root endpoint with API information
- POST /predict - Image classification endpoint
- GET /health - Health check
- GET /metrics - Prometheus metrics
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

# TODO: Import your model loading utilities
# from .model import ModelInference, load_model

# TODO: Import Prometheus metrics library
# from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="REST API for serving image classification models",
    version="1.0.0"
)

# ==============================================================================
# TODO: Initialize Prometheus Metrics
# ==============================================================================
"""
Create the following Prometheus metrics:

1. request_count - Counter for total requests
   Labels: endpoint, method, status_code

2. request_duration - Histogram for request duration
   Labels: endpoint, method

3. prediction_count - Counter for successful predictions

4. error_count - Counter for errors
   Labels: error_type

Example:
from prometheus_client import Counter

request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status_code']
)
"""

# TODO: Create Prometheus metrics here
# request_count = Counter(...)
# request_duration = Histogram(...)
# prediction_count = Counter(...)
# error_count = Counter(...)


# ==============================================================================
# Pydantic Models for Request/Response
# ==============================================================================

class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint

    TODO: Define the structure of your prediction request

    Fields to include:
    - image_url: Optional[str] - URL to image (if using URL-based prediction)
    - top_k: int - Number of top predictions to return (default 5)
    - threshold: float - Confidence threshold (default 0.0)

    Example:
    class PredictionRequest(BaseModel):
        image_url: Optional[str] = None
        top_k: int = Field(default=5, ge=1, le=10)
        threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    """
    # TODO: Implement request model fields
    pass


class Prediction(BaseModel):
    """
    Single prediction result

    TODO: Define the structure of a single prediction

    Fields to include:
    - class_id: int - Predicted class ID
    - class_name: str - Human-readable class name
    - confidence: float - Prediction confidence score

    Example:
    class Prediction(BaseModel):
        class_id: int
        class_name: str
        confidence: float = Field(..., ge=0.0, le=1.0)
    """
    # TODO: Implement prediction model fields
    pass


class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint

    TODO: Define the structure of your prediction response

    Fields to include:
    - predictions: List[Prediction] - List of top predictions
    - inference_time_ms: float - Time taken for inference in milliseconds
    - model_version: str - Version of the model used

    Example:
    class PredictionResponse(BaseModel):
        predictions: List[Prediction]
        inference_time_ms: float
        model_version: str
    """
    # TODO: Implement response model fields
    pass


class HealthResponse(BaseModel):
    """
    Health check response

    TODO: Define health check response structure

    Fields to include:
    - status: str - "healthy" or "unhealthy"
    - model_loaded: bool - Whether model is loaded
    - version: str - API version
    - uptime_seconds: float - Time since startup
    """
    # TODO: Implement health response fields
    pass


# ==============================================================================
# Application State and Startup
# ==============================================================================

# TODO: Store application state
"""
Create a global variable or use app.state to store:
- model: The loaded ML model
- start_time: Application startup time
- request_count_local: Local request counter

Example:
app.state.model = None
app.state.start_time = time.time()
"""


@app.on_event("startup")
async def startup_event():
    """
    TODO: Load ML model at application startup

    Steps to implement:
    1. Log startup message
    2. Load model from disk or model registry
    3. Store model in app.state
    4. Validate model loads correctly
    5. Warm up model (run dummy inference)
    6. Log successful loading

    Example:
    logger.info("Loading ML model...")
    model = load_model("path/to/model")
    app.state.model = model
    logger.info("Model loaded successfully")

    Error Handling:
    - If model fails to load, log error and potentially exit
    - Ensure app stays healthy for Kubernetes probes
    """
    logger.info("Starting ML Model Serving API...")

    # TODO: Implement model loading
    # try:
    #     logger.info("Loading model...")
    #     app.state.model = load_model()
    #     app.state.start_time = time.time()
    #     logger.info("Model loaded successfully")
    # except Exception as e:
    #     logger.error(f"Failed to load model: {e}")
    #     raise

    pass


@app.on_event("shutdown")
async def shutdown_event():
    """
    TODO: Cleanup on application shutdown

    Tasks:
    - Log shutdown message
    - Clean up model resources if needed
    - Close any open connections
    """
    logger.info("Shutting down ML Model Serving API...")
    # TODO: Implement cleanup
    pass


# ==============================================================================
# Middleware for Request Tracking
# ==============================================================================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """
    TODO: Implement request tracking middleware

    This middleware should:
    1. Record start time
    2. Process the request
    3. Calculate request duration
    4. Update Prometheus metrics
    5. Log request information

    Example:
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Update metrics
    request_count.labels(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code
    ).inc()

    request_duration.labels(
        endpoint=request.url.path,
        method=request.method
    ).observe(duration)

    return response
    """
    # TODO: Implement request tracking
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # TODO: Update Prometheus metrics
    # TODO: Log request
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")

    return response


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - API information

    TODO: Return API information

    Should include:
    - API name
    - Version
    - Status
    - Documentation link

    Example:
    return {
        "name": "ML Model Serving API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }
    """
    # TODO: Implement root endpoint
    return {
        "message": "ML Model Serving API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    TODO: Implement comprehensive health check

    Health check should verify:
    1. API is running
    2. Model is loaded
    3. Dependencies are available
    4. System resources are sufficient (optional)

    Return:
    - HealthResponse with status and details
    - HTTP 200 if healthy
    - HTTP 503 if unhealthy

    Example:
    model_loaded = app.state.model is not None

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0",
        uptime_seconds=time.time() - app.state.start_time
    )

    Kubernetes Usage:
    - Used by liveness probe (is app running?)
    - Used by readiness probe (can app serve traffic?)
    """
    # TODO: Implement health check logic
    # TODO: Check if model is loaded
    # TODO: Return appropriate status

    # Placeholder implementation
    return HealthResponse(
        status="healthy",
        model_loaded=False,  # TODO: Check actual model status
        version="1.0.0",
        uptime_seconds=0.0  # TODO: Calculate actual uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = 5):
    """
    TODO: Implement prediction endpoint

    Main prediction endpoint that:
    1. Accepts image file upload
    2. Validates input
    3. Preprocesses image
    4. Runs model inference
    5. Post-processes predictions
    6. Returns formatted response

    Steps to implement:

    1. Input Validation:
       - Check file type (accept image/* MIME types)
       - Check file size (limit to reasonable size, e.g., 10MB)
       - Validate top_k parameter (1-10)

    2. Image Preprocessing:
       - Read image bytes
       - Convert to PIL Image
       - Apply transforms (resize, normalize, etc.)
       - Convert to tensor

    3. Model Inference:
       - Check if model is loaded
       - Run inference (with torch.no_grad())
       - Measure inference time

    4. Post-processing:
       - Get top-k predictions
       - Map class IDs to class names
       - Format confidence scores

    5. Response:
       - Return predictions with metadata
       - Update Prometheus metrics

    Error Handling:
    - Invalid file format → HTTP 400
    - File too large → HTTP 413
    - Model not loaded → HTTP 503
    - Inference error → HTTP 500

    Example Implementation Structure:

    try:
        # Validate input
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(400, "Invalid image format")

        # Read and preprocess
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        # Run inference
        start_time = time.time()
        predictions = app.state.model.predict(image, top_k=top_k)
        inference_time = (time.time() - start_time) * 1000  # ms

        # Update metrics
        prediction_count.inc()

        # Format response
        return PredictionResponse(
            predictions=predictions,
            inference_time_ms=inference_time,
            model_version="1.0.0"
        )

    except ValueError as e:
        error_count.labels(error_type="validation").inc()
        raise HTTPException(400, str(e))
    except Exception as e:
        error_count.labels(error_type="inference").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, "Inference failed")
    """
    # TODO: Implement prediction logic

    # Placeholder - remove and implement
    raise HTTPException(501, "Prediction endpoint not implemented yet")

    # Your implementation here:
    # 1. Validate input
    # 2. Preprocess image
    # 3. Run inference
    # 4. Format response
    # 5. Update metrics


@app.get("/metrics")
async def metrics():
    """
    TODO: Implement Prometheus metrics endpoint

    This endpoint returns metrics in Prometheus format for scraping.

    Steps:
    1. Import prometheus_client.generate_latest
    2. Generate metrics in Prometheus text format
    3. Return with correct content type

    Example:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    metrics_data = generate_latest()
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

    Metrics to track:
    - api_requests_total - Total API requests
    - api_request_duration_seconds - Request duration histogram
    - api_predictions_total - Total successful predictions
    - api_errors_total - Total errors by type

    Prometheus Configuration:
    In prometheus.yml, add scrape config:
    scrape_configs:
      - job_name: 'ml-api'
        static_configs:
          - targets: ['ml-api:8000']
        metrics_path: '/metrics'
    """
    # TODO: Implement metrics endpoint
    # TODO: Return Prometheus-formatted metrics

    # Placeholder
    return {"message": "Metrics endpoint not implemented"}


# ==============================================================================
# Helper Functions
# ==============================================================================

def preprocess_image(image_bytes: bytes):
    """
    TODO: Implement image preprocessing

    Steps:
    1. Convert bytes to PIL Image
    2. Resize to model input size (e.g., 224x224)
    3. Normalize pixel values
    4. Convert to tensor
    5. Add batch dimension

    Example:
    from PIL import Image
    import io
    from torchvision import transforms

    image = Image.open(io.BytesIO(image_bytes))

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch
    """
    # TODO: Implement preprocessing
    pass


def format_predictions(outputs, top_k: int):
    """
    TODO: Implement prediction formatting

    Takes model outputs and formats them into Prediction objects

    Steps:
    1. Apply softmax to get probabilities
    2. Get top-k predictions
    3. Map class indices to class names
    4. Create Prediction objects

    Example:
    import torch.nn.functional as F

    probabilities = F.softmax(outputs, dim=1)
    top_probs, top_indices = torch.topk(probabilities, top_k)

    predictions = []
    for i in range(top_k):
        predictions.append(Prediction(
            class_id=int(top_indices[0][i]),
            class_name=get_class_name(int(top_indices[0][i])),
            confidence=float(top_probs[0][i])
        ))

    return predictions
    """
    # TODO: Implement prediction formatting
    pass


def get_class_name(class_id: int) -> str:
    """
    TODO: Map class ID to class name

    Load ImageNet class names or your model's class mapping

    Example:
    # Load class names from file
    with open('imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    return classes[class_id]
    """
    # TODO: Implement class name mapping
    return f"class_{class_id}"  # Placeholder


# ==============================================================================
# Development Server
# ==============================================================================

if __name__ == "__main__":
    """
    Run the application for local development

    TODO: Configure uvicorn server settings
    - host: 0.0.0.0 (accessible from outside container)
    - port: 8000
    - reload: True (for development)
    - workers: 1 (for development, increase for production)
    - log_level: "info"

    For production, use:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
    """
    import uvicorn

    # TODO: Configure uvicorn settings
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
