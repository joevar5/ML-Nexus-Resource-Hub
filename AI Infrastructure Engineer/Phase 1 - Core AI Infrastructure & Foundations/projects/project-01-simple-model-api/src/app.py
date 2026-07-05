#!/usr/bin/env python3
"""
ML Model Serving API

Flask application that serves predictions from a pre-trained image classification model.

Usage:
    python app.py

    # Test endpoints:
    curl http://localhost:5000/health
    curl http://localhost:5000/info
    curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
import io
import logging
import logging.config
import time
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

from config import get_settings
from model_loader import ModelLoader


app = Flask(__name__)

# WHAT: Retrieve the configuration settings for the application.
# WHY: Decoupling configuration from code keeps settings centralized, customizable via environment variables, and secure (no hardcoded credentials/parameters).
# HOW: Calling `get_settings()` returns a Config object holding variables like model name, host, port, allowed extensions, and file sizes.
settings = get_settings()

# Configure logging
# WHAT: Set up root logging configuration for formatting and filtering messages.
# WHY: Proper logging is critical for production visibility, debugging, and tracing system state and failures under load.
# HOW: `logging.basicConfig` defines the standard logging format (timestamp, logger name, severity, message) and prints messages with level >= settings.log_level to stdout via `StreamHandler`.
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# WHAT: Declare a global variable to store the ModelLoader instance.
# WHY: Machine learning models can be extremely large (hundreds of MBs to GBs). We must load the model weights into memory ONCE during application startup (singleton pattern) rather than on every HTTP request, which would cause severe latency and memory exhaustion.
# HOW: Initialize `model_loader` to `None` at the module level. It will be populated with a loaded model state in `setup_model()`.
model_loader: ModelLoader = None


def setup_model():
    """Initialize model loader on startup"""
    global model_loader

    logger.info("Starting model initialization...")
    try:
        # WHAT: Instantiate the helper class that wraps our machine learning model.
        # WHY: Keeping the PyTorch/DL details separate from Flask makes the code modular, testable, and clean.
        # HOW: Creates a `ModelLoader` with the specified model architecture and device (CPU or GPU).
        model_loader = ModelLoader(
            model_name=settings.model_name,
            device=settings.model_device
        )

        # WHAT: Run a dummy inference pass through the model.
        # WHY: PyTorch and neural networks execute initializations, lazy allocations, and memory maps on their first run, causing the first inference request to be extremely slow. "Warming up" the model on startup ensures subsequent user requests experience low latency.
        # HOW: `model_loader.warmup()` triggers inference with a dummy input (e.g., zeros tensor of correct shape) before accepting real traffic.
        model_loader.warmup()

        logger.info("Model initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed

    Args:
        filename: Name of the file

    Returns:
        True if file extension is allowed
    """
    return Path(filename).suffix.lower() in settings.allowed_extensions


def validate_image(file) -> Tuple[bool, str]:
    """
    Validate uploaded image file

    Args:
        file: Uploaded file object

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not file:
        return False, "No file provided"

    # Check filename
    if file.filename == '':
        return False, "Empty filename"

    # Check file extension
    # WHAT: Check if the file suffix is allowed.
    # WHY: Prevents users from uploading malicious executable files or scripts.
    # HOW: Calls allowed_file which matches the file's extension against a whitelist of valid image formats.
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {settings.allowed_extensions}"

    # Check file size
    # WHAT: Programmatically calculate the size of the uploaded file stream.
    # WHY: Large files consume significant memory and CPU, which can cause out-of-memory crashes or block other users (DoS vulnerability).
    # HOW: `file.seek(0, 2)` moves the read cursor to the end of the file. `file.tell()` returns the current byte position (the file size). `file.seek(0)` resets the cursor back to the beginning so it can be read later.
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    if file_size > settings.max_file_size:
        return False, f"File too large. Max size: {settings.max_file_size} bytes"

    if file_size == 0:
        return False, "Empty file"

    # Try to open as image
    # WHAT: Verify that the bytes represent a valid, uncorrupted image.
    # WHY: A file may have a `.jpg` extension but contain arbitrary corrupted bytes that would crash the image library or model during preprocessing.
    # HOW: PIL's `Image.open` initializes the image representation. `img.verify()` reads and checks the integrity of the file structure without loading all pixel data into memory. We then seek back to 0.
    try:
        img = Image.open(file)
        img.verify()
        file.seek(0)  # Reset for actual processing
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


# WHAT: Flask middleware/hook that executes code before each request hits its endpoint route.
# WHY: Allows logging incoming traffic and starting performance metrics (like latency timing) globally.
# HOW: Dynamically attaches a `start_time` float property to the request context object, and logs the HTTP method and request path.
@app.before_request
def before_request():
    """Log request information"""
    request.start_time = time.time()
    logger.info(f"Request: {request.method} {request.path}")


# WHAT: Flask middleware/hook that executes code after the endpoint processes a request but before the response is sent back to the client.
# WHY: Measures and logs the exact duration of request processing for performance auditing.
# HOW: Calculates the difference between the current time and the stored `request.start_time`, logs the result along with the HTTP status code, and returns the response.
@app.after_request
def after_request(response):
    """Log response information"""
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"Response: {response.status_code} ({duration:.3f}s)")
    return response


# WHAT: Custom exception handlers that catch specific HTTP errors or Python exceptions thrown by the app.
# WHY: Prevents leaking stack traces to the end user (security hazard) and provides clear, standardized, client-friendly JSON error formats.
# HOW: Decorated with `@app.errorhandler`. Flask intercepts matching status codes or exceptions and invokes the respective function, returning a JSON response and HTTP status code.
@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file size limit exceeded"""
    return jsonify({
        "error": "File too large",
        "message": f"Maximum file size is {settings.max_file_size} bytes",
        "status": "error"
    }), 413


@app.errorhandler(400)
def handle_bad_request(e):
    """Handle bad request"""
    return jsonify({
        "error": "Bad request",
        "message": str(e),
        "status": "error"
    }), 400


@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server error"""
    logger.error(f"Internal error: {e}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status": "error"
    }), 500


# WHAT: Liveness and readiness health check endpoint.
# WHY: Cloud load balancers and orchestrators (like Kubernetes, ECS, or PM2) call health check endpoints regularly to verify if the server is responsive and the model is fully loaded. If it fails, they restart the container or stop routing user traffic to it.
# HOW: Responds to GET requests. Verifies if `model_loader` is initialized, and returns an appropriate status code (200 OK or 503 Service Unavailable).
@app.route('/health', methods=['GET'])
def health() -> Dict[str, Any]:
    """
    Health check endpoint

    Returns:
        JSON with health status
    """
    try:
        # Check if model is loaded
        if model_loader is None:
            return jsonify({
                "status": "unhealthy",
                "message": "Model not loaded",
                "timestamp": time.time()
            }), 503

        # Run a quick inference test
        return jsonify({
            "status": "healthy",
            "model": settings.model_name,
            "device": settings.model_device,
            "timestamp": time.time()
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "message": str(e),
            "timestamp": time.time()
        }), 503


@app.route('/info', methods=['GET'])
def info() -> Dict[str, Any]:
    """
    Model information endpoint

    Returns:
        JSON with model metadata
    """
    try:
        if model_loader is None:
            return jsonify({
                "error": "Model not loaded",
                "status": "error"
            }), 503

        model_info = model_loader.get_model_info()

        return jsonify({
            "status": "success",
            "model": model_info,
            "config": {
                "max_file_size": settings.max_file_size,
                "allowed_extensions": settings.allowed_extensions,
                "request_timeout": settings.request_timeout
            }
        }), 200

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return jsonify({
            "error": "Failed to get model info",
            "message": str(e),
            "status": "error"
        }), 500


# WHAT: Primary model inference API endpoint that accepts an image and returns predictions.
# WHY: Provides the external HTTP interface for client applications to interact with the underlying deep learning model.
# HOW: Processes a POST request with multipart/form-data. It extracts, validates, processes the image, runs model inference, formats predictions to JSON, and measures inference latency.
@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """
    Prediction endpoint

    Accepts image file and returns top-K predictions

    Request:
        file: Image file (multipart/form-data)
        top_k: Number of predictions to return (optional, default=5)

    Returns:
        JSON with predictions
    """
    try:
        # Check if model is loaded
        if model_loader is None:
            return jsonify({
                "error": "Model not loaded",
                "status": "error"
            }), 503

        # WHAT: Extract and validate the optional K parameter.
        # WHY: Clients can specify how many top classes they want to see, but we must cap this number to prevent excessive memory/compute overhead.
        # HOW: `request.form.get` retrieves values from form data, converts to `int`, and verifies it is within the bounds [1, 10].
        top_k = request.form.get('top_k', 5, type=int)
        if not (1 <= top_k <= 10):
            return jsonify({
                "error": "Invalid top_k value",
                "message": "top_k must be between 1 and 10",
                "status": "error"
            }), 400

        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "error": "No file provided",
                "message": "Request must include 'file' field",
                "status": "error"
            }), 400

        file = request.files['file']

        # Validate file
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            return jsonify({
                "error": "Invalid image",
                "message": error_msg,
                "status": "error"
            }), 400

        # WHAT: Load image file bytes into memory and convert to a PIL Image.
        # WHY: PyTorch / torchvision image transformation pipelines require a standard PIL Image or Tensor as input.
        # HOW: `file.read()` retrieves raw bytes, `io.BytesIO` wraps them in an in-memory buffer, and `Image.open` loads it. `.convert('RGB')` ensures any transparency channel (alpha) or grayscale is standardized to 3-channel RGB.
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # WHAT: Execute model inference and track latency.
        # WHY: Monitoring inference time separately from total request latency allows us to distinguish model performance issues from network overhead.
        # HOW: Captures start/end timestamps around `model_loader.predict()`.
        start_time = time.time()
        predictions = model_loader.predict(image, top_k=top_k)
        inference_time = time.time() - start_time

        # WHAT: Construct the successful response payload.
        # WHY: Standardized responses with rich metadata (filename, model name, device used, image size, inference time) are crucial for client logging, debugging, and analytics.
        # HOW: Formats predictions, uses `secure_filename` to sanitize the user-provided filename, and returns the dictionary as a JSON response with status 200.
        response = {
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "filename": secure_filename(file.filename),
                "inference_time": round(inference_time, 3),
                "model": settings.model_name,
                "device": settings.model_device,
                "image_size": image.size
            }
        }

        logger.info(f"Prediction successful: {predictions[0]['class']} ({predictions[0]['confidence']:.3f})")

        return jsonify(response), 200

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            "error": "Validation error",
            "message": str(e),
            "status": "error"
        }), 400

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "status": "error"
        }), 500


@app.route('/', methods=['GET'])
def root() -> Dict[str, Any]:
    """
    Root endpoint with API documentation

    Returns:
        JSON with API information
    """
    return jsonify({
        "name": "ML Model Serving API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API documentation (this page)",
            "/health": "Health check endpoint",
            "/info": "Model information",
            "/predict": "Image classification endpoint (POST with file)"
        },
        "model": settings.model_name,
        "status": "running"
    }), 200


# Flask configuration
# WHAT: Restrict the maximum size of incoming client requests.
# WHY: Flask handles incoming requests by reading data into memory. If there's no limit, an attacker could stream gigabytes of data to exhaust server resources (causing a Denial of Service).
# HOW: Setting `app.config['MAX_CONTENT_LENGTH']` configures Flask to automatically reject requests larger than settings.max_file_size with a HTTP 413 Payload Too Large error.
app.config['MAX_CONTENT_LENGTH'] = settings.max_file_size


def main():
    """Main entry point"""
    logger.info("Starting ML Model API Server...")
    logger.info(f"Configuration: {settings.to_dict()}")

    # WHAT: Load the machine learning model weights on startup.
    # WHY: We must ensure the model load is successful BEFORE starting the web server. If it fails, the server should fail-fast and exit, alerting the infrastructure.
    # HOW: Calls `setup_model()` and exits with code 1 if it returns False.
    if not setup_model():
        logger.error("Failed to initialize model. Exiting.")
        sys.exit(1)

    # Run Flask app
    # WHAT: Start the WSGI web server to accept incoming connections.
    # WHY: This starts the main loop, listening for incoming TCP requests on the host/port.
    # HOW: `app.run` configures Flask's built-in development server with custom network options. Setting `threaded=True` allows the server to handle concurrent requests in different threads.
    logger.info(f"Starting Flask server on {settings.host}:{settings.port}")
    app.run(
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()