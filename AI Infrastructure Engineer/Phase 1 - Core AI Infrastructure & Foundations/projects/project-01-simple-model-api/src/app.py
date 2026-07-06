#!/usr/bin/env python3
"""
ML Model Serving API (FastAPI Implementation)

FastAPI application that serves predictions from a pre-trained image classification model.

Usage:
    python app.py

    # Test endpoints:
    curl http://localhost:5000/health
    curl http://localhost:5000/info
    curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
"""

from fastapi import FastAPI, Request, File, UploadFile, Form, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from werkzeug.utils import secure_filename
from PIL import Image
import io
import logging
import logging.config
import time
from typing import Dict, Any, Tuple, Optional
import sys
import uuid
from pathlib import Path

from config import get_settings
from model_loader import ModelLoader

# Retrieve the configuration settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global model loader instance
model_loader: Optional[ModelLoader] = None


def setup_model() -> bool:
    """Initialize model loader on startup"""
    global model_loader

    logger.info("Starting model initialization...")
    try:
        model_loader = ModelLoader(
            model_name=settings.model_name,
            device=settings.model_device
        )
        model_loader.warmup()
        logger.info("Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown execution"""
    logger.info("Starting ML Model API Server...")
    logger.info(f"Configuration: {settings.to_dict()}")

    if not setup_model():
        logger.error("Failed to initialize model. Exiting.")
        sys.exit(1)

    yield


app = FastAPI(
    title="ML Model Serving API",
    version="1.0.0",
    lifespan=lifespan
)


# Middleware: Log request/response and measure duration
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request.state.correlation_id = f"req-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path} [Correlation ID: {request.state.correlation_id}]")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(f"Response: {response.status_code} ({duration:.3f}s)")
    response.headers["X-Correlation-ID"] = request.state.correlation_id
    return response


# Middleware: Enforce maximum upload file size
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/predict":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.max_file_size:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "success": False,
                            "status": "error",
                            "error": {
                                "code": "FILE_TOO_LARGE",
                                "message": f"Maximum file size is {settings.max_file_size} bytes"
                            }
                        }
                    )
            except ValueError:
                pass
    return await call_next(request)


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    message = errors[0]["msg"] if errors else "Validation error"
    loc = errors[0]["loc"] if errors else []
    
    code = "INVALID_PARAMETER"
    if "file" in loc:
        code = "MISSING_FILE"
        message = "Request must include 'file' field"
        
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "status": "error",
            "error": {
                "code": code,
                "message": message
            }
        }
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "status": "error",
            "error": {
                "code": "NOT_FOUND",
                "message": "The requested URL was not found on the server."
            }
        }
    )


@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=405,
        content={
            "success": False,
            "status": "error",
            "error": {
                "code": "METHOD_NOT_ALLOWED",
                "message": "The method is not allowed for the requested URL."
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "status": "error",
            "error": {
                "code": "ERROR",
                "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            }
        }
    )


@app.exception_handler(Exception)
async def custom_500_handler(request: Request, exc: Exception):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "status": "error",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred"
            }
        }
    )


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in settings.allowed_extensions


def validate_image(file: UploadFile) -> Tuple[bool, str]:
    """Validate uploaded image file"""
    if not file or not file.filename:
        return False, "No file provided"

    # Check file extension
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {settings.allowed_extensions}"

    # Check file size (via seek/tell)
    try:
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
    except Exception as e:
        return False, f"Could not read file size: {e}"

    if file_size > settings.max_file_size:
        return False, f"File too large. Max size: {settings.max_file_size} bytes"

    if file_size == 0:
        return False, "Empty file"

    # Try to open as image
    try:
        img = Image.open(file.file)
        img.verify()
        file.file.seek(0)
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


# Endpoints
@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint with API documentation"""
    return {
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
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint"""
    if model_loader is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "message": "Model not loaded",
                "timestamp": time.time()
            }
        )

    return {
        "status": "healthy",
        "model": settings.model_name,
        "model_name": settings.model_name,
        "model_loaded": True,
        "device": settings.model_device,
        "timestamp": time.time()
    }


@app.get("/info")
def info() -> Dict[str, Any]:
    """Model information endpoint"""
    if model_loader is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Model not loaded",
                "status": "error"
            }
        )

    try:
        model_info = model_loader.get_model_info()
        return {
            "status": "success",
            "model": model_info,
            "api": {
                "version": "1.0.0"
            },
            "limits": {
                "max_file_size_mb": settings.max_file_size / (1024 * 1024),
                "timeout_seconds": settings.request_timeout
            },
            "config": {
                "max_file_size": settings.max_file_size,
                "allowed_extensions": settings.allowed_extensions,
                "request_timeout": settings.request_timeout
            }
        }

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to get model info",
                "message": str(e),
                "status": "error"
            }
        )


@app.post("/predict")
def predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    top_k: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Prediction endpoint that accepts an image and returns predictions
    """
    if model_loader is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Model not loaded",
                "status": "error"
            }
        )

    # 1. Validate 'file' parameter
    if file is None or file.filename == '':
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "status": "error",
                "error": {
                    "code": "MISSING_FILE",
                    "message": "Request must include 'file' field"
                }
            }
        )

    # 2. Parse and Validate 'top_k' parameter
    parsed_top_k = 5
    if top_k is not None:
        try:
            parsed_top_k = int(top_k)
            if not (1 <= parsed_top_k <= 10):
                raise ValueError()
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "status": "error",
                    "error": {
                        "code": "INVALID_PARAMETER",
                        "message": "top_k must be an integer between 1 and 10"
                    }
                }
            )

    # 3. Validate image content
    is_valid, error_msg = validate_image(file)
    if not is_valid:
        code = "INVALID_IMAGE_FORMAT"
        if "File too large" in error_msg:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "success": False,
                    "status": "error",
                    "error": {
                        "code": "FILE_TOO_LARGE",
                        "message": error_msg
                    }
                }
            )
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "status": "error",
                "error": {
                    "code": code,
                    "message": error_msg
                }
            }
        )

    try:
        # Load image bytes and convert to RGB
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Run inference
        start_time = time.time()
        predictions = model_loader.predict(image, top_k=parsed_top_k)
        inference_time = time.time() - start_time
        latency_ms = round(inference_time * 1000, 3)

        # Retrieve correlation ID from request state
        correlation_id = getattr(request.state, "correlation_id", "req-unknown")

        response = {
            "success": True,
            "status": "success",
            "predictions": predictions,
            "latency_ms": latency_ms,
            "correlation_id": correlation_id,
            "metadata": {
                "filename": secure_filename(file.filename),
                "inference_time": round(inference_time, 3),
                "model": settings.model_name,
                "device": settings.model_device,
                "image_size": image.size
            }
        }

        logger.info(f"Prediction successful: {predictions[0]['class']} ({predictions[0]['confidence']:.3f})")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "status": "error",
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": str(e)
                }
            }
        )


def main():
    """Main entry point"""
    import uvicorn
    # Serve the app directly
    logger.info(f"Starting FastAPI server on {settings.host}:{settings.port}")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None
    )


if __name__ == '__main__':
    main()