# Exercise 01: Build Your First ML Dockerfile

**Estimated Time:** 2-3 hours
**Difficulty:** Beginner
**Prerequisites:** Docker installed, basic Python knowledge

## Objective

Create a Dockerfile for a FastAPI application that serves a simple machine learning model (image classification with ResNet).

## Learning Goals

- Write a production-ready Dockerfile
- Install ML dependencies correctly
- Handle model files appropriately
- Test the containerized application

## Project Structure

Create the following structure:

```
ml-api-exercise/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── app.py
├── model.py
└── README.md
```

## Step 1: Create Application Files

**requirements.txt:**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
torchvision==0.16.0
pillow==10.1.0
python-multipart==0.0.6
pydantic==2.5.0
```

**model.py:**
```python
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io

class ImageClassifier:
    def __init__(self):
        # Load pre-trained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ImageNet class labels
        self.labels = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]

    def predict(self, image_bytes: bytes) -> dict:
        """Predict image class"""
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        results = []
        for i in range(5):
            results.append({
                "class": self.labels[top5_catid[i]],
                "probability": float(top5_prob[i])
            })

        return results
```

**app.py:**
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from model import ImageClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="Image Classification API", version="1.0.0")

# Load model at startup
classifier = None

@app.on_event("startup")
async def startup_event():
    global classifier
    logger.info("Loading model...")
    classifier = ImageClassifier()
    logger.info("Model loaded successfully")

@app.get("/")
async def root():
    return {"message": "Image Classification API", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "resnet18"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict image class"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    image_bytes = await file.read()

    # Get prediction
    try:
        results = classifier.predict(image_bytes)
        return {
            "filename": file.filename,
            "predictions": results
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Step 2: Write Dockerfile

**TODO: Complete this Dockerfile following best practices**

```dockerfile
# TODO: Choose appropriate base image
# Consider: python:3.11-slim for smaller size


# TODO: Set working directory


# TODO: Install system dependencies (if needed)


# TODO: Copy requirements.txt and install Python packages
# Remember: Copy requirements BEFORE code for better caching


# TODO: Copy application code


# TODO: Create non-root user
# Security best practice!


# TODO: Switch to non-root user


# TODO: Set environment variables
# PYTHONUNBUFFERED=1 is recommended


# TODO: Expose port


# TODO: Add health check


# TODO: Set startup command
# Use exec form: CMD ["python", "app.py"]

```

## Step 3: Create .dockerignore

**TODO: Create .dockerignore file**

```
# Add files/directories to exclude from Docker build
# Examples: __pycache__, .git, *.pyc, .venv, etc.

```

## Step 4: Build and Test

```bash
# Build the image
docker build -t ml-api:v1.0 .

# Check image size
docker images ml-api:v1.0

# Run the container
docker run -d -p 8000:8000 --name ml-api ml-api:v1.0

# Check if running
docker ps

# View logs
docker logs ml-api

# Test health endpoint
curl http://localhost:8000/health

# Test prediction (download a test image first)
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict

# Stop and remove
docker stop ml-api
docker rm ml-api
```

## Success Criteria

Your Dockerfile should meet these requirements:

- [ ] Image builds successfully without errors
- [ ] Image size is < 2GB (preferably < 1GB)
- [ ] Container starts and stays running
- [ ] Health check endpoint returns 200 OK
- [ ] Prediction endpoint works with a test image
- [ ] Runs as non-root user
- [ ] Uses layer caching effectively (requirements before code)
- [ ] Includes .dockerignore file

## Stretch Goals

1. **Multi-stage build**: Reduce image size further
2. **Metrics endpoint**: Add Prometheus metrics
3. **Async predictions**: Use FastAPI background tasks
4. **Model versioning**: Accept model version as environment variable
5. **Docker Compose**: Add redis for caching predictions

## Solution Hints

<details>
<summary>Click to reveal hints</summary>

**Base Image:**
```dockerfile
FROM python:3.11-slim
```

**Non-root user:**
```dockerfile
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
```

**Health check:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

**Install curl for health check:**
```dockerfile
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
```

</details>

## Submission

Create a GitHub repository with:
1. Complete Dockerfile
2. All application files
3. README.md with:
   - Build instructions
   - Run instructions
   - Example API usage
   - Image size achieved

## Common Issues and Solutions

**Issue: "torch not found"**
- Solution: Ensure requirements.txt is installed before copying code

**Issue: "Permission denied"**
- Solution: Check that files are owned by appuser, or copy with --chown

**Issue: "Health check failing"**
- Solution: Ensure curl is installed, or use python for health check

**Issue: "Image too large (>3GB)"**
- Solution: Use python:3.11-slim instead of python:3.11, use --no-cache-dir with pip

## Additional Challenges

1. **Optimize for CPU**: Install CPU-only PyTorch to reduce size
2. **Add logging**: Implement structured JSON logging
3. **Environment config**: Make port and host configurable via env vars
4. **Batch prediction**: Support uploading multiple images

---

**Next Exercise:** [exercise-02-multi-container.md](./exercise-02-multi-container.md) (multi-container Docker Compose stacks)
**Related Lab:** [lab-01-dockerfile-optimization.md](../labs/lab-01-dockerfile-optimization.md) (deeper Dockerfile optimization techniques)
