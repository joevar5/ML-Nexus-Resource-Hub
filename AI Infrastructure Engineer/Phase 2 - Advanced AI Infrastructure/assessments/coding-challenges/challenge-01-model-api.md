# Coding Challenge 01: Build a Model Serving API

**Difficulty:** Intermediate
**Time Limit:** 2 hours
**Topics:** FastAPI, Docker, Model Serving

---

## Challenge Overview

Build a production-ready REST API for serving a machine learning model. The API should include proper error handling, logging, metrics, and containerization.

## Requirements

### Functional Requirements

1. **API Endpoints:**
   - `POST /predict` - Accept input and return predictions
   - `GET /health` - Health check endpoint
   - `GET /metrics` - Prometheus metrics endpoint
   - `GET /model/info` - Return model metadata

2. **Model:**
   - Use any pre-trained model (sklearn, PyTorch, TensorFlow)
   - Load model on startup
   - Cache model in memory

3. **Input Validation:**
   - Validate request payload structure
   - Return clear error messages for invalid input
   - Handle missing fields gracefully

4. **Logging:**
   - Log all requests with timestamp
   - Log errors with stack traces
   - Use structured logging (JSON format)

5. **Metrics:**
   - Track request count
   - Track request latency (histogram)
   - Track error rate
   - Expose metrics in Prometheus format

### Non-Functional Requirements

1. **Containerization:**
   - Create optimized Dockerfile
   - Use multi-stage build
   - Run as non-root user
   - Image size < 500MB (if possible)

2. **Performance:**
   - API response time < 200ms (p95)
   - Support concurrent requests
   - Implement timeout handling

3. **Code Quality:**
   - Type hints for all functions
   - Docstrings for public functions
   - Follow PEP 8 style guide
   - No hardcoded values (use config)

## Starter Code

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

app = FastAPI()

class PredictionRequest(BaseModel):
    # TODO: Define request schema
    pass

class PredictionResponse(BaseModel):
    # TODO: Define response schema
    pass

@app.on_event("startup")
async def startup_event():
    # TODO: Load model
    pass

@app.post("/predict")
async def predict(request: PredictionRequest) -> PredictionResponse:
    # TODO: Implement prediction
    pass

@app.get("/health")
async def health():
    # TODO: Implement health check
    pass

# TODO: Add metrics endpoint
# TODO: Add model info endpoint
```

```dockerfile
# Dockerfile
FROM python:3.11-slim as builder

# TODO: Implement multi-stage build
# Stage 1: Install dependencies
# Stage 2: Copy only necessary files

# TODO: Run as non-root user
# TODO: Expose port
# TODO: Set CMD
```

## Evaluation Criteria

### Code Quality (30 points)
- [ ] Type hints used (5 pts)
- [ ] Proper docstrings (5 pts)
- [ ] Follows PEP 8 (5 pts)
- [ ] No hardcoded values (5 pts)
- [ ] Error handling (10 pts)

### Functionality (40 points)
- [ ] All endpoints working (15 pts)
- [ ] Input validation (10 pts)
- [ ] Logging implemented (5 pts)
- [ ] Metrics working (10 pts)

### Containerization (20 points)
- [ ] Dockerfile builds successfully (5 pts)
- [ ] Multi-stage build (5 pts)
- [ ] Non-root user (5 pts)
- [ ] Optimized image size (5 pts)

### Performance (10 points)
- [ ] Meets latency requirement (5 pts)
- [ ] Handles concurrent requests (5 pts)

**Total: 100 points**

## Submission

Submit:
1. Source code (`app/` directory)
2. `Dockerfile`
3. `requirements.txt`
4. `README.md` with:
   - How to build and run
   - API documentation
   - Example requests
   - Performance benchmarks

## Testing Your Solution

```bash
# Build image
docker build -t model-api .

# Run container
docker run -p 8000:8000 model-api

# Test endpoints
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [1, 2, 3, 4]}'

curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/model/info
```

## Bonus Challenges (+10 points each)

1. **Rate Limiting** - Implement rate limiting (100 requests/minute)
2. **Caching** - Cache predictions for identical inputs
3. **Batch Prediction** - Support batch predictions
4. **Authentication** - Add API key authentication
5. **OpenAPI Docs** - Customize OpenAPI documentation

---

**Time's up?** Submit what you have. Partial solutions receive partial credit!
