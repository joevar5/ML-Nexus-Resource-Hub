# API Documentation - ML Model Serving

## Base URL

- Local: `http://localhost:8000`
- Production: `https://api.your-domain.com`

## Authentication

Currently no authentication required. TODO: Add API key authentication for production.

---

## Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check if the service is healthy and ready to serve requests.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is not ready

---

### 2. Metrics

**Endpoint:** `GET /metrics`

**Description:** Prometheus-formatted metrics for monitoring.

**Response:** (Prometheus format)
```
# HELP predictions_total Total number of predictions
# TYPE predictions_total counter
predictions_total{status="success"} 1234
predictions_total{status="error"} 56

# HELP prediction_duration_seconds Time spent on inference
# TYPE prediction_duration_seconds histogram
prediction_duration_seconds_bucket{le="0.01"} 100
...
```

---

### 3. Predict (Image Classification)

**Endpoint:** `POST /v1/predict`

**Description:** Classify an uploaded image.

**Request:**

*Content-Type:* `multipart/form-data`

**Parameters:**
- `file` (required): Image file (JPEG, PNG)
- `top_k` (optional): Number of top predictions to return (default: 5, max: 10)

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -F "file=@cat.jpg" \
  -F "top_k=3"
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/v1/predict"
files = {"file": open("cat.jpg", "rb")}
data = {"top_k": 3}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response (Success):**
```json
{
  "predictions": [
    {
      "class_id": 281,
      "class_name": "tabby_cat",
      "probability": 0.95
    },
    {
      "class_id": 282,
      "class_name": "tiger_cat",
      "probability": 0.03
    },
    {
      "class_id": 285,
      "class_name": "Egyptian_cat",
      "probability": 0.01
    }
  ],
  "inference_time_ms": 25,
  "model_version": "resnet18-v1",
  "request_id": "uuid-1234-5678"
}
```

**Response (Error):**
```json
{
  "detail": "Invalid image format. Supported formats: JPEG, PNG",
  "error_code": "INVALID_INPUT"
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid input
- `413 Payload Too Large` - Image file too large (max 10MB)
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Model not loaded

---

### 4. Batch Predict (TODO)

**Endpoint:** `POST /v1/batch-predict`

**Description:** Classify multiple images in a single request.

**Request:**
*Content-Type:* `multipart/form-data`

**Parameters:**
- `files`: Array of image files

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/batch-predict" \
  -F "files=@cat1.jpg" \
  -F "files=@cat2.jpg" \
  -F "files=@dog1.jpg"
```

**Response:**
```json
{
  "results": [
    {"filename": "cat1.jpg", "predictions": [...]},
    {"filename": "cat2.jpg", "predictions": [...]},
    {"filename": "dog1.jpg", "predictions": [...]}
  ],
  "total_inference_time_ms": 75
}
```

---

## Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `INVALID_INPUT` | Invalid request format | Check request format |
| `FILE_TOO_LARGE` | File exceeds size limit | Reduce file size |
| `UNSUPPORTED_FORMAT` | Unsupported file format | Use JPEG or PNG |
| `MODEL_NOT_LOADED` | Model not ready | Wait and retry |
| `INFERENCE_FAILED` | Model inference error | Report to support |
| `INTERNAL_ERROR` | Server error | Report to support |

---

## Rate Limits

- **Development:** No limits
- **Production:** 60 requests per minute per IP (TODO: Configure)

Exceeded rate limit response:
```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds.",
  "retry_after": 60
}
```

---

## ImageNet Classes

The model is trained on ImageNet with 1000 classes. See [ImageNet Class Labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

Common classes:
- 281: tabby cat
- 282: tiger cat
- 207-294: various cat breeds
- 151-268: various dog breeds
- ...

---

## Interactive API Documentation

Visit `/docs` for interactive Swagger UI documentation:
```
http://localhost:8000/docs
```

Visit `/redoc` for ReDoc documentation:
```
http://localhost:8000/redoc
```

---

## Client Libraries

### Python

```python
# TODO: Create Python client library
```

### JavaScript

```javascript
// TODO: Create JavaScript client library
```

---

## Changelog

### v1.0.0 (2025-10-15)
- Initial release
- Image classification endpoint
- Health check and metrics

### Future Releases
- v1.1.0: Batch prediction
- v1.2.0: Video classification
- v1.3.0: Model versioning and A/B testing
