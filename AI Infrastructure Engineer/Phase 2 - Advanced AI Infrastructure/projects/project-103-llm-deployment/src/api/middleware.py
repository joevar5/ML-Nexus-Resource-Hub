"""
API Middleware for Rate Limiting, Logging, and Authentication

This module implements middleware for LLM API production features:
- Rate limiting to prevent abuse
- Request/response logging
- Authentication and authorization
- Request validation
- Error handling
- CORS configuration

Learning Objectives:
- Understand middleware patterns in FastAPI
- Implement rate limiting strategies
- Learn authentication best practices
- Build production-ready logging
- Handle API security concerns

Key Concepts:
- Middleware execution order
- Rate limiting algorithms (token bucket, sliding window)
- API key authentication
- Request context and tracing
- Security headers
"""

import time
import logging
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import secrets

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


# ============================================================================
# RATE LIMITING MIDDLEWARE
# ============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting middleware.

    Implements per-IP and per-API-key rate limiting to prevent abuse.

    TODO: Implement rate limiting using token bucket algorithm
    - Track requests per client (IP or API key)
    - Allow burst traffic up to bucket size
    - Refill tokens at constant rate
    - Return 429 when rate exceeded
    - Include rate limit headers in response

    Token Bucket Algorithm:
    - Each client has a bucket with capacity N tokens
    - Tokens refill at rate R per second
    - Each request consumes 1 token
    - Request rejected if no tokens available

    Learning Resources:
    - https://en.wikipedia.org/wiki/Token_bucket
    - https://redis.io/glossary/rate-limiting/
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        """
        Initialize rate limiter.

        TODO: Implement initialization
        - Store rate limit configuration
        - Initialize client buckets storage
        - Set up token refill rate

        Args:
            app: FastAPI application
            requests_per_minute: Sustained request rate
            burst_size: Maximum burst requests
        """
        super().__init__(app)
        # TODO: Store configuration
        # self.requests_per_minute = requests_per_minute
        # self.burst_size = burst_size
        # self.refill_rate = requests_per_minute / 60.0  # tokens per second

        # TODO: Initialize storage
        # Storage format: {client_id: {"tokens": float, "last_update": float}}
        # self.buckets: Dict[str, Dict[str, float]] = {}

        pass

    def _get_client_id(self, request: Request) -> str:
        """
        Get unique client identifier.

        TODO: Implement client identification
        - Check for API key in headers
        - Fall back to IP address
        - Hash for privacy

        Args:
            request: Incoming request

        Returns:
            Client identifier string
        """
        # TODO: Check for API key
        # api_key = request.headers.get("X-API-Key")
        # if api_key:
        #     return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        # TODO: Use IP address
        # ip = request.client.host
        # return f"ip:{ip}"

        pass

    def _get_tokens(self, client_id: str) -> float:
        """
        Get current token count for client.

        TODO: Implement token bucket logic
        - Get bucket for client (create if new)
        - Calculate tokens to add since last request
        - Cap at burst_size
        - Update last_update time

        Args:
            client_id: Client identifier

        Returns:
            Current number of tokens
        """
        # TODO: Implement token bucket
        # current_time = time.time()
        #
        # if client_id not in self.buckets:
        #     # New client - give full bucket
        #     self.buckets[client_id] = {
        #         "tokens": self.burst_size,
        #         "last_update": current_time
        #     }
        #     return self.burst_size
        #
        # bucket = self.buckets[client_id]
        # time_passed = current_time - bucket["last_update"]
        # tokens_to_add = time_passed * self.refill_rate
        #
        # # Update tokens (capped at burst_size)
        # bucket["tokens"] = min(
        #     bucket["tokens"] + tokens_to_add,
        #     self.burst_size
        # )
        # bucket["last_update"] = current_time
        #
        # return bucket["tokens"]

        pass

    def _consume_token(self, client_id: str) -> bool:
        """
        Attempt to consume one token.

        TODO: Implement token consumption
        - Get current tokens
        - If >= 1, consume and return True
        - If < 1, return False (rate limited)

        Args:
            client_id: Client identifier

        Returns:
            True if token consumed, False if rate limited
        """
        # TODO: Implement consumption
        # tokens = self._get_tokens(client_id)
        # if tokens >= 1.0:
        #     self.buckets[client_id]["tokens"] -= 1.0
        #     return True
        # return False

        pass

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting.

        TODO: Implement middleware dispatch
        - Get client ID
        - Try to consume token
        - If success, process request
        - If failure, return 429 error
        - Add rate limit headers to response

        Args:
            request: Incoming request
            call_next: Next middleware/endpoint

        Returns:
            Response with rate limit headers
        """
        # TODO: Skip rate limiting for health checks
        # if request.url.path in ["/health", "/metrics"]:
        #     return await call_next(request)

        # TODO: Get client ID and check rate limit
        # client_id = self._get_client_id(request)
        #
        # if not self._consume_token(client_id):
        #     # Rate limited
        #     return JSONResponse(
        #         status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        #         content={
        #             "error": "Rate limit exceeded",
        #             "error_code": "rate_limit_exceeded",
        #             "retry_after": 60
        #         },
        #         headers={
        #             "X-RateLimit-Limit": str(self.requests_per_minute),
        #             "X-RateLimit-Remaining": "0",
        #             "Retry-After": "60"
        #         }
        #     )

        # TODO: Process request
        # response = await call_next(request)

        # TODO: Add rate limit headers
        # tokens = self._get_tokens(client_id)
        # response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        # response.headers["X-RateLimit-Remaining"] = str(int(tokens))

        # return response

        pass


# ============================================================================
# LOGGING MIDDLEWARE
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all requests and responses with timing.

    TODO: Implement request/response logging
    - Log request details (method, path, client)
    - Measure request duration
    - Log response status and size
    - Include request ID for tracing
    - Structured logging for analysis

    Best Practices:
    - Use structured logging (JSON)
    - Include correlation IDs
    - Don't log sensitive data (auth tokens, PII)
    - Log at appropriate levels
    - Include timing information
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response.

        TODO: Implement logging
        - Generate request ID
        - Log incoming request
        - Measure duration
        - Log response
        - Handle errors

        Args:
            request: Incoming request
            call_next: Next middleware/endpoint

        Returns:
            Response with request ID header
        """
        # TODO: Generate request ID
        # request_id = secrets.token_hex(8)

        # TODO: Log incoming request
        # logger.info(
        #     "Incoming request",
        #     extra={
        #         "request_id": request_id,
        #         "method": request.method,
        #         "path": request.url.path,
        #         "client": request.client.host,
        #         "user_agent": request.headers.get("user-agent")
        #     }
        # )

        # TODO: Process and time request
        # start_time = time.time()
        # try:
        #     response = await call_next(request)
        #     duration = time.time() - start_time
        #
        #     # Log response
        #     logger.info(
        #         "Request completed",
        #         extra={
        #             "request_id": request_id,
        #             "status_code": response.status_code,
        #             "duration_ms": duration * 1000,
        #             "response_size": response.headers.get("content-length", 0)
        #         }
        #     )
        #
        #     response.headers["X-Request-ID"] = request_id
        #     return response
        #
        # except Exception as e:
        #     duration = time.time() - start_time
        #     logger.error(
        #         "Request failed",
        #         extra={
        #             "request_id": request_id,
        #             "error": str(e),
        #             "duration_ms": duration * 1000
        #         },
        #         exc_info=True
        #     )
        #     raise

        pass


# ============================================================================
# AUTHENTICATION MIDDLEWARE
# ============================================================================

class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    API key authentication middleware.

    TODO: Implement API key authentication
    - Check for API key in headers
    - Validate key against database/cache
    - Track key usage
    - Support key rotation
    - Rate limit per key

    Security Considerations:
    - Hash API keys in database
    - Use HTTPS only
    - Implement key expiration
    - Monitor for suspicious activity
    - Support key revocation
    """

    def __init__(self, app, api_keys: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize authentication.

        TODO: Implement initialization
        - Load API keys (from database/config)
        - Set up key validation
        - Configure public endpoints

        Args:
            app: FastAPI application
            api_keys: Dict mapping API keys to metadata
                     Format: {key: {"name": "...", "tier": "...", "limits": {...}}}
        """
        super().__init__(app)
        # TODO: Store API keys (hashed)
        # self.api_keys = api_keys or {}

        # TODO: Define public endpoints (no auth required)
        # self.public_endpoints = {"/health", "/docs", "/openapi.json"}

        pass

    def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key and return metadata.

        TODO: Implement key validation
        - Hash provided key
        - Look up in database/cache
        - Check expiration
        - Return key metadata if valid

        Args:
            api_key: API key from request

        Returns:
            Key metadata if valid, None otherwise
        """
        # TODO: Hash key
        # key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # TODO: Look up in database
        # key_data = self.api_keys.get(key_hash)
        # if not key_data:
        #     return None

        # TODO: Check expiration
        # if "expires_at" in key_data:
        #     if datetime.fromisoformat(key_data["expires_at"]) < datetime.now():
        #         return None

        # return key_data

        pass

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Authenticate request.

        TODO: Implement authentication
        - Skip public endpoints
        - Extract API key from header
        - Validate key
        - Attach key metadata to request
        - Return 401 if invalid

        Args:
            request: Incoming request
            call_next: Next middleware/endpoint

        Returns:
            Response or 401 error
        """
        # TODO: Skip public endpoints
        # if request.url.path in self.public_endpoints:
        #     return await call_next(request)

        # TODO: Extract API key
        # api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
        #
        # if not api_key:
        #     return JSONResponse(
        #         status_code=status.HTTP_401_UNAUTHORIZED,
        #         content={
        #             "error": "Missing API key",
        #             "error_code": "missing_api_key"
        #         }
        #     )

        # TODO: Validate key
        # key_data = self._validate_api_key(api_key)
        # if not key_data:
        #     return JSONResponse(
        #         status_code=status.HTTP_401_UNAUTHORIZED,
        #         content={
        #             "error": "Invalid API key",
        #             "error_code": "invalid_api_key"
        #         }
        #     )

        # TODO: Attach metadata to request state
        # request.state.api_key_data = key_data

        # TODO: Process request
        # return await call_next(request)

        pass


# ============================================================================
# COST TRACKING MIDDLEWARE
# ============================================================================

class CostTrackingMiddleware(BaseHTTPMiddleware):
    """
    Track API usage costs per request.

    TODO: Implement cost tracking
    - Calculate tokens used
    - Estimate cost based on model
    - Track costs per API key
    - Enforce budget limits
    - Generate billing data

    This is crucial for:
    - Billing customers
    - Budget management
    - Cost optimization
    - Usage analytics
    """

    def __init__(self, app, cost_per_1k_tokens: Dict[str, float]):
        """
        Initialize cost tracking.

        Args:
            app: FastAPI application
            cost_per_1k_tokens: Pricing per model
        """
        super().__init__(app)
        # TODO: Store pricing
        # self.pricing = cost_per_1k_tokens

        # TODO: Initialize cost storage
        # Storage format: {api_key: {"total_cost": float, "requests": int}}
        # self.costs: Dict[str, Dict[str, Any]] = defaultdict(
        #     lambda: {"total_cost": 0.0, "requests": 0}
        # )

        pass

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Track request cost.

        TODO: Implement cost tracking
        - Process request
        - Extract token usage from response
        - Calculate cost
        - Update running totals
        - Add cost header to response
        - Enforce budget limits

        Args:
            request: Incoming request
            call_next: Next middleware/endpoint

        Returns:
            Response with cost headers
        """
        # TODO: Process request
        # response = await call_next(request)

        # TODO: Extract usage from response (if available)
        # This requires modifying response body, which is complex
        # Alternative: Use request.state to pass usage data

        # TODO: Calculate cost
        # cost = calculate_cost(prompt_tokens, completion_tokens, model)

        # TODO: Add cost header
        # response.headers["X-Request-Cost"] = f"{cost:.6f}"

        # return response

        pass


# ============================================================================
# CORS MIDDLEWARE
# ============================================================================

def setup_cors(app, allowed_origins: list = None):
    """
    Configure CORS middleware.

    TODO: Implement CORS setup
    - Allow specified origins
    - Configure allowed methods
    - Set allowed headers
    - Handle preflight requests

    Args:
        app: FastAPI application
        allowed_origins: List of allowed origins
    """
    # TODO: Set default origins
    # if allowed_origins is None:
    #     allowed_origins = ["*"]  # WARNING: Restrict in production!

    # TODO: Add CORS middleware
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=allowed_origins,
    #     allow_credentials=True,
    #     allow_methods=["*"],
    #     allow_headers=["*"],
    #     expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
    # )

    pass


# ============================================================================
# UTILITIES
# ============================================================================

def generate_api_key() -> str:
    """
    Generate a secure API key.

    TODO: Implement key generation
    - Use cryptographically secure random
    - Include prefix for identification
    - Sufficient length (32+ characters)

    Returns:
        New API key string

    Format: llm_<40 hex characters>
    """
    # TODO: Generate key
    # prefix = "llm_"
    # random_part = secrets.token_hex(20)
    # return prefix + random_part

    pass


def hash_api_key(api_key: str) -> str:
    """
    Hash API key for storage.

    TODO: Implement key hashing
    - Use SHA-256 or stronger
    - Return hex digest

    Args:
        api_key: Plain text API key

    Returns:
        Hashed key
    """
    # TODO: Hash key
    # return hashlib.sha256(api_key.encode()).hexdigest()

    pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example FastAPI Setup:

from fastapi import FastAPI
from .middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    APIKeyAuthMiddleware,
    setup_cors
)

app = FastAPI()

# Add middleware (order matters! First added = outermost layer)
# Execution order: CORS -> Rate Limit -> Logging -> Auth -> Endpoint

setup_cors(app, allowed_origins=["https://example.com"])

app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=60,
    burst_size=10
)

app.add_middleware(RequestLoggingMiddleware)

api_keys = {
    hash_api_key("llm_test_key_123"): {
        "name": "Test User",
        "tier": "free",
        "limits": {"requests_per_day": 1000}
    }
}

app.add_middleware(
    APIKeyAuthMiddleware,
    api_keys=api_keys
)

@app.get("/generate")
async def generate(request: Request):
    # Access API key metadata
    key_data = request.state.api_key_data
    # ... process request
    pass


Example API Key Generation:

key = generate_api_key()
print(f"API Key: {key}")
# Output: llm_a1b2c3d4e5f6...

# Store hashed version
key_hash = hash_api_key(key)
print(f"Hash: {key_hash}")
# Store key_hash in database


Example Rate Limiting Response:

HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
Retry-After: 60

{
    "error": "Rate limit exceeded",
    "error_code": "rate_limit_exceeded",
    "retry_after": 60
}
"""
