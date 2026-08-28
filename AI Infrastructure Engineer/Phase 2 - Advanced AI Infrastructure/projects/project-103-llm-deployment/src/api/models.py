"""
Pydantic Models for LLM API Requests and Responses

This module defines the data models used for API request validation and response serialization.
Using Pydantic provides automatic validation, serialization, and documentation generation.

Learning Objectives:
- Understand API contract design for LLM systems
- Learn request/response validation patterns
- Implement proper type hints and data validation
- Design extensible API models

Key Concepts:
- Pydantic BaseModel for schema definition
- Field validation and constraints
- Optional vs required fields
- Nested model structures
- Response streaming models
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ============================================================================
# REQUEST MODELS
# ============================================================================

class GenerateRequest(BaseModel):
    """
    Standard text generation request for LLM.

    TODO: Complete this model with proper validation
    - Add field for prompt (required, non-empty string)
    - Add max_tokens with default 512, range 1-4096
    - Add temperature with default 0.7, range 0.0-2.0
    - Add top_p for nucleus sampling (default 0.95)
    - Add stop_sequences (list of strings)
    - Add streaming flag (boolean)

    Fields to implement:
    - prompt: The input text to generate from
    - max_tokens: Maximum number of tokens to generate
    - temperature: Controls randomness (higher = more random)
    - top_p: Nucleus sampling threshold
    - top_k: Top-k sampling parameter
    - stop_sequences: Sequences that stop generation
    - stream: Whether to stream response via SSE
    - presence_penalty: Penalty for token presence
    - frequency_penalty: Penalty for token frequency
    """

    # TODO: Implement prompt field with validation
    # prompt: str = Field(
    #     ...,  # Required field
    #     min_length=1,
    #     max_length=8192,
    #     description="The input prompt for text generation"
    # )

    # TODO: Implement max_tokens with constraints
    # Hint: Use Field() with ge (greater/equal) and le (less/equal)

    # TODO: Implement temperature with validation
    # Should be between 0.0 and 2.0

    # TODO: Implement top_p (nucleus sampling)
    # Should be between 0.0 and 1.0

    # TODO: Implement top_k (optional)
    # Should be positive integer or None

    # TODO: Implement stop_sequences
    # Optional list of strings

    # TODO: Implement stream flag
    # Boolean, default False

    # TODO: Implement presence_penalty and frequency_penalty
    # Both between -2.0 and 2.0

    @validator('temperature')
    def validate_temperature(cls, v):
        """
        TODO: Implement temperature validation
        - Ensure it's between 0.0 and 2.0
        - Provide helpful error message if invalid
        """
        pass

    @validator('prompt')
    def validate_prompt(cls, v):
        """
        TODO: Implement prompt validation
        - Ensure prompt is not empty or just whitespace
        - Optionally check for maximum length
        - Strip leading/trailing whitespace
        """
        pass

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "stream": False
            }
        }


class RAGGenerateRequest(BaseModel):
    """
    RAG-augmented generation request.

    This extends standard generation with retrieval parameters.

    TODO: Complete this model for RAG requests
    - Include all fields from GenerateRequest
    - Add retrieval-specific parameters:
      - query: The retrieval query (can differ from prompt)
      - top_k: Number of documents to retrieve (default 3)
      - collection_name: Vector DB collection to search
      - filter: Optional metadata filters
      - rerank: Whether to rerank retrieved documents
      - min_relevance_score: Minimum similarity threshold
    """

    # TODO: Implement query field
    # This is the text used for retrieval (may differ from final prompt)

    # TODO: Implement prompt field (optional)
    # If not provided, will be constructed from query + context

    # TODO: Implement top_k for retrieval
    # Number of documents to retrieve (default 3, range 1-20)

    # TODO: Implement collection_name
    # Vector database collection to search

    # TODO: Implement metadata_filter
    # Dict for filtering by document metadata
    # Example: {"source": "documentation", "date": "2024"}

    # TODO: Implement rerank flag
    # Whether to rerank retrieved documents

    # TODO: Implement min_relevance_score
    # Minimum cosine similarity threshold (0.0-1.0)

    # TODO: Implement include_sources flag
    # Whether to include source documents in response

    # Standard generation parameters
    # TODO: Include max_tokens, temperature, etc.

    class Config:
        schema_extra = {
            "example": {
                "query": "How do I deploy a model to Kubernetes?",
                "top_k": 3,
                "collection_name": "documentation",
                "max_tokens": 512,
                "temperature": 0.7,
                "include_sources": True
            }
        }


class BatchGenerateRequest(BaseModel):
    """
    Batch generation request for processing multiple prompts.

    TODO: Implement batch request model
    - List of prompts (required)
    - Shared generation parameters
    - Optional per-prompt overrides
    - Batch processing settings

    This is useful for:
    - Processing multiple requests efficiently
    - Amortizing overhead across requests
    - Maximizing GPU utilization
    """

    # TODO: Implement prompts field
    # List of strings, max 100 prompts per batch

    # TODO: Implement shared generation parameters
    # max_tokens, temperature, etc. applied to all prompts

    # TODO: Implement per_prompt_overrides
    # Optional list of dicts with per-prompt parameters

    # TODO: Implement batch_size
    # How many prompts to process in parallel

    pass


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class GeneratedText(BaseModel):
    """
    Single generated text completion.

    TODO: Implement completion model
    - text: Generated text
    - finish_reason: Why generation stopped
    - tokens_used: Number of tokens generated
    - logprobs: Optional log probabilities
    """

    # TODO: Implement text field
    # The generated completion

    # TODO: Implement finish_reason
    # One of: "stop", "length", "content_filter", "error"

    # TODO: Implement tokens_used
    # Number of tokens in this completion

    # TODO: Implement logprobs
    # Optional token-level log probabilities

    pass


class GenerateResponse(BaseModel):
    """
    Response for standard text generation.

    TODO: Complete response model with:
    - generated_text: The completion(s)
    - model: Model name used
    - usage: Token usage statistics
    - latency_ms: Time to generate
    - request_id: Unique request identifier
    """

    # TODO: Implement generated_text
    # Can be single string or list for multiple completions

    # TODO: Implement model field
    # Name/version of model used

    # TODO: Implement usage statistics
    # prompt_tokens, completion_tokens, total_tokens

    # TODO: Implement latency_ms
    # Time taken for generation

    # TODO: Implement request_id
    # Unique ID for tracing/debugging

    # TODO: Implement finish_reason
    # Why generation stopped

    class Config:
        schema_extra = {
            "example": {
                "generated_text": "Quantum computing uses quantum bits...",
                "model": "llama-2-7b-chat",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 95,
                    "total_tokens": 107
                },
                "latency_ms": 342,
                "request_id": "req_abc123"
            }
        }


class SourceDocument(BaseModel):
    """
    Retrieved source document from RAG.

    TODO: Implement source document model
    - content: Document text/chunk
    - metadata: Document metadata (source, title, etc.)
    - relevance_score: Similarity score
    - chunk_id: Unique chunk identifier
    """

    # TODO: Implement content field
    # The retrieved text chunk

    # TODO: Implement metadata
    # Dict with source, title, url, date, etc.

    # TODO: Implement relevance_score
    # Cosine similarity or reranking score

    # TODO: Implement chunk_id
    # Unique identifier for this chunk

    pass


class RAGGenerateResponse(BaseModel):
    """
    Response for RAG-augmented generation.

    TODO: Extend GenerateResponse with:
    - sources: List of retrieved documents
    - retrieval_latency_ms: Time for retrieval
    - num_sources_retrieved: Number of sources used
    """

    # TODO: Include all fields from GenerateResponse

    # TODO: Implement sources field
    # List of SourceDocument objects

    # TODO: Implement retrieval_latency_ms
    # Time spent on vector search

    # TODO: Implement num_sources_retrieved
    # How many documents were retrieved

    # TODO: Implement context_length
    # Total tokens in retrieved context

    pass


class StreamChunk(BaseModel):
    """
    Single chunk in a streaming response.

    TODO: Implement streaming chunk model
    - chunk: Text chunk
    - finish_reason: Optional finish reason (last chunk only)
    - delta_tokens: Tokens in this chunk
    """

    # TODO: Implement chunk field
    # The text delta for this chunk

    # TODO: Implement finish_reason
    # Only present in final chunk

    # TODO: Implement delta_tokens
    # Number of tokens in this chunk

    # TODO: Implement request_id
    # Request ID for correlation

    pass


# ============================================================================
# MONITORING AND METADATA MODELS
# ============================================================================

class TokenUsage(BaseModel):
    """
    Token usage statistics.

    TODO: Implement token usage tracking
    - prompt_tokens: Tokens in input
    - completion_tokens: Tokens in output
    - total_tokens: Sum of both
    - cost_usd: Estimated cost (optional)
    """

    # TODO: Implement prompt_tokens

    # TODO: Implement completion_tokens

    # TODO: Implement total_tokens
    # Should be prompt_tokens + completion_tokens

    # TODO: Implement cost_usd
    # Optional cost calculation

    @validator('total_tokens', always=True)
    def calculate_total(cls, v, values):
        """
        TODO: Auto-calculate total_tokens
        Sum prompt_tokens and completion_tokens
        """
        pass


class HealthStatus(BaseModel):
    """
    Health check response.

    TODO: Implement health status model
    - status: "healthy", "degraded", "unhealthy"
    - model_loaded: Whether model is ready
    - gpu_available: GPU status
    - uptime_seconds: Service uptime
    - version: API version
    """

    # TODO: Implement status field
    # Literal type with allowed values

    # TODO: Implement model_loaded
    # Boolean indicating if LLM is loaded

    # TODO: Implement gpu_available
    # GPU availability status

    # TODO: Implement uptime_seconds
    # Time since service started

    # TODO: Implement version
    # API version string

    # TODO: Implement timestamp
    # Current timestamp

    pass


class ErrorResponse(BaseModel):
    """
    Standardized error response.

    TODO: Implement error model
    - error: Error message
    - error_code: Machine-readable code
    - details: Additional context
    - request_id: For debugging
    """

    # TODO: Implement error field
    # Human-readable error message

    # TODO: Implement error_code
    # Machine-readable error code
    # Examples: "invalid_request", "rate_limit_exceeded", "model_error"

    # TODO: Implement details
    # Optional additional context

    # TODO: Implement request_id
    # For tracing and debugging

    # TODO: Implement timestamp
    # When error occurred

    class Config:
        schema_extra = {
            "example": {
                "error": "Maximum token limit exceeded",
                "error_code": "invalid_request",
                "details": "Requested 5000 tokens, maximum is 4096",
                "request_id": "req_xyz789",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_generation_params(
    temperature: float,
    top_p: float,
    max_tokens: int
) -> None:
    """
    Validate generation parameters.

    TODO: Implement parameter validation
    - Check temperature range (0.0-2.0)
    - Check top_p range (0.0-1.0)
    - Check max_tokens > 0
    - Raise ValueError with helpful message if invalid

    Args:
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_tokens: Maximum tokens to generate

    Raises:
        ValueError: If any parameter is invalid
    """
    pass


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str
) -> float:
    """
    Estimate generation cost in USD.

    TODO: Implement cost estimation
    - Define cost per token for different models
    - Calculate total cost based on usage
    - Return estimated cost in USD

    Typical pricing (2024):
    - Llama 2 7B (self-hosted): ~$0.0002/1K tokens
    - GPT-3.5 Turbo: $0.0015/1K input, $0.002/1K output
    - GPT-4: $0.03/1K input, $0.06/1K output

    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        model_name: Name of the model

    Returns:
        Estimated cost in USD
    """
    # TODO: Define pricing table
    pricing = {
        "llama-2-7b": {"input": 0.0002, "output": 0.0002},
        "llama-2-13b": {"input": 0.0003, "output": 0.0003},
        # Add more models
    }

    # TODO: Calculate cost
    # cost = (prompt_tokens / 1000) * input_price + (completion_tokens / 1000) * output_price

    pass


# ============================================================================
# EXAMPLES AND DOCUMENTATION
# ============================================================================

"""
Example Usage:

# Create a generation request
request = GenerateRequest(
    prompt="Explain neural networks",
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    stream=False
)

# Create a RAG request
rag_request = RAGGenerateRequest(
    query="How do I fine-tune a model?",
    top_k=3,
    collection_name="ml_docs",
    max_tokens=512,
    include_sources=True
)

# Parse a response
response = GenerateResponse(
    generated_text="Neural networks are...",
    model="llama-2-7b-chat",
    usage=TokenUsage(
        prompt_tokens=5,
        completion_tokens=50,
        total_tokens=55
    ),
    latency_ms=230
)

# Handle errors
error = ErrorResponse(
    error="Model temporarily unavailable",
    error_code="service_unavailable",
    request_id="req_123"
)
"""
