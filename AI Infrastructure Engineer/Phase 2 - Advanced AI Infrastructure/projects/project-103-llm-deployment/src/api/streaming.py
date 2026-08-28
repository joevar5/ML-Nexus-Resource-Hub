"""
Server-Sent Events (SSE) Streaming for LLM Responses

This module implements streaming responses for real-time text generation.
SSE allows the server to push updates to the client as tokens are generated.

Learning Objectives:
- Understand streaming patterns for LLMs
- Implement SSE (Server-Sent Events) protocol
- Handle backpressure and flow control
- Implement proper error handling in streams
- Learn async iteration patterns

Key Concepts:
- Server-Sent Events (SSE) protocol
- Async generators in Python
- Streaming vs batch generation
- Token-by-token vs chunk streaming
- Error handling in streams

Benefits of Streaming:
- Lower perceived latency (users see results immediately)
- Better user experience for long generations
- Ability to cancel long-running requests
- Progressive rendering of responses
"""

import asyncio
import json
from typing import AsyncIterator, Optional, Dict, Any
from fastapi.responses import StreamingResponse
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SSE UTILITIES
# ============================================================================

def format_sse(data: str, event: Optional[str] = None, id: Optional[str] = None) -> str:
    """
    Format data as Server-Sent Event.

    SSE format:
    - event: event_name
    - data: json_data
    - id: event_id
    - (blank line to end event)

    TODO: Implement SSE formatting
    - Format event line if event provided
    - Format data line (required)
    - Format id line if id provided
    - Add blank line at end
    - Handle multi-line data

    Args:
        data: The data to send (will be sent as JSON)
        event: Optional event name
        id: Optional event ID

    Returns:
        Formatted SSE string

    Example:
        format_sse('{"text": "Hello"}', event="chunk", id="1")
        # Returns:
        # event: chunk
        # data: {"text": "Hello"}
        # id: 1
        #
    """
    # TODO: Implement SSE formatting
    # message = ""
    # if event:
    #     message += f"event: {event}\n"
    # message += f"data: {data}\n"
    # if id:
    #     message += f"id: {id}\n"
    # message += "\n"
    # return message
    pass


def create_sse_response(
    generator: AsyncIterator[str],
    status_code: int = 200
) -> StreamingResponse:
    """
    Create FastAPI StreamingResponse for SSE.

    TODO: Implement SSE response creation
    - Set proper content-type header (text/event-stream)
    - Set cache-control to no-cache
    - Set connection to keep-alive
    - Set X-Accel-Buffering to no (for nginx)
    - Return StreamingResponse with generator

    Args:
        generator: Async generator yielding SSE-formatted strings
        status_code: HTTP status code

    Returns:
        StreamingResponse configured for SSE
    """
    # TODO: Implement headers
    headers = {
        # "Content-Type": "text/event-stream",
        # "Cache-Control": "no-cache",
        # "Connection": "keep-alive",
        # "X-Accel-Buffering": "no",  # Disable nginx buffering
    }

    # TODO: Return StreamingResponse
    # return StreamingResponse(
    #     generator,
    #     status_code=status_code,
    #     headers=headers,
    #     media_type="text/event-stream"
    # )
    pass


# ============================================================================
# STREAMING GENERATORS
# ============================================================================

async def stream_llm_response(
    llm_generator: AsyncIterator[str],
    request_id: str,
    include_metadata: bool = True
) -> AsyncIterator[str]:
    """
    Stream LLM tokens as SSE events.

    This wraps the raw LLM generator and formats output as SSE.

    TODO: Implement streaming wrapper
    - Iterate over LLM token generator
    - Format each token as SSE event
    - Include metadata (tokens count, latency, etc.)
    - Handle errors gracefully
    - Send final "done" event

    Args:
        llm_generator: Async generator yielding tokens
        request_id: Unique request identifier
        include_metadata: Whether to include timing/token metadata

    Yields:
        SSE-formatted strings

    Event Types:
    - "chunk": Text chunk/token
    - "metadata": Token count, timing info
    - "done": Generation complete
    - "error": Error occurred
    """
    start_time = datetime.now()
    token_count = 0

    try:
        # TODO: Send initial event
        # yield format_sse(
        #     json.dumps({"request_id": request_id, "status": "started"}),
        #     event="start"
        # )

        # TODO: Stream tokens
        # async for token in llm_generator:
        #     token_count += 1
        #
        #     # Format token as chunk event
        #     chunk_data = {
        #         "text": token,
        #         "token_count": token_count
        #     }
        #     yield format_sse(json.dumps(chunk_data), event="chunk")
        #
        #     # Optionally send periodic metadata
        #     if include_metadata and token_count % 10 == 0:
        #         # Send timing info every 10 tokens
        #         pass

        # TODO: Send completion event
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000

        # done_data = {
        #     "status": "complete",
        #     "token_count": token_count,
        #     "latency_ms": latency_ms,
        #     "request_id": request_id
        # }
        # yield format_sse(json.dumps(done_data), event="done")

    except Exception as e:
        # TODO: Send error event
        logger.error(f"Error in stream {request_id}: {e}")
        # error_data = {
        #     "error": str(e),
        #     "request_id": request_id
        # }
        # yield format_sse(json.dumps(error_data), event="error")

    pass


async def stream_rag_response(
    rag_pipeline,
    query: str,
    request_id: str,
    **generation_params
) -> AsyncIterator[str]:
    """
    Stream RAG-augmented generation.

    This performs retrieval first, then streams generation.

    TODO: Implement RAG streaming
    - Send "retrieving" event
    - Perform vector search
    - Send "sources" event with retrieved docs
    - Stream generation with context
    - Send "done" event with metadata

    Args:
        rag_pipeline: RAG pipeline instance
        query: User query
        request_id: Request identifier
        **generation_params: Generation parameters

    Yields:
        SSE-formatted strings

    Event Flow:
    1. "start" - Request started
    2. "retrieving" - Performing vector search
    3. "sources" - Retrieved documents
    4. "generating" - Starting generation
    5. "chunk" - Text chunks (multiple)
    6. "done" - Complete with metadata
    """
    try:
        # TODO: Send start event
        # yield format_sse(
        #     json.dumps({"status": "started", "request_id": request_id}),
        #     event="start"
        # )

        # TODO: Send retrieving event
        # yield format_sse(
        #     json.dumps({"status": "retrieving"}),
        #     event="retrieving"
        # )

        # TODO: Perform retrieval
        # sources = await rag_pipeline.retrieve(query)

        # TODO: Send sources event
        # sources_data = {
        #     "sources": [
        #         {
        #             "content": doc.content[:200],  # Truncate
        #             "score": doc.score,
        #             "metadata": doc.metadata
        #         }
        #         for doc in sources
        #     ],
        #     "num_sources": len(sources)
        # }
        # yield format_sse(json.dumps(sources_data), event="sources")

        # TODO: Build prompt with context
        # prompt = rag_pipeline.build_prompt(query, sources)

        # TODO: Send generating event
        # yield format_sse(
        #     json.dumps({"status": "generating"}),
        #     event="generating"
        # )

        # TODO: Stream generation
        # llm_generator = rag_pipeline.generate_stream(prompt, **generation_params)
        # async for chunk in llm_generator:
        #     yield format_sse(
        #         json.dumps({"text": chunk}),
        #         event="chunk"
        #     )

        # TODO: Send done event
        # yield format_sse(
        #     json.dumps({"status": "complete", "request_id": request_id}),
        #     event="done"
        # )

    except Exception as e:
        logger.error(f"RAG stream error {request_id}: {e}")
        # yield format_sse(
        #     json.dumps({"error": str(e)}),
        #     event="error"
        # )

    pass


async def stream_with_heartbeat(
    generator: AsyncIterator[str],
    heartbeat_interval: int = 30
) -> AsyncIterator[str]:
    """
    Wrap generator with heartbeat events.

    Sends periodic heartbeat events to keep connection alive.
    Useful for preventing timeouts during slow generation.

    TODO: Implement heartbeat wrapper
    - Use asyncio.wait_for with timeout
    - Send heartbeat if no data for N seconds
    - Pass through actual data immediately
    - Handle generator completion

    Args:
        generator: Underlying event generator
        heartbeat_interval: Seconds between heartbeats

    Yields:
        SSE-formatted strings (including heartbeats)
    """
    # TODO: Implement heartbeat logic
    # while True:
    #     try:
    #         # Wait for next item with timeout
    #         item = await asyncio.wait_for(
    #             generator.__anext__(),
    #             timeout=heartbeat_interval
    #         )
    #         yield item
    #     except asyncio.TimeoutError:
    #         # Send heartbeat
    #         yield format_sse(
    #             json.dumps({"type": "heartbeat"}),
    #             event="heartbeat"
    #         )
    #     except StopAsyncIteration:
    #         break
    pass


# ============================================================================
# BATCH STREAMING
# ============================================================================

async def stream_batch_responses(
    requests: list,
    llm_server,
    max_concurrent: int = 3
) -> AsyncIterator[str]:
    """
    Stream multiple requests concurrently.

    TODO: Implement batch streaming
    - Process multiple requests in parallel
    - Multiplex results into single stream
    - Include request_id with each event
    - Handle per-request errors
    - Send completion summary

    Args:
        requests: List of generation requests
        llm_server: LLM server instance
        max_concurrent: Max concurrent requests

    Yields:
        SSE events from all requests (with request_id)

    Event Format:
    {
        "request_id": "req_123",
        "request_index": 0,
        "text": "generated text"
    }
    """
    # TODO: Implement concurrent batch streaming
    # Use asyncio.gather or asyncio.as_completed
    # Track which request each event belongs to
    pass


# ============================================================================
# ERROR HANDLING AND RECOVERY
# ============================================================================

async def safe_stream_wrapper(
    generator: AsyncIterator[str],
    request_id: str,
    on_error: Optional[callable] = None
) -> AsyncIterator[str]:
    """
    Wrap stream with error handling and recovery.

    TODO: Implement safe streaming wrapper
    - Catch exceptions from generator
    - Send error event to client
    - Optionally attempt retry
    - Log errors for debugging
    - Ensure proper cleanup

    Args:
        generator: Underlying generator
        request_id: Request identifier
        on_error: Optional error callback

    Yields:
        SSE-formatted strings
    """
    try:
        async for item in generator:
            yield item
    except asyncio.CancelledError:
        # TODO: Handle cancellation
        logger.info(f"Stream {request_id} cancelled")
        yield format_sse(
            json.dumps({"status": "cancelled"}),
            event="cancelled"
        )
        raise
    except Exception as e:
        # TODO: Handle errors
        logger.error(f"Stream {request_id} error: {e}")
        yield format_sse(
            json.dumps({"error": str(e), "error_type": type(e).__name__}),
            event="error"
        )
        if on_error:
            await on_error(e)


# ============================================================================
# UTILITIES
# ============================================================================

class StreamBuffer:
    """
    Buffer for aggregating tokens before sending.

    This can reduce overhead by sending chunks instead of individual tokens.

    TODO: Implement token buffering
    - Buffer tokens until size threshold or timeout
    - Flush buffer to stream
    - Handle word boundaries (don't split mid-word)
    """

    def __init__(self, buffer_size: int = 5, flush_timeout: float = 0.1):
        """
        Initialize buffer.

        Args:
            buffer_size: Number of tokens to buffer
            flush_timeout: Max seconds to wait before flushing
        """
        # TODO: Implement initialization
        pass

    async def add_token(self, token: str) -> Optional[str]:
        """
        Add token to buffer.

        TODO: Implement token addition
        - Add token to buffer
        - Check if should flush
        - Return buffered content if flushing

        Returns:
            Buffered content if flushing, None otherwise
        """
        pass

    async def flush(self) -> str:
        """
        Flush buffer contents.

        TODO: Implement flush
        - Return all buffered tokens
        - Clear buffer
        """
        pass


def calculate_stream_metrics(
    start_time: datetime,
    token_count: int,
    first_token_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Calculate streaming performance metrics.

    TODO: Implement metrics calculation
    - Time to first token (TTFT)
    - Tokens per second (throughput)
    - Total latency
    - Average inter-token latency

    Args:
        start_time: When request started
        token_count: Total tokens generated
        first_token_time: When first token was generated

    Returns:
        Dict with metrics
    """
    # TODO: Calculate metrics
    metrics = {
        # "ttft_ms": ...,  # Time to first token
        # "tokens_per_second": ...,
        # "total_latency_ms": ...,
        # "avg_inter_token_ms": ...
    }
    pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example FastAPI Endpoint:

from fastapi import FastAPI
from .streaming import stream_llm_response, create_sse_response

app = FastAPI()

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    # Get LLM generator
    llm_generator = llm_server.generate_stream(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    # Wrap with SSE formatting
    sse_generator = stream_llm_response(
        llm_generator,
        request_id=generate_request_id(),
        include_metadata=True
    )

    # Return streaming response
    return create_sse_response(sse_generator)


Example Client (JavaScript):

const eventSource = new EventSource('/generate/stream');

eventSource.addEventListener('chunk', (e) => {
    const data = JSON.parse(e.data);
    console.log('Token:', data.text);
});

eventSource.addEventListener('done', (e) => {
    const data = JSON.parse(e.data);
    console.log('Complete:', data.token_count, 'tokens');
    eventSource.close();
});

eventSource.addEventListener('error', (e) => {
    console.error('Error:', e.data);
    eventSource.close();
});


Example Client (Python):

import requests

response = requests.post(
    'http://localhost:8000/generate/stream',
    json={"prompt": "Hello", "max_tokens": 100},
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            print(data.get('text', ''), end='', flush=True)
"""
