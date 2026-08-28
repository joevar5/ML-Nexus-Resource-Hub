"""
Document Chunking Module

Implements various strategies for splitting documents into chunks for RAG:
- Fixed-size chunking
- Recursive character splitting
- Semantic chunking
- Sentence-aware chunking
- Markdown-aware chunking

Learning Objectives:
1. Understand chunking strategies and trade-offs
2. Handle different document formats
3. Preserve context across chunks
4. Optimize chunk size for retrieval
5. Implement overlap strategies

References:
- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a document chunk.

    Attributes:
        content: Chunk text content
        metadata: Associated metadata
        chunk_index: Index within parent document
        start_char: Starting character position
        end_char: Ending character position
    """
    content: str
    metadata: dict
    chunk_index: int
    start_char: int
    end_char: int


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Input text
            metadata: Optional metadata to attach

        Returns:
            List of chunks
        """
        pass


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size chunks with overlap.

    Simple but effective for uniform documents.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function: callable = len
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Size of each chunk (in characters or tokens)
            chunk_overlap: Overlap between chunks
            length_function: Function to measure length (len for chars, token counter for tokens)

        TODO:
        1. Validate chunk_size > chunk_overlap
        2. Store parameters
        3. Set up length function
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Split text into fixed-size chunks.

        Args:
            text: Input text
            metadata: Optional metadata

        Returns:
            List of chunks

        TODO: Implement fixed-size chunking:
        1. Calculate step size (chunk_size - chunk_overlap)
        2. Iterate through text with sliding window
        3. Create chunks maintaining overlap
        4. Track start/end positions
        5. Attach metadata to each chunk
        6. Handle edge cases (last chunk, empty text)

        Example:
        - Text: "ABCDEFGHIJ" (10 chars)
        - chunk_size: 5, overlap: 2
        - Chunks: "ABCDE", "DEFGH", "GHIJ"
        """
        if not text:
            return []

        chunks = []
        # TODO: Implement chunking logic
        # step = self.chunk_size - self.chunk_overlap
        # for i in range(0, len(text), step):
        #     chunk_text = text[i:i + self.chunk_size]
        #     chunk = Chunk(
        #         content=chunk_text,
        #         metadata=metadata or {},
        #         chunk_index=len(chunks),
        #         start_char=i,
        #         end_char=i + len(chunk_text)
        #     )
        #     chunks.append(chunk)

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursively split text using multiple separators.

    Tries to keep paragraphs, sentences, and words together.
    This is the most commonly used chunker in practice.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            separators: List of separators to try (in order)

        TODO:
        1. Store parameters
        2. Set default separators if not provided:
           ["\n\n", "\n", ". ", " ", ""]
           (paragraph, line, sentence, word, character)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Recursively split text.

        Args:
            text: Input text
            metadata: Optional metadata

        Returns:
            List of chunks

        TODO: Implement recursive chunking:
        1. Try first separator
        2. If chunks are too large, recursively split with next separator
        3. Merge small chunks if possible
        4. Add overlap between chunks
        5. Preserve natural boundaries (paragraphs, sentences)

        Algorithm:
        - Split by first separator (e.g., "\n\n")
        - For each piece:
          - If too large: recursively split with next separator
          - If too small: merge with adjacent pieces
        - Add overlap at chunk boundaries
        """
        if not text:
            return []

        # TODO: Implement recursive splitting
        chunks = self._recursive_split(text, self.separators)

        # TODO: Convert to Chunk objects with metadata
        return self._create_chunks(chunks, metadata)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text.

        Args:
            text: Text to split
            separators: Remaining separators to try

        Returns:
            List of text chunks

        TODO: Implement recursive splitting logic
        """
        # TODO: Implement
        pass

    def _create_chunks(self, texts: List[str], metadata: Optional[dict]) -> List[Chunk]:
        """
        Convert text chunks to Chunk objects.

        Args:
            texts: List of text chunks
            metadata: Metadata to attach

        Returns:
            List of Chunk objects

        TODO: Create Chunk objects with proper indexing
        """
        # TODO: Implement
        return []


class SemanticChunker(BaseChunker):
    """
    Split text based on semantic similarity.

    Uses embeddings to determine natural breakpoints.
    More sophisticated but slower than character-based chunking.
    """

    def __init__(
        self,
        embedding_model,
        breakpoint_threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        """
        Initialize semantic chunker.

        Args:
            embedding_model: Model for computing embeddings
            breakpoint_threshold: Similarity threshold for splitting
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size

        TODO:
        1. Store embedding model
        2. Set thresholds
        3. Initialize sentence splitter
        """
        self.embedding_model = embedding_model
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Split text semantically.

        Args:
            text: Input text
            metadata: Optional metadata

        Returns:
            List of chunks

        TODO: Implement semantic chunking:
        1. Split text into sentences
        2. Compute embedding for each sentence
        3. Calculate cosine similarity between consecutive sentences
        4. Insert breakpoints where similarity drops below threshold
        5. Merge small chunks, split large chunks
        6. Return semantic chunks

        Algorithm:
        - For consecutive sentences i and i+1:
        - If similarity(sent_i, sent_i+1) < threshold:
          - Insert breakpoint
        - Ensures chunks are semantically coherent
        """
        # TODO: Implement semantic chunking
        pass

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences

        TODO: Implement sentence splitting:
        - Use regex or spaCy/NLTK
        - Handle edge cases (abbreviations, decimals)
        """
        # Simple regex-based sentence splitting
        # TODO: Improve with NLP library
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class MarkdownChunker(BaseChunker):
    """
    Markdown-aware chunking that respects structure.

    Preserves headers, code blocks, lists, etc.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        preserve_headers: bool = True
    ):
        """
        Initialize markdown chunker.

        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            preserve_headers: Include parent headers in each chunk

        TODO:
        1. Store parameters
        2. Set up markdown parser
        3. Define structural elements to preserve
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_headers = preserve_headers

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Split markdown text.

        Args:
            text: Markdown text
            metadata: Optional metadata

        Returns:
            List of chunks

        TODO: Implement markdown chunking:
        1. Parse markdown structure (headers, code blocks, lists)
        2. Split at natural boundaries (headers, horizontal rules)
        3. Keep code blocks together
        4. Preserve header context if enabled
        5. Handle nested lists and quotes

        Preservation strategy:
        - If preserve_headers=True, prepend parent headers to chunks
        - Example:
          # Chapter 1
          ## Section 1.1
          Content here...

          Chunk: "# Chapter 1\n## Section 1.1\nContent here..."
        """
        # TODO: Implement markdown-aware chunking
        pass

    def _extract_headers(self, text: str) -> List[tuple]:
        """
        Extract headers and their positions.

        Args:
            text: Markdown text

        Returns:
            List of (level, text, position) tuples

        TODO: Parse markdown headers (# ## ###)
        """
        # TODO: Implement header extraction
        pass


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_optimal_chunk_size(
    documents: List[str],
    embedding_model,
    target_retrieval_size: int = 5
) -> int:
    """
    Estimate optimal chunk size for a document collection.

    Args:
        documents: Sample documents
        embedding_model: Embedding model
        target_retrieval_size: Number of chunks to retrieve

    Returns:
        Recommended chunk size

    TODO: Implement optimization:
    1. Try different chunk sizes
    2. Measure retrieval quality
    3. Consider context window size
    4. Balance between granularity and context
    5. Return optimal size

    Factors:
    - Model context window (e.g., 4K tokens)
    - Number of retrieved chunks
    - Document structure
    - Query types
    """
    # TODO: Implement chunk size optimization
    pass


def count_tokens(text: str, tokenizer) -> int:
    """
    Count tokens in text.

    Args:
        text: Input text
        tokenizer: Tokenizer to use

    Returns:
        Number of tokens

    TODO: Implement token counting using tokenizer
    """
    # TODO: Implement
    pass
