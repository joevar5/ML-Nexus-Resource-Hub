# Day 14: Chunking and Retrieval Quality

## Concept

Chunking — how you split documents before embedding — has an outsized effect on retrieval quality, more than most people expect given how simple it sounds. Two failure directions:

- **Chunks too large.** A 2000-token chunk covering 5 different subtopics gets one embedding that's an average of all 5 — a query about subtopic 3 may not score highly against it even though the answer is in there, because the embedding is diluted by the other 4 subtopics' content.
- **Chunks too small.** A 50-token chunk may retrieve well (it's about exactly one thing) but not contain enough surrounding context for the model to answer fully — e.g. a chunk with a number but not the sentence explaining what the number means.

Common strategies, in order of sophistication:

- **Fixed-size splitting** (e.g. 512 tokens, no regard for content boundaries). Simple, but can split a sentence or table row in half.
- **Recursive/structure-aware splitting.** Split on paragraph or section boundaries first, falling back to sentence boundaries only if a paragraph is still too large. Respects the document's actual structure.
- **Overlap.** Adjacent chunks share a small window of text (e.g. last 50 tokens of chunk N = first 50 tokens of chunk N+1) so a fact that would otherwise be split at a chunk boundary appears intact in at least one chunk.
- **Semantic chunking.** Split based on embedding similarity between consecutive sentences — a large similarity drop marks a topic boundary. More expensive to compute, better topic coherence.

Retrieval quality also depends on what you search *with*, not just what you search *over*. Retrieving with the raw user question sometimes underperforms retrieving with a rewritten or expanded version of it (e.g. resolving "it" to what it refers to from conversation history, or generating a hypothetical answer and embedding that instead — HyDE). This matters because embedding similarity is a proxy for relevance, not relevance itself, and short or ambiguous queries embed poorly.

## Coding Problem

Write `chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]` that splits `text` on paragraph boundaries (`\n\n`), and for any paragraph longer than `max_chars`, further splits it on sentence boundaries (naive: split on `". "`). Each returned chunk should be at most `max_chars` long (combine adjacent short paragraphs up to that limit), with `overlap_chars` of text repeated from the end of chunk N at the start of chunk N+1 (except the first chunk). Return the list of chunks.

## Quiz

### Question 1: Chunks Too Large

**Why does a chunk covering 5 different subtopics tend to retrieve poorly for a query about just one of them?**

A) Large chunks are always rejected by vector databases
B) The chunk's single embedding is an average influenced by all 5 subtopics, diluting its similarity to a query about just one of them — even though the answer is present in the text
C) Large chunks cost more to store but retrieve identically to small chunks
D) This isn't a real effect — chunk size doesn't influence embedding quality

**Answer**: B

**Explanation**: A chunk's embedding represents its content as a whole. Mixing multiple subtopics into one chunk means no single subtopic dominates the embedding, which can push it below the similarity threshold for a query specifically about one of those subtopics, even though the right text is technically present.

### Question 2: Purpose of Overlap

**What problem does adding overlap between adjacent chunks solve?**

A) It makes the vector database run faster
B) It prevents a fact or sentence that spans a chunk boundary from being split across two chunks with neither containing it whole
C) It removes the need for embeddings entirely
D) It has no effect on retrieval, only storage size

**Answer**: B

**Explanation**: Without overlap, a fact that happens to fall right at a chunk boundary can end up split between two chunks, with neither containing the complete, coherent statement. Overlap ensures boundary-spanning content appears intact in at least one chunk.

### Question 3: Query Rewriting

**Why might retrieving with a rewritten/expanded version of the user's question outperform retrieving with the raw question?**

A) Rewriting is required by all vector database APIs
B) Embedding similarity is a proxy for relevance, not relevance itself — short, ambiguous, or pronoun-laden queries ("what about it?") embed poorly and can be improved by resolving context or expanding the query first
C) Raw questions are always too long to embed
D) Rewriting has no measurable effect on retrieval

**Answer**: B

**Explanation**: A vector search finds text with embeddings numerically close to the query's embedding — but a short or ambiguous query embeds ambiguously too. Techniques like resolving referents from conversation history or generating a hypothetical answer to embed (HyDE) give the search a richer, more specific target to match against.

## Interview Practice

**1.** Retrieval quality on your RAG system dropped noticeably after ingesting a new batch of documents. Walk through how you'd determine whether the cause is a chunking problem, an embedding problem, or something else.

**2.** A teammate proposes using the largest possible chunk size "so we never lose context." Argue against this using the dilution mechanism from this lesson, and propose what you'd measure to find the right chunk size instead.

**3.** Design a chunking strategy for a codebase (not prose documents) — where should chunk boundaries fall, and why does a fixed-token-count splitter perform especially poorly on code compared to natural language?
