# GenAI Atlas

**Status:** Partially unlocked — the Agents track is live.

An atlas maps terrain too large to see all at once — you open it to the page you need, not front to back. GenAI Atlas does that for generative AI: seven tracks, each covering one region and pointing at how it borders the others. Find the track that matches what you're building, start there.

Everything here is downstream of one thing — a base LLM predicting the next token. Fine-tuning changes what it knows by default. Prompt engineering changes what you ask it in one turn. RAG hands it facts it wasn't trained on. Agents wrap that whole loop in a system that can act and iterate on its own. Evals tell you whether any of it actually worked.

## Tracks

| # | Track | Covers | Status |
|---|-------|--------|--------|
| 1 | [LLM Foundations](1.%20LLM%20Foundations/README.md) | Architecture, attention, tokenization, context windows | Locked |
| 2 | [Fine-Tuning & Alignment](2.%20Fine-Tuning%20%26%20Alignment/README.md) | LoRA/QLoRA, RLHF, DPO, when to fine-tune vs. prompt | Locked |
| 3 | [Agents](Agents/README.md) | Tool-calling, memory, planning/ReAct, multi-agent orchestration — the [30-Day Agentic AI Interview Prep Path](Agents/README.md) | **Live** |
| 4 | [RAG & Retrieval](4.%20RAG%20%26%20Retrieval/README.md) | Chunking, embeddings, vector DBs, reranking, evaluation | Locked |
| 5 | [Prompt Engineering](5.%20Prompt%20Engineering/README.md) | Few-shot, chain-of-thought, structured output, prompt injection defense | Locked |
| 6 | [Model Landscape](6.%20Model%20Landscape/README.md) | Living comparison of GPT, Claude, Gemini, Llama, DeepSeek — when to use which | Locked |
| 7 | [Evals & Observability](7.%20Evals%20%26%20Observability/README.md) | LLM-as-judge, benchmarks, tracing/logging for GenAI apps | Locked |

Locked tracks unlock over time — check back. If you're building tool-calling loops, RAG-backed assistants, or prepping for agent interviews, start with **Agents**.
