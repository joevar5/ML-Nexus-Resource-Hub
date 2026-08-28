# Prompt Templates — Project 103

This directory contains reusable prompt templates for the LLM deployment project. Templates are stored in `templates.yaml` and consumed by the serving code under `../src/`.

## Why a separate prompts directory

Prompts are infrastructure, not inline strings. Treating them as versioned artifacts gives you:

- **Reproducibility** — a model + prompt combination is the actual unit of behavior. Pinning only the model is half the story.
- **Auditability** — when output regresses, you need to know whether the model, the prompt, or both changed.
- **Iteration speed** — non-engineers (PM, applied scientists) can edit YAML in a PR without touching Python.
- **Multi-tenant isolation** — different tenants get different system prompts without a code path per tenant.

## File layout

```text
prompts/
├── README.md           ← this file
└── templates.yaml      ← all prompt templates, keyed by id
```

## Template format

`templates.yaml` uses the following shape:

```yaml
templates:
  - id: chat-system-default
    description: Default system prompt for chat completions.
    version: 1.2.0
    audience: end-user
    model_compat: [llama-3.1-8b-instruct, llama-3.1-70b-instruct, mistral-7b-instruct]
    body: |
      You are a helpful, accurate assistant. If you don't know something,
      say so plainly. Decline requests that are harmful or illegal. Prefer
      concise answers; expand only when the user explicitly asks for depth.

  - id: rag-answer
    description: Answer a user question using only the provided context.
    version: 1.0.0
    audience: end-user
    model_compat: [llama-3.1-8b-instruct, llama-3.1-70b-instruct]
    variables:
      - name: question
        required: true
      - name: context_chunks
        required: true
        type: list[str]
    body: |
      You are answering a user's question using ONLY the context provided.
      If the answer is not in the context, say "I don't have enough
      information to answer that."

      Context:
      {context_chunks_joined}

      Question: {question}

      Answer:
```

## Naming conventions

- `id`: kebab-case, prefixed by use-case (`chat-`, `rag-`, `summarize-`, `extract-`, `classify-`).
- `version`: semver. Bump major on backward-incompatible semantic changes.
- `model_compat`: list of model IDs that have been evaluated with this prompt. Don't ship a template against an untested model.
- `variables`: explicit list. Templating engine (the one in `../src/prompt_loader.py`) validates that all required variables are passed.

## Adding a new template

1. Edit `templates.yaml`.
2. Bump the version (or assign a new id for a net-new template).
3. Add an evaluation run in `../tests/test_prompts.py` covering the template on at least one model.
4. Update `model_compat` based on the eval results.
5. Open a PR — the CI evaluation suite runs against a small canary set.

## Anti-patterns to avoid

- **Inline prompt strings in Python**: makes prompt changes invisible in code review and impossible to A/B test.
- **Embedding business logic in the prompt**: "Always reply in JSON with these 7 keys" works for one-off scripts and breaks at scale. Use structured output (function calling / JSON mode) instead.
- **Unbounded user input concatenation**: if a user message can be pasted directly into a system-prompt slot, you have a prompt-injection vulnerability. Always wrap and label user input clearly.
- **Skipping eval on new template versions**: a prompt change is a model behavior change. Treat it as such.

## See also

- `../docs/OPTIMIZATION.md` — performance + latency considerations for prompts
- `../docs/RAG.md` — retrieval patterns and prompt construction for grounded answers
- `../src/prompt_loader.py` — the template loader + validator the serving code uses
