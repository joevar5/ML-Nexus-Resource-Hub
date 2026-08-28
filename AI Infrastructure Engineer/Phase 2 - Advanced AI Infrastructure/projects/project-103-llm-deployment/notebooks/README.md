# Jupyter Notebooks for LLM Experimentation

This directory contains Jupyter notebooks for experimenting with and evaluating the LLM deployment platform components.

## 📓 Available Notebooks

### 1. `01-llm-generation-testing.ipynb`
Test LLM text generation capabilities:
- Basic text completion
- Parameter tuning (temperature, top_p, top_k)
- Token generation optimization
- Streaming vs batch inference
- Performance benchmarking

### 2. `02-rag-experimentation.ipynb`
Experiment with RAG (Retrieval-Augmented Generation):
- Document ingestion and chunking
- Embedding generation
- Vector similarity search
- Retrieval quality evaluation
- RAG vs baseline comparison

### 3. `03-embedding-evaluation.ipynb`
Evaluate and compare embedding models:
- Semantic similarity testing
- Embedding visualization (t-SNE, UMAP)
- Clustering analysis
- Embedding model comparison
- Performance vs quality trade-offs

### 4. `04-prompt-engineering.ipynb`
Prompt engineering experiments:
- Prompt template testing
- Few-shot learning examples
- Chain-of-thought prompting
- System message optimization
- Output formatting techniques

## 🚀 Getting Started

### Prerequisites
```bash
# Install Jupyter and dependencies
pip install jupyter ipykernel ipywidgets

# Install notebook requirements
pip install -r ../requirements.txt

# Register kernel
python -m ipykernel install --user --name=llm-deployment
```

### Running Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Environment Setup
Make sure your `.env` file is configured with:
- `OPENAI_API_KEY` or model endpoints
- `VECTOR_DB_URL` (Pinecone, Weaviate, or Milvus)
- `EMBEDDING_MODEL` path or name
- `LLM_MODEL` path or name

## 📊 Example Workflows

### Testing Basic Generation
```python
from notebooks.utils import test_llm_generation

result = test_llm_generation(
    prompt="Explain machine learning in simple terms",
    temperature=0.7,
    max_tokens=200
)
print(result['text'])
print(f"Latency: {result['latency_ms']}ms")
```

### Testing RAG Pipeline
```python
from notebooks.utils import test_rag_pipeline

result = test_rag_pipeline(
    query="What is the capital of France?",
    top_k=3
)
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## 🔧 Utilities

The `utils.py` file provides helper functions for:
- Loading and testing models
- Benchmarking performance
- Visualizing results
- Comparing outputs

## 📈 Metrics to Track

When experimenting, track these metrics:
- **Latency**: Time to first token, total generation time
- **Quality**: BLEU, ROUGE, human evaluation scores
- **Cost**: Token usage, API costs, GPU utilization
- **Relevance**: For RAG - retrieval precision, answer accuracy

## 💡 Tips

1. **Start small**: Test with small datasets before scaling
2. **Cache results**: Save expensive API calls and embeddings
3. **Version experiments**: Use MLflow to track experiments
4. **Compare systematically**: Keep other variables constant when testing
5. **Document findings**: Add markdown cells with insights

## 📚 Resources

- [LLM Fine-tuning Guide](../docs/OPTIMIZATION.md)
- [RAG Best Practices](../docs/RAG.md)
- [Cost Optimization](../docs/COST.md)
- [Prompt Engineering Guide](../prompts/README.md)

---

**Note:** These notebooks are for experimentation and development. For production use, implement the findings in the main codebase.
