# Resources: LLM Infrastructure

## Official Documentation

### Serving
- [vLLM Documentation](https://docs.vllm.ai/) · [GitHub](https://github.com/vllm-project/vllm) · [Performance Tuning](https://docs.vllm.ai/en/latest/serving/performance.html)
- [Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA's optimized inference engine
- [Ray Serve](https://docs.ray.io/en/latest/serve/) — distributed multi-model serving

### RAG & Application Frameworks
- [LangChain Documentation](https://python.langchain.com/docs/) — [RAG tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) · [PEFT (LoRA)](https://huggingface.co/docs/peft/) · [TRL](https://huggingface.co/docs/trl/)

### Vector Databases
- [Qdrant Docs](https://qdrant.tech/documentation/) — [Optimization guide](https://qdrant.tech/documentation/guides/optimization/)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [Pinecone Docs](https://docs.pinecone.io/) (managed)
- [Chroma Docs](https://docs.trychroma.com/) (lightweight/embeddable)
- [Milvus Docs](https://milvus.io/docs) (enterprise scale)

### Model Optimization
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 8-bit quantization
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) · [AutoAWQ](https://github.com/mit-han-lab/llm-awq) — 4-bit quantization
- [llama.cpp / GGUF](https://github.com/ggerganov/llama.cpp) — CPU/edge inference

---

## Books

1. **"Build a Large Language Model (From Scratch)"** — Sebastian Raschka (Manning, 2024) — LLM internals from the ground up
2. **"Natural Language Processing with Transformers"** — Tunstall, von Werra, Wolf (O'Reilly, 2022) — Hugging Face ecosystem
3. **"Designing Machine Learning Systems"** — Chip Huyen (O'Reilly, 2022) — deployment & serving chapters especially relevant
4. **"Generative AI with LangChain"** — Ben Auffarth (Packt, 2024)
5. **"Programming Massively Parallel Processors"** — Kirk & Hwu (Morgan Kaufmann, 4th ed.) — CUDA fundamentals for the curious

---

## Online Courses

- **[Hugging Face LLM Course](https://huggingface.co/learn/nlp-course/)** (Free) — transformers to fine-tuning to deployment, 20+ hours
- **[LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)** (DeepLearning.AI, free, ~2 hrs)
- **[Vector Databases: Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)** (DeepLearning.AI, free, ~2 hrs)
- **[Introduction to Large Language Models](https://www.cloudskillsboost.google/paths/118)** (Google Cloud Skills Boost, free)
- **MLOps for Scaling LLMs and Generative AI** (Coursera, Duke University, 4 weeks) — deployment, scaling, monitoring

---

## Interactive Learning

- **[Hugging Face Spaces](https://huggingface.co/spaces)** — try models before deploying
- **[Google Colab](https://colab.research.google.com/)** / **[Kaggle Notebooks](https://www.kaggle.com/code)** — free GPU experimentation
- **[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)** — compare model quality
- **[LMSys Chatbot Arena](https://chat.lmsys.org/)** — head-to-head model comparisons
- **[Artificial Analysis](https://artificialanalysis.ai/)** — compare LLM API pricing and throughput

---

## Tools & Libraries

```bash
# Serving & core
pip install vllm transformers torch

# RAG & applications
pip install langchain llama-index

# Vector databases
pip install qdrant-client weaviate-client chromadb pinecone-client

# Fine-tuning
pip install peft bitsandbytes trl datasets

# Quantization
pip install auto-gptq autoawq optimum

# Evaluation & monitoring
pip install ragas deepeval prometheus-client
```

| Category | Options |
|---|---|
| Fine-tuning frameworks | [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) |
| RAG evaluation | [RAGAS](https://github.com/explodinggradients/ragas), [DeepEval](https://github.com/confident-ai/deepeval) |
| LLM observability | [LangSmith](https://docs.smith.langchain.com/), [Phoenix](https://github.com/Arize-ai/phoenix) |
| GPU monitoring | [NVIDIA DCGM](https://github.com/NVIDIA/dcgm-exporter), [nvitop](https://github.com/XuehaiPan/nvitop) |

---

## GitHub Repositories

- **[vLLM Examples](https://github.com/vllm-project/vllm/tree/main/examples)** — official patterns
- **[RAG Techniques](https://github.com/NirDiamant/RAG_Techniques)** — comprehensive RAG implementation patterns
- **[Awesome LLMOps](https://github.com/tensorchord/Awesome-LLMOps)** — curated infra/tooling list
- **[FastChat](https://github.com/lm-sys/FastChat)** — production serving platform behind Chatbot Arena
- **[QLoRA](https://github.com/artidoro/qlora)** / **[LLaMA Recipes](https://github.com/facebookresearch/llama-recipes)** — fine-tuning references

---

## Research Papers

| Paper | Year | Why it matters |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Foundation of every modern LLM |
| [Efficient Memory Management... (PagedAttention)](https://arxiv.org/abs/2309.06180) | 2023 | The vLLM paper |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2021 | Parameter-efficient fine-tuning |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 2023 | Fine-tuning large models on consumer GPUs |
| [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | 2020 | Foundation of RAG |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | Faster, memory-efficient attention |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 2022 | 4-bit post-training quantization |

---

## Blogs & Articles

- **[vLLM Blog](https://blog.vllm.ai/)** — benchmarks, new features
- **[Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** — best visual intro to attention
- **[Eugene Yan — Patterns for Building LLM Systems](https://eugeneyan.com/writing/llm-patterns/)**
- **[Chip Huyen et al. — What We Learned from a Year of Building with LLMs](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms/)**
- **[Anyscale — Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)**

---

## Video

- **[Andrej Karpathy — Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)** (1 hr) — best single overview
- **[Andrej Karpathy — State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A)** (45 min) — training → fine-tuning → deployment
- **[Jeremy Howard — A Hackers' Guide to Language Models](https://www.youtube.com/watch?v=jkrNMKz9pWU)** (1.5 hrs)
- **YouTube channels:** [AI Makerspace](https://www.youtube.com/@AIMakerspace) (RAG/apps), [Sam Witteveen](https://www.youtube.com/@samwitteveenai) (LangChain/RAG), [Weights & Biases](https://www.youtube.com/@WeightsBiases) (training/MLOps)

---

## Communities & Forums

- **Discord:** [Hugging Face](https://hf.co/join/discord), [LangChain](https://discord.gg/langchain), [vLLM](https://discord.gg/vllm), [Qdrant](https://qdrant.to/discord)
- **Reddit:** [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) (very active, running/optimizing LLMs), [r/MachineLearning](https://www.reddit.com/r/MachineLearning/), [r/LLMDevs](https://www.reddit.com/r/LLMDevs/)
- **[MLOps Community](https://mlops.community/)** — Slack + events for production ML/LLM practitioners

---

## Newsletters & Podcasts

- **[The Batch](https://www.deeplearning.ai/the-batch/)** (weekly) · **[TLDR AI](https://tldr.tech/ai)** (daily) — news roundups
- **[Latent Space](https://www.latent.space/podcast)** — LLMs and AI engineering, weekly

---

## Cloud & GPU Providers

| Category | Providers |
|---|---|
| Major clouds | AWS (Bedrock, SageMaker, P4/P5), GCP (Vertex AI, A100/H100), Azure (OpenAI Service, ND-series) |
| GPU clouds (cheaper) | [Lambda Labs](https://lambdalabs.com/service/gpu-cloud), [RunPod](https://www.runpod.io/), [Vast.ai](https://vast.ai/), [CoreWeave](https://www.coreweave.com/), [Together.ai](https://www.together.ai/) |
| Serverless inference | [Modal](https://modal.com/), [Replicate](https://replicate.com/) |

---

## Practice Projects

**Beginner:** deploy Llama 2 7B with vLLM · build a simple RAG system over your own notes with Chroma · run a GGUF model locally with llama.cpp

**Intermediate:** production RAG with reranking + evaluation on K8s · fine-tune Llama 2 7B with LoRA on domain data · build a Prometheus/Grafana LLM monitoring dashboard

**Advanced:** multi-model platform with intelligent routing + semantic caching · domain assistant combining RAG + fine-tuning + agents · full K8s deployment with autoscaling on queue depth and blue-green rollouts

---

## Datasets & Model Registries

- **[Hugging Face Datasets](https://huggingface.co/datasets)** — Alpaca, Dolly, OpenOrca for instruction tuning
- **[Hugging Face Model Hub](https://huggingface.co/models)** · **[Ollama Library](https://ollama.ai/library)** (local deployment)

---

## What's Next

**Next Module:** none — this is the last module in Phase 2. Build **Project 103 (LLM Deployment Platform)** to bring everything in this curriculum together, or go deeper into fine-tuning, serving optimization, or platform design.

**Keep learning:** build projects and share the code, read the papers behind the tools you use, and follow [vLLM](https://github.com/vllm-project/vllm/releases) / [Transformers](https://github.com/huggingface/transformers/releases) release notes — this space moves fast.
