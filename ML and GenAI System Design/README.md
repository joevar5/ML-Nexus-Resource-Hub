# ML & GenAI System Design Repository

Welcome! If you've ever frozen in an interview trying to whiteboard "design YouTube's recommendation system" or "design a RAG pipeline for enterprise search," you're in the right place.

---

## What is ML/GenAI System Design?

**Machine Learning & GenAI System Design** is the discipline of translating a fuzzy product requirement ("recommend videos," "answer questions over our docs") into a concrete, production-grade architecture — covering problem framing, data pipelines, model choice, training, evaluation, and low-latency serving at scale.

Simply put: **It bridges the gap between "we should use ML for this" and a system that actually ships, scales, and gets evaluated correctly in a production interview or a real product.**

---

## The Story Behind This Repository

Let's be honest: most ML/GenAI system design prep material is either a **shallow blog post** that skips the hard tradeoffs, or **locked behind expensive interview-prep paywalls**. I wanted a better way. I built this repository by collaborating with AI assistants — burning my own API tokens to synthesize, structure, and document deep, end-to-end system designs. It serves as both my personal interview-prep study manual and a practical reference for anyone learning how real recommendation, search, and LLM-powered systems are built.

## The Two Learning Tracks

This repository is organized into two complementary tracks — one for classic ML systems, one for LLM/GenAI-powered systems:

*   **Track 1: MLSD — Classic ML System Design:** Recommendation, search & retrieval, ranking, trust & safety, computer vision, and forecasting systems, all built on the same 7-step design framework.

    <div style="text-align: center; font-size: 26px; margin: 18px 0; color: var(--color-primary); text-shadow: 0 0 8px var(--color-primary), 0 0 15px var(--color-primary); line-height: 1; font-weight: bold;">↓</div>

*   **Track 2: GenAI SD — LLM-Powered System Design:** Systems built around large language models — retrieval-augmented generation, personalized text generation, and other LLM-native product architectures.

---

## Track 1: MLSD — Machine Learning System Design

A structured collection of real-world ML System Design questions, organized by product category, each following the same rigorous end-to-end blueprint.

**[Start Learning: MLSD Overview](MLSD/README.md)**

**Framework:** Learn the **7-Step End-to-End Blueprint** — from requirement gathering to real-time monitoring — in [ML System Design Framework](MLSD/0.%20Foundations/ML%20System%20Design%20Framework.md).

**Explore by Category:**

| # | Category | System Designs |
|---|----------|-----------------|
| 1 | Recommendation & Personalization | [Video Recommendation System](MLSD/1.%20Recommendation%20&%20Personalization/Video%20Recommendation%20System.md) · [People You May Know](MLSD/1.%20Recommendation%20&%20Personalization/People%20You%20May%20Know.md) |
| 2 | Search & Retrieval | *in progress* |
| 3 | Ranking & Prediction | *in progress* |
| 4 | Trust & Safety | *in progress* |
| 5 | Computer Vision | *in progress* |
| 6 | Forecasting & Time Series | *in progress* |

---

## Track 2: GenAI SD — GenAI System Design

Systems built around large language models — from grounding responses in external knowledge to generating personalized, context-aware text.

**[Start Learning: GenAI SD Overview](GENAI%20SD/README.md)**

| # | System Design |
|---|----------------|
| 1 | [Retrieval-Augmented Generation (RAG) System Design](GENAI%20SD/Retrieval-Augmented%20Generation.md) |
| 2 | [Gmail Smart Compose System Design](GENAI%20SD/Gmail%20Smart%20Compose.md) |
| 3 | *more in progress* |
