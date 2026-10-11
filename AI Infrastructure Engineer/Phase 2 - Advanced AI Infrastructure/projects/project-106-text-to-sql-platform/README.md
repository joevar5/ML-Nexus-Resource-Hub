# Project 106: Capstone, Text-to-SQL Platform

Wire the fine-tuned model, schema retrieval, gateway, and monitoring into one deployed product.

**What's new vs Phase 1:** Fine-tuned model + RAG schema retrieval + gateway + monitoring
**Prep:** Projects 101 to 105

## Lessons that prepare you

- [Lessons for Projects 101 to 105 (above)](../project-101-llm-serving-platform/README.md)
- [Lesson 08: LLM Infrastructure for Agents](../../lessons/mod-10-llm-infrastructure/08-llm-infrastructure-for-agents.md)
- [Exercise 12: LLM Evaluation Harness](../../lessons/mod-10-llm-infrastructure/exercises/exercise-12-llm-evaluation-harness/README.md)
- [Exercise 11: LLM Guardrails](../../lessons/mod-10-llm-infrastructure/exercises/exercise-11-llm-guardrails/README.md)
- [Lesson 10: Text-to-SQL and Evaluation by Execution](../../lessons/mod-10-llm-infrastructure/10-text-to-sql-and-execution-eval.md)

## Steps

- [ ] Retrieve relevant tables for each question (103)
- [ ] Generate SQL with the fine-tuned model (102)
- [ ] Serve through the gateway with guardrails (101, 104)
- [ ] Add monitoring and a CI gate that fails if accuracy drops
- [ ] Write a decision record: why fine-tune instead of a hosted API

## Result

End-to-end demo plus accuracy, latency, and cost per 1,000 queries. *(fill in when done)*

## Links

| What | Link |
| :--- | :--- |
| Kaggle notebook | *(add)* |
| Hugging Face model or dataset | *(add)* |
| W&B project | *(add)* |

Put Kaggle notebooks in `notebooks/` and keep real code in `src/`, with tests in `tests/`, as the earlier projects do.
