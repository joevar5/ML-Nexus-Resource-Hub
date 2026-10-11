# Project 102: Text-to-SQL Fine-Tuning

Fine-tune a small code model with LoRA/QLoRA to turn business questions into safe SQL, and prove it beats the untrained model.

**What's new vs Phase 1:** LoRA/QLoRA, eval harness, failure analysis, guardrails
**Prep:** Lesson 05 of mod-10


## Lessons that prepare you

- [Lesson 05: LLM Fine-Tuning Infrastructure](../../lessons/mod-10-llm-infrastructure/05-llm-fine-tuning-infrastructure.md)
- [MLflow Experiment Tracking](../../lessons/mod-06-mlops/01-mlflow-experiment-tracking.md)
- [ML Model Observability](../../lessons/mod-08-monitoring-observability/03-ml-model-observability.md)
- [Lesson 10: Text-to-SQL and Evaluation by Execution](../../lessons/mod-10-llm-infrastructure/10-text-to-sql-and-execution-eval.md)
- [Lesson 12: Preference Tuning With DPO (optional)](../../lessons/mod-10-llm-infrastructure/12-preference-tuning-dpo.md)

## Steps

- [ ] Load a text-to-SQL dataset, split train/test, write `run_sql` (SQLite execution scorer)
- [ ] Score the untrained model (baseline)
- [ ] LoRA fine-tune on Kaggle, log runs to W&B
- [ ] Group wrong answers by type and fix the biggest group
- [ ] Add guardrails (block writes, read-only DB, timeouts)
- [ ] Push the adapter to Hugging Face and serve it

## Result

Execution accuracy: baseline vs fine-tuned on held-out questions. *(fill in when done)*

## Links

| What | Link |
| :--- | :--- |
| Kaggle notebook | *(add)* |
| Hugging Face model or dataset | *(add)* |
| W&B project | *(add)* |

Put Kaggle notebooks in `notebooks/` and keep real code in `src/`, with tests in `tests/`, as the earlier projects do.
