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

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the components, data flow, scoring, training, serving and safety design.

## Plan

Work one phase at a time, with a check at the end of each. Tick a box only when its "done when" is true.

| Phase | What you do | Done when | Where |
| :--- | :--- | :--- | :--- |
| 1. Data and scorer | Load the dataset, split it, write `run_sql` to run queries on SQLite | You can run a query and get rows or an error | Mac |
| 2. Baseline | Make the untrained model write SQL for 200 held-out questions and score it | You have a baseline accuracy number | Colab (GPU) |
| 3. Train | LoRA fine-tune the model, log runs to W&B | Loss falls and the adapter is saved | Colab |
| 4. Evaluate and fix | Score the tuned model, group wrong answers by type, fix the biggest group | Accuracy beats baseline and you can say why it failed before | Colab |
| 5. Guardrails | Allow only a single `SELECT`, use a read-only database, add a timeout | `DROP` and `DELETE` are rejected | Mac |
| 6. Publish and serve | Upload the adapter to Hugging Face, serve it with vLLM, write the README results | A `curl` returns valid SQL and your links are in the README | Colab |

## Progress

- [x] **Phase 1:** data and scorer
  - [x] 1.1 Set up the environment and explore the dataset
  - [x] 1.2 Dedupe and split into train and test
  - [x] 1.3 Write `run_sql` and test it
- [ ] **Phase 2:** baseline
  - [x] 2.1 `src/prompt.py`: one prompt format and an SQL extractor
  - [x] 2.2 `src/eval.py`: score a model on the 200 test questions
  - [ ] 2.3 Pick a small Qwen coder model and run the baseline on Colab
- [ ] **Phase 3:** train
- [ ] **Phase 4:** evaluate and fix
- [ ] **Phase 5:** guardrails
- [ ] **Phase 6:** publish and serve

## Phase 1 steps

### 1.1 Set up the environment and explore the dataset

In a terminal inside this folder:

```
python3 -m venv .venv
source .venv/bin/activate
pip install datasets
```

Create `src/explore.py`. It should load the dataset `b-mc2/sql-create-context` (train split) and print:

- the number of rows
- the column names
- the first 3 rows

Run `python src/explore.py` and check the output.

*Hints:* the loader is `load_dataset(name, split="train")` from `datasets`. A loaded dataset supports `len(ds)`, `ds.column_names` and `ds[0]`.

*Done when:* you can see 3 rows and say what each column holds.

*Think about:* no column holds real table data, only a schema. What problem does that cause when we check whether a query is right?

### 1.2 Dedupe and split into train and test

Create `src/data.py` with a function `load_split(n_test=200, seed=42)` that:

1. loads the dataset
2. removes rows with a duplicate `question`
3. splits into train and test, with 200 test rows and a fixed seed
4. returns both

Add a `__main__` block that prints both sizes and checks that no question appears in both splits.

*Hints:* a dataset has `.select(indices)` and `.train_test_split(test_size=..., seed=...)`. A Python `set` is a quick way to spot duplicates.

*Done when:* it prints the train and test sizes, and the overlap check passes.

*Think about:* why must the same question not appear in both train and test?

## Notes

Keep a log of numbers, errors, and fixes in [`notes.md`](./notes.md). It becomes the write-up at the end.

## Result

Execution accuracy: baseline vs fine-tuned on held-out questions. *(fill in when done)*

## Links

| What | Link |
| :--- | :--- |
| Colab notebook | *(add)* |
| Hugging Face model or dataset | *(add)* |
| W&B project | *(add)* |

Put Colab notebooks in `notebooks/` and keep real code in `src/`, with tests in `tests/`, as the earlier projects do.
