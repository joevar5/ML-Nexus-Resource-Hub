# Notes

One entry per session: what you did, the numbers, any error and how you fixed it.

## Session log

| Date | Step | What happened | Numbers or errors | Fix or lesson |
| :--- | :--- | :--- | :--- | :--- |
| | | | | |
| 2026-10-11 | 1.1 | Loaded `b-mc2/sql-create-context` and printed rows | 78,577 rows; columns answer, question, context; schema only, no table data | Run from the project `.venv`; the repo `venv` lacked `datasets` |
| 2026-10-11 | 1.2 | Wrote `src/data.py`: dedupe by question, 200-row test split, seed 42 | 78,577 to 78,311 rows (266 duplicate questions); train 78,111, test 200 | Overlap and reproducibility checks pass |
| 2026-10-11 | 1.2 (rework) | Production pass: pinned dataset revision, manifest with sha256, logging, 7 tests | Revision 9d80a6a1; `make test` and `make data` | Pinned data so splits are reproducible |
| 2026-10-11 | 1.3 | Wrote `src/run_sql.py` (sandboxed SQLite runner, row seeding, timeout, `score`) and `src/check_references.py` | 20 tests pass. With 30 random rows 112/200 references returned nothing; seeding from query literals and 100 rows cut that to 6/200. 4/200 references and about 2.1% of the dataset were invalid SQL | Filter unscorable rows before splitting: 78,577 to 76,910 scorable, then train 76,446 / test 200 |
| 2026-10-11 | Setup | Kaggle notebook `text-to-sql-model-nb` with 2x Tesla T4 (15,360 MiB each), internet on; `HF_TOKEN` and `WANDB_API_KEY` stored as Kaggle secrets and read back as found | 30 GPU hours per week available | Notebook stays private until we confirm no secret appears in any output |
| 2026-10-11 | Setup | `pip install -r requirements.txt` on Kaggle pinned `huggingface_hub==2.2.0`, which conflicts with Kaggle preinstalled transformers, gradio and diffusers (need hub below 2.0) | Tests: 20 passed on Kaggle | Removed the hub pin; `datasets==5.1.0` accepts hub below 3.0. Pin only direct dependencies |
| 2026-10-11 | 2 (data check) | Rebuilt data on Kaggle and compared manifests | Same row counts and same sorted-question hashes, but different sha256: field order inside each JSON row differed (Mac answer,question,context; Kaggle question,context,answer) | `save` now writes fields in a fixed order; Mac and Kaggle both give train f8e91b15…7b68, test 2ac0aade…75aa. Added a test for it. 21 tests pass |
| 2026-10-11 | 2.1, 2.2 | Wrote `src/prompt.py`, `src/eval.py`, `src/generate.py` (HF generator, untested until Kaggle) and 7 tests | 28 tests pass. Sanity checks on the 200 test questions: giving the reference answers scores 1.00; always answering `SELECT 1` scores 0.005; `COUNT(*)` of the first table scores 0.000; uninformative share 3% | Scorer is not fooled by trivial answers. Work now goes on branch `project-102-text-to-sql` so the Pages deploy does not run |
| 2026-10-11 | 2.3 | `src.eval` on Kaggle failed: ImportError, transformers needs huggingface-hub below 2.0 but 2.2.0 was installed | pip upgraded hub to 2.2.0 because the preinstalled one was older than the 1.31 that datasets needs | requirements.txt now says `huggingface_hub>=1.31,<2.0` (a range both libraries accept). Mac: hub 1.33.0, 28 tests pass, same data checksums |
| 2026-10-11 | 2.3 | Smoke test on Kaggle: 20 questions, Qwen2.5-Coder-0.5B-Instruct | accuracy 0.25 (5/20), invalid SQL 0.10, wrong result 13, syntax error 1, unknown column 1, uninformative 0.15 | Works end to end; 20 questions is too few to trust. Moved the steps into `notebooks/text_to_sql_kaggle.ipynb` |
