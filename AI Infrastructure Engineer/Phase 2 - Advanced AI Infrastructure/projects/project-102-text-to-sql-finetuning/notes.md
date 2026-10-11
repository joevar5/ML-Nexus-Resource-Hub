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
