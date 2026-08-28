# Exercise 13: ML Reproducibility Audit

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 02, 06 (DVC), 03

## Objective

Pick a model trained > 30 days ago. Without consulting your colleagues, try to reproduce its training exactly: same data, same code, same dependencies, same hyperparameters, same outputs. Document every gap. Use the gaps to harden your reproducibility process.

## Why this matters

"Can you reproduce that result?" is the question most ML teams fail. Reproducibility audits surface the gaps cheaply, before regulators or new joiners discover them painfully.

## Requirements

For an in-production model:
1. **Inventory** what should be reproducible.
2. **Attempt** the reproduction.
3. **Score** the gap (0=no, 1=close, 2=exact).
4. **Document** every gap with: what's missing, how to close it, effort.
5. **Implement** fixes for the top 3 gaps.

## Step-by-step

### Step 1 — Pick a target (15 min)
A model currently in Production, ideally trained > 30 days ago by someone other than yourself.

### Step 2 — Inventory (15 min)
You need to reproduce:
- [ ] Code (specific commit)
- [ ] Dependencies (pinned versions)
- [ ] Training data (exact rows)
- [ ] Hyperparameters
- [ ] Random seeds
- [ ] Hardware (sometimes affects float ops)
- [ ] Training environment (Docker image, CUDA, drivers)
- [ ] Output: model artifact bit-identical, or metrics within ε

### Step 3 — Attempt to reproduce (60 min)
Without asking anyone:
1. Find the MLflow run. Note its tags + params.
2. Find the git SHA from MLflow tags (or fail loudly).
3. Check out that SHA.
4. Find the Dockerfile / requirements at that SHA.
5. Build the image.
6. Find the training data. (DVC? S3 with a known prefix? A query?)
7. Run training.
8. Compare outputs (model file hash, metrics).

Score each step 0/1/2.

### Step 4 — Document gaps (30 min)
For each step that wasn't 2:
- What was missing?
- How would you have known where to look?
- What's the fix?
- How much effort?

Sample:
```markdown
## Gap 1: Data version
- Score: 0
- What's missing: MLflow run tags don't include the DVC commit hash. I found the parquet file in S3 but it had been overwritten since training.
- Fix: Add a tag `data_version: <dvc-sha>` to every training run; never overwrite data files (use versioned object storage).
- Effort: 1 day (training script change + S3 bucket versioning enable).

## Gap 2: Dependencies
- Score: 1
- What's missing: `requirements.txt` had `scikit-learn>=1.4`; at training time was 1.4.0, now 1.5.1.
- Fix: Use `requirements.lock` produced by `pip-compile`; commit and use in CI.
- Effort: 4 hours.
```

### Step 5 — Implement top 3 fixes (30 min)
Pick 3 highest-impact gaps. Implement at least the data-versioning + dependency-pinning gaps.

### Step 6 — Re-audit (15 min)
After fixes, choose a more recent model. Reproduce. Score. Compare to your first attempt's scores.

## Deliverables

1. `AUDIT.md` for the first reproduction attempt.
2. Top 3 fixes implemented (commits in your repo).
3. `AUDIT.md` for the re-audit after fixes.
4. `REPRODUCIBILITY_PLAYBOOK.md`: the team's standard for what a "reproducible" model means.

## Validation

- [ ] First audit identifies at least 3 gaps.
- [ ] At least 3 fixes implemented and merged.
- [ ] Re-audit on a newer model shows improvement.

## Stretch goals

- Add a **`reproducibility check`** to your CI: rebuild + re-train on demand; fail if it produces a different model hash.
- Implement a **bit-identical** reproduction test for at least one model (requires deterministic everything; harder than it sounds).
- Add **provenance attestation** for every model artifact (SLSA-style — see mod-03 ex-10).

## Common pitfalls

- **Floating-point non-determinism on GPU** — Same code, same data, different GPU model → slightly different weights. Need PyTorch's deterministic mode for bit-identity.
- **Hidden environment** — A package installed via system pip not in requirements.txt. Always start from a clean container.
- **Mutable training data** — S3 bucket without versioning + overwriting partition: prior data gone. Version your training data.
- **"It works on my machine"** — The audit isn't done until a colleague can reproduce from your fixes.

## Solutions

There is no reference solution — every team's reproducibility gaps are unique.
