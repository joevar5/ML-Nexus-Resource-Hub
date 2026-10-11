"""Step 1.2: dedupe by question and make a fixed, reproducible train/test split."""
import hashlib
import json
import logging
from pathlib import Path

from datasets import Dataset, load_dataset

from src.run_sql import is_scorable

log = logging.getLogger(__name__)

DATASET = "b-mc2/sql-create-context"
REVISION = "9d80a6a118b838d9defc3798d659a54a2ac2ff37"  # pinned: a changed upstream dataset must not silently change results
FIELDS = ("question", "context", "answer")  # fixed order: column order from the loader differs between machines
OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def dedupe_by_question(ds: Dataset) -> Dataset:
    """Keep the first row for each question, so a question can't land in both splits."""
    seen, keep = set(), []
    for i, question in enumerate(ds["question"]):
        if question not in seen:
            seen.add(question)
            keep.append(i)
    return ds.select(keep)


def split(ds: Dataset, n_test: int, seed: int) -> tuple[Dataset, Dataset]:
    if not 0 < n_test < len(ds):
        raise ValueError(f"n_test must be between 1 and {len(ds) - 1}, got {n_test}")
    parts = dedupe_by_question(ds).train_test_split(test_size=n_test, seed=seed)
    train, test = parts["train"], parts["test"]
    overlap = set(train["question"]) & set(test["question"])
    if overlap:
        raise RuntimeError(f"{len(overlap)} questions appear in both splits")
    return train, test


def load_split(n_test: int = 200, seed: int = 42) -> tuple[Dataset, Dataset]:
    raw = load_dataset(DATASET, split="train", revision=REVISION)
    clean = raw.filter(lambda r: is_scorable(r["context"], r["answer"]))  # drop rows whose schema or reference SQL fails in SQLite
    train, test = split(clean, n_test, seed)
    log.info("raw=%d scorable=%d train=%d test=%d", len(raw), len(clean), len(train), len(test))
    return train, test


def save(train: Dataset, test: Dataset, out_dir: Path = OUT_DIR) -> dict:
    """Write jsonl files plus a manifest with checksums, so any later run can prove it used the same data."""
    out_dir.mkdir(exist_ok=True)
    manifest = {"dataset": DATASET, "revision": REVISION}
    for name, d in (("train", train), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps({k: r[k] for k in FIELDS}) + "\n" for r in d))
        manifest[name] = {"rows": len(d), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")  # libraries stay quiet
    log.setLevel(logging.INFO)  # only this module logs progress
    train, test = load_split()
    print(json.dumps(save(train, test), indent=2))
