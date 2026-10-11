"""Score a model on the held-out test set by execution accuracy."""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable

from src.prompt import build_prompt, extract_sql
from src.run_sql import score

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "data" / "test.jsonl"


def failure_type(correct: bool, error: str | None) -> str:
    if correct:
        return "correct"
    if error is None:
        return "wrong_result"
    e = error.lower()
    if "no such column" in e:
        return "unknown_column"
    if "no such table" in e:
        return "unknown_table"
    if "syntax error" in e:
        return "syntax_error"
    return "other_error"


def evaluate(rows: list[dict], generate: Callable[[list[str]], list[str]], batch_size: int = 8) -> dict:
    """`generate` maps a list of prompts to a list of raw model outputs."""
    results = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        outputs = generate([build_prompt(r["context"], r["question"]) for r in batch])
        for r, raw in zip(batch, outputs):
            sql = extract_sql(raw)
            s = score(r["context"], sql, r["answer"])
            results.append({"question": r["question"], "predicted": sql, "reference": r["answer"],
                            "correct": s.correct, "error": s.error, "informative": s.informative,
                            "failure": failure_type(s.correct, s.error)})
    n = len(results)
    counts = Counter(r["failure"] for r in results)
    return {"n": n,
            "execution_accuracy": counts["correct"] / n if n else 0.0,
            "invalid_sql_rate": sum(1 for r in results if r["error"]) / n if n else 0.0,
            "uninformative_share": sum(1 for r in results if not r["informative"]) / n if n else 0.0,
            "failure_counts": dict(counts), "results": results}


def load_test(path: Path = TEST, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    return rows[:limit] if limit else rows


def write_report(report: dict, out: Path, tag: str) -> None:
    """Save per-question results and the summary. `report` loses its per-question list."""
    out.mkdir(parents=True, exist_ok=True)  # parents: `results/smoke` must work when `results` does not exist yet
    (out / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in report.pop("results")))
    (out / f"{tag}.summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model id, or 'reference' to check the loop with the right answers")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    rows = load_test(limit=a.limit)
    if a.model == "reference":
        lookup = {build_prompt(r["context"], r["question"]): r["answer"] for r in rows}
        gen = lambda prompts: [lookup[p] for p in prompts]
    else:
        from src.generate import make_generator  # needs torch and transformers (Kaggle)
        gen = make_generator(a.model)
    report = evaluate(rows, gen)
    write_report(report, Path(a.out), a.model.replace("/", "_"))
    print(json.dumps(report, indent=2))
