"""Sanity check on the real test set: every reference query must run, and most should return rows."""
import json
from pathlib import Path

from src.run_sql import build_db, run_query

rows = [json.loads(l) for l in (Path(__file__).resolve().parent.parent / "data" / "test.jsonl").read_text().splitlines()]
errors, empty = [], 0
for r in rows:
    con = build_db(r["context"], seed_query=r["answer"])
    got, err = run_query(con, r["answer"])
    con.close()
    if err:
        errors.append((r["answer"], err))
    elif not got:
        empty += 1
print(f"reference queries: {len(rows)}  errors: {len(errors)}  empty results: {empty}")
for q, e in errors[:5]:
    print("  ERR", e, "|", q)
