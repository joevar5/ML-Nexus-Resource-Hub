"""Step 1.3: run SQL on a throwaway SQLite database and compare results (execution accuracy)."""
import random
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass

BASE_TEXT = ["alpha", "beta", "gamma", "delta", "epsilon"]
INT_TYPES = ("INT",)
REAL_TYPES = ("REAL", "FLOA", "DOUB", "NUM", "DEC")


def literals(query: str) -> tuple[list[str], list[float]]:
    """String and number literals in a query, used to seed table rows so filters can match."""
    quoted = r"'([^']*)'|\"([^\"]*)\""  # SQLite also accepts "double quotes" as strings
    strings = [a or b for a, b in re.findall(quoted, query)]
    bare = re.sub(quoted, "", query)
    numbers = [float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", bare)]
    return strings, numbers


def build_db(schema: str, seed_query: str = "", rows: int = 100, seed: int = 0) -> sqlite3.Connection:
    """In-memory database from a trusted `schema`, filled with deterministic rows.

    Values come from small pools plus the literals in `seed_query`, so a filter like
    WHERE name = 'zed' has rows to match. Without this, wrong queries often "match"
    the reference by both returning nothing.
    """
    con = sqlite3.connect(":memory:")
    con.executescript(schema)
    strings, numbers = literals(seed_query)
    text_pool = BASE_TEXT + strings * 3  # weight query literals so multi-condition filters still find rows
    num_pool = list(range(10)) + [int(n) + d for n in numbers for d in (-1, 0, 1)] * 3
    rng = random.Random(seed)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in tables:
        cols = con.execute(f'PRAGMA table_info("{t}")').fetchall()
        for _ in range(rows):
            vals = []
            for col in cols:
                ctype = (col[2] or "").upper()
                if any(k in ctype for k in INT_TYPES):
                    vals.append(rng.choice(num_pool))
                elif any(k in ctype for k in REAL_TYPES):
                    vals.append(float(rng.choice(num_pool)))
                else:
                    vals.append(str(rng.choice(num_pool + text_pool)))  # text columns hold numbers too (VARCHAR age)
            con.execute(f'INSERT INTO "{t}" VALUES ({",".join("?" * len(vals))})', vals)
    con.execute("PRAGMA query_only = ON")  # from here on, writes are rejected
    return con


def run_query(con: sqlite3.Connection, query: str, timeout_s: float = 2.0, max_rows: int = 1000):
    """Return (rows, None) on success or (None, error_text) on failure or timeout."""
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)  # nonzero return aborts the query
    try:
        return con.execute(query).fetchmany(max_rows), None
    except Exception as e:
        return None, str(e)
    finally:
        con.set_progress_handler(None, 0)


def run_sql(schema: str, query: str, **kw):
    con = build_db(schema, seed_query=query)
    try:
        return run_query(con, query, **kw)
    finally:
        con.close()


def is_scorable(schema: str, reference: str) -> bool:
    """True when the schema builds and the reference query runs. Dataset noise fails this."""
    try:
        con = build_db(schema, seed_query=reference)
    except sqlite3.Error:
        return False
    try:
        return run_query(con, reference)[1] is None
    finally:
        con.close()


@dataclass
class Score:
    correct: bool
    error: str | None
    informative: bool  # False when the reference returns no rows: many wrong queries also match then


def same_result(a, b, ordered: bool) -> bool:
    if a is None or b is None:
        return False
    return a == b if ordered else Counter(a) == Counter(b)


def score(schema: str, predicted: str, reference: str) -> Score:
    con = build_db(schema, seed_query=reference)  # both queries see the same rows
    try:
        want, _ = run_query(con, reference)
        got, err = run_query(con, predicted)
    finally:
        con.close()
    ordered = "order by" in reference.lower()
    return Score(correct=same_result(got, want, ordered), error=err, informative=bool(want))
