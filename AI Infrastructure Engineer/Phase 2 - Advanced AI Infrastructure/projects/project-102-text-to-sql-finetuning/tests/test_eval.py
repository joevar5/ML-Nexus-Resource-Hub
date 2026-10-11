from src.eval import evaluate, failure_type
from src.prompt import build_prompt, extract_sql

ROWS = [
    {"question": "names?", "context": "CREATE TABLE emp (name TEXT, age INT)", "answer": "SELECT name FROM emp"},
    {"question": "ages?", "context": "CREATE TABLE emp (name TEXT, age INT)", "answer": "SELECT age FROM emp"},
]


def test_prompt_ends_where_sql_starts():
    p = build_prompt("CREATE TABLE t (a INT)", "how many?")
    assert p.endswith("### SQL\n") and "CREATE TABLE t" in p and "how many?" in p


def test_extract_sql_handles_fences_prose_and_extra_statements():
    assert extract_sql("```sql\nSELECT a FROM t;\n```") == "SELECT a FROM t"
    assert extract_sql("SELECT a FROM t; SELECT b FROM t") == "SELECT a FROM t"
    assert extract_sql("SELECT a FROM t\n\n### Question\nnext") == "SELECT a FROM t"
    assert extract_sql("  SELECT a\nFROM t  ") == "SELECT a\nFROM t"


def test_failure_types():
    assert failure_type(True, None) == "correct"
    assert failure_type(False, None) == "wrong_result"
    assert failure_type(False, "no such column: x") == "unknown_column"
    assert failure_type(False, "no such table: x") == "unknown_table"
    assert failure_type(False, 'near "SELEC": syntax error') == "syntax_error"
    assert failure_type(False, "boom") == "other_error"


def test_perfect_model_scores_one():
    answers = {build_prompt(r["context"], r["question"]): r["answer"] for r in ROWS}
    rep = evaluate(ROWS, lambda ps: [answers[p] for p in ps])
    assert rep["execution_accuracy"] == 1.0 and rep["invalid_sql_rate"] == 0.0


def test_broken_model_scores_zero_and_reports_failures():
    rep = evaluate(ROWS, lambda ps: ["SELEC nonsense" for _ in ps])
    assert rep["execution_accuracy"] == 0.0 and rep["invalid_sql_rate"] == 1.0
    assert rep["failure_counts"] == {"syntax_error": 2}


def test_wrong_column_model_is_wrong_not_invalid():
    rep = evaluate(ROWS, lambda ps: ["SELECT age FROM emp" for _ in ps])  # right for row 2 only
    assert rep["execution_accuracy"] == 0.5 and rep["invalid_sql_rate"] == 0.0
    assert rep["failure_counts"]["wrong_result"] == 1


def test_batching_covers_every_row():
    rep = evaluate(ROWS * 5, lambda ps: ["SELECT name FROM emp" for _ in ps], batch_size=3)
    assert rep["n"] == 10
