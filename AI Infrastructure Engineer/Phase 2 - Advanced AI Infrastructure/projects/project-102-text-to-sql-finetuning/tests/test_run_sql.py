from src.run_sql import is_scorable, run_sql, score

SCHEMA = "CREATE TABLE emp (name TEXT, age INT, dept TEXT);"


def test_identical_query_is_correct():
    assert score(SCHEMA, "SELECT name FROM emp", "SELECT name FROM emp").correct


def test_different_text_same_result_is_correct():
    assert score(SCHEMA, "SELECT emp.name FROM emp", "SELECT name FROM emp").correct


def test_wrong_column_is_wrong():
    assert not score(SCHEMA, "SELECT dept FROM emp", "SELECT name FROM emp").correct


def test_invalid_sql_is_wrong_and_reports_error():
    s = score(SCHEMA, "SELEC name FROM emp", "SELECT name FROM emp")
    assert not s.correct and s.error


def test_row_order_ignored_without_order_by():
    assert score(SCHEMA, "SELECT age FROM emp ORDER BY age DESC", "SELECT age FROM emp").correct


def test_row_order_enforced_with_order_by():
    assert not score(SCHEMA, "SELECT age FROM emp ORDER BY age DESC", "SELECT age FROM emp ORDER BY age ASC").correct


def test_reference_literal_makes_rows_exist():
    s = score(SCHEMA, "SELECT name FROM emp WHERE name = 'zed'", "SELECT name FROM emp WHERE name = 'zed'")
    assert s.correct and s.informative


def test_wrong_filter_is_caught_not_matched_on_empty():
    assert not score(SCHEMA, "SELECT name FROM emp WHERE name = 'other'", "SELECT name FROM emp WHERE name = 'zed'").correct


def test_numeric_text_column_compares():
    schema = "CREATE TABLE head (age VARCHAR)"
    assert score(schema, "SELECT COUNT(*) FROM head WHERE age > 56", "SELECT COUNT(*) FROM head WHERE age > 56").correct


def test_writes_are_rejected():
    rows, err = run_sql(SCHEMA, "DELETE FROM emp")
    assert rows is None and err


def test_runaway_query_times_out():
    q = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) SELECT count(*) FROM r"
    rows, err = run_sql(SCHEMA, q, timeout_s=0.5)
    assert rows is None and err


def test_is_scorable_accepts_good_row():
    assert is_scorable(SCHEMA, "SELECT name FROM emp")


def test_is_scorable_rejects_bad_reference_and_bad_schema():
    assert not is_scorable(SCHEMA, "SELECT name FROM emp OFFSET 5")
    assert not is_scorable("CREATE TABLE t (a INT); CREATE TABLE t (b INT)", "SELECT a FROM t")
    assert not is_scorable("CREATE TABLE order (a INT)", "SELECT a FROM t")
