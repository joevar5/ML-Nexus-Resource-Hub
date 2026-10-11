# Lesson 10: Text-to-SQL and Evaluation by Execution

[Lesson 05](./05-llm-fine-tuning-infrastructure.md) taught you to fine-tune a model, and its evaluation step scored answers by exact match. That works when there is one right string. SQL isn't like that: `SELECT name FROM users WHERE age > 30` and `SELECT u.name FROM users u WHERE u.age > 30` are different text with the same answer. This lesson covers how to score SQL by what it *returns*, how to find out why a model fails, and how to stop it from running anything dangerous.

It is the working guide for [Project 102](../../projects/project-102-text-to-sql-finetuning/README.md) and the capstone.

**Prerequisites:** [Lesson 05](./05-llm-fine-tuning-infrastructure.md) (fine-tuning and held-out evaluation), basic SQL.

### Contents

1. [What Text-to-SQL Is All About](#what-text-to-sql-is-all-about)
2. [Core Logic: Score the Result, Not the Text](#core-logic-score-the-result-not-the-text)
3. [Step-by-Step: Build the Evaluation](#step-by-step-build-the-evaluation)
4. [Practical Exercise](#practical-exercise)
5. [Key Takeaways](#key-takeaways)
6. [Additional Resources](#additional-resources)

---

## What Text-to-SQL Is All About

A business user asks "how many orders did we get last month?" and an analyst writes SQL to answer it. A text-to-SQL system does the analyst's job: it takes the question and the database **schema** (table and column definitions) and returns a query.

Why teams fine-tune a small model for this instead of calling a large hosted one:

| Reason | What it means |
|---|---|
| **Privacy** | Schemas and questions stay on your own hardware |
| **Cost** | A small model costs far less per query at volume |
| **Latency** | A small model on your own GPU is fast and predictable |
| **Control** | You can measure it, retrain it on its failures, and block unsafe output |

The risk is the reverse: a wrong query returns a confident, wrong number, and a malicious one could delete data. So the two jobs of this lesson are **measuring correctness honestly** and **making unsafe queries impossible**.

---

## Core Logic: Score the Result, Not the Text

### Execution Accuracy

A prediction counts as correct when running it on the database returns the **same result** as running the reference query:

```
execution accuracy = (questions where result(predicted) == result(reference)) / total questions
```

Exact string match under-counts, because many different queries are right. Execution accuracy fixes that, but it has its own traps:

| Trap | Why it misleads | Handling |
|---|---|---|
| **Row order** | Without `ORDER BY`, order is arbitrary | Compare as unordered multisets unless the reference has `ORDER BY` |
| **Lucky matches** | A wrong query can return the same rows on a tiny database (for example both return empty) | Use a database with enough varied data, and skip or flag questions whose answer is empty |
| **Column names and order** | `SELECT a, b` vs `SELECT b, a` | Compare values by position, ignore column aliases |
| **Errors** | An invalid query has no result | Count it as wrong, and keep the error text for failure analysis |
| **Slow queries** | A missing join condition can run forever | Set a timeout and count it as wrong |

### Where Failures Come From

Wrong answers fall into a small number of groups, and the group tells you the fix:

| Failure type | Example | Likely fix |
|---|---|---|
| Invalid SQL | Syntax error, unknown column | More training data, a retry loop with the error message |
| Wrong table or column | Uses `order_date` when the column is `created_at` | Better schema in the prompt, schema retrieval for large databases |
| Wrong join | Missing or wrong join condition | Training examples that need joins |
| Wrong aggregation | `COUNT` vs `COUNT(DISTINCT ...)`, missing `GROUP BY` | Targeted examples |
| Wrong filter | Wrong date range or value | Examples with filters and dates |

Always report accuracy **broken down by difficulty** (single table, joins, aggregations), not only one overall number.

### Safety Is a Separate Problem

A fine-tuned model won't reliably refuse to write `DROP TABLE`. Never rely on the model for safety. Enforce it outside the model, in layers:

1. **Read-only database connection.** The strongest layer. The database itself rejects writes.
2. **Parse and allow-list.** Accept only a single `SELECT` statement.
3. **Timeout and row limit.** Stops runaway queries.
4. **Table allow-list.** Block restricted tables.

---

## Step-by-Step: Build the Evaluation

### Step 1: Load and Split the Data

Use a public text-to-SQL dataset from the Hugging Face Hub (for example `b-mc2/sql-create-context`, which pairs a question, a `CREATE TABLE` schema, and an answer query; check its fields and license before use). Hold out about 200 rows that the model never trains on.

```python
from datasets import load_dataset

ds = load_dataset("b-mc2/sql-create-context", split="train").train_test_split(test_size=200, seed=42)
train, test = ds["train"], ds["test"]
```

Dedupe by question before splitting, so the same question can't sit in both sets.

### Step 2: Run SQL Safely

Build an in-memory SQLite database from the schema, open it read-only for the model's query, and capture errors instead of crashing:

```python
import sqlite3
import time

def run_sql(schema: str, query: str, timeout_s: float = 2.0, max_rows: int = 1000):
    """Return (rows, None) on success or (None, error_text) on failure."""
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(schema)                 # trusted reference schema only
        con.execute("PRAGMA query_only = ON")     # reject writes from here on
        deadline = time.monotonic() + timeout_s
        con.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)  # abort when over time
        return con.execute(query).fetchmany(max_rows), None
    except Exception as e:
        return None, str(e)
    finally:
        con.close()
```

`PRAGMA query_only` makes SQLite refuse writes on that connection. For a real database, create a database user with `SELECT` permission only, which is the same idea at the server level. Datasets like this one give only a schema, so the tables are empty. Insert a few rows per table before scoring, or your "same result" check will compare empty lists and pass wrong queries.

### Step 3: Compare Results

```python
from collections import Counter

def same_result(a, b, ordered: bool) -> bool:
    if a is None or b is None:
        return False
    return a == b if ordered else Counter(a) == Counter(b)

def is_correct(schema, predicted, reference):
    got, err = run_sql(schema, predicted)
    want, _ = run_sql(schema, reference)
    ordered = "order by" in reference.lower()
    return same_result(got, want, ordered), err
```

A small self-check keeps the scorer honest. Test it before you trust any model number:

```python
schema = "CREATE TABLE t(a INT, b INT); INSERT INTO t VALUES (1,2),(3,4);"
assert is_correct(schema, "SELECT a FROM t", "SELECT a FROM t")[0]
assert is_correct(schema, "SELECT t.a FROM t", "SELECT a FROM t")[0]       # different text, same result
assert not is_correct(schema, "SELECT b FROM t", "SELECT a FROM t")[0]     # different result
assert not is_correct(schema, "SELEC a FROM t", "SELECT a FROM t")[0]      # invalid SQL counts as wrong
```

### Step 4: Score the Baseline, Then the Fine-Tuned Model

Run the untrained model on the held-out set first. You need a number to beat. Then run the fine-tuned model on the same rows with the same prompt format, and record both in a table:

| | Baseline | Fine-tuned |
|---|---|---|
| Execution accuracy | | |
| Invalid SQL rate | | |

Generate with greedy decoding (`temperature=0`) so results are repeatable, and set a stop token so the model ends after the query.

### Step 5: Analyze the Failures

Keep every wrong prediction with its error text, then group them by the failure types above. Fix the **largest group** first, retrain, and re-measure on the same held-out set. Never tune against the held-out set itself: if you do, set aside a second untouched set for the final number.

### Step 6: Add Guardrails

Validate before running anything the model wrote. Parsing with a SQL library is more reliable than string matching, which a comment or odd spacing defeats. This example uses the `sqlglot` package:

```python
import sqlglot
from sqlglot import exp

def is_safe(query: str, allowed_tables: set[str]) -> bool:
    try:
        trees = sqlglot.parse(query, read="sqlite")
    except sqlglot.errors.ParseError:
        return False
    if len(trees) != 1 or not isinstance(trees[0], exp.Select):
        return False                                   # one SELECT statement only
    tables = {t.name for t in trees[0].find_all(exp.Table)}
    return tables <= allowed_tables
```

```python
assert is_safe("SELECT a FROM t", {"t"})
assert not is_safe("DROP TABLE t", {"t"})
assert not is_safe("SELECT a FROM t; DELETE FROM t", {"t"})
assert not is_safe("SELECT a FROM secrets", {"t"})
```

This check is one layer. The read-only connection from Step 2 stays on regardless, because a parser can miss something.

### Step 7: Retry With the Error (Optional Agent Loop)

When a query fails, send the error back and let the model try again once or twice:

```python
def generate_with_retry(ask_model, question, schema, retries=2):
    prompt = f"Schema:\n{schema}\nQuestion: {question}\nSQL:"
    for _ in range(retries + 1):
        sql = ask_model(prompt)
        rows, err = run_sql(schema, sql)
        if err is None:
            return sql
        prompt += f"\n{sql}\nThat failed with: {err}\nFixed SQL:"
    return sql
```

Count retries in your results. A retry costs latency and tokens, so report accuracy with and without it.

---

## Practical Exercise

Build the scorer and baseline for your own model:

1. Load the dataset, dedupe, and hold out 200 rows
2. Write `run_sql` and `same_result`, and make the four self-check asserts pass
3. Insert sample rows per table so results aren't empty
4. Score an untrained small code model and write the baseline down
5. Group its wrong answers by failure type and name the largest group

*Hint: if your baseline is near 0%, check the prompt format and stop token before blaming the model. A model that keeps generating after the query will fail every comparison.*

---

## Key Takeaways

1. Score SQL by what it returns (execution accuracy), not by its text
2. Compare results unordered unless the reference has `ORDER BY`, and populate tables so wrong queries can't match on empty results
3. Invalid SQL and timeouts count as wrong, and keep their error text
4. Report accuracy by difficulty, and group failures by type: the biggest group is your next fix
5. Never tune against the held-out set, or your final number is meaningless
6. Enforce safety outside the model: read-only connection first, then parse and allow-list, timeout, and table limits
7. A retry-with-error loop often helps, but it costs latency, so measure it

---

## Additional Resources

- [Spider benchmark](https://yale-lily.github.io/spider) and [BIRD benchmark](https://bird-bench.github.io/): standard text-to-SQL evaluation sets
- [sqlglot](https://github.com/tobymao/sqlglot): SQL parser and transpiler
- [SQLite PRAGMA query_only](https://www.sqlite.org/pragma.html#pragma_query_only)
- [Lesson 05: LLM Fine-Tuning Infrastructure](./05-llm-fine-tuning-infrastructure.md)

---

**Next Lesson:** [11-autoscaling-on-queue-depth-keda.md](./11-autoscaling-on-queue-depth-keda.md): Autoscaling LLM Serving on Queue Depth
