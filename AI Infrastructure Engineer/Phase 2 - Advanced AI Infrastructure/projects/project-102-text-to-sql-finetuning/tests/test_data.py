import pytest
from datasets import Dataset

import json

from src.data import FIELDS, dedupe_by_question, save, split


def make(n, dupes=0):
    rows = [{"question": f"q{i}", "context": "c", "answer": "a"} for i in range(n)]
    rows += [{"question": "q0", "context": "c2", "answer": "b"}] * dupes
    return Dataset.from_list(rows)


def test_dedupe_keeps_first_row_per_question():
    out = dedupe_by_question(make(5, dupes=3))
    assert len(out) == 5 and out[0]["context"] == "c"


def test_split_sizes_and_no_overlap():
    train, test = split(make(100, dupes=10), n_test=20, seed=1)
    assert len(test) == 20 and len(train) == 80
    assert not set(train["question"]) & set(test["question"])


def test_split_is_reproducible():
    assert split(make(100), 20, 7)[1]["question"] == split(make(100), 20, 7)[1]["question"]


def test_different_seed_gives_different_split():
    assert split(make(100), 20, 1)[1]["question"] != split(make(100), 20, 2)[1]["question"]


@pytest.mark.parametrize("n_test", [0, 100, 150])
def test_bad_test_size_rejected(n_test):
    with pytest.raises(ValueError):
        split(make(100), n_test, 1)


def test_save_checksum_ignores_column_order(tmp_path):
    a = Dataset.from_list([{"answer": "a", "question": "q", "context": "c"}])
    b = Dataset.from_list([{"question": "q", "context": "c", "answer": "a"}])
    ma = save(a, a, tmp_path / "a")
    mb = save(b, b, tmp_path / "b")
    assert ma["train"]["sha256"] == mb["train"]["sha256"]
    assert list(json.loads((tmp_path / "a" / "train.jsonl").read_text().splitlines()[0])) == list(FIELDS)
