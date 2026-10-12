import json

from src.prompt import build_prompt
from src.train import build_example, load_examples


class FakeTok:  # one token per character; eos is 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


ROW = {"question": "how many?", "context": "CREATE TABLE t (a INT)", "answer": "SELECT COUNT(*) FROM t"}


def test_only_the_sql_and_stop_token_are_learned():
    ex = build_example(FakeTok(), ROW, max_len=1000)
    n_prompt = len(build_prompt(ROW["context"], ROW["question"]))
    assert ex["labels"][:n_prompt] == [-100] * n_prompt
    assert ex["labels"][n_prompt:] == ex["input_ids"][n_prompt:]
    assert ex["input_ids"][-1] == 0 and len(ex["input_ids"]) == n_prompt + len(ROW["answer"]) + 1


def test_too_long_example_is_dropped_not_cut():
    assert build_example(FakeTok(), ROW, max_len=10) is None


def test_val_and_train_do_not_overlap(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("".join(json.dumps({**ROW, "question": f"q{i}"}) + "\n" for i in range(30)))
    train, val = load_examples(FakeTok(), n_train=20, n_val=5, max_len=1000, seed=1, path=p)
    assert (len(train), len(val)) == (20, 5)
    assert not {tuple(e["input_ids"]) for e in train} & {tuple(e["input_ids"]) for e in val}
