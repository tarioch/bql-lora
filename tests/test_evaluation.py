"""The training data and the evaluation set: what the unsloth-lora and ollama-model skills rely on."""

import json
from pathlib import Path

from bql_lora.evaluation import EVAL_LEDGER_SEED, extract_sql, same_rows

ROOT = Path(__file__).resolve().parent.parent


def _lines(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_training_data_has_no_system_message():
    # See README, "Why the data has no system prompt": the chat template's default system line is what the
    # model trains on, and the skills' run.json records it for the Modelfile.
    for name in ("train.jsonl", "val.jsonl"):
        for row in _lines(ROOT / "data" / name):
            assert [m["role"] for m in row["messages"]] == ["user", "assistant"], f"data/{name} has a system message"


def test_eval_set_is_held_out_from_training():
    training = {
        row["messages"][0]["content"] for name in ("train.jsonl", "val.jsonl") for row in _lines(ROOT / "data" / name)
    }
    eval_questions = [row["messages"][0]["content"] for row in _lines(ROOT / "eval" / "eval.jsonl")]
    assert eval_questions
    assert not training & set(eval_questions)


def test_every_expected_answer_passes_the_check():
    for row in _lines(ROOT / "eval" / "eval.jsonl"):
        prompt, expected = row["messages"][:-1], row["messages"][-1]["content"]
        assert same_rows(prompt, expected, expected), row["messages"][0]["content"]


def test_check_rejects_wrong_or_missing_queries():
    expected = "```sql\nSELECT account, sum(position) WHERE account ~ '^Expenses' GROUP BY account\n```"
    reordered = "```sql\nSELECT account, sum(position) AS total WHERE account ~ '^Expenses' GROUP BY 1\n```\n\nok"
    assert same_rows([], expected, reordered)  # an alias or another spelling is not a mistake
    assert not same_rows(
        [], expected, "```sql\nSELECT account, sum(position) WHERE account ~ '^Income' GROUP BY account\n```"
    )
    assert not same_rows([], expected, "```sql\nSELECT nonsense FROM\n```")
    assert not same_rows([], expected, "SELECT account WHERE account ~ '^Expenses'")  # no sql block


def test_row_order_counts_only_with_order_by():
    asc = "```sql\nSELECT account GROUP BY account ORDER BY account\n```"
    desc = "```sql\nSELECT account GROUP BY account ORDER BY account DESC\n```"
    unordered = "```sql\nSELECT account GROUP BY account\n```"
    assert not same_rows([], asc, desc)  # same rows, other order
    assert same_rows([], unordered, desc)


def test_a_string_where_an_amount_is_expected_is_wrong_not_an_error():
    # Amount.__eq__ raises AttributeError for a string, which crashed the check instead of scoring the answer
    expected = "```sql\nSELECT date, currency, amount FROM #prices ORDER BY date DESC LIMIT 1\n```"
    answer = "```sql\nSELECT date, currency, currency FROM #prices ORDER BY date DESC LIMIT 1\n```"
    assert not same_rows([], expected, answer)
    assert same_rows([], expected, expected)


def test_extract_sql_takes_the_first_block():
    assert extract_sql("text\n```sql\nSELECT 1\n```\nmore\n```sql\nSELECT 2\n```") == "SELECT 1"
    assert extract_sql("no block") is None


def test_eval_ledger_is_not_a_training_ledger():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_eval import training_ledger_seeds

    assert EVAL_LEDGER_SEED not in training_ledger_seeds(1, 350)


def test_validation_prompts_are_not_in_training():
    # the unsloth-lora skill's check_data.py fails on these: the model has seen the prompt, so its validation
    # loss there measures recall
    train = {json.dumps(row["messages"][:-1]) for row in _lines(ROOT / "data" / "train.jsonl")}
    val = [json.dumps(row["messages"][:-1]) for row in _lines(ROOT / "data" / "val.jsonl")]
    assert val
    assert not train & set(val)


def test_split_keeps_each_prompt_on_one_side():
    import importlib.util

    spec = importlib.util.spec_from_file_location("generate_dataset", ROOT / "scripts" / "generate_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def ex(q, a):
        return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}

    examples = [ex("a", "1"), ex("b", "1"), ex("c", "1"), ex("a", "2"), ex("d", "1"), ex("b", "2")]
    val, train = module.split_by_prompt(examples, 3)
    assert [e["messages"][0]["content"] for e in val] == ["a", "a", "b", "b"]  # whole prompts, at least 3 examples
    assert [e["messages"][0]["content"] for e in train] == ["c", "d"]
    val, train = module.split_by_prompt(examples, 1)
    assert [e["messages"][0]["content"] for e in val] == ["a", "a"]
    assert [e["messages"][0]["content"] for e in train] == ["b", "c", "d", "b"]  # training keeps the shuffled order
