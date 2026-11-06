"""The evaluation check for the ollama-model skill's evaluate.py, which loads it by path:

    uv run python <ollama-model>/scripts/evaluate.py --eval eval/eval.jsonl --check scripts/eval_check.py:same_rows ...

Run evaluate.py with this repo's environment (``uv run``), the check executes the queries with beanquery.
"""

from bql_lora.evaluation import same_rows

__all__ = ["same_rows"]
