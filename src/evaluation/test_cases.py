"""Evaluation case schema and the evaluation dataset."""

from typing import List

from pydantic import BaseModel


class EvaluationCase(BaseModel):
    query: str
    expected_category: str
    expected_tools: List[str]
    requires_approval: bool = False
    thread_id: str = "eval_thread"


EVAL_DATASET: List[EvaluationCase] = [
    EvaluationCase(
        query="Where is my order DF1617?",
        expected_category="order_status",
        expected_tools=["get_order_status"],
        requires_approval=False,
        thread_id="eval_001_order_status",
    ),
    EvaluationCase(
        query="What is the price of the Wireless Headphones?",
        expected_category="product_information",
        expected_tools=["get_product_price"],
        requires_approval=False,
        thread_id="eval_002_product_price",
    ),
    EvaluationCase(
        query="How is the weather in Mingora?",
        expected_category="weather_information",
        expected_tools=["get_weather_info"],
        requires_approval=False,
        thread_id="eval_003_weather",
    ),
    EvaluationCase(
        query="Please cancel my order DF1617.",
        expected_category="order_status",
        expected_tools=["cancel_order"],
        requires_approval=True,
        thread_id="eval_004_cancel_order",
    ),
]
