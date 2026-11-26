"""Shared fixtures: a scripted fake LLM so tests run offline."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import graph as agent_graph
from src.agent.memory import reset_memory_manager
from src.agent.state import (
    Classification,
    MemoryExtraction,
    NewMemory,
    Plan,
    Reflection,
    ToolSelection,
)
from src.api.dependencies import reset_container

# keyword -> (category, tool, tool args)
SCRIPT = {
    "cancel": ("order_status", "cancel_order", {"order_id": "DF1617"}),
    "where": ("order_status", "get_order_status", {"order_id": "DF1617"}),
    "price": ("product_information", "get_product_price", {"product_name": "Wireless Headphones"}),
    "weather": ("weather_information", "get_weather_info", {"place_name": "Mingora"}),
}


def _scenario(messages):
    text = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            text = m.content.lower()
    for keyword, scenario in SCRIPT.items():
        if keyword in text:
            return scenario
    return SCRIPT["where"]


class _Structured:
    def __init__(self, schema, llm):
        self.schema = schema.__name__
        self.llm = llm

    def invoke(self, messages):
        if self.schema == "MemoryExtraction":
            if self.llm.fail_extraction:
                raise RuntimeError("extractor unavailable")
            text = messages[-1].content.lower()
            if "prefer sms" in text:
                return MemoryExtraction(
                    memories=[NewMemory(memory_type="preference", content="Prefers SMS updates")]
                )
            return MemoryExtraction(memories=[])
        category, tool_name, _ = _scenario(messages)
        if self.schema == "Classification":
            return Classification(category=category)
        if self.schema == "ToolSelection":
            return ToolSelection(tools=[tool_name])
        if self.schema == "Plan":
            return Plan(steps=[f"call {tool_name}", "answer the user"])
        return Reflection(status="approved", feedback="looks good")


class _ToolBound:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, messages):
        self.llm.agent_prompts.append(messages[0].content)
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"Result: {messages[-1].content}")
        _, tool_name, args = _scenario(messages)
        return AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": args, "id": "call_1", "type": "tool_call"}],
        )


class FakeLLM:
    def __init__(self):
        self.agent_prompts: list[str] = []   # system prompts the agent node received
        self.fail_extraction = False

    def with_structured_output(self, schema):
        return _Structured(schema, self)

    def bind_tools(self, tools):
        return _ToolBound(self)

    def invoke(self, messages):
        return AIMessage(content="Sorry, I could not complete that request.")


@pytest.fixture(autouse=True)
def fake_llm():
    llm = FakeLLM()
    agent_graph.configure_models(llm)
    reset_memory_manager()
    reset_container()
    yield llm
    reset_memory_manager()
    reset_container()


@pytest.fixture
def client():
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client
