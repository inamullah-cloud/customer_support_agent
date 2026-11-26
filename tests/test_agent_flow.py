"""End-to-end graph tests with the fake LLM."""

from langgraph.types import Command

from src.agent.graph import app


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _run(query, thread_id):
    app.invoke({"messages": [("user", query)], "retry_count": 0, "tool_iterations": 0},
               config=_config(thread_id))
    return app.get_state(_config(thread_id))


def test_read_only_tool_runs_without_approval():
    snapshot = _run("Where is my order DF1617?", "flow-1")
    values = snapshot.values

    assert snapshot.interrupts == ()
    assert values["category"] == "order_status"
    assert values["executed_tools"] == ["get_order_status"]
    assert "out for delivery" in values["response"]
    assert values["observability_report"] is not None
    assert "classify" in values["observability_report"].node_durations


def test_cancel_pauses_then_approval_executes():
    snapshot = _run("Please cancel my order DF1617", "flow-2")
    assert len(snapshot.interrupts) == 1
    assert snapshot.interrupts[0].value["pending_tool_calls"][0]["name"] == "cancel_order"

    app.invoke(Command(resume=True), config=_config("flow-2"))
    values = app.get_state(_config("flow-2")).values
    assert values["executed_tools"] == ["cancel_order"]
    assert "cancelled" in values["response"]


def test_cancel_rejected():
    _run("Please cancel my order DF1617", "flow-3")
    app.invoke(Command(resume="no"), config=_config("flow-3"))
    values = app.get_state(_config("flow-3")).values

    assert values["executed_tools"] == []
    assert values["approval_status"] == "rejected"
    assert values["observability_report"] is not None  # rejection path is observed too


def test_trace_resets_between_turns():
    _run("Where is my order DF1617?", "flow-4")
    first = len(app.get_state(_config("flow-4")).values["execution_trace"])
    _run("What is the price of the headphones?", "flow-4")
    second = len(app.get_state(_config("flow-4")).values["execution_trace"])
    assert abs(second - first) < first  # not accumulating across turns
