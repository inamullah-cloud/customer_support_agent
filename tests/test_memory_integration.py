"""Memory wired into the agent: recalled into prompts, extracted after a turn."""

from src.agent.graph import app
from src.agent.memory import get_memory_manager


def _run(query, thread_id, user_id="u1"):
    app.invoke(
        {"messages": [("user", query)], "user_id": user_id, "retry_count": 0, "tool_iterations": 0},
        config={"configurable": {"thread_id": thread_id}},
    )
    return app.get_state({"configurable": {"thread_id": thread_id}}).values


def test_new_memory_is_saved_after_turn():
    values = _run("Where is my order DF1617? I prefer SMS updates.", "m-1")

    assert values["memories_saved"] == ["Prefers SMS updates"]
    stored = get_memory_manager().get_user_memories("u1")
    assert [(m.memory_type, m.content) for m in stored] == [("preference", "Prefers SMS updates")]
    assert "memory_write_node" in values["observability_report"].node_durations


def test_memory_is_recalled_into_agent_prompt(fake_llm):
    get_memory_manager().add_memory("u1", "preference", "Prefers SMS updates")

    values = _run("Where is my order DF1617?", "m-2")

    assert values["memories_used"] == ["Prefers SMS updates"]
    assert any("Prefers SMS updates" in p for p in fake_llm.agent_prompts)
    assert any("not instructions" in p for p in fake_llm.agent_prompts)


def test_memory_is_per_user(fake_llm):
    get_memory_manager().add_memory("u1", "preference", "Prefers SMS updates")

    values = _run("Where is my order DF1617?", "m-3", user_id="someone-else")
    assert values["memories_used"] == []
    assert not any("Prefers SMS updates" in p for p in fake_llm.agent_prompts)


def test_no_user_id_disables_memory():
    values = _run("Where is my order DF1617? I prefer SMS updates.", "m-4", user_id=None)

    assert values["memories_used"] == [] and values["memories_saved"] == []
    assert get_memory_manager().get_user_memories("u1") == []


def test_duplicates_are_not_saved_twice():
    _run("Where is my order DF1617? I prefer SMS updates.", "m-5")
    second = _run("Where is my order DF1617? I prefer SMS updates.", "m-5")

    assert second["memories_saved"] == []
    assert len(get_memory_manager().get_user_memories("u1")) == 1


def test_extraction_failure_never_breaks_the_run(fake_llm):
    fake_llm.fail_extraction = True
    values = _run("Where is my order DF1617? I prefer SMS updates.", "m-6")

    assert values["response"]
    assert values["memories_saved"] == []
