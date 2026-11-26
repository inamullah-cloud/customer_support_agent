import json


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_chat_completes(client):
    r = client.post("/api/chat", json={"message": "Where is my order DF1617?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["category"] == "order_status"
    assert body["executed_tools"] == ["get_order_status"]
    assert body["thread_id"]
    assert body["observability"]["total_duration_ms"] >= 0


def test_chat_validation(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_approval_flow_approve(client):
    r = client.post("/api/chat", json={"message": "Please cancel my order DF1617", "thread_id": "api-1"})
    body = r.json()
    assert body["status"] == "awaiting_approval"
    assert body["approval"]["pending_tool_calls"][0] == {
        "name": "cancel_order", "args": {"order_id": "DF1617"}
    }

    # cannot send another message while paused
    assert client.post("/api/chat", json={"message": "hi", "thread_id": "api-1"}).status_code == 409

    done = client.post("/api/chat/api-1/approval", json={"approved": True}).json()
    assert done["status"] == "completed"
    assert done["executed_tools"] == ["cancel_order"]

    # nothing left to approve
    assert client.post("/api/chat/api-1/approval", json={"approved": True}).status_code == 409


def test_approval_flow_reject(client):
    client.post("/api/chat", json={"message": "Please cancel my order DF1617", "thread_id": "api-2"})
    done = client.post("/api/chat/api-2/approval", json={"approved": False}).json()
    assert done["status"] == "completed"
    assert done["executed_tools"] == []
    assert "not executed" in done["response"]


def test_thread_history(client):
    client.post("/api/chat", json={"message": "How is the weather in Mingora?", "thread_id": "api-3"})
    body = client.get("/api/chat/api-3").json()
    roles = [m["role"] for m in body["messages"]]
    assert roles[0] == "user" and "tool" in roles and roles[-1] == "assistant"
    assert client.get("/api/chat/does-not-exist").status_code == 404


def test_stream(client):
    with client.stream("POST", "/api/chat/stream", json={"message": "Where is my order DF1617?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())

    events = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
    assert events[0] == "thread" and events[-1] == "final"
    assert "node" in events

    final = json.loads([l for l in text.splitlines() if l.startswith("data: ")][-1][6:])
    assert final["status"] == "completed"


def test_memory_crud(client):
    base = "/api/users/u1/memories"
    created = client.post(base, json={"memory_type": "preference", "content": "Likes email"})
    assert created.status_code == 201
    memory_id = created.json()["memory_id"]

    assert client.post(base, json={"memory_type": "preference", "content": "likes EMAIL"}).status_code == 409
    assert client.get(f"{base}/{memory_id}").json()["content"] == "Likes email"
    assert len(client.get(base, params={"q": "email"}).json()) == 1
    assert client.get(base, params={"memory_type": "goal"}).json() == []

    patched = client.patch(f"{base}/{memory_id}", json={"content": "Likes SMS"})
    assert patched.json()["content"] == "Likes SMS"
    assert client.patch(f"{base}/{memory_id}", json={}).status_code == 422

    assert client.get(f"/api/users/other/memories/{memory_id}").status_code == 404
    assert client.delete(f"{base}/{memory_id}").status_code == 204
    assert client.get(f"{base}/{memory_id}").status_code == 404


def test_observability_dashboard(client):
    empty = client.get("/api/observability/dashboard").json()
    assert empty["summary"]["total_runs"] == 0

    client.post("/api/chat", json={"message": "Where is my order DF1617?", "thread_id": "obs-1"})
    dashboard = client.get("/api/observability/dashboard").json()
    assert dashboard["summary"]["total_runs"] == 1
    assert client.get("/api/observability/threads/obs-1").status_code == 200
    assert client.get("/api/observability/threads/nope").status_code == 404


def test_evaluation_endpoints(client):
    assert client.get("/api/evaluation/dashboard").status_code == 404

    first = client.post("/api/evaluation/run")
    assert first.status_code == 200
    body = first.json()
    assert body["metrics"]["total_cases"] == 4
    assert body["metrics"]["overall_success_rate"] == 100.0   # includes auto-approved cancel
    assert body["dashboard"]["health"]["status"] == "healthy"
    assert body["dashboard"]["regression"]["regression_detected"] is False

    second = client.post("/api/evaluation/run").json()
    assert second["dashboard"]["regression"]["message"] != "No previous runs available for comparison."

    assert client.get("/api/evaluation/reliability").json()["status"] == "healthy"


def test_chat_with_user_id_saves_and_recalls_memory(client):
    first = client.post("/api/chat", json={
        "message": "Where is my order DF1617? I prefer SMS updates.", "user_id": "u9",
    }).json()
    assert first["memories_saved"] == ["Prefers SMS updates"]

    listed = client.get("/api/users/u9/memories").json()
    assert [m["content"] for m in listed] == ["Prefers SMS updates"]

    second = client.post("/api/chat", json={"message": "Where is my order DF1617?", "user_id": "u9"}).json()
    assert second["memories_used"] == ["Prefers SMS updates"]
    assert second["memories_saved"] == []


def test_approval_stream(client):
    client.post("/api/chat", json={"message": "Please cancel my order DF1617", "thread_id": "api-s"})

    with client.stream("POST", "/api/chat/api-s/approval/stream", json={"approved": True}) as r:
        text = "".join(r.iter_text())

    events = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
    assert events[0] == "thread" and events[-1] == "final"
    assert "node" in events
    assert client.post("/api/chat/api-s/approval/stream", json={"approved": True}).status_code == 409


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
    assert client.get("/health").json()["status"] == "ok"   # API routes still win
